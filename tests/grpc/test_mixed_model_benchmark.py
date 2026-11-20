import asyncio
import logging
import socket
from functools import partial
from pathlib import Path

import aiohttp
import pytest
import uvicorn
from aiohttp import web
from aiohttp.test_utils import TestServer

from benchmarking.mixed_model.report import summary
from benchmarking.mixed_model.run import Scenario, check_output, database, run
from mlserver.grpc import dataplane_pb2 as pb
from mlserver.parallel import InferencePoolRegistry
from mlserver.rest.app import create_app


def scenario(name="A", version="1", **kwargs):
    tensor = dict(name="total", shape=[1, 1], datatype="FP32", data=[3])
    config = dict(
        rate=4,
        duration_s=1,
        warmup_s=0,
        deadline_s=1,
        concurrency=1,
        seed=1,
        deployment={"workers": 1},
        initial_load=[],
        actions=[],
        models=[
            dict(
                name=name,
                version=version,
                artifact_sha256="0" * 64,
                request={
                    "inputs": [
                        dict(name="input-0", shape=[1, 2], datatype="FP32", data=[1, 2])
                    ]
                },
                expected=[tensor],
            )
        ],
    )
    return Scenario.model_validate(dict(config, **kwargs))


def reply(model, version, request_id, value):
    return pb.ModelInferResponse(
        id=request_id,
        model_name=model,
        model_version=version,
        outputs=[
            pb.ModelInferResponse.InferOutputTensor(
                name="total",
                shape=[1, 1],
                datatype="FP32",
                contents=pb.InferTensorContents(fp32_contents=[value]),
            )
        ],
    )


@pytest.mark.parametrize(
    "model,version,value", [("B", "1", 3), ("A", "2", 3), ("A", "1", 4)]
)
def test_distinguishes_model_version_and_result(model, version, value):
    target = scenario().models[0]
    check_output(reply("A", "1", "id", 3), target, "id")
    with pytest.raises(ValueError):
        check_output(reply(model, version, "id", value), target, "id")


async def test_generator_accounts_for_arrivals_while_response_is_delayed(tmp_path):
    release = asyncio.Event()

    class Client:
        async def ModelInfer(self, request, timeout):
            await release.wait()
            return reply(request.model_name, request.model_version, request.id, 3)

    db = database(tmp_path / "run.db")
    job = asyncio.create_task(run(scenario(), Client(), db, "test"))
    try:
        async with asyncio.timeout(8):
            while not db.execute(
                "SELECT count(*)=3 FROM attempt WHERE outcome LIKE 'generator_%'"
            ).fetchone()[0]:
                await asyncio.sleep(0.01)
            release.set()
            await job
        all_results = summary(db)["models"][0]["windows"]["all"]
        assert all_results["offered"] == 4
        assert all_results["outcomes"] == {"correct": 1, "generator_full": 3}
        assert (
            db.execute(
                "SELECT COUNT(*) FROM attempt WHERE submitted IS NOT NULL"
            ).fetchone()[0]
            == 1
        )
    finally:
        release.set()
        await job
        db.close()


async def _gate_worker(model, directory):
    # Use the normal worker and model. Only the test controls when predict returns.
    logger = logging.getLogger("mlserver.parallel.execution")
    handler = logging.FileHandler(directory / "worker.log")
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    predict = model.predict

    async def gated(request):
        (directory / "entered").touch()
        async with asyncio.timeout(12):
            while not (directory / "release").exists():
                await asyncio.sleep(0.01)
        return await predict(request)

    setattr(model, "predict", gated)
    return model


@pytest.fixture
async def inference_pool_registry(settings, tmp_path, prometheus_registry):
    settings.parallel_workers = 1
    pool = InferencePoolRegistry(
        settings, on_worker_load=[partial(_gate_worker, directory=tmp_path)]
    )
    await pool.start()
    try:
        yield pool
    finally:
        (tmp_path / "release").touch()
        await pool.close()


@pytest.fixture
def sum_model_settings(sum_model_settings):
    sum_model_settings.max_batch_size = 0
    return sum_model_settings


@pytest.fixture
def settings(settings, unused_tcp_port):
    settings.grpc_port = unused_tcp_port
    return settings


@pytest.fixture(params=["grpc", "rest"])
async def benchmark_client(
    request, inference_service_stub, settings, data_plane, model_repository_handlers
):
    if request.param == "grpc":
        yield inference_service_stub
        return

    app = create_app(settings, data_plane, model_repository_handlers)
    server = uvicorn.Server(
        uvicorn.Config(app, log_config=None, access_log=False, lifespan="off")
    )
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        address = f"http://127.0.0.1:{sock.getsockname()[1]}"
        serving = asyncio.create_task(server.serve(sockets=[sock]))
        try:
            async with asyncio.timeout(8):
                while not server.started:
                    if serving.done():
                        await serving
                    await asyncio.sleep(0.01)
            async with aiohttp.ClientSession(address) as client:
                yield client
        finally:
            server.should_exit = True
            await serving


async def test_client_deadline_precedes_real_worker_completion(
    tmp_path: Path, benchmark_client, sum_model_settings
):
    config = scenario(
        sum_model_settings.name,
        sum_model_settings.version,
        rate=1,
        duration_s=0.1,
        deadline_s=1,
    )
    db = database(tmp_path / "run.db")
    task = asyncio.create_task(run(config, benchmark_client, db, "loopback"))
    log = tmp_path / "worker.log"
    try:
        async with asyncio.timeout(8):
            while not (tmp_path / "entered").exists():
                await asyncio.sleep(0.01)
            await task
            events = summary(db, log)["observed_worker_events"]
            assert [(e["client_outcome"], e["worker_event"]) for e in events] == [
                ("deadline", "started")
            ]
            (tmp_path / "release").touch()
            while '"event": "returned"' not in log.read_text():
                await asyncio.sleep(0.01)
            assert {
                e["worker_event"] for e in summary(db, log)["observed_worker_events"]
            } == {"started", "returned"}
    finally:
        (tmp_path / "release").touch()
        await task
        db.close()


async def test_rest_records_management_status_and_inference_errors(tmp_path):
    async def manage(request):
        return web.Response(status=404)

    async def infer(request):
        body = await request.json()
        assert body["inputs"][0]["data"] == [1, 2]
        seq = int(body["id"].rsplit(".", 1)[-1])
        if seq == 0:
            return web.Response(status=429)
        if seq == 1:
            return web.Response(status=503)
        if seq == 2:
            return web.Response(body=b"{}", content_type="application/json")
        return web.json_response(
            {
                "id": body["id"],
                "model_name": "A",
                "model_version": "1",
                "outputs": [
                    {
                        "name": "total",
                        "datatype": "FP32",
                        "shape": [1, 1],
                        "data": ["invalid-float"],
                    }
                ],
            }
        )

    app = web.Application()
    app.router.add_post("/v2/repository/models/missing/unload", manage)
    app.router.add_post("/v2/models/A/versions/1/infer", infer)
    config = scenario(
        actions=[
            dict(
                at_s=0, operation="unload", model="missing", expected_status="HTTP_404"
            )
        ]
    )
    db = database(tmp_path / "rest.sqlite")
    try:
        async with TestServer(app) as server:
            address = str(server.make_url("/"))
            async with aiohttp.ClientSession(address) as client:
                await run(config, client, db, address)
        result = summary(db)
        assert result["deployment"]["protocol"] == "rest"
        assert result["actions"][0]["status"] == result["actions"][0]["expected"]
        assert result["models"][0]["windows"]["all"]["outcomes"] == {
            "rejected": 1,
            "http_error": 1,
            "wrong_output": 2,
        }
    finally:
        db.close()
