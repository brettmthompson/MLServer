import asyncio
import importlib
import os
import pytest
import signal
import sys

from aiohttp.client_exceptions import ClientResponseError
from click.testing import CliRunner
from pathlib import Path
from subprocess import Popen, TimeoutExpired
from typing import Tuple
from unittest.mock import Mock, AsyncMock

from mlserver.cli.main import root
from mlserver.settings import ModelSettings, Settings, TRUSTED_RUNTIMES_ARTIFACT_PATH
from mlserver.types import InferenceRequest

from ..utils import (
    RESTClient,
    TEST_TRUSTED_RUNTIMES_ARTIFACT_ENV,
    get_available_ports,
)
from .test_start_cases import case_custom_module, case_sum_model


def _spawn_mlserver(folder: str) -> Popen:
    # Use the same interpreter as the running test env so imports resolve
    # consistently across tox and local runs.
    # This fixture depends on repository-root `conftest.py` bootstrap setup,
    # which pre-populates PYTHONPATH and trusted-runtime artifact env for
    # spawned subprocesses.
    repo_root = str(Path(__file__).resolve().parents[2])
    subprocess_env = {
        key: value for key, value in os.environ.items() if key != "PYTHONHOME"
    }
    if "PYTHONPATH" not in subprocess_env:
        raise RuntimeError("Missing PYTHONPATH test bootstrap env.")
    if TEST_TRUSTED_RUNTIMES_ARTIFACT_ENV not in subprocess_env:
        raise RuntimeError("Missing trusted-runtimes test artifact env.")
    return Popen(
        [sys.executable, "-m", "mlserver.cli.main", "start", folder],
        cwd=repo_root,
        start_new_session=True,
        env=subprocess_env,
    )


def _stop_mlserver(process: Popen) -> None:
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
    except ProcessLookupError:
        # Process may have already exited before fixture teardown runs.
        pass
    try:
        process.wait(timeout=10)
    except TimeoutExpired:
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=5)


@pytest.fixture
def settings(settings: Settings, free_ports: Tuple[int, int]) -> Settings:
    http_port, grpc_port, metrics_port = free_ports

    settings.http_port = http_port
    settings.grpc_port = grpc_port
    settings.metrics_port = metrics_port

    return settings


@pytest.fixture
def mlserver_start_sum_model(
    tmp_path: str, settings: Settings, sum_model_settings: ModelSettings
) -> Popen:
    # Baseline scenario: importable runtime (`tests.fixtures.SumModel`).
    sum_model_folder = case_sum_model(tmp_path, settings, sum_model_settings)
    p = _spawn_mlserver(sum_model_folder)

    yield p

    _stop_mlserver(p)


@pytest.fixture
def mlserver_start_custom_module(
    tmp_path: str, settings: Settings, sum_model_settings: ModelSettings
) -> Popen:
    # Security scenario: model-folder module (`custom.SumModel`) should not load.
    custom_module_folder = case_custom_module(tmp_path, settings, sum_model_settings)
    p = _spawn_mlserver(custom_module_folder)

    yield p

    _stop_mlserver(p)


@pytest.fixture
async def rest_client(settings: Settings) -> RESTClient:
    http_server = f"127.0.0.1:{settings.http_port}"
    client = RESTClient(http_server)

    yield client

    await client.close()


@pytest.mark.usefixtures("mlserver_start_sum_model")
async def test_live(rest_client: RESTClient):
    await rest_client.wait_until_live()
    is_live = await rest_client.live()
    assert is_live

    # Assert that the server is live, but some models are still loading
    with pytest.raises(ClientResponseError):
        await rest_client.ready()


@pytest.mark.usefixtures("mlserver_start_sum_model")
async def test_infer(
    rest_client: RESTClient,
    sum_model_settings: ModelSettings,
    inference_request: InferenceRequest,
):
    await rest_client.wait_until_model_ready(sum_model_settings.name)
    response = await rest_client.infer(sum_model_settings.name, inference_request)

    assert len(response.outputs) == 1


@pytest.mark.usefixtures("mlserver_start_custom_module")
async def test_custom_module_fails_closed(
    rest_client: RESTClient,
    sum_model_settings: ModelSettings,
):
    # Fail closed when runtime points to a non-importable model-folder module.
    await rest_client.wait_until_live()
    with pytest.raises(ClientResponseError):
        await rest_client.wait_until_model_ready(sum_model_settings.name)


@pytest.mark.usefixtures("mlserver_start_sum_model")
async def test_spawned_workers_load_allowlisted_runtime_via_bootstrap(
    settings: Settings,
    rest_client: RESTClient,
    sum_model_settings: ModelSettings,
):
    # This verifies the trusted-runtime bootstrap is applied inside spawned
    # worker processes (parallel_workers > 0). Without that bootstrap, runtime
    # `tests.fixtures.SumModel` would be rejected by worker-side validation.
    assert settings.parallel_workers > 0
    await rest_client.wait_until_live()
    await rest_client.wait_until_model_ready(sum_model_settings.name)


async def test_concurrent_mlserver_start_spawns_workers(
    tmp_path: str,
    settings: Settings,
    sum_model_settings: ModelSettings,
):
    # Start multiple MLServer instances concurrently to stress worker spawn
    # and trusted-runtime bootstrap application under parallel startup.
    instances: list[tuple[Popen, RESTClient]] = []
    used_ports: set[int] = set()

    try:
        for idx in range(2):
            instance_settings = settings.model_copy(deep=True)
            instance_ports = get_available_ports(3)
            if used_ports.intersection(instance_ports):
                raise AssertionError(
                    "Concurrent test allocated duplicate ports across instances."
                )
            used_ports.update(instance_ports)
            instance_settings.http_port = instance_ports[0]
            instance_settings.grpc_port = instance_ports[1]
            instance_settings.metrics_port = instance_ports[2]

            folder = os.path.join(str(tmp_path), f"instance-{idx}")
            os.makedirs(folder, exist_ok=True)
            case_sum_model(folder, instance_settings, sum_model_settings)

            process = _spawn_mlserver(folder)
            client = RESTClient(f"127.0.0.1:{instance_settings.http_port}")
            instances.append((process, client))

        await asyncio.gather(*[client.wait_until_live() for _, client in instances])
        await asyncio.gather(
            *[
                client.wait_until_model_ready(sum_model_settings.name)
                for _, client in instances
            ]
        )
    finally:
        await asyncio.gather(*[client.close() for _, client in instances])
        for process, _ in instances:
            _stop_mlserver(process)


# Unit tests for start command with custom runtime flags


@pytest.fixture
def cli_runner():
    """Click CLI test runner."""
    return CliRunner()


@pytest.fixture
def cli_main():
    """Import mlserver.cli.main module for mocking."""
    return importlib.import_module("mlserver.cli.main")


def test_start_with_allow_runtime_when_artifact_exists(
    cli_runner, tmp_path, monkeypatch, cli_main
):
    """When trusted-runtimes.json exists, --allow-runtime should be ignored with warning."""
    # Create a fake runtime file so Click's path validation passes
    runtime_file = Path(tmp_path) / "custom.py"
    runtime_file.write_text("class MyRuntime: pass")

    # Mock the artifact file to exist
    monkeypatch.setattr(
        "os.path.isfile", lambda path: path == TRUSTED_RUNTIMES_ARTIFACT_PATH
    )

    # Mock the setup function to verify it's NOT called
    mock_setup = Mock()
    monkeypatch.setattr(cli_main, "_setup_custom_runtimes", mock_setup)

    # Mock load_settings and MLServer to prevent actual server start
    mock_settings = Mock()
    mock_models = []
    mock_load = AsyncMock(return_value=(mock_settings, mock_models))
    monkeypatch.setattr(cli_main, "load_settings", mock_load)

    mock_server = Mock()
    mock_server.start = AsyncMock()
    mock_server_class = Mock(return_value=mock_server)
    monkeypatch.setattr(cli_main, "MLServer", mock_server_class)

    # Run the command with custom runtime flags
    result = cli_runner.invoke(
        root,
        [
            "start",
            str(tmp_path),
            "--allow-runtime",
            "custom.MyRuntime",
            "--runtime-path",
            str(runtime_file),
        ],
    )

    # Verify command succeeded
    assert result.exit_code == 0

    # Verify setup was NOT called (the key behavior when artifact exists)
    mock_setup.assert_not_called()


def test_start_with_allow_runtime_when_artifact_missing(
    cli_runner, tmp_path, monkeypatch, cli_main
):
    """When trusted-runtimes.json doesn't exist, --allow-runtime should work."""
    # Mock the artifact file to NOT exist
    monkeypatch.setattr("os.path.isfile", lambda path: False)

    # Create a fake runtime file
    runtime_file = Path(tmp_path) / "custom.py"
    runtime_file.write_text("class MyRuntime: pass")

    # Mock the setup function to verify it IS called
    mock_setup = Mock()
    monkeypatch.setattr(cli_main, "_setup_custom_runtimes", mock_setup)

    # Mock load_settings and MLServer to prevent actual server start
    mock_settings = Mock()
    mock_models = []
    mock_load = AsyncMock(return_value=(mock_settings, mock_models))
    monkeypatch.setattr(cli_main, "load_settings", mock_load)

    mock_server = Mock()
    mock_server.start = AsyncMock()
    mock_server_class = Mock(return_value=mock_server)
    monkeypatch.setattr(cli_main, "MLServer", mock_server_class)

    # Run the command with custom runtime flags
    result = cli_runner.invoke(
        root,
        [
            "start",
            str(tmp_path),
            "--allow-runtime",
            "custom.MyRuntime",
            "--runtime-path",
            str(runtime_file),
        ],
    )

    # Verify no warning about ignored flags
    assert "ignored" not in result.output.lower()

    # Verify setup WAS called with correct arguments
    mock_setup.assert_called_once()
    call_args = mock_setup.call_args
    assert call_args[0][0] == ("custom.MyRuntime",)  # allow_runtime_import_paths
    assert call_args[0][1] == (str(runtime_file),)  # runtime_source_paths
    assert call_args[0][2] == str(tmp_path)  # folder (models directory)


def test_start_without_custom_runtime_flags(cli_runner, tmp_path, monkeypatch, cli_main):
    """When no custom runtime flags provided, setup should not be called."""
    # Mock the setup function to verify it's NOT called
    mock_setup = Mock()
    monkeypatch.setattr(cli_main, "_setup_custom_runtimes", mock_setup)

    # Mock load_settings and MLServer to prevent actual server start
    mock_settings = Mock()
    mock_models = []
    mock_load = AsyncMock(return_value=(mock_settings, mock_models))
    monkeypatch.setattr(cli_main, "load_settings", mock_load)

    mock_server = Mock()
    mock_server.start = AsyncMock()
    mock_server_class = Mock(return_value=mock_server)
    monkeypatch.setattr(cli_main, "MLServer", mock_server_class)

    # Run the command without custom runtime flags
    result = cli_runner.invoke(root, ["start", str(tmp_path)])

    # Verify setup was NOT called
    mock_setup.assert_not_called()


def test_start_with_multiple_allow_runtime_flags(
    cli_runner, tmp_path, monkeypatch, cli_main
):
    """Multiple --allow-runtime flags should all be passed to setup."""
    # Mock the artifact file to NOT exist
    monkeypatch.setattr("os.path.isfile", lambda path: False)

    # Mock the setup function
    mock_setup = Mock()
    monkeypatch.setattr(cli_main, "_setup_custom_runtimes", mock_setup)

    # Mock load_settings and MLServer
    mock_settings = Mock()
    mock_models = []
    mock_load = AsyncMock(return_value=(mock_settings, mock_models))
    monkeypatch.setattr(cli_main, "load_settings", mock_load)

    mock_server = Mock()
    mock_server_class = Mock(return_value=mock_server)
    monkeypatch.setattr(cli_main, "MLServer", mock_server_class)

    # Run with multiple --allow-runtime flags
    result = cli_runner.invoke(
        root,
        [
            "start",
            str(tmp_path),
            "--allow-runtime",
            "custom.Runtime1",
            "--allow-runtime",
            "custom.Runtime2",
        ],
    )

    # Verify setup WAS called with all runtimes
    mock_setup.assert_called_once()
    call_args = mock_setup.call_args
    assert call_args[0][0] == ("custom.Runtime1", "custom.Runtime2")
