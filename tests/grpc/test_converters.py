import pytest
import numpy as np

from google.protobuf import json_format

from mlserver import types
from mlserver.types import Datatype
from mlserver.grpc.converters import (
    ModelInferRequestConverter,
    ModelInferResponseConverter,
    ServerMetadataResponseConverter,
    ModelMetadataResponseConverter,
    RepositoryIndexRequestConverter,
    RepositoryIndexResponseConverter,
    ParametersConverter,
    InferTensorContentsConverter,
    InferOutputTensorConverter,
)
from mlserver.grpc import dataplane_pb2 as pb
from mlserver.errors import InferenceError


def test_servermetadataresponse_from_types(metadata_server_response):
    metadata = ServerMetadataResponseConverter.from_types(metadata_server_response)

    assert metadata.name == metadata_server_response.name
    assert metadata.version == metadata_server_response.version
    assert metadata.extensions == metadata_server_response.extensions


def test_servermetadataresponse_to_types(metadata_server_response):
    server_metadata_response = ServerMetadataResponseConverter.from_types(
        metadata_server_response
    )
    metadata = ServerMetadataResponseConverter.to_types(server_metadata_response)

    assert metadata == metadata_server_response


def test_modelmetadataresponse_from_types(metadata_model_response):
    metadata = ModelMetadataResponseConverter.from_types(metadata_model_response)

    assert metadata.name == metadata_model_response.name
    assert metadata.versions == metadata_model_response.versions
    assert metadata.platform == metadata_model_response.platform

    for pb_obj, ty_obj in zip(metadata.inputs, metadata_model_response.inputs):
        assert pb_obj.name == ty_obj.name
        assert pb_obj.datatype == ty_obj.datatype
        assert pb_obj.shape == ty_obj.shape

    for pb_obj, ty_obj in zip(metadata.outputs, metadata_model_response.outputs):
        assert pb_obj.name == ty_obj.name
        assert pb_obj.datatype == ty_obj.datatype
        assert pb_obj.shape == ty_obj.shape


def test_modelmetadataresponse_to_types(metadata_model_response):
    model_metadata_response = ModelMetadataResponseConverter.from_types(
        metadata_model_response
    )
    model_metadata = ModelMetadataResponseConverter.to_types(model_metadata_response)

    assert model_metadata == metadata_model_response


@pytest.mark.parametrize(
    "model_infer_request",
    [
        pb.ModelInferRequest(
            id="",
            inputs=[
                pb.ModelInferRequest.InferInputTensor(
                    name="input-0",
                    datatype="INT32",
                    shape=[1, 3],
                    parameters={"content_type": pb.InferParameter(string_param="np")},
                    contents=pb.InferTensorContents(int_contents=[1, 2, 3]),
                )
            ],
        ),
        pb.ModelInferRequest(
            id="",
            inputs=[
                pb.ModelInferRequest.InferInputTensor(
                    name="input-0",
                    datatype="INT32",
                    shape=[1, 3],
                    parameters={"content_type": pb.InferParameter(string_param="np")},
                    contents=pb.InferTensorContents(),
                )
            ],
            raw_input_contents=[b"\x01\x00\x00\x00\x02\x00\x00\x00\x03\x00\x00\x00"],
        ),
    ],
)
def test_modelinferrequest_to_types(model_infer_request):
    inference_request = ModelInferRequestConverter.to_types(model_infer_request)

    expected = types.InferenceRequest(
        id="",
        inputs=[
            types.RequestInput(
                name="input-0",
                datatype="INT32",
                shape=[1, 3],
                data=types.TensorData.model_validate([1, 2, 3]),
                parameters=types.Parameters(content_type="np"),
            )
        ],
    )

    assert type(inference_request) is types.InferenceRequest
    assert dict(inference_request) == dict(expected)


@pytest.mark.parametrize(
    "use_raw, expected",
    [
        (
            False,
            pb.ModelInferRequest(
                id="123",
                model_name="sum-model",
                inputs=[
                    pb.ModelInferRequest.InferInputTensor(
                        name="input-0",
                        datatype="FP32",
                        shape=[1],
                        contents=pb.InferTensorContents(fp32_contents=[21.0]),
                    )
                ],
            ),
        ),
        (
            True,
            pb.ModelInferRequest(
                id="123",
                model_name="sum-model",
                inputs=[
                    pb.ModelInferRequest.InferInputTensor(
                        name="input-0",
                        datatype="FP32",
                        shape=[1],
                    )
                ],
                raw_input_contents=[b"\x00\x00\xa8A"],
            ),
        ),
    ],
)
def test_modelinferrequest_from_types(use_raw, expected):
    inference_request = types.InferenceRequest(
        id="123",
        inputs=[
            types.RequestInput(
                name="input-0",
                datatype="FP32",
                shape=[1],
                data=types.TensorData.model_validate([21.0]),
            )
        ],
    )

    model_infer_request = ModelInferRequestConverter.from_types(
        inference_request, model_name="sum-model", use_raw=use_raw
    )

    assert type(model_infer_request) is pb.ModelInferRequest
    assert json_format.MessageToDict(model_infer_request) == json_format.MessageToDict(
        expected
    )


@pytest.mark.parametrize(
    "use_raw, expected",
    [
        (
            False,
            pb.ModelInferResponse(
                model_name="sum-model",
                id="123",
                outputs=[
                    pb.ModelInferResponse.InferOutputTensor(
                        name="output-0",
                        datatype="FP32",
                        shape=[1],
                        contents=pb.InferTensorContents(fp32_contents=[21.0]),
                    )
                ],
            ),
        ),
        (
            True,
            pb.ModelInferResponse(
                model_name="sum-model",
                id="123",
                outputs=[
                    pb.ModelInferResponse.InferOutputTensor(
                        name="output-0",
                        datatype="FP32",
                        shape=[1],
                    )
                ],
                raw_output_contents=[b"\x00\x00\xa8A"],
            ),
        ),
    ],
)
def test_modelinferresponse_from_types(inference_response, use_raw, expected):
    model_infer_response = ModelInferResponseConverter.from_types(
        inference_response, use_raw=use_raw
    )

    assert type(model_infer_response) is pb.ModelInferResponse
    assert json_format.MessageToDict(model_infer_response) == json_format.MessageToDict(
        expected
    )


def test_modelinferrequest_fp16_uses_raw_contents():
    values = np.asarray([1.5, 2.0], dtype=np.float16)
    request = types.InferenceRequest(
        inputs=[
            types.RequestInput(
                name="input-0",
                datatype="FP16",
                shape=[2],
                data=types.TensorData.model_validate(values.tolist()),
            )
        ]
    )

    converted = ModelInferRequestConverter.from_types(request, model_name="model")

    assert converted.raw_input_contents == [values.tobytes()]
    assert not converted.inputs[0].contents.ListFields()


def test_modelinferresponse_fp16_uses_raw_contents():
    values = np.asarray([1.5, 2.0], dtype=np.float16)
    response = types.InferenceResponse(
        model_name="model",
        outputs=[
            types.ResponseOutput(
                name="output-0",
                datatype="FP16",
                shape=[2],
                data=types.TensorData.model_validate(values.tolist()),
            )
        ],
    )

    converted = ModelInferResponseConverter.from_types(response)

    assert converted.raw_output_contents == [values.tobytes()]
    assert not converted.outputs[0].contents.ListFields()


def test_modelinferrequest_mixed_datatypes_use_ordered_raw_contents():
    fp16_values = np.asarray([1.5, 2.0], dtype=np.float16)
    fp32_values = np.asarray([3.0], dtype=np.float32)
    request = types.InferenceRequest(
        inputs=[
            types.RequestInput(
                name="fp16-input",
                datatype="FP16",
                shape=[2],
                data=types.TensorData.model_validate(fp16_values.tolist()),
            ),
            types.RequestInput(
                name="fp32-input",
                datatype="FP32",
                shape=[1],
                data=types.TensorData.model_validate(fp32_values.tolist()),
            ),
        ]
    )

    converted = ModelInferRequestConverter.from_types(request, model_name="model")

    assert converted.raw_input_contents == [
        fp16_values.tobytes(),
        fp32_values.tobytes(),
    ]
    assert all(not tensor.contents.ListFields() for tensor in converted.inputs)


def test_modelinferresponse_mixed_datatypes_use_ordered_raw_contents():
    fp32_values = np.asarray([3.0], dtype=np.float32)
    fp16_values = np.asarray([1.5, 2.0], dtype=np.float16)
    response = types.InferenceResponse(
        model_name="model",
        outputs=[
            types.ResponseOutput(
                name="fp32-output",
                datatype="FP32",
                shape=[1],
                data=types.TensorData.model_validate(fp32_values.tolist()),
            ),
            types.ResponseOutput(
                name="fp16-output",
                datatype="FP16",
                shape=[2],
                data=types.TensorData.model_validate(fp16_values.tolist()),
            ),
        ],
    )

    converted = ModelInferResponseConverter.from_types(response)

    assert converted.raw_output_contents == [
        fp32_values.tobytes(),
        fp16_values.tobytes(),
    ]
    assert all(not tensor.contents.ListFields() for tensor in converted.outputs)


@pytest.mark.parametrize(
    "model_infer_response",
    [
        pb.ModelInferResponse(
            id="",
            model_name="foo",
            outputs=[
                pb.ModelInferResponse.InferOutputTensor(
                    name="input-0",
                    datatype="INT32",
                    shape=[1, 3],
                    parameters={"content_type": pb.InferParameter(string_param="np")},
                    contents=pb.InferTensorContents(int_contents=[1, 2, 3]),
                )
            ],
        ),
        pb.ModelInferResponse(
            id="",
            model_name="foo",
            outputs=[
                pb.ModelInferResponse.InferOutputTensor(
                    name="input-0",
                    datatype="INT32",
                    shape=[1, 3],
                    parameters={"content_type": pb.InferParameter(string_param="np")},
                    contents=pb.InferTensorContents(),
                )
            ],
            raw_output_contents=[b"\x01\x00\x00\x00\x02\x00\x00\x00\x03\x00\x00\x00"],
        ),
    ],
)
def test_modelinferresponse_to_types(model_infer_response):
    expected = types.InferenceResponse(
        model_name="foo",
        model_version=None,
        id="",
        parameters=None,
        outputs=[
            types.ResponseOutput(
                name="input-0",
                shape=[1, 3],
                datatype="INT32",
                parameters=types.Parameters(content_type="np", headers=None),
                data=[1, 2, 3],
            )
        ],
    )

    inference_response = ModelInferResponseConverter.to_types(model_infer_response)

    assert inference_response == expected


def test_parameters_to_types(grpc_parameters):
    parameters = ParametersConverter.to_types(grpc_parameters)

    assert parameters.content_type == grpc_parameters["content_type"].string_param
    assert parameters.foo == grpc_parameters["foo"].bool_param
    assert parameters.bar == grpc_parameters["bar"].int64_param


def test_parameters_from_types(grpc_parameters):
    parameters = ParametersConverter.to_types(grpc_parameters)
    conv_parameters = ParametersConverter.from_types(parameters)

    assert conv_parameters == grpc_parameters


@pytest.mark.parametrize(
    "response_output, expected",
    [
        (
            types.ResponseOutput(
                name="output-0", datatype="FP32", shape=[1], data=[21.0]
            ),
            pb.ModelInferResponse.InferOutputTensor(
                name="output-0",
                datatype="FP32",
                shape=[1],
                contents=pb.InferTensorContents(fp32_contents=[21.0]),
            ),
        ),
        (
            types.ResponseOutput(
                name="output-0",
                datatype="BYTES",
                shape=[2],
                data=[b"hey", b"hello world"],
            ),
            pb.ModelInferResponse.InferOutputTensor(
                name="output-0",
                datatype="BYTES",
                shape=[2],
                contents=pb.InferTensorContents(
                    bytes_contents=[b"hey", b"hello world"]
                ),
            ),
        ),
        (
            types.ResponseOutput(
                name="output-0", datatype="FP16", shape=[2], data=[1.5, 2.0]
            ),
            pb.ModelInferResponse.InferOutputTensor(
                name="output-0",
                datatype="FP16",
                shape=[2],
            ),
        ),
    ],
)
def test_inferoutputtensor_from_types(
    response_output: types.ResponseOutput,
    expected: pb.ModelInferResponse.InferOutputTensor,
):
    infer_output_tensor = InferOutputTensorConverter.from_types(response_output)
    assert infer_output_tensor == expected


def test_inferoutputtensor_rejects_fp16_typed_contents():
    tensor = pb.ModelInferResponse.InferOutputTensor(
        name="output-0",
        datatype="FP16",
        shape=[2],
        contents=pb.InferTensorContents(
            bytes_contents=[np.asarray([1.5, 2.0], dtype=np.float16).tobytes()]
        ),
    )

    with pytest.raises(InferenceError, match="FP16 tensor contents must use raw"):
        InferOutputTensorConverter.to_types(tensor)


def test_infertensorcontents_fp16_empty_to_types():
    contents = pb.InferTensorContents(bytes_contents=[])

    output = InferTensorContentsConverter.to_types(contents, datatype=Datatype.FP16)

    assert output.root == []


def test_infertensorcontents_fp16_invalid_length_to_types():
    contents = pb.InferTensorContents(bytes_contents=[b"\x00"])

    with pytest.raises(InferenceError, match="FP16 tensor contents must use raw"):
        InferTensorContentsConverter.to_types(contents, datatype=Datatype.FP16)


def test_infertensorcontents_fp16_from_types_requires_raw_contents():
    with pytest.raises(InferenceError, match="FP16 tensor contents must use raw"):
        InferTensorContentsConverter.from_types(
            types.TensorData.model_validate([1.5, 2.0]), Datatype.FP16
        )


def test_repositoryindexrequest_to_types(grpc_repository_index_request):
    repository_index_request = RepositoryIndexRequestConverter.to_types(
        grpc_repository_index_request
    )

    assert repository_index_request.ready == grpc_repository_index_request.ready


def test_repositoryindexresponse_from_types(repository_index_response):
    grpc_repository_index_response = RepositoryIndexResponseConverter.from_types(
        repository_index_response
    )

    assert isinstance(grpc_repository_index_response, pb.RepositoryIndexResponse)
    assert len(grpc_repository_index_response.models) == len(repository_index_response)

    for expected, grpc_model in zip(
        repository_index_response, grpc_repository_index_response.models
    ):
        assert isinstance(grpc_model, pb.RepositoryIndexResponse.ModelIndex)
        assert expected.name == grpc_model.name
        assert expected.version == grpc_model.version
        assert expected.state == grpc_model.state
        assert expected.reason == grpc_model.reason
