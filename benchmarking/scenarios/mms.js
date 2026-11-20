/*
 * This benchmark expects a pre-generated large-ish set of models.
 * This can be generated using the `clone-models.sh` script.
 */

import { GrpcClient } from "../common/grpc.js";
import { readTestData } from "../common/helpers.js";
import { RestClient } from "../common/rest.js";

const TestData = {
  iris: readTestData("iris"),
  sum_model: readTestData("sum-model"),
};

const protocol = __ENV.PROTOCOL || "rest";
if (protocol !== "rest" && protocol !== "grpc") {
  throw new Error("PROTOCOL must be rest or grpc");
}
const client = protocol === "grpc" ? new GrpcClient() : new RestClient();

const ModelCount = 10;

export const options = {
  thresholds: { checks: ["rate==1"] },
  scenarios: {
    sum_model: {
      executor: "per-vu-iterations",
      vus: 1,
      iterations: ModelCount,
      tags: { model_name: "sum-model" },
      env: { MODEL_NAME: "sum-model" },
    },
    iris: {
      executor: "per-vu-iterations",
      vus: 1,
      iterations: ModelCount,
      tags: { model_name: "iris" },
      env: { MODEL_NAME: "iris" },
    },
  },
};

export function setup() {
  return TestData;
}

export default function (data) {
  const modelBasename = __ENV.MODEL_NAME;

  const modelData = data[modelBasename.replace("-", "_")];
  const modelName = `${modelBasename}-${__ITER + 1}`;

  client.loadModel(modelName);
  if (protocol === "grpc") {
    // Fixtures use both protobuf JSON spellings of the model name.
    const request = Object.assign({}, modelData.grpc);
    delete request.model_name;
    request.modelName = modelName;
    client.infer(request);
  } else {
    client.infer(modelName, modelData.rest);
  }
}

export function teardown(data) {}
