import json
import os
from deployment.model.transfer_worker import TransferValidator, hint_keys
from cosmos_transfer1.utils import log
from cosmos_gradio.deployment_env import DeploymentEnv
from cosmos_gradio.model_ipc.model_server import ModelServer


def get_spec(spec_file):
    with open(spec_file, "r") as f:
        controlnet_specs = json.load(f)
    return controlnet_specs


def test_model_server():
    folder = "outputs_4gpu/"
    cfg = DeploymentEnv()

    with ModelServer(
        num_gpus=cfg.num_gpus, factory_module="deployment.model.transfer_worker", factory_function="create_worker"
    ) as pipeline:
        validator = TransferValidator(hint_keys=hint_keys)
        log.info("Inference start****************************************")

        model_params = validator.parse_and_validate(
            controlnet_specs=get_spec("assets/inference_cosmos_transfer1_multi_control.json"),
        )
        model_params["output_dir"] = f"{folder}/multi_control_3/"
        pipeline.infer(model_params)

        log.info("Inference complete****************************************")
        # model_params = validator.parse_and_validate(
        #     controlnet_specs=get_spec("assets/inference_cosmos_transfer1_uniform_weights.json"),
        # )
        # model_params["output_dir"] = f"{folder}/multi_control_4/"
        # pipeline.infer(model_params)
        # log.info("Inference complete****************************************")

        # model_params = validator.parse_and_validate(
        #     controlnet_specs=get_spec("assets/inference_cosmos_transfer1_single_control_vis.json"),
        # )
        # model_params["output_dir"] = f"{folder}/single_control/"
        # pipeline.infer(model_params)
        # log.info("Inference complete****************************************")

        # model_params = validator.parse_and_validate(
        #     controlnet_specs=get_spec("assets/inference_cosmos_transfer1_uniform_weights.json"),
        # )
        # model_params["output_dir"] = f"{folder}/multi_control_4_2/"
        # pipeline.infer(model_params)
        # log.info("Inference complete****************************************")


if __name__ == "__main__":
    test_model_server()
