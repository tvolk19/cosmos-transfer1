import json
import os
from deployment.model.transfer_worker import (
    TransferValidator,
)
from imaginaire.utils import log
from cosmos_gradio.model_ipc.server_config import Config
from cosmos_gradio.model_ipc.model_server import ModelServer


def get_spec(spec_file):
    with open(spec_file, "r") as f:
        controlnet_specs = json.load(f)
    return controlnet_specs


d1 = "assets/robot_example/depth/robot_depth_spec.json"
e1 = "assets/robot_example/edge/robot_edge_spec.json"
s1 = "assets/robot_example/seg/robot_seg_spec.json"


def test_model_server():

    # test os.environ["FACTORY_MODULE"] = "cosmos_gradio.model_ipc.model_worker"
    os.environ["FACTORY_MODULE"] = "deployment.model.transfer_worker"

    folder = "outputs/"
    with ModelServer(cfg=Config()) as pipeline:
        validator = TransferValidator()

        log.info("Inference start****************************************")

        model_params = validator.parse_and_validate(
            controlnet_specs=get_spec(e1),
        )
        model_params["output_dir"] = f"{folder}/"
        pipeline.infer(model_params)
        log.info("Inference complete****************************************")


if __name__ == "__main__":
    test_model_server()
