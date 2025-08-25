from deployment.model.transfer_worker import (
    TransferWorker,
    TransferValidator,
)
from imaginaire.utils import log
from server.deploy_config import Config
import json


def get_spec(spec_file):
    with open(spec_file, "r") as f:
        controlnet_specs = json.load(f)
    return controlnet_specs


d1 = "assets/robot_example/depth/robot_depth_spec.json"
e1 = "assets/robot_example/edge/robot_edge_spec.json"
s1 = "assets/robot_example/seg/robot_seg_spec.json"

t2 = "assets/robot_example/robot_multi_modal_on_the_fly_spec.json"


def test_transfer():
    validator = TransferValidator()
    model_params = validator.parse_and_validate(
        controlnet_specs=get_spec(d1),
    )
    pipeline = TransferWorker(num_gpus=Config.num_gpus, checkpoint_dir=Config.checkpoint_dir)

    log.info("Inference start****************************************")

    model_params["output_dir"] = "outputs/"
    pipeline.infer(
        model_params,
    )
    log.info("Inference complete****************************************")


if __name__ == "__main__":
    test_transfer()
