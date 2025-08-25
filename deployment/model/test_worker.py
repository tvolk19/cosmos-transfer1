from deployment.model.transfer_worker import (
    TransferPipeline,
    TransferValidator,
    hint_keys_av,
    hint_keys,
)
from cosmos_transfer1.utils import log
from deployment.model.transfer_worker import BASE_7B_CHECKPOINT_AV_SAMPLE_PATH
from cosmos_gradio.deployment_env import DeploymentEnv
import json
import os


def get_spec(spec_file):
    with open(spec_file, "r") as f:
        controlnet_specs = json.load(f)
    return controlnet_specs


def test_transfer_AV():
    pipeline = TransferPipeline(
        num_gpus=int(os.environ.get("NUM_GPU", 1)),
        checkpoint_name=BASE_7B_CHECKPOINT_AV_SAMPLE_PATH,
        hint_keys=hint_keys_av,
    )
    validator = TransferValidator(hint_keys=hint_keys_av)

    model_params = validator.prune_and_validate(
        prompt="The video is captured from a camera mounted on a car. The camera is facing forward. The video showcases a scenic golden-hour drive through a suburban area, bathed in the warm, golden hues of the setting sun. The dashboard camera captures the play of light and shadow as the sun’s rays filter through the trees, casting elongated patterns onto the road. The streetlights remain off, as the golden glow of the late afternoon sun provides ample illumination. The two-lane road appears to shimmer under the soft light, while the concrete barrier on the left side of the road reflects subtle warm tones. The stone wall on the right, adorned with lush greenery, stands out vibrantly under the golden light, with the palm trees swaying gently in the evening breeze. Several parked vehicles, including white sedans and vans, are seen on the left side of the road, their surfaces reflecting the amber hues of the sunset. The trees, now highlighted in a golden halo, cast intricate shadows onto the pavement. Further ahead, houses with red-tiled roofs glow warmly in the fading light, standing out against the sky, which transitions from deep orange to soft pastel blue. As the vehicle continues, a white sedan is seen driving in the same lane, while a black sedan and a white van move further ahead. The road markings are crisp, and the entire setting radiates a peaceful, almost cinematic beauty. The golden light, combined with the quiet suburban landscape, creates an atmosphere of tranquility and warmth, making for a mesmerizing and soothing drive.",
        sigma_max=80,
        controlnet_specs=get_spec("assets/sample_av_multi_control_spec.json"),
    )

    pipeline.infer(model_params)


def test_transfer():
    validator = TransferValidator(hint_keys=hint_keys)

    cfg = DeploymentEnv()
    print(cfg)
    pipeline = TransferPipeline(num_gpus=cfg.num_gpus, checkpoint_dir=cfg.checkpoint_dir)

    log.info("Inference start****************************************")
    # model_params = validator.parse_and_validate(
    #     controlnet_specs=get_spec("assets/inference_cosmos_transfer1_single_control_vis.json"),
    # )
    # model_params["output_dir"] = "outputs/vis/"
    # pipeline.infer(model_params)
    # log.info("Inference complete****************************************")

    # model_params = validator.parse_and_validate(
    #     controlnet_specs=get_spec("assets/preload_vis.json"),
    # )
    # model_params["output_dir"] = "outputs/test1/"
    # pipeline.infer(model_params)
    # log.info("Inference complete****************************************")

    model_params = validator.parse_and_validate(
        controlnet_specs=get_spec("assets/inference_cosmos_transfer1_multi_control.json"),
    )
    model_params["output_dir"] = "outputs/multi_control/"
    pipeline.infer(model_params)
    log.info("Inference complete****************************************")


if __name__ == "__main__":
    test_transfer()
