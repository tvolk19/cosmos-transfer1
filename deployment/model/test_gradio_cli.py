import json
from pathlib import Path

from deployment.model.transfer_worker import TransferValidator, hint_keys
from cosmos_transfer1.utils import log
from server.deploy_config import Config
from server.gradio_app_cli import GradioCLIApp
from server.model_server import ModelServer


def get_spec(spec_file):
    with open(spec_file, "r") as f:
        controlnet_specs = json.load(f)
    return controlnet_specs


def test_model_server_cli():
    folder = "outputs/"
    app = GradioCLIApp(num_workers=Config.num_gpus, checkpoint_dir=Config.checkpoint_dir)

    log.info("Inference start****************************************")
    controlnet_specs = get_spec("assets/inference_cosmos_transfer1_multi_control.json")
    app.infer(controlnet_specs)


av_prompt = "The video is captured from a camera mounted on a car. The camera is facing forward. The video showcases a scenic golden-hour drive through a suburban area, bathed in the warm, golden hues of the setting sun. The dashboard camera captures the play of light and shadow as the sun’s rays filter through the trees, casting elongated patterns onto the road. The streetlights remain off, as the golden glow of the late afternoon sun provides ample illumination. The two-lane road appears to shimmer under the soft light, while the concrete barrier on the left side of the road reflects subtle warm tones. The stone wall on the right, adorned with lush greenery, stands out vibrantly under the golden light, with the palm trees swaying gently in the evening breeze. Several parked vehicles, including white sedans and vans, are seen on the left side of the road, their surfaces reflecting the amber hues of the sunset. The trees, now highlighted in a golden halo, cast intricate shadows onto the pavement. Further ahead, houses with red-tiled roofs glow warmly in the fading light, standing out against the sky, which transitions from deep orange to soft pastel blue. As the vehicle continues, a white sedan is seen driving in the same lane, while a black sedan and a white van move further ahead. The road markings are crisp, and the entire setting radiates a peaceful, almost cinematic beauty. The golden light, combined with the quiet suburban landscape, creates an atmosphere of tranquility and warmth, making for a mesmerizing and soothing drive."


def get_parameters(json_file):
    if "sample_av_mv" in json_file.name:
        log.info(f"Skipping {json_file.name} (contains 'sample_av_mv')")
        return None

    with open(json_file, "r") as f:
        json_content = f.read()

    request_data = json.loads(json_content)

    if "sample_av" in json_file.name:
        request_data["prompt"] = av_prompt
        request_data["is_av_sample"] = True
        request_data["sigma_max"] = 80

    return request_data


def test_model_server_cli_batch(input_folder):
    app = GradioCLIApp(num_workers=Config.num_gpus, checkpoint_dir=Config.checkpoint_dir)
    app1 = GradioCLIApp(num_workers=1, checkpoint_dir=Config.checkpoint_dir)

    # Get all JSON files from the input folder
    input_path = Path(input_folder)
    if not input_path.exists():
        log.error(f"Input folder does not exist: {input_folder}")
        return

    json_files = list(input_path.glob("*.json"))

    if not json_files:
        log.warning(f"No JSON files found in {input_folder}")
        return

    log.info(f"Found {len(json_files)} JSON files to process")

    # Process each JSON file
    for json_file in json_files:
        log.info(f"Processing {json_file.name}...")

        # if "inference_cosmos" in json_file.name:
        #     continue
        try:
            json_content = get_parameters(json_file)
            if json_content is None:
                continue

            # Create output folder based on JSON file name (without extension)
            output_base_name = json_file.stem
            output_folder = f"outputs/{output_base_name}"

            log.info(f"Starting inference for {json_file.name}, output to {output_folder}")

            # Call the app.infer function with JSON content
            result = app.infer_dict(json_content, output_folder)

            output_folder = f"{output_folder}_1gpu"
            app1.infer_dict(json_content, output_folder)

        except Exception as e:
            log.error(f"Error processing {json_file.name}: {str(e)}")
            continue

    log.info("Batch processing complete")


if __name__ == "__main__":
    test_model_server()
    # test_model_server_cli_batch("assets/")
