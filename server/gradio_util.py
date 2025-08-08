import os
from datetime import datetime
from cosmos_transfer1.utils import log


def get_output_folder(output_dir):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_folder = os.path.join(output_dir, f"generation_{timestamp}")
    os.makedirs(output_folder, exist_ok=True)
    return output_folder


def get_outputs(output_folder):
    # Check if output was generated
    output_path = os.path.join(output_folder, "output.mp4")
    if os.path.exists(output_path):
        # Read the generated prompt
        prompt_path = os.path.join(output_folder, "output.txt")
        final_prompt = ""
        if os.path.exists(prompt_path):
            with open(prompt_path, "r", encoding="utf-8") as f:
                final_prompt = f.read().strip()

        return (
            output_path,
            f"Video generated successfully!\nOutput saved to: {output_folder}\nFinal prompt: {final_prompt}",
        )
    else:
        return None, f"Generation failed - no output video was created\nCheck folder: {output_folder}"


def create_worker_pipeline(cfg, create_model=True):
    module = __import__(cfg.factory_module, fromlist=[cfg.factory_function])
    factory_function = getattr(module, cfg.factory_function)
    log.info(f"initializing model using {cfg.factory_module}.{cfg.factory_function}")
    return factory_function(cfg, create_model=create_model)
