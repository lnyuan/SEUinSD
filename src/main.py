import datetime
import time
import numpy as np
import torch
from diffusers import DiffusionPipeline
from .custom_pipeline import CustomStableDiffusionPipeline, register_custom_pipeline
from .fault_injection import error_free, inj_Unet_d_attn, inj_Unet_d_res, inj_Unet_u_attn, inj_Unet_u_res
from .utils import setup_logging

def main():
    """Main function to run fault injection experiments."""
    current_run_time = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    print(f"*/-------------------------------------------------------------------------------/*\ncurrent:{current_run_time}")

    # Setup logging
    log_path = f'./error_detect/main_output_log_{current_run_time}.txt'
    logger = setup_logging(log_path)

    # Register custom pipeline
    register_custom_pipeline()

    # Load model
    model_id = "./stable-diffusion-2-1"
    try:
        pipe = DiffusionPipeline.from_pretrained(
            model_id,
            custom_pipeline="custom/stable-diffusion",
            torch_dtype=torch.float16,
            use_safetensors=True,
        )
        pipe = pipe.to("cuda")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Load dataset
    try:
        test_file = np.load('./prompts/data_erin.npy', allow_pickle=True)
        test_count = len(test_file)
        print(f"Loaded {test_count} prompts from data_erin.npy")
    except Exception as e:
        print(f"Error loading prompts: {e}")
        return

    # Experiment settings
    save_type = ['x_t']
    save_iter = [1]
    print(f'save_type:{save_type}; save_iter:{save_iter}')

    # Run fault injection experiments
    test_start_time = time.time()
    inj_Unet_d_attn(
        pipe=pipe,
        test_file=test_file,
        save_type=save_type,
        save_iter=save_iter,
        fixed_latents=True,
        blockname="down_blocks",
        partname="attentions",
        blockseq=[1],
        layerseq=[1],
        bitseq=[1, 2],
        weight_num=50,
        weight_num_save=25,
        param_type=["attn2_v", "attn1_v"],
        prompt_num=range(5, 10),
    )
    test_end_time = time.time()
    test_time = test_end_time - test_start_time
    print(f"test time:{test_time}s = {test_time/3600}h")

    # Close logger
    logger.close()

if __name__ == "__main__":
    main()