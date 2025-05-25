# Stable Diffusion Fault Injection and Detection

This repository contains the code for fault injection experiments on Stable Diffusion models, as described in [PAPER TITLE]. The code implements a custom `StableDiffusionPipeline` and performs fault injection in the UNet components to study their impact on image generation quality.

## Prerequisites

- Python 3.8+
- PyTorch 2.0.1+ with CUDA support
- A GPU with at least 8GB VRAM
- The `data_erin.npy` prompt dataset (not included; users must provide their own)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/[YOUR_USERNAME]/stable-diffusion-fault-injection.git
   cd stable-diffusion-fault-injection
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Download the Stable Diffusion model:
   - Place the `stable-diffusion-2-1` model in the root directory or update `model_id` in `src/main.py`.
   - Alternatively, use a Hugging Face model ID (e.g., `stabilityai/stable-diffusion-2-1`).

4. Provide required modules:


5. Place the prompt dataset:
   - Copy `data_erin.npy` to the `prompts/` directory.

## Usage

Run the main experiment script:
```bash
python src/main.py
```

This will:
- Load the custom pipeline and model.
- Perform fault injection in UNet's `down_blocks.attentions`.
- Save intermediate outputs to `error_detect/` and images to `Images_error/`.
- Log results to `error_detect/main_output_log_[TIMESTAMP].txt`.

To modify experiments, edit `src/main.py`. For example, to run `inj_Unet_d_res`:
```python
inj_Unet_d_res(
    pipe=pipe,
    test_file=test_file,
    save_type=save_type,
    save_iter=save_iter,
    fixed_latents=True,
    blockname="down_blocks",
    partname="resnets",
    blockseq=[3],
    layerseq=[0, 1],
    bitseq=[1, 2],
    weight_num=50,
    weight_num_save=25,
    param_type=["norm2"],
    prompt_num=range(5, 10),
)
```

## Directory Structure

- `src/`: Source code (custom pipeline, fault injection, utilities).
- `prompts/`: Directory for prompt dataset (`data_erin.npy`).
- `error_detect/`: Output directory for intermediate results and logs.
- `Images_error/`: Output directory for generated images.
- `requirements.txt`: Python dependencies.
- `LICENSE`: MIT License.
- `.gitignore`: Git ignore file.

## Citation

If you use this code in your research, please cite our paper:

[Insert citation details here, e.g.,]
```bibtex
@article{yourpaper2025,
  title={Fault Injection in Stable Diffusion Models},
  author={Your Name and Co-authors},
  journal={TBD},
  year={2025},
}
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Notes

- The `Clip` module is a placeholder. Replace with a CLIP implementation (e.g., `open_clip`).
- Ensure sufficient disk space for outputs in `error_detect/` and `Images_error/`.
- Fault injection requires `injectorbit` and `injector0`. Contact the authors for details if unavailable.
