# Stable Diffusion Fault Injection and Detection

This repository contains the code for fault injection experiments and fault detection in Stable Diffusion models, as described in [Dependability Evaluation and error toleration of Stable Diffusion with Soft Errors on the Model Parameters]. The code implements a custom `StableDiffusionPipeline` for fault injection in UNet components to study their impact on image generation quality and provides a classifier module to detect faults using intermediate outputs and CLIP scores.

## Prerequisites

- **Python**: 3.10 or higher
- **PyTorch**: 2.2.0 or higher with CUDA support
- **Hardware**: A GPU with at least 8GB VRAM
- **Datasets**:
  - `data_erin.npy` prompt dataset for fault injection (not included; users must provide their own)
  - Fault detection datasets in `./dataset/` (see Fault Detection section)
- **Other Dependencies**: See `requirements.txt` and `classifier/requirements.txt`

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/[YOUR_USERNAME]/stable-diffusion-fault-injection.git
   cd stable-diffusion-fault-injection
   ```

2. **Create a virtual environment and install dependencies**:

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   pip install -r classifier/requirements.txt
   ```

3. **Download the Stable Diffusion model**:

   - Place the `stable-diffusion-2-1` model in the root directory or update `model_id` in `src/main.py`.
   - Alternatively, use a Hugging Face model ID (e.g., `stabilityai/stable-diffusion-2-1`).

4. **Provide required datasets**:

   - Copy `data_erin.npy` to the `prompts/` directory for fault injection experiments.
   - Place fault detection datasets in the `dataset/` directory (see Fault Detection section below).

## Usage

### Fault Injection Experiments

Run the main fault injection script to perform fault injection in the Stable Diffusion UNet:

```bash
python src/main.py
```

This will:

- Load the custom pipeline and Stable Diffusion model.
- Perform fault injection in UNet.
- Save intermediate outputs to `error_detect/` and generated images to `Images_error/`.
- Log results to `error_detect/main_output_log_[TIMESTAMP].txt`.

To customize the fault injection experiment, edit `src/main.py`. For example, to inject faults in `down_blocks.resnets`:

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

### Fault Detection with Classifiers

The `classifier/` module provides tools to train and evaluate an AdaBoost classifier for detecting faults in Stable Diffusion's UNet outputs. The classifier uses intermediate data (e.g., `x_t` tensors) and CLIP scores to identify errors introduced during fault injection.

#### Dataset Preparation

To run the fault detection classifier, place the following datasets in the `dataset/` directory:

- **JSON Files** (CLIP scores for fault-injected and error-free samples):
  - `error_clipsore_dict_down_blocks_attentions.json`
  - `error_clipsore_dict_down_blocks_resnets.json`
  - `error_clipsore_dict_up_blocks_attentions.json`
  - `error_clipsore_dict_up_blocks_resnets.json`
  - `unfixed_error_clipsore_dict.json` (for unfixed latent experiments)
- **PyTorch Files** (UNet intermediate outputs):
  - `.pth` files in subdirectories.
  - Similar subdirectories for other data types if needed.

These datasets contain CLIP scores and UNet outputs for fault-injected and error-free samples. Users must generate these datasets using the fault injection pipeline (`src/main.py`) or obtain them from the authors. The JSON files map error types to CLIP scores, and the `.pth` files store tensor outputs from the UNet.

#### Running the Classifier

Run the fault detection experiment:

```bash
python classifier/main_classifier.py
```

This will:

- Load data from `dataset/`.
- Train an AdaBoost classifier with specified hyperparameters.
- Evaluate classification performance (accuracy, loss, etc.).
- Save results to `error_detect_results/[type_name]_results/`, including:
  - Trained classifier models (`.pkl`).
  - Accuracy and loss plots (`.svg`).
  - Data distribution visualizations (`.svg`) if enabled.
  - Saved data in `.npy` and `.mat` formats for further analysis.

To customize the experiment, modify `classifier/main_classifier.py`. Example configuration:

```python
run_experiment(
    type_name='xt',                 # Data type: 'xt', 'ut', 'ut_text', etc.
    iterations=[1],                # Iteration indices
    n_slice=0,                     # Data slice index
    threshold=0.96,                # CLIP score threshold for labeling errors
    n_estimators=range(39, 41),    # Number of AdaBoost estimators
    learning_rates=[10],           # Learning rates
    test_param_types=['attn1_v', 'fc1', 'attn2_v'],  # UNet parameter types
    dataset_root='./dataset',      # Dataset directory
    result_dir='error_detect_results',  # Output directory
    data_mode='only_fixed',        # Data mode: 'all', 'only_fixed', 'only_unfixed'
    train_label0_ratio=0.4,        # Label 0 ratio in training set
    test_label0_ratio=0.4,         # Label 0 ratio in test set
    data_distribution=True         # Enable distribution plots
)
```

#### Expected Outputs

- **Classifier Models**: Saved as `.pkl` files, e.g., `adaboost_[params]_bestclf_ACC0.XXXX_[timestamp].pkl`.
- **Plots**:
  - Accuracy vs. number of estimators (`_Acc_[timestamp].svg`).
  - Training and testing loss curves (`_Loss_[timestamp].svg`).
  - Data distribution histograms and PDFs (`distributionHist_*.svg`, `pdf_*.svg`) if `data_distribution=True`.
- **Data Files**: Feature data and results in `.npy` and `.mat` formats for MATLAB compatibility.
- **Logs**: Detailed logs in `error_detect_results/[type_name]_log.txt` with timestamped outputs.

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{yourpaper2025,
  title={Fault Injection and Detection in Stable Diffusion Models},
  author={Your Name and Co-authors},
  journal={TBD},
  year={2025},
}
```

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Notes

- **CLIP Module**: The CLIP module is a placeholder. Replace with a CLIP implementation (e.g., `open_clip`) for computing CLIP scores.
- **Disk Space**: Ensure sufficient disk space for outputs in `error_detect/`, `error_detect_results/`, and `Images_error/`.
- **Fault Injection Dependencies**: Fault injection requires `injectorbit` and `injector0`. Contact the authors for details if unavailable.
- **Fault Detection Datasets**: The classifier assumes preprocessed data in `dataset/`. Users must generate these datasets using the fault injection pipeline or obtain them from the authors.
- **Python Version**: The code has been tested with Python 3.10. Ensure compatibility with dependencies in `requirements.txt`.

