# Legacy Training Notebooks

This directory contains archived training experiments from the Aurigen fine-tuning process. These notebooks are **not production code** — they represent earlier iterations of the training pipeline that were superseded by the final training run that produced `fine-tuned-weights/unet_epoch_3.pth`. They are preserved here for reference and reproducibility, but should not be run without reviewing and addressing the known issues listed below.

## Known Issues

| File | Severity | Description |
|------|----------|-------------|
| `old_training_file.ipynb` | Critical | **Double normalization bug** (lines ~45) — the transform normalizes pixel values, then `(x*2)-1` is applied again, pushing inputs outside the expected `[-1, 1]` range and corrupting training signal |
| `old_training_file.ipynb` | High | **`GradScaler('cuda')` hardcoded** — raises an error on CPU-only environments; must be wrapped in a device guard |
| `old_training_file.ipynb` | High | **Silent zeros for corrupt images** — corrupt or unreadable images are replaced with zero tensors without logging, silently degrading training batches |
| `new_training_file.ipynb` | High | **Windows backslash paths** — hardcoded path separators (`\`) break on macOS and Linux |
| `new_training_file.ipynb` | High | **Hardcoded LoRA checkpoint path** `"output/checkpoint-9000"` — fails if training runs for fewer steps or uses a different output directory |

## Contents

| File | Description |
|------|-------------|
| `old_training_file.ipynb` | Initial fine-tuning experiment on SDXL UNet with ControlNet conditioning |
| `new_training_file.ipynb` | Revised training run with updated hyperparameters and dataset pipeline |

Production inference code lives in `models/` and `utils/`.
