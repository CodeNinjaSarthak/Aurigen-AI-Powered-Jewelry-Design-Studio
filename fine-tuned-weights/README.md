# Fine-Tuned Weights

This directory holds the fine-tuned UNet checkpoint used by the Aurigen inference pipeline.

## Download

The checkpoint is hosted on Google Drive due to its size and is not tracked in this repository.

**Download link:** https://drive.google.com/drive/folders/13bx0xMu9Py2vFqFG8ocny2YVamw7EQOX

After downloading, place `unet_epoch_3.pth` directly inside this `fine-tuned-weights/` directory.

## Expected Structure After Download

```
fine-tuned-weights/
└── unet_epoch_3.pth
```

## What Happens If the File Is Missing

The app does not crash. `models/sdxl_pipeline.py` wraps weight loading in a `try/except` block and logs a warning:

```
WARNING — UNet checkpoint not found at fine-tuned-weights/unet_epoch_3.pth — using base SDXL weights.
```

Generation continues using the base `stabilityai/stable-diffusion-xl-base-1.0` weights. Output quality will be lower for jewelry-specific prompts, but the app remains fully functional.

## Do Not Commit `.pth` Files

`.pth` files are excluded by `.gitignore`. Do not force-add them — they are large binaries (several GB) that will bloat the repository history permanently.

## Training Details

- Base model: `stabilityai/stable-diffusion-xl-base-1.0`
- Trained component: UNet only
- Dataset: ~6,157 jewelry images (see `Dataset/README.md`)
- Training notebooks: see `legacy/` (archived — contains known issues)
