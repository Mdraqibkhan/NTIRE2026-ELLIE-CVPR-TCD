# NTIRE 2026 Inference Package

## Contents
```
ntire2026_inference_package/
├── run.py                        # Inference script (self-contained)
├── greya_scalequery_reduced.py   # Model architecture
├── model.pth                     # Best checkpoint (PSNR 26.10, SSIM 0.9969)
├── requirements.txt              # pip dependencies
└── README.md                     # This file
```

## Quick Start

```bash
# run this to reproduce result
python run.py 


## Output
- Enhanced images saved to `submission_ntire/`
- ZIP file `ntire2026_submission.zip` created automatically (CodaBench ready)

## Arguments
| Arg | Default | Description |
|-----|---------|-------------|
| `--input_dir` | low | Folder of dark input images |
| `--model_path` | `model.pth` | Path to checkpoint |
| `--output_dir` | `submission_ntire` | Output folder |
| `--zip` | `ntire2026_submission.zip` | ZIP filename |
