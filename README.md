# NTIRE 2026 Inference Package

## Contents
```
NTIRE2026-ELLIE-CVPR-TCD/
├── run.py                        # Inference script (self-contained)
├── greya_scalequery_reduced.py   # Model architecture
├── model.pth                     # Best checkpoint 
├── requirements.txt              # pip dependencies
└── README.md                     # This file
```

## Quick Start

```bash
# run this to reproduce result
python run.py 


## Output
- Enhanced images saved to `results_LL/`

## Arguments
| Arg | Default | Description |
|-----|---------|-------------|
| `--input_dir` | low | Folder of dark input images |
| `--model_path` | `model.pth` | Path to checkpoint |
| `--output_dir` | results_LL` | Output folder |

