# Totems: Physical Objects for Verifying Visual Integrity
### [Project Page](https://jingweim.github.io/totems/) | [arXiv](https://arxiv.org/abs/2209.13032) | [Paper](https://arxiv.org/pdf/2209.13032.pdf) | [Data (coming soon)](https://jingweim.github.io/totems/)

This repository contains the official code release for Totems: Physical Objects for Verifying Visual Integrity. The code was based on this NeRF implementation [here](https://github.com/yenchenlin/nerf-pytorch/).

## Installation
```
git clone https://github.com/jingweim/totems.git
cd totems

# Make a conda environment
conda create --name totems python=3.6
conda activate totems

# Install requirements
pip install -r requirements.txt
```

## Pre-compiled dataset
### Real
```
# Dataset structure
```


### Synthetic
Coming soon

## How to run

### Reconstruction
```
# Fit NeRF to scene (jointly optimizing totem poses and NeRF)
python run.py --config configs/{DATASET}.txt

# Fit NeRF to scene (fixed totem poses)
Coming soon

# Check reconstruction results and save intermediate files for detection stage
python run.py --config configs/{DATASET}.txt --render_only --render_cam --render_totem --protect_mask
```

### Detection
```

```

### Custom Dataset
Coming soon


