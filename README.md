# Totems: Physical Objects for Verifying Visual Integrity
### [Project Page](https://jingweim.github.io/totems/) | [arXiv](https://arxiv.org/abs/2209.13032) | [Paper](https://arxiv.org/pdf/2209.13032.pdf) | [Data](https://drive.google.com/drive/folders/1xyCeLqfkL3h1KPFDkcNvDNjRBivbW0Jw?usp=sharing)

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

## Dataset
### Raw data (coming soon)
The raw dataset includes JPG and CR2 image files straight from the camera, annotated totem masks, and the checkerboard images used for camera calibration.

### Pre-compiled data ([link](https://drive.google.com/drive/folders/1xyCeLqfkL3h1KPFDkcNvDNjRBivbW0Jw?usp=sharing))
The pre-compiled dataset has gone through pre-processing (i.e. undistortion, computing initial totem pose) and is ready to run with the reconstruction and detection code. 
```
data/
    real/
        calib.npy                       # Camera calibration
        JT8A8283/
            JT8A8283.JPG                # image from camera
            image.png                   # undistorted using calib.npy
            initial_totem_pose.npy      # totem poses used for initialization, estimated from totem masks
            totem_masks/                # annotated totem masks, 0-n = left-right
                totem_000.png           # white = totem, black = scene
                totem_001.png
                ...
```

### Manipulated data (coming soon)
This dataset contains the 4 types of manipulations in the paper: 1) randomly added color patches, 2) image splice, 3) Photoshop content aware fill, 4) reference shift. The manipulated images and the ground truth masks of the manipulation are provided.
```
data-manipulated/
    color-patch/                        # Randomly added color patches
        JT8A8283_000.png                # Manipulated image
        JT8A8283_000_mask.png           # Mask of the manipulation
        ...
    splice/                             # Image splicing, copying content from source image to target image
        JT8A8283_JT8A8292.png           # Manipulated image, naming = {tgt}_{src}.png
        JT8A8283_JT8A8292_mask.png      # Mask of the manipulation
        ...
    content-aware-fill/                 # Photoshop's content-aware-fill
        JT8A8283_000.png                # Manipulated image
        JT8A8283_000_mask.png           # Mask of the manipulation
        ...
    reference_shift/                    # Shifting person in both camera and totem views to a scene reference point
        JT8A8283_000.png                # Manipulated image
        JT8A8283_000_mask.png           # Mask of the manipulation
        ...
```

## How to run

### Reconstruction
#### Example commands
```
# Fit NeRF to scene (jointly optimizing totem poses and NeRF)
python run.py --config configs/real/JT8A8283_joint_pose.txt

# Fit NeRF to scene (fixed totem poses)
python run.py --config configs/real/JT8A8283_initial_pose.txt

# Monitor reconstruction progress
tensorboard --logdir logs/summaries

# Check reconstruction results and save intermediate files for detection stage
python run.py --config configs/real/JT8A8283_joint_pose.txt \
              --render_only --render_cam --render_totem --export_detect
```

### Detection
```

```

### How to create custom dataset
Coming soon


