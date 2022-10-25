# Totems: Physical Objects for Verifying Visual Integrity
### [Project Page](https://jingweim.github.io/totems/) | [Video](https://www.youtube.com/watch?v=xjyVAgOM5E4) | [arXiv](https://arxiv.org/abs/2209.13032) | [Paper](https://arxiv.org/pdf/2209.13032.pdf) | [Data (coming soon)](https://github.com/jingweim/totems)
<!-- (https://drive.google.com/drive/folders/1xyCeLqfkL3h1KPFDkcNvDNjRBivbW0Jw?usp=sharing) -->

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

## Dataset (coming soon)
<!-- ### Raw data (coming soon)
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
``` -->

## How to run

### Reconstruction
#### Example commands
```
# Fit NeRF to scene (jointly optimizing totem poses and NeRF)
python run.py --config configs/real/example_joint_pose.txt

# Fit NeRF to scene (fixed totem poses)
python run.py --config configs/real/example_initial_pose.txt

# Monitor reconstruction progress
tensorboard --logdir logs/summaries

# Check reconstruction results and save intermediate files for detection stage
python run.py --config configs/real/example_joint_pose.txt \
              --render_only --render_cam --render_totem --export_detect
```

### Detection (coming soon)
```

```

### Output Folder
```
logs/
    ${EXP1_NAME}/                           # Experiment folder
        010000.tar                          # Model at iteration 10000
        ...
        050000.tar                          # Model at iteration 50000
        args.txt                            # All arguments and values
        config.txt                          # A copy of the config file
        loss.txt                            # Losses every 100 steps
        render_050000/                      # Camera and totem view renderings
            camera_view.png  
            totem_views/ 
        detect_050000/                      # Detection intermediate files and results
            intermediate/                   # Intermediate files generated from dataset and trained model
                image.png                   # Input image
                recon.png                   # Reconstructed image
                totem_mask.png              # Merging individual totem masks
                protect_mask.png            # Generated protect mask
                protect_mask/               # Protect mask intermediate files
            results/                        # Detection results
                L1/
                    heatmap.npy             # grid_size x grid_size numpy array
                    heatmap_vis.png         # Resized to image size and color mapped with matplotlib 'jet'
                    heatmap_overlay.png     # Overlayed on top of manipulated(?) image
                    metrics.npy             # Dictionary storing image-wise L1 and LPIPS score, patch-wise L1 and LPIPS mAP, totem pose L1 (if applicable)
                LPIPS/                      # Same as the above folder, using LPIPS metric instead
                    ...
    ${EXP2_NAME}/
    ...
    summaries/                  # Tensorboard files
        ${EXP1_NAME}/
        ${EXP2_NAME}/
        ...
```


<!-- ### How to create custom dataset
Coming soon
 -->
 
## Citing
If you find our work useful, please consider citing:
```BibTeX
@article{ma2022totems
  author    = {Ma, Jingwei 
               and Chai, Lucy 
               and Huh, Minyoung 
               and Wang, Tongzhou 
               and Lim, Ser-Nam 
               and Isola, Phillip 
               and Torralba, Antonio},
  title     = {Totems: Physical Objects for Verifying Visual Integrity},
  journal   = {ECCV},
  year      = {2022},
}
