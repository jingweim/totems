# Totems: Physical Objects for Verifying Visual Integrity
### [Project Page](https://jingweim.github.io/totems/) | [Video](https://www.youtube.com/watch?v=xjyVAgOM5E4) | [arXiv](https://arxiv.org/abs/2209.13032) | [Paper](https://arxiv.org/pdf/2209.13032.pdf) | [Data](https://drive.google.com/drive/folders/1JBWJrT4PzAPysaGYPJvfi-t_Xm4HrBBY?usp=share_link)

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

### Pre-compiled data ([link](https://drive.google.com/drive/folders/1JBWJrT4PzAPysaGYPJvfi-t_Xm4HrBBY?usp=share_link))
The pre-compiled dataset contains manipulated and unmanipulated photos that have gone through pre-processing (i.e. undistortion, computing initial totem pose) and are ready to run with the reconstruction and detection code. In the current release we only have one manipulation type: randomly added color patches. The other manipulation types are coming soon.
```
data-compiled/
    calib.npy                           # Shared camera calibration
    unmanipulated/
        JT8A8283/
            JT8A8283.JPG                # image from camera
            image.png                   # undistorted using calib.npy
            initial_totem_pose.npy      # totem poses used for initialization, estimated from totem masks
            totem_masks/                # annotated+undistorted totem masks, 0-n = left-right
                totem_000.png           # white = totem, black = scene
                totem_001.png
                ...
    color-patch/                        # Randomly added color patches
        JT8A8283_000/
            JT8A8283_000.png            # Manipulated image
            JT8A8283_000_mask.png       # Manipulation mask
            image.png                   # Maipulated image (undistorted)
            manip_mask.png              # Manipulation mask (undistorted)
            initial_totem_pose.npy      # totem poses used for initialization, estimated from totem masks
            totem_masks/                # annotated+undistorted totem masks, 0-n = left-right
                totem_000.png           # white = totem, black = scene
                totem_001.png
                ...
        ...
```

### Raw data (coming soon)
The raw dataset includes JPG files straight from a Canon EOS 5D Mark III camera, annotated totem masks, the checkerboard images used for camera calibration, and the computed calibration.

### Preprocess code (coming soon)
Scripts that preprocess the raw data into the compiled version, compute camera calibration, or generate manipulations.


<!-- ### Raw data
The raw dataset includes JPG files straight from a Canon EOS 5D Mark III camera, annotated totem masks, the checkerboard images used for camera calibration, and the computed calibration.

raw files: /data/vision/phillipi/gan-training/totem/resources/totems/data
calibration files: /data/vision/torralba/virtualhome/realvirtualhome/realvirtualhome/totem/totems-misc/calib/calib_data/11-04-2021
```
data-raw/
    JT8A8282/
        JT8A8282.JPG                    # Totem-protected photo
        totem_masks/                    # Annotated totem masks
            totem_000.png               # white = totem, black = scene
            totem_001.png
            ...
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
```
# Fit NeRF to scene (jointly optimizing totem poses and NeRF)
python run.py --config configs/real/example_joint_pose.txt

# Fit NeRF to scene (fixed totem poses)
python run.py --config configs/real/example_initial_pose.txt

# Monitor reconstruction progress
tensorboard --logdir logs/summaries

# Check reconstruction results and save intermediate files for detection stage
python run.py --config configs/real/example_joint_pose.txt \
              --render_only --render_cam --render_totem
```

### Detection
```
python run.py --config configs/real/example_joint_pose.txt --detect_only
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
                    metrics.npy             # Dictionary storing image-wise and patch-wise metrics
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
