import os
import imageio
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn import metrics


def get_patches_numpy(imA, imB, y_valid, grid_size, patch_size):
    
    # Crop and rescale to [-1, 1]
    imA = imA[:y_valid, :, :].astype(float)/255*2-1
    imB = imB[:y_valid, :, :].astype(float)/255*2-1
    H, W, _ = imA.shape

    patchesA = np.empty((grid_size ** 2, patch_size, patch_size, 3))
    patchesB = np.empty((grid_size ** 2, patch_size, patch_size, 3))
    
    idx = 0
    for i in np.linspace(0, H - patch_size, grid_size).astype(int):
        for j in np.linspace(0, W - patch_size, grid_size).astype(int):
            patchesA[idx] = imA[i:i+patch_size, j:j+patch_size]
            patchesB[idx] = imB[i:i+patch_size, j:j+patch_size]
            idx += 1

    return patchesA, patchesB


def run_metrics(image_patches, recon_patches, image, protect_mask, grid_size, y_valid, out_dir, metric_name="L1", manip_mask=None):
 
    # Output paths
    heatmap_path = os.path.join(out_dir, 'heatmap.npy')
    vis_path = os.path.join(out_dir, 'heatmap_vis.png')
    overlay_path = os.path.join(out_dir, 'heatmap_overlay.png')
    metrics_path = os.path.join(out_dir, 'metrics.npy')

    # Save heatmap
    heatmap = np.abs(image_patches - recon_patches)
    heatmap = np.reshape(heatmap, (grid_size ** 2, -1))
    heatmap = np.reshape(np.mean(heatmap, axis=-1), (grid_size, grid_size))
    np.save(heatmap_path, heatmap)

    # Save color heatmap
    heatmap_resized = cv2.resize(heatmap, (image.shape[1], y_valid), cv2.INTER_CUBIC)
    plt.imsave(vis_path, heatmap_resized, cmap='jet')

    # Save overlay
    alpha = 0.3 # for colormap
    bn = 0.6 # for unprotected region
    heatmap_cmap = imageio.imread(vis_path)[..., :3]
    top_im = image[:y_valid]
    top_im = cv2.cvtColor(cv2.cvtColor(top_im, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR) # 3-channel grayscale
    protect_mask_valid = protect_mask[:y_valid]
    top = top_im * (1-protect_mask_valid[..., None]) * bn + (top_im * (1-alpha) + heatmap_cmap * alpha) * protect_mask_valid[..., None]

    overlay = image.copy()
    overlay = cv2.cvtColor(cv2.cvtColor(overlay, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR) # 3-channel grayscale
    overlay = overlay * bn
    overlay[:y_valid] = top
    overlay = overlay.astype('uint8')
    imageio.imwrite(overlay_path, overlay)
    print(f'Detection results saved in: {out_dir}')

    # Save metrics: patch-wise mAP
    if manip_mask:
        manip_gt = (manip_mask[:y_valid, :, 0] == 255)
        assert np.sum(manip_gt) > 0, "Manipulation mask must not be blank"
        manip_est = heatmap_resized.copy()
        protect_filter = np.where(protect_mask_valid)

        # Only evaluate protected pixels
        gt = manip_gt[protect_filter]
        est = manip_est[protect_filter]
        mAP = metrics.average_precision_score(gt, est)

        print(f'Patch-wise mAP {metric_name}: {mAP}')

        out = dict()
        out[f'patch_mAP_{metric_name}'] = mAP
        np.save(metrics_path, out)
        

def run_detect(out_dir, image, recon, protect_mask, totem_mask, manip_mask=None):
    '''
        Crop out totem region
        Sample image patches
        Compute patch-wise L1 and LPIPS
        Overlay heatmap

        
    '''

    # Load crop factor
    ys, _ = np.where(totem_mask) 
    y_valid = np.min(ys)

    # Load protect mask as {0,1} binary mask
    protect_mask_binary = (protect_mask/255).astype('uint8')

    # Get patches and patch label (if manipulation mask exists)
    grid_size = 30
    patch_size = 64
    image_patches, recon_patches = get_patches_numpy(image, recon, y_valid, grid_size, patch_size)

    # L1 metrics
    run_metrics(image_patches, recon_patches, image, protect_mask_binary, grid_size, y_valid, out_dir)

    # # LPIPS metrics (coming soon)
    # run_metrics(image_patches, recon_patches, image, protect_mask, grid_size, y_valid, out_dir, metric_name='LPIPS')