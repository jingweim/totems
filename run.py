import os, sys
import numpy as np
import imageio
import json
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils
from tqdm import tqdm, trange
import cv2
import matplotlib.pyplot as plt

from run_nerf_helpers import *
from joint_helpers import *

from PIL import Image
import yaml
from projections import *
import pidfile
import time

# from load_llff import load_llff_data
# from load_deepvoxels import load_dv_data
# from load_blender import load_blender_data
# from load_blender import pose_spherical

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
DEBUG = False


def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn
    def ret(inputs):
        return torch.cat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
    return ret


def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024*64):
    """Prepares inputs and applies network 'fn'.
    """
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    embedded = embed_fn(inputs_flat)

    if viewdirs is not None:
        input_dirs = viewdirs[:,None].expand(inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat([embedded, embedded_dirs], -1)

    outputs_flat = batchify(fn, netchunk)(embedded)
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs


def batchify_rays(rays_flat, chunk=1024*32, **kwargs):
    """Render rays in smaller minibatches to avoid OOM.
    """
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(rays_flat[i:i+chunk], **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k : torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret


def render(chunk=1024*32, rays=None, c2w=None, 
           near=0., far=1., use_viewdirs=False, c2w_staticcam=None, 
           scene_params=None, proj_params=None, **kwargs):
    """Render rays
    Args:
      H: int. Height of image in pixels.
      W: int. Width of image in pixels.
      focal: float. Focal length of pinhole camera.
      chunk: int. Maximum number of rays to process simultaneously. Used to
        control maximum memory usage. Does not affect final results.
      rays: array of shape [2, batch_size, 3]. Ray origin and direction for
        each example in batch.
      c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
      ndc: bool. If True, represent ray origin, direction in NDC coordinates.
      near: float or array of shape [batch_size]. Nearest distance for a ray.
      far: float or array of shape [batch_size]. Farthest distance for a ray.
      use_viewdirs: bool. If True, use viewing direction of a point in space in model.
      c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for 
       camera while using other c2w argument for viewing directions.
    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays.
      disp_map: [batch_size]. Disparity map. Inverse of depth.
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
      extras: dict with everything returned by render_rays().
    """
    if c2w is not None:
        # special case to render full image
        rays_o, rays_d = get_rays(c2w)
    else:
        # use provided ray batch
        rays_o, rays_d = rays

    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        if c2w_staticcam is not None:
            # special case to visualize effect of viewdirs
            rays_o, rays_d = get_rays(c2w_staticcam)
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1,3]).float()

    sh = rays_d.shape # [..., 3]
    if proj_params['ndc']:
        # note: I assume far can be set the same way for all projections
        # since ndc is linear in disparity, should increase the far bound
        # so that it samples enough points within the scene range
        far = 1000*far # note: changed it from 20
        rays_o, rays_d, th = my_ndc_rays(scene_params['cam_sensor_h'],
                                         scene_params['cam_sensor_w'],
                                         scene_params['focal'],
                                         near, far, rays_o, rays_d)
        # if using ndc, rays shoudl be filtered beforehand
        assert(torch.sum(th).item() == len(th))
        # in their ndc implementation, they assume the near plane=1
        # rays_o, rays_d = ndc_rays(1., rays_o, rays_d)
        # the difference in my implementation is recompute ndc at
        # the specified near and far planes

    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1,3]).float()
    rays_d = torch.reshape(rays_d, [-1,3]).float()


    #far = 6. * torch.ones_like(rays_d[...,:1])
    #near = rays_o[:, 2:]
    near, far = near * torch.ones_like(rays_d[...,:1]), far * torch.ones_like(rays_d[...,:1])
    rays = torch.cat([rays_o, rays_d, near, far], -1)
    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)

    # Render and reshape
    kwargs['scene_params'] = scene_params
    kwargs['proj_params'] = proj_params
    all_ret = batchify_rays(rays, chunk, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map', 'disp_map', 'acc_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k : all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]


def render_path(render_poses, chunk, render_kwargs, gt_imgs=None, savedir=None, render_factor=0):


    if render_factor!=0:
        # Render downsampled for speed
        H = H//render_factor
        W = W//render_factor
        focal = focal/render_factor

    rgbs = []
    disps = []

    t = time.time()
    for i, c2w in enumerate(tqdm(render_poses)):
        print(i, time.time() - t)
        t = time.time()
        rgb, disp, acc, _ = render(chunk=chunk, c2w=c2w[:3,:4], **render_kwargs)
        rgbs.append(rgb.cpu().numpy())
        disps.append(disp.cpu().numpy())
        if i==0:
            print(rgb.shape, disp.shape)

        """
        if gt_imgs is not None and render_factor==0:
            p = -10. * np.log10(np.mean(np.square(rgb.cpu().numpy() - gt_imgs[i])))
            print(p)
        """

        if savedir is not None:
            rgb8 = to8b(rgbs[-1])
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)


    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)

    return rgbs, disps


def create_nerf(args):
    """Instantiate NeRF's MLP model.
    """
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed)

    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed)
    output_ch = 5 if args.N_importance > 0 else 4
    skips = [4]
    model = NeRF(D=args.netdepth, W=args.netwidth,
                 input_ch=input_ch, output_ch=output_ch, skips=skips,
                 input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
    grad_vars = list(model.parameters())

    model_fine = None
    if args.N_importance > 0:
        model_fine = NeRF(D=args.netdepth_fine, W=args.netwidth_fine,
                          input_ch=input_ch, output_ch=output_ch, skips=skips,
                          input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
        grad_vars += list(model_fine.parameters())

    network_query_fn = lambda inputs, viewdirs, network_fn : run_network(inputs, viewdirs, network_fn,
                                                                embed_fn=embed_fn,
                                                                embeddirs_fn=embeddirs_fn,
                                                                netchunk=args.netchunk)

    # Create optimizer
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    start = 0
    basedir = args.basedir
    expname = args.expname

    ##########################

    # Load checkpoints
    if args.ft_path is not None and args.ft_path!='None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if 'tar' in f]

    print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[args.ckpt_idx]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)

        start = ckpt['global_step']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        # Load model
        model.load_state_dict(ckpt['network_fn_state_dict'])
        if model_fine is not None:
            model_fine.load_state_dict(ckpt['network_fine_state_dict'])

    ##########################

    render_kwargs_train = {
        'network_query_fn' : network_query_fn,
        'perturb' : args.perturb,
        'N_importance' : args.N_importance,
        'network_fine' : model_fine,
        'N_samples' : args.N_samples,
        'network_fn' : model,
        'use_viewdirs' : args.use_viewdirs,
        'white_bkgd' : args.white_bkgd,
        'raw_noise_std' : args.raw_noise_std,
        'proj_params': dict(ndc=args.use_ndc, cube=args.use_cube,
                            lindisp=args.lindisp,
                            shift_o_normalize_z=args.shift_o_normalize_z)
    }

    # change to set these directly using args instead
    ## NDC only good for LLFF-style forward facing data
    #if args.dataset_type != 'llff' or args.no_ndc:
    #    print('Not ndc!')
    #    render_kwargs_train['ndc'] = False
    #    render_kwargs_train['lindisp'] = args.lindisp


    render_kwargs_test = {k : render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    if len(ckpts) > 0 and not args.no_reload:
        render_kwargs_train['totem_pos'] = ckpt['totem_pos'].detach().cpu().numpy()

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer


def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False,
                pts=None, proj_params=None, pytest=False):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)

    # import pdb; pdb.set_trace()
    if proj_params['cube']:
        pts_norm = torch.norm(pts[:, :-1, :] - pts[:, 1:, :], dim=-1)
        # print(pts_norm)
        # add an arbitary large constant to the end
        dists = torch.cat([pts_norm, torch.Tensor([1e10]).expand(pts_norm[...,:1].shape)], -1)
        # note that z_vals was never scaled, but it uses scaled pts to
        # determine attentuation
    else:
        dists = z_vals[...,1:] - z_vals[...,:-1]
        dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape)], -1)  # [N_rays, N_samples]
        # import pdb; pdb.set_trace()
        dists = dists * torch.norm(rays_d[...,None,:], dim=-1)
        # print(dists)
        pts_norm = torch.norm(pts[:, :-1, :] - pts[:, 1:, :], dim=-1)
        # note that dists should measure distance between points
        # so this double checks it
        # print(pts_norm)
        # if not torch.allclose(dists[:, :-1], pts_norm, atol=1e-3):
        #     print("here")
        #     print(torch.max(torch.abs(dists[:, :-1] - pts_norm)))
        #     from IPython import embed; embed()
        #     print("hello")


    rgb = torch.sigmoid(raw[...,:3])  # [N_rays, N_samples, 3]
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[...,3].shape) * raw_noise_std

        # Overwrite randomly sampled data if pytest
        if pytest:
            np.random.seed(0)
            noise = np.random.rand(*list(raw[...,3].shape)) * raw_noise_std
            noise = torch.Tensor(noise)

    alpha = raw2alpha(raw[...,3] + noise, dists)  # [N_rays, N_samples]
    # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]

    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
    depth_map_norm = depth_map / torch.sum(weights, -1)
    acc_map = torch.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1.-acc_map[...,None])

    # changed to return depth_map_norm rather than depth_map
    return rgb_map, disp_map, acc_map, weights, depth_map_norm


def render_rays(ray_batch,
                network_fn,
                network_query_fn,
                N_samples,
                retraw=False,
                lindisp=False,
                perturb=0.,
                N_importance=0,
                network_fine=None,
                white_bkgd=False,
                raw_noise_std=0.,
                scene_params=None,
                proj_params=None,
                save_rays=None,
                rays_test_tstart=None,
                verbose=False,
                pytest=False):
    """Volumetric rendering.
    Args:
      ray_batch: array of shape [batch_size, ...]. All information necessary
        for sampling along a ray, including: ray origin, ray direction, min
        dist, max dist, and unit-magnitude viewing direction.
      network_fn: function. Model for predicting RGB and density at each point
        in space.
      network_query_fn: function used for passing queries to network_fn.
      N_samples: int. Number of different times to sample along each ray.
      retraw: bool. If True, include model's raw, unprocessed predictions.
      lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
      perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
        random points in time.
      N_importance: int. Number of additional times to sample along each ray.
        These samples are only passed to network_fine.
      network_fine: "fine" network with same spec as network_fn.
      white_bkgd: bool. If True, assume a white background.
      raw_noise_std: ...
      verbose: bool. If True, print more debugging info.
    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
      disp_map: [num_rays]. Disparity map. 1 / depth.
      acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
      raw: [num_rays, num_samples, 4]. Raw predictions from model.
      rgb0: See rgb_map. Output for coarse model.
      disp0: See disp_map. Output for coarse model.
      acc0: See acc_map. Output for coarse model.
      z_std: [num_rays]. Standard deviation of distances along ray for each
        sample.
    """
    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:,0:3], ray_batch[:,3:6] # [N_rays, 3] each
    viewdirs = ray_batch[:,-3:] if ray_batch.shape[-1] > 8 else None
    bounds = torch.reshape(ray_batch[...,6:8], [-1,1,2])
    near, far = bounds[...,0], bounds[...,1] # [-1,1]

    # added
    # import pdb; pdb.set_trace()
    if proj_params['ndc']:
        # ndc is linear in disparity  by default
        # nerf assumes far = infinity, near = 1, so max_t = 1
        # since we use a set far bound, we need to compute max_t = 2/dz
        # t = 0 starts at the shifted origin oz=near
        t_start = 0. if rays_test_tstart is None else rays_test_tstart
        t_vals = torch.linspace(t_start, 1., steps=N_samples)
        # in ndc, dz is the same for all rays, so just use the first
        max_t = 2 / rays_d[0, -1]
        z_vals = t_vals * max_t
    else:
        t_start = 0. if rays_test_tstart is None else rays_test_tstart
        t_vals = torch.linspace(t_start, 1., steps=N_samples)
        if not proj_params['lindisp']:
            z_vals = near * (1.-t_vals) + far * (t_vals)
        else:
            z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

    # z_vals is the time along the ray
    z_vals = z_vals.expand([N_rays, N_samples])

    if perturb > 0.:
        # get intervals between samples
        mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        upper = torch.cat([mids, z_vals[...,-1:]], -1)
        lower = torch.cat([z_vals[...,:1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape)

        # Pytest, overwrite u with numpy's fixed random numbers
        if pytest:
            np.random.seed(0)
            t_rand = np.random.rand(*list(z_vals.shape))
            t_rand = torch.Tensor(t_rand)

        z_vals = lower + (upper - lower) * t_rand

    # ADDED:
    # import pdb; pdb.set_trace()
    if proj_params['ndc']:
        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None]
    elif proj_params['cube']:
        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None]
        pts = euc_to_cube(scene_params['cam_sensor_h'], 
                          scene_params['cam_sensor_w'], 
                          scene_params['focal'], near, far, pts)
    else:
        # note: jingwei added this --> enforces sampling at equal depths
        # along each ray
        t_vals = (z_vals - rays_o[..., 2:]) / rays_d[..., 2:]
        if proj_params['shift_o_normalize_z']:
            # if rays are shifted origin = 0 and rays_dz= 1
            # it doesn't do anything -- check that
            assert(torch.allclose(t_vals, z_vals, atol=1e-4))
        pts = rays_o[...,None,:] + rays_d[...,None,:] * t_vals[...,:,None] # [N_rays, N_samples, 3]

#     raw = run_network(pts)
    if save_rays:
        save_dict = {
            'pts': pts.cpu(),
            'scene_params': scene_params,
            'proj_params': proj_params
        }
        torch.save(save_dict, save_rays)
    # import pdb;  pdb.set_trace()
    raw = network_query_fn(pts, viewdirs, network_fn)
    rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(
        raw, z_vals, rays_d, raw_noise_std, white_bkgd,
        proj_params=proj_params, pts=pts, pytest=pytest)

    if N_importance > 0:

        rgb_map_0, disp_map_0, acc_map_0, depth_map_0 = rgb_map, disp_map, acc_map, depth_map

        z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        z_samples = sample_pdf(z_vals_mid, weights[...,1:-1], N_importance, det=(perturb==0.), pytest=pytest)
        z_samples = z_samples.detach()

        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)

        # ADDED: transform the reprojected points
        if proj_params['ndc']:
            pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None]
        elif proj_params['cube']:
            pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None]
            pts = euc_to_cube(scene_params['cam_sensor_h'], 
                              scene_params['cam_sensor_w'], 
                              scene_params['focal'], near, far, pts)
        else:
            # note: jingwei added this --> enforces sampling at equal depths
            # along each ray
            t_vals = (z_vals - rays_o[..., 2:]) / rays_d[..., 2:]
            if proj_params['shift_o_normalize_z']:
                # if rays are shifted origin = 0 and rays_dz= 1
                # it doesn't do anything -- check that
                assert(torch.allclose(t_vals, z_vals, atol=1e-4))
            pts = rays_o[...,None,:] + rays_d[...,None,:] * t_vals[...,:,None] # [N_rays, N_samples, 3]


        run_fn = network_fn if network_fine is None else network_fine
#         raw = run_network(pts, fn=run_fn)
        raw = network_query_fn(pts, viewdirs, run_fn)

        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(
            raw, z_vals, rays_d, raw_noise_std, white_bkgd,
            proj_params=proj_params, pts=pts, pytest=pytest)
    # import pdb; pdb.set_trace()
    ret = {'rgb_map' : rgb_map, 'disp_map' : disp_map, 'acc_map' : acc_map, 'depth_map': depth_map}
    if retraw:
        ret['raw'] = raw
    if N_importance > 0:
        ret['rgb0'] = rgb_map_0
        ret['disp0'] = disp_map_0
        ret['acc0'] = acc_map_0
        ret['depth0'] = depth_map_0
        ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]

    for k in ret:
        if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
            print(f"! [Numerical Error] {k} contains nan or inf.")

    return ret


def config_parser():

    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, 
                        help='config file path')
    parser.add_argument("--expname", type=str, 
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/', 
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='./data/llff/fern', 
                        help='input data directory')

    # training options
    parser.add_argument("--netdepth", type=int, default=8, 
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256, 
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int, default=8, 
                        help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256, 
                        help='channels per layer in fine network')
    parser.add_argument("--N_rand", type=int, default=32*32*4, 
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float, default=5e-4, 
                        help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=250, 
                        help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--totem_lrate", type=float, default=5e-4, 
                        help='totem pose residual learning rate')
    parser.add_argument("--chunk", type=int, default=1024*32, 
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024*64, 
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_batching", action='store_true', 
                        help='only take random rays from 1 image at a time')
    parser.add_argument("--no_reload", action='store_true', 
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None, 
                        help='specific weights npy file to reload for coarse network')

    # rendering options
    parser.add_argument("--N_samples", type=int, default=64, 
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_true', 
                        help='use full 5D input instead of 3D')
    parser.add_argument("--i_embed", type=int, default=0, 
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10, 
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4, 
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0., 
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')

    parser.add_argument("--render_only", action='store_true', 
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_cam", action='store_true', 
                        help='render camera view')
    parser.add_argument("--render_totem", action='store_true', 
                        help='render totem views')
    parser.add_argument("--render_test", action='store_true', 
                        help='render the test set instead of render_poses path')
    parser.add_argument("--render_factor", type=int, default=0, 
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')
    parser.add_argument("--ckpt_idx", type=int, default=-1, 
                        help='checkpoint to load')

    # training options
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops') 

    # dataset options
    parser.add_argument("--dataset_type", type=str, default='llff', 
                        help='options: llff / blender / deepvoxels')
    parser.add_argument("--data_folder", type=str, required=True, 
                        help='folder of preprocessed npy files')
    parser.add_argument("--near", type=float, required=True, 
                        help='manually defined min depth bound')
    parser.add_argument("--far", type=float, required=True, 
                        help='manually defined max depth bound')
    parser.add_argument("--testskip", type=int, default=8, 
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')
    parser.add_argument("--path_totem_pos", type=str, required=True, 
                        help='path to npy file that stores totem positions')
    parser.add_argument("--path_totem_gt", type=str,
                        help='path to npy file that stores groundtruth totem positions')
    parser.add_argument("--folder_mask", type=str, required=True, 
                        help='folder of totem masks')
    parser.add_argument("--totem_radius", type=float, required=True, 
                        help='totem radius in unit of meters')
    parser.add_argument("--path_im", type=str, required=True, 
                        help='path to image')
    parser.add_argument("--iou_loss_weight", type=float, required=True, 
                        help='weight to be multiplied to the iou loss')
    parser.add_argument("--img_loss_weight", type=float, required=True, 
                        help='weight to be multiplied to the iou loss')
    parser.add_argument("--totem_gamma", type=float, required=True, 
                        help='gamma for totem pose lr decay')
    parser.add_argument("--num_iterations_stage_one", type=int,  
                        help='number of iterations before training totem pos and nerf jointly')
    parser.add_argument("--calib_npy", type=str,  
                        help='camera calibration parameters')
    parser.add_argument("--path_glare_mask", type=str, 
                        help='mask of glare pixels so we can ignore them')
    parser.add_argument("--n_totem", type=float, default=1.52,
                        help='totem index of refraction, generally use 1.52 unless known')

    ## deepvoxels flags
    parser.add_argument("--shape", type=str, default='greek', 
                        help='options : armchair / cube / greek / vase')

    ## blender flags
    parser.add_argument("--white_bkgd", action='store_true', 
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--half_res", action='store_true', 
                        help='load blender synthetic data at 400x400 instead of 800x800')

    ## llff flags
    parser.add_argument("--factor", type=int, default=8, 
                        help='downsample factor for LLFF images')
    parser.add_argument("--no_ndc", action='store_true', 
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.add_argument("--lindisp", action='store_true', 
                        help='sampling linearly in disparity rather than depth')
    parser.add_argument("--spherify", action='store_true', 
                        help='set for spherical 360 scenes')
    parser.add_argument("--llffhold", type=int, default=8, 
                        help='will take every 1/N images as LLFF test set, paper uses 8')

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=100, 
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_img",     type=int, default=500, 
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=50000, 
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=25000, 
                        help='frequency of testset saving')
    parser.add_argument("--i_video",   type=int, default=50000, 
                        help='frequency of render_poses video saving')

    # ADDITIONAL ADDED FLAGS
    parser.add_argument("--which_rays", type=str, nargs='+',
                        default=['totem_000', 'totem_001', 'totem_002', 'totem_003'],
                        help="which rays to use for training, e.g. cam_train, totem_XXX")
    parser.add_argument('--shift_o_normalize_z', action='store_true',
                        help="shifts all rays origin to z=0, direction z velocity=1 (this is like nerf's cam rays)")
    parser.add_argument('--filter_view', action='store_true',
                        help="remove totem rays that are outside the camera view frustum")
    parser.add_argument('--use_ndc', action='store_true',
                        help="use the ndc projection of rays --> maps to cube with z linear in *disparity*")
    parser.add_argument('--use_cube', action='store_true',
                        help="use the cube projection of rays --> maps to cube with z linear in *distance*")
    parser.add_argument("--midas_npy", type=str, 
                        help='path to midas rays npy file')
    parser.add_argument("--w_depth", type=float, default= 0.04,
                        help="loss weight for depth regularization")

    # test options
    parser.add_argument('--is_test', action='store_true',
                        help="run test() instead of train")
    parser.add_argument("--cam_npy", type=str, 
                        help='path to camera rays npy file')
    parser.add_argument("--test_mask", type=str, 
                        help='path to mask for computing a filtered test MSE')
    parser.add_argument("--mask_folder", type=str, 
                        help='path to folder with totem masks')
    parser.add_argument("--im_path", type=str, 
                        help='path to original image with groundtruth cam and totems')
    parser.add_argument("--totems_npy_folder", type=str, 
                        help='path to folder with totem rays npy files')
    return parser


def train(args):
    
    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    pidfile.exit_if_job_done(os.path.join(basedir, expname))
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    # Load data
    if args.dataset_type == 'real' or args.dataset_type == 'mitsuba':
        # near, far = min and max depth range
        near = args.near
        far = args.far
   
        # load the params file
        with open(os.path.join(args.data_folder, 'params.yml')) as file:
            scene_params = yaml.load(file, Loader=yaml.FullLoader)
            print(scene_params)

        # downsample test mask and get filter idxs
        ds_factor = int(args.cam_npy.split('.')[0][-1])
        w_ds, h_ds = scene_params['image_width']//ds_factor, scene_params['image_height']//ds_factor
        test_mask_ds = cv2.resize(cv2.imread(args.test_mask), (w_ds, h_ds), interpolation=cv2.INTER_NEAREST)
        test_mask_ds = test_mask_ds.reshape(-1, 3)
        test_filter = torch.from_numpy((test_mask_ds == 0).astype(int)).to(device)

    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return

    # sanity checks
    if args.filter_view:
        # may be fine without this one
        assert(args.shift_o_normalize_z)
    if args.use_ndc:
        assert(args.shift_o_normalize_z)
        assert(args.filter_view)
        assert(not args.use_cube)
    if args.use_cube:
        assert(args.shift_o_normalize_z)
        # assert(args.filter_view)
        assert(not args.use_ndc)

    # Create nerf model
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args)
    global_step = start
    bds_dict = {
        'near' : near,
        'far' : far,
    }
    # note: this is where far is set if specified, 
    # otherwise (e.g. in original nerf llff) it is set to 1 in render
    # which corresponds to infinite depth using NDC
    # add bounds to scene params for vis_rays_batch.ipynb
    scene_params.update(bds_dict)
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    # added:
    # bundle in scene params too
    render_kwargs_train.update({'scene_params': scene_params})
    render_kwargs_test.update({'scene_params': scene_params})

    # Create totem pos parameters
    # Two stage training -> initially parameters not set to train
    from torch.optim.lr_scheduler import MultiStepLR
    init_totem_pos = np.load(args.path_totem_pos)/100
    N_totems = len(init_totem_pos)
    if 'totem_pos' in render_kwargs_train.keys():
        totem_pos_residual = render_kwargs_train['totem_pos'] - init_totem_pos
    else:
        totem_pos_residual = np.zeros((N_totems, 3))
    totem_pos_net = LearnTotemPos(init_totem_pos, totem_pos_residual, req_grad=False, device=device).to(device)
    opt_totem_pos = torch.optim.Adam(totem_pos_net.parameters(), lr=args.totem_lrate)
    scheduler_totem_pos = MultiStepLR(opt_totem_pos, milestones=list(range(0, 50000-args.num_iterations_stage_one, 100)), gamma=args.totem_gamma)

    # Parameters for getting training rays
    path_im = args.path_im
    im = cv2.imread(path_im)[..., ::-1]
    folder_mask = args.folder_mask
    fnames_mask = sorted(os.listdir(folder_mask))
    paths_mask = [os.path.join(folder_mask, fname) for fname in fnames_mask]
    masks = [cv2.imread(paths_mask[i])[..., 0] for i in range(N_totems)]
    if args.path_glare_mask:
        glare_mask = 1-cv2.imread(args.path_glare_mask)/255
        totem_pixs = [np.where(mask*glare_mask[..., 0] > 0) for mask in masks]
    else:
        totem_pixs = [np.where(mask) for mask in masks]
    N_totem_pixs = [len(ys) for (ys, _) in totem_pixs]
    totem_ids = np.arange(N_totems)
    boxBs = [[min(xs), min(ys), max(xs), max(ys)] for (ys, xs) in totem_pixs]
    
    # Get camera rays
    H, W = scene_params['image_height'], scene_params['image_width']
    mtx = np.load(args.calib_npy, allow_pickle=True).item()['mtx']
    cx, cy = mtx[0][2], mtx[1][2]
    cam_ray_ds_unnorm, cam_ray_ds = get_cam_ray_ds_calib(W, H, mtx)
    pos_map = cam_ray_ds
    cam_ray_o = torch.tensor([0,0,0], dtype=torch.float32).to(device)
    pinhole_pos = np.array([0,0,0])
    totem_radius = args.totem_radius # this /100 already
    n_air = 1.0
    n_totem = args.n_totem
    save_totem_pos = True # visualize totem position in xy and xz planes
    print_time = False # flag for speed-up logging

    # create folders for visualizing totem pose progression
    if save_totem_pos:
        '''
        import colorsys
        HSV_tuples = [(x*1.0/N_totems, 0.5, 0.5) for x in range(N_totems)]
        RGB_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)
        totem_colors = list(RGB_tuples)
        # visualize totem positions in xy and xz planes here 
        folder_totem_pos_vis = os.path.join(basedir, expname, 'totem_pos_vis')
        os.makedirs(folder_totem_pos_vis, exist_ok=True)
        '''
        # saving totem positions in npy files here
        folder_totem_pos = os.path.join(basedir, expname, 'totem_pos')
        os.makedirs(folder_totem_pos, exist_ok=True)


    # Train
    N_rand = args.N_rand
    N_iters = 50000 + 1
    print('Begin')
    loss_txt = os.path.join(basedir, expname, 'loss.txt')
    f = open(loss_txt, 'w+')
    f.close()

    writer = SummaryWriter(os.path.join(basedir, 'summaries', expname))

    start = start + 1
    for i in trange(start, N_iters):
        # Block 0
        start0 = time.time()
        if i >= args.num_iterations_stage_one and (not totem_pos_net.totem_pos_residual.requires_grad):
            totem_pos_net.requires_grad_()
            totem_pos_net.train()
        end0 = '%.4f seconds' % (time.time()-start0)
        if print_time:
            print('Block 0: %s' % end0)

        # Block 1
        time0 = time.time()
        start1 = time.time()
        if i % N_totems == 0:
            np.random.shuffle(totem_ids)
        totem_id = totem_ids[i % N_totems]
        totem_pos = totem_pos_net(totem_id) # learnable parameters
        totem_pos_np = totem_pos.detach().cpu().numpy()
        end1 = '%.4f seconds' % (time.time()-start1)
        if print_time:
            print('Block 1: %s' % end0)

        # Block 2 (Block 3 merged)
        start2 = time.time()
        # Shuffle all cam rays (mask pixels)
        N_pixs = N_totem_pixs[totem_id]         
        pix_ids = np.arange(N_pixs)
        np.random.shuffle(pix_ids)
        # [Here]
        # [SpeedUp] Divide shuffled cam rays into chunks, take on extra chunks if not enough rays
        size_chunk = N_rand * 2
        N_chunks = N_pixs // size_chunk + 1
        ys, xs = [], []
        ray_os_np, ray_ds_np = [], []
        for chunk_idx in range(N_chunks):
            start_idx = chunk_idx*size_chunk
            end_idx = min((chunk_idx+1)*size_chunk, N_pixs)
            pix_ids_chunk = pix_ids[start_idx:end_idx]
            ys_chunk = totem_pixs[totem_id][0][pix_ids_chunk]
            xs_chunk = totem_pixs[totem_id][1][pix_ids_chunk]
            cam_ray_ds_chunk = pos_map[ys_chunk, xs_chunk]
            cam_ray_os_chunk = np.broadcast_to(pinhole_pos, cam_ray_ds_chunk.shape)

            # Refraction(numpy)
            # Filter1: cam rays that don't intersect with the totem
            D, valid_idx_1 = line_sphere_intersection_v2(totem_pos_np, totem_radius, cam_ray_ds_chunk, cam_ray_os_chunk)
            OD = (D-pinhole_pos)/LA.norm(D-pinhole_pos, axis=1)[:, None]
            AD = (D-totem_pos_np)/LA.norm(D-totem_pos_np, axis=1)[:, None]
            DE= get_refracted_ray(OD, AD, n_air, n_totem)
            E, valid_idx_2 = line_sphere_intersection_v2(totem_pos_np, totem_radius, DE, D, use_min=False)
            EA = (totem_pos_np-E)/LA.norm(totem_pos_np-E, axis=1)[:, None]
            direction = get_refracted_ray(DE, EA, n_totem, n_air)
            ray_os_np_chunk = E
            ray_ds_np_chunk = direction
            ys_chunk = ys_chunk[valid_idx_1][valid_idx_2]
            xs_chunk = xs_chunk[valid_idx_1][valid_idx_2]

            # Filter2: totem rays that are too noisy by NDC threshold
            if args.filter_view:
                _, _, rays_threshold = my_ndc_rays_np(
                    scene_params['cam_sensor_h'], scene_params['cam_sensor_w'],
                    scene_params['focal'],  near, far, ray_os_np_chunk, ray_ds_np_chunk, thresh=2.0)
                valid_idx_3 = np.where(rays_threshold == 1)[0]
                ys_chunk = ys_chunk[valid_idx_3]
                xs_chunk = xs_chunk[valid_idx_3]
                ray_os_np_chunk = ray_os_np_chunk[valid_idx_3]
                ray_ds_np_chunk = ray_ds_np_chunk[valid_idx_3]

            # Merge chunk with previous data
            if len(ys): # no longer an empty list
                ys = np.concatenate([ys, ys_chunk], axis=0)
                xs = np.concatenate([xs, xs_chunk], axis=0)
                ray_os_np = np.concatenate([ray_os_np, ray_os_np_chunk], axis=0)
                ray_ds_np = np.concatenate([ray_ds_np, ray_ds_np_chunk], axis=0)
            else:
                ys = ys_chunk
                xs = xs_chunk
                ray_os_np = ray_os_np_chunk
                ray_ds_np = ray_ds_np_chunk

            # If accumulated > N_rand rays, exit loop
            if len(ys) > N_rand:
                break

        end2 = '%.4f seconds' % (time.time()-start2)
        if print_time:
            print('Block 2: %s' % end2)
        

        # Block 5
        start5 = time.time()
        # Next, sample N_rand rays from filtered cam_rays
        # Everything below is still numpy
        assert len(ys) >= N_rand, "totem pose too far off from mask, not enough rays to sample"
        ys_selected = ys[:N_rand]
        xs_selected = xs[:N_rand]
        cam_ray_ds_selected = pos_map[ys_selected, xs_selected]
        cam_ray_ds_selected = torch.from_numpy(cam_ray_ds_selected.astype('float32')).to(device)

        # torch from here on
        # Refraction and check for nans in forward calculations
        D = line_sphere_intersection_torch(totem_pos, totem_radius, cam_ray_ds_selected, cam_ray_o)
        OD = (D-cam_ray_o)/torch.unsqueeze(torch.norm(D-cam_ray_o, dim=1), dim=1)
        AD = (D-totem_pos)/torch.unsqueeze(torch.norm(D-totem_pos, dim=1), dim=1)
        DE= get_refracted_ray_torch(OD, AD, n_air, n_totem)
        E = line_sphere_intersection_torch(totem_pos, totem_radius, DE, D, use_min=False)
        EA = (totem_pos-E)/torch.unsqueeze(torch.norm(totem_pos-E, dim=1), dim=1)
        direction = get_refracted_ray_torch(DE, EA, n_totem, n_air)
        if E.isnan().any() or direction.isnan().any():
            import pdb; pdb.set_trace()
        end5 = '%.4f seconds' % (time.time()-start5)
        if print_time:
            print('Block 5: %s' % end5)

        # Block 6
        start6 = time.time()
        # Get colors of these rays
        rays_o = E
        rays_d = direction
        target_s = im[ys_selected, xs_selected]
        # rgb to [0,1]
        target_s = target_s.astype('float32')
        target_s /= 255
        target_s = torch.Tensor(target_s).to(device)
        

        # Check if calculations match numpy version
        if not np.allclose(rays_o.detach().cpu().numpy(), ray_os_np[:N_rand], atol=1e-5):
            import pdb; pdb.set_trace()        
        if not np.allclose(rays_d.detach().cpu().numpy(), ray_ds_np[:N_rand], atol=1e-3):
            import pdb; pdb.set_trace()  
        end6 = '%.4f seconds' % (time.time()-start6)
        if print_time:
            print('Block 6: %s' % end6)      
        
        # Block 7
        start7 = time.time()
        # added:
        # if necessary, shift the rays s.t. origin is zero, and z velocity
        # (z component of rays_d) = 1. this is how nerf is set up
        if args.shift_o_normalize_z:
            if i%args.i_print==0:
                print("shifting origin")
            origin_z = 0.
            t_shift = -rays_o[:, [2]] / rays_d[:, [2]]
            rays_o = rays_o + t_shift * rays_d
            d_scale = 1/rays_d[:, [2]]
            rays_d = rays_d * d_scale

        batch_rays = torch.stack([rays_o, rays_d], 0)
        end7 = '%.4f seconds' % (time.time()-start7)
        if print_time:
            print('Block 7: %s' % end7)

        # Block 8
        start8 = time.time()
        #####  Core optimization loop  #####
        # path to save first batch of rays, for debugging
        save_rays = os.path.join(basedir, expname, 'rays.pth') if i == start else None
        optimizer.zero_grad()
        if i >= args.num_iterations_stage_one and totem_pos_net.totem_pos_residual.requires_grad:
            opt_totem_pos.zero_grad()
        rgb, disp, acc, extras = render(chunk=args.chunk, rays=batch_rays,
                                        verbose=i < 10, retraw=True,
                                        save_rays=save_rays,
                                        **render_kwargs_train)
        end8 = '%.4f seconds' % (time.time()-start8)
        if print_time:
            print('Block 8: %s' % end8)

        # Block 9
        start9 = time.time()
        # compute image losses
        img_loss = img2mse(rgb, target_s)
        trans = extras['raw'][...,-1]
        iou_loss = totem_pose_iou_loss_calib(totem_pos, totem_radius, cx, cy, scene_params, boxBs[totem_id])
        if i >= args.num_iterations_stage_one and totem_pos_net.totem_pos_residual.requires_grad:
            loss = img_loss * args.img_loss_weight + iou_loss * args.iou_loss_weight
        else:
            loss = img_loss
        psnr = mse2psnr(img_loss)

        if 'rgb0' in extras:
            img_loss0 = img2mse(extras['rgb0'], target_s)
            loss = loss + img_loss0
            psnr0 = mse2psnr(img_loss0)

        # compute depth losses
        if args.midas_npy and global_step > 50:
            w_depth_initial = args.w_depth
            decay_rate = 10
            decay_iteration = 25 # note: changed it from 50
            divsor = global_step // (decay_iteration * 1000)
            w_depth = w_depth_initial/(decay_rate ** divsor)
            if not args.use_ndc:
                depth_loss = w_depth * compute_depth_loss(cam_disp, target_disp)
            else:
                # if ndc: should be 1/cam_disp and -target_disp
                depth_loss = w_depth * compute_depth_loss(cam_extras['depth_map'], -target_disp)
            if torch.isinf(depth_loss) or torch.isnan(depth_loss):
                print("depth loss invalid")
                from IPython import embed; embed()
            loss = loss + depth_loss
            if 'disp0' in cam_extras:
                if not args.use_ndc:
                    depth_loss0 = w_depth * compute_depth_loss(
                        cam_extras['disp0'], target_disp)
                else:
                    # if ndc: should be 1/cam_disp and -target_disp
                    depth_loss0 = w_depth * compute_depth_loss(
                        cam_extras['depth0'], -target_disp)
                if torch.isinf(depth_loss0) or torch.isnan(depth_loss0):
                    print("depth loss 0 invalid")
                    from IPython import embed; embed()
                loss  = loss + depth_loss0
        else:
            depth_loss = torch.Tensor([0.])
            w_depth = 0.
        end9 = '%.4f seconds' % (time.time()-start9)
        if print_time:
            print('Block 9: %s' % end9)

        # Block 10
        start10 = time.time()
        loss.backward()
        optimizer.step()
        if i >= args.num_iterations_stage_one and totem_pos_net.totem_pos_residual.requires_grad:
            opt_totem_pos.step()

        # NOTE: IMPORTANT!
        ###   update learning rate   ###
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate
        if i >= args.num_iterations_stage_one and totem_pos_net.totem_pos_residual.requires_grad:
            scheduler_totem_pos.step()
        ################################
        end10 = '%.4f seconds' % (time.time()-start10)
        if print_time:
            print('Block 10: %s' % end10)

        # Block 11
        start11 = time.time()
        dt = time.time()-time0
        # print(f"Step: {global_step}, Loss: {loss}, Time: {dt}")
        #####           end            #####

        # Rest is logging
        # learned_totem_pos_torch = torch.stack([totem_pos_net(i) for i in range(N_totems)])
        # learned_totem_pos_numpy = learned_totem_pos_torch.detach().cpu().numpy()
        # if save_totem_pos:
        #     path_totem_pos = os.path.join(folder_totem_pos, 'Iter%06d.npy' % (i))
        #     np.save(path_totem_pos, learned_totem_pos_numpy)

        if i%args.i_weights==0:
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
            torch.save({
                'global_step': global_step,
                'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                'network_fine_state_dict': render_kwargs_train['network_fine'].state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'totem_pos': learned_totem_pos_torch,
            }, path)
            print('Saved checkpoints at', path)

        if i%args.i_print==0:
            if args.path_totem_gt:
                totem_pos_gt = np.load(args.path_totem_gt)/100
                totem_l1_loss = np.mean(np.abs(totem_pos_gt-learned_totem_pos_numpy))


            iou_loss_print = np.mean([(totem_pose_iou_loss_calib(learned_totem_pos_torch[n], totem_radius, cx, cy, scene_params, boxBs[n])).item() for n in range(N_totems)])
            tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item():0.8f} MSE:{img_loss.item():0.8f} IoU: {iou_loss_print}  PSNR: {psnr.item():0.8f} DISP: {depth_loss.item():0.8f}")
            for n in range(N_totems):
                tqdm.write(f"[TRAIN] Iter: {i} Totem{n+1}: {learned_totem_pos_numpy[n]}")
            with open(loss_txt, 'a') as f:
                f.write(f"[TRAIN] Iter: {i} Loss: {loss.item():0.8f} MSE:{img_loss.item():0.8f} IoU: {iou_loss_print}  PSNR: {psnr.item():0.8f} DISP: {depth_loss.item():0.8f}\n")
                for n in range(N_totems):
                    f.write(f"[TRAIN] Iter: {i} Totem{n+1}: {learned_totem_pos_numpy[n]}\n")

            # summary writer: write loss and learning rate 
            writer.add_scalar('losses/loss', loss.item(), i)
            writer.add_scalar('losses/mse', img_loss.item(), i)
            writer.add_scalar('losses/iou', iou_loss_print, i)
            if args.path_totem_gt:
                writer.add_scalar('losses/totem_l1', totem_l1_loss, i)
            writer.add_scalar('losses/psnr', psnr.item(), i)
            writer.add_scalar('losses/depth', depth_loss.item(), i)
            opt_lr = optimizer.param_groups[0]['lr']
            opt_totem_lr = scheduler_totem_pos.get_last_lr()[0]
            writer.add_scalar('learning_rates/optim', opt_lr, i)
            writer.add_scalar('learning_rates/optim_totem', opt_totem_lr, i)
            writer.add_scalar('learning_rates/w_depth', w_depth, i)
        if i % args.i_testset==0:
            rays_o, rays_d, target_s = np.load(args.cam_npy)
            # scale rgb to (0, 1)
            target_s /= 255
            # Preprocessed data in centimeter, now in meter after dividing by 100
            rays_o /= 100
            h_test, w_test, _ = rays_o.shape
            rays_o = rays_o.reshape(-1, 3)
            rays_d = rays_d.reshape(-1, 3)
            target_s = target_s.reshape(-1, 3)
            if args.shift_o_normalize_z:
                print("shifting origin for test")
                # it shouldn't need to shift, but it needs to scale d
                assert(np.ndim(rays_o) == 2)
                assert(np.ndim(rays_d) == 2)
                origin_z = 0.
                t_shift = -rays_o[:, [2]] / rays_d[:, [2]]
                rays_o_shifted = rays_o + t_shift * rays_d
                d_scale = 1/rays_d[:, [2]]
                rays_d_normalized = rays_d * d_scale
                rays_o = rays_o_shifted
                rays_d = rays_d_normalized
            rays_o = torch.from_numpy(rays_o).to(device)
            rays_d = torch.from_numpy(rays_d).to(device)
            target_s = torch.Tensor(target_s).to(device)
            rays = torch.stack([rays_o, rays_d], dim=0)
            with torch.no_grad():
                rgb, disp, acc, extras = render(rays=rays, chunk=args.chunk,
                                                verbose=i<10, retraw=False,
                                                  **render_kwargs_test)
                img_loss = img2mse(rgb*test_filter, target_s*test_filter)
                psnr = mse2psnr(img_loss)
                rgb = rgb.view(h_test, w_test, -1)
                acc = acc.view(h_test, w_test)
                disp = disp.view(h_test, w_test)
            rgb = torchvision.utils.make_grid(
                rgb.cpu().permute(2, 0, 1)[None],
                normalize=True, range=(0, 1))
            disp = torchvision.utils.make_grid(
                disp.cpu()[None, None], normalize=True)
            acc = torchvision.utils.make_grid(
                acc.cpu()[None, None], normalize=True)
            writer.add_scalar('test_losses/mse', img_loss.item(), i)
            writer.add_scalar('test_losses/psnr', psnr.item(), i)
            writer.add_image('test_image', rgb, i)
            writer.add_image('test_disp', disp, i)
            writer.add_image('test_acc', acc, i)
        end11 = '%.4f seconds' % (time.time()-start11)
        if print_time:
            print('Block 11: %s' % end11)

        global_step += 1
    print("done")
    pidfile.mark_job_done(os.path.join(basedir, expname))

##### TEST FUNCTIONS ###### 
def test(args):
    # assert(False) # have not modified/debugged for test option
    # moved test pipeline to notebooks/render_test_views.ipynb
    # added by Jingwei Nov.8th 16:18 PT
    near = args.near
    far = args.far

    # load the params file
    with open(os.path.join(args.data_folder, 'params.yml')) as file:
        scene_params = yaml.load(file, Loader=yaml.FullLoader)

    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args)
    global_step = start
    bds_dict = {
        'near' : near,
        'far' : far,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)
    render_kwargs_train.update({'scene_params': scene_params})
    render_kwargs_test.update({'scene_params': scene_params})
    debug = True

    if debug:
         testsavedir = os.path.join(args.basedir, args.expname, 'renderonly_debug_{:06d}'.format(start))
    else:
         testsavedir = os.path.join(args.basedir, args.expname, 'renderonly_path_{:06d}'.format(start))

    os.makedirs(testsavedir, exist_ok=True)

    if args.render_cam:
        print(args.cam_npy)
        rays_o, rays_d, target_s = np.load(args.cam_npy)
        # Preprocessed data in centimeter, now in meter after dividing by 100
        rays_o /= 100
        h_test, w_test, _ = rays_o.shape
        rays_o = rays_o.reshape(-1, 3)
        rays_d = rays_d.reshape(-1, 3)

        if args.shift_o_normalize_z:
            print("shifting origin for test")
            # it shouldn't need to shift, but it needs to scale d
            assert(np.ndim(rays_o) == 2)
            assert(np.ndim(rays_d) == 2)
            origin_z = 0.
            t_shift = -rays_o[:, [2]] / rays_d[:, [2]]
            rays_o_shifted = rays_o + t_shift * rays_d
            d_scale = 1/rays_d[:, [2]]
            rays_d_normalized = rays_d * d_scale
            rays_o = rays_o_shifted
            if not debug:
                rays_d = rays_d_normalized
            rays_o = torch.from_numpy(rays_o).to(device)
            rays_d = torch.from_numpy(rays_d).to(device)
            rays = torch.stack([rays_o, rays_d], dim=0)
            with torch.no_grad():
                rgb, disp, acc, extras = render(rays=rays, chunk=args.chunk,
                                                verbose=False, retraw=False,
                                                  **render_kwargs_test)
                rgb = rgb.view(h_test, w_test, -1)
                rgb = rgb.cpu().numpy()
                rgb8 = to8b(rgb)
                near = int(near * 100)
                filename = os.path.join(testsavedir, 'near_%03d.png' % near)
                imageio.imwrite(filename, rgb8)

    if args.render_totem:
        img = cv2.imread(args.path_im)[..., ::-1]
        n_air = 1.0
        n_totem = args.n_totem
        filenames = ['totem_%03d' % totem_idx for totem_idx in range(4)] # output fnames
        savedir = testsavedir.split('/')
        savedir[-1] = 'totems_'+savedir[-1]
        savedir = '/'.join(savedir)
        os.makedirs(savedir, exist_ok=True)
        near = int(near * 100)
        mtx = (np.load(args.calib_npy, allow_pickle=True)).item()['mtx']
        all_totem_pos = render_kwargs_train['totem_pos']
        folder_mask = args.folder_mask
        fnames_mask = sorted(os.listdir(folder_mask))
        mask_paths = [os.path.join(folder_mask, fname) for fname in fnames_mask]
        pinhole_pos = np.array([0,0,0])
        totem_radius = args.totem_radius
        H, W = scene_params['image_height'], scene_params['image_width']
        _, pos_map = get_cam_ray_ds_calib(W, H, mtx)
        for (name, path_mask, totem_pos) in zip(filenames, mask_paths, all_totem_pos):
            mask = cv2.imread(path_mask)[..., 0]
            ys, xs = np.where(mask)
            selected_cam_ray_ds = pos_map[ys, xs]
            selected_cam_ray_os = np.broadcast_to(pinhole_pos, selected_cam_ray_ds.shape)

            # Refraction(numpy)
            # Filter1: cam rays that don't intersect with the totem
            D, valid_idx_1 = line_sphere_intersection_v2(totem_pos, totem_radius, selected_cam_ray_ds, selected_cam_ray_os)
            OD = (D-pinhole_pos)/LA.norm(D-pinhole_pos, axis=1)[:, None]
            AD = (D-totem_pos)/LA.norm(D-totem_pos, axis=1)[:, None]
            DE= get_refracted_ray(OD, AD, n_air, n_totem)
            E, valid_idx_2 = line_sphere_intersection_v2(totem_pos, totem_radius, DE, D, use_min=False)
            EA = (totem_pos-E)/LA.norm(totem_pos-E, axis=1)[:, None]
            direction = get_refracted_ray(DE, EA, n_totem, n_air)
            ray_os_np = E
            ray_ds_np = direction
            ys = ys[valid_idx_1][valid_idx_2]
            xs = xs[valid_idx_1][valid_idx_2]

            # Filter2: totem rays that are too noisy by NDC threshold
            if args.filter_view:
                _, _, rays_threshold = my_ndc_rays_np(
                    scene_params['cam_sensor_h'], scene_params['cam_sensor_w'],
                    scene_params['focal'],  near, far, ray_os_np, ray_ds_np, thresh=2.0)
                valid_idx_3 = np.where(rays_threshold == 1)[0]
                ys = ys[valid_idx_3]
                xs = xs[valid_idx_3]
                ray_os_np = ray_os_np[valid_idx_3]
                ray_ds_np = ray_ds_np[valid_idx_3]

            ray_os = torch.from_numpy(ray_os_np).to(device)
            ray_ds = torch.from_numpy(ray_ds_np).to(device)
            rays = torch.stack([ray_os, ray_ds], dim=0)
            with torch.no_grad():
                rgb, disp, acc, extras = render(rays=rays, chunk=args.chunk,
                                                verbose=False, retraw=False,
                                                  **render_kwargs_test)
                rgb = rgb.cpu().numpy()
                rgb8 = to8b(rgb)
                filename = os.path.join(savedir, 'near_%03d_%s.png' % (near, name))
                out = np.zeros((H, W, 3), dtype='uint8')
                out[ys, xs] = rgb8
                imageio.imwrite(filename, out)
                print(name, 'done!')

            # Saving cropped totems and diff from groundtruth
            margin = 16
            up, down, left, right = np.min(ys), np.max(ys), np.min(xs), np.max(xs)
            up, down, left, right = max(0, up-margin), min(H, down+margin), max(0, left-margin), min(W, right+margin)
            crop = out[up:down, left:right][::-1, ::-1]
            diff = np.zeros((H, W, 3), dtype='uint8')
            diff[ys, xs] = (np.abs(rgb8.astype(int) - img[ys, xs].astype(int))).astype('uint8')
            diff_crop = diff[up:down, left:right][::-1, ::-1]
            out_folder = os.path.join(savedir,'totem_crops')
            os.makedirs(out_folder, exist_ok=True)
            filename = os.path.join(out_folder, 'near_%03d_%s.png' % (near, name))
            imageio.imwrite(filename, crop)
            filename = os.path.join(out_folder, 'diff_near_%03d_%s.png' % (near, name))
            imageio.imwrite(filename, diff_crop)

           

if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    parser = config_parser()
    args = parser.parse_args()
    if args.render_only:
        test(args)
    else:
        train(args)
