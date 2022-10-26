import os
import time
import numpy as np
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange

from run_nerf_helpers import *
from run_totem_helpers import *
from run_detect_helpers import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)


def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0,
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

    pts_norm = torch.norm(pts[:, :-1, :] - pts[:, 1:, :], dim=-1)
    # print(pts_norm)
    # add an arbitary large constant to the end
    dists = torch.cat([pts_norm, torch.Tensor([1e10]).expand(pts_norm[...,:1].shape)], -1)
    # note that z_vals was never scaled, but it uses scaled pts to
    # determine attentuation

    rgb = torch.sigmoid(raw[...,:3])  # [N_rays, n_samples, 3]
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[...,3].shape) * raw_noise_std

        # Overwrite randomly sampled data if pytest
        if pytest:
            np.random.seed(0)
            noise = np.random.rand(*list(raw[...,3].shape)) * raw_noise_std
            noise = torch.Tensor(noise)

    alpha = raw2alpha(raw[...,3] + noise, dists)  # [N_rays, n_samples]
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]

    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)

    # changed to return depth_map_norm rather than depth_map
    return rgb_map, disp_map, acc_map, weights, depth_map


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


def render_rays(ray_batch,
                network_fn,
                network_query_fn,
                n_samples,
                data,
                retraw=False,
                perturb=0.,
                n_importance=0,
                network_fine=None,
                raw_noise_std=0.,
                scene_params=None,
                proj_params=None,
                save_rays=None,
                rays_test_tstart=None,
                pytest=False, **kwargs):
    """Volumetric rendering.
    Args:
      ray_batch: array of shape [batch_size, ...]. All information necessary
        for sampling along a ray, including: ray origin, ray direction, min
        dist, max dist, and unit-magnitude viewing direction.
      network_fn: function. Model for predicting RGB and density at each point
        in space.
      network_query_fn: function used for passing queries to network_fn.
      n_samples: int. Number of different times to sample along each ray.
      retraw: bool. If True, include model's raw, unprocessed predictions.
      lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
      perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
        random points in time.
      n_importance: int. Number of additional times to sample along each ray.
        These samples are only passed to network_fine.
      network_fine: "fine" network with same spec as network_fn.
      raw_noise_std: ...
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

    t_start = 0. if rays_test_tstart is None else rays_test_tstart
    t_vals = torch.linspace(t_start, 1., steps=n_samples)
    z_vals = near * (1.-t_vals) + far * (t_vals) # sample in linear depth

    # z_vals is the time along the ray
    z_vals = z_vals.expand([N_rays, n_samples])

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

    pts_euc = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None]
    pts = euc_to_cube(data['W'], data['H'], data['K'], near, far, pts_euc)

    raw = network_query_fn(pts, viewdirs, network_fn)
    rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(
        raw, z_vals, rays_d, raw_noise_std,
        proj_params=proj_params, pts=pts, pytest=pytest)

    if n_importance > 0:

        rgb_map_0, disp_map_0, acc_map_0, weights_0, depth_map_0 = rgb_map, disp_map, acc_map, weights, depth_map
        pts_euc_0 = pts_euc

        z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        z_samples = sample_pdf(z_vals_mid, weights[...,1:-1], n_importance, det=(perturb==0.), pytest=pytest)
        z_samples = z_samples.detach()

        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)

        # ADDED: transform the reprojected points
        pts_euc = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None]
        pts = euc_to_cube(data['W'], data['H'], data['K'], near, far, pts_euc)

        run_fn = network_fn if network_fine is None else network_fine
        raw = network_query_fn(pts, viewdirs, run_fn)

        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(
            raw, z_vals, rays_d, raw_noise_std,
            proj_params=proj_params, pts=pts, pytest=pytest)

    ret = {'rgb_map' : rgb_map, 'disp_map' : disp_map, 'acc_map' : acc_map, 'depth_map': depth_map}
    if retraw:
        ret['raw'] = raw
    if n_importance > 0:
        ret['rgb0'] = rgb_map_0
        ret['disp0'] = disp_map_0
        ret['acc0'] = acc_map_0
        ret['depth0'] = depth_map_0
        ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]

    # For protect mask
    weights = torch.cat([weights_0, weights], dim=1)
    pts_euc = torch.cat([pts_euc_0, pts_euc], dim=1)
    wmax_idxs = torch.argmax(weights, dim=1)
    n_rays = len(wmax_idxs)
    wmax_pts = pts_euc[range(n_rays), wmax_idxs]
    ret['wmax_pts'] = wmax_pts

    return ret


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


def render(chunk=1024*32, rays=None, 
           near=0., far=1., use_viewdirs=False, **kwargs):
    """Render rays
    Args:
      H: int. Height of image in pixels.
      W: int. Width of image in pixels.
      focal: float. Focal length of pinhole camera.
      chunk: int. Maximum number of rays to process simultaneously. Used to
        control maximum memory usage. Does not affect final results.
      rays: array of shape [2, batch_size, 3]. Ray origin and direction for
        each example in batch.
      ndc: bool. If True, represent ray origin, direction in NDC coordinates.
      near: float or array of shape [batch_size]. Nearest distance for a ray.
      far: float or array of shape [batch_size]. Farthest distance for a ray.
      use_viewdirs: bool. If True, use viewing direction of a point in space in model.
    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays.
      disp_map: [batch_size]. Disparity map. Inverse of depth.
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
      extras: dict with everything returned by render_rays().
    """
    rays_o, rays_d = rays
    sh = rays_d.shape # [..., 3]

    # provide ray directions as input
    if use_viewdirs:
        viewdirs = rays_d
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1,3]).float()

    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1,3]).float()
    rays_d = torch.reshape(rays_d, [-1,3]).float()


    near, far = near * torch.ones_like(rays_d[...,:1]), far * torch.ones_like(rays_d[...,:1])
    rays = torch.cat([rays_o, rays_d, near, far], -1)
    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)

    # Render and reshape
    all_ret = batchify_rays(rays, chunk, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map', 'disp_map', 'acc_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k : all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]


def create_nerf(args):
    """Instantiate NeRF's MLP model.
    """
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed)

    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed)
    output_ch = 5 if args.n_importance > 0 else 4
    skips = [4]
    model = NeRF(D=args.netdepth, W=args.netwidth,
                 input_ch=input_ch, output_ch=output_ch, skips=skips,
                 input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
    grad_vars = list(model.parameters())

    model_fine = None
    if args.n_importance > 0:
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
    base_dir = args.base_dir
    exp_name = args.exp_name

    ##########################

    # Load checkpoints
    ckpts = [os.path.join(base_dir, exp_name, f) for f in sorted(os.listdir(os.path.join(base_dir, exp_name))) if 'tar' in f]

    print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)

        start = ckpt['global_step'] + 1
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        # Load model
        model.load_state_dict(ckpt['network_fn_state_dict'])
        if model_fine is not None:
            model_fine.load_state_dict(ckpt['network_fine_state_dict'])

    ##########################

    render_kwargs_train = {
        'network_query_fn' : network_query_fn,
        'perturb' : args.perturb,
        'n_importance' : args.n_importance,
        'network_fine' : model_fine,
        'n_samples' : args.n_samples,
        'network_fn' : model,
        'use_viewdirs' : args.use_viewdirs,
        'raw_noise_std' : args.raw_noise_std,
        'near' : args.near,
        'far' : args.far,
    }

    render_kwargs_test = {k : render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.
    if len(ckpts) > 0 and not args.no_reload and args.optimize_totems:
        totem_pos = ckpt['totem_pos'].detach().cpu().numpy()
        render_kwargs_train['totem_pos'] = totem_pos
        render_kwargs_test['totem_pos'] = totem_pos

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer


def config_parser():
    import configargparse
    parser = configargparse.ArgumentParser()

    parser.add_argument('--config', is_config_file=True, required=True, 
                        help='config file path')
    parser.add_argument("--exp_name", type=str, required=True, 
                        help='experiment name')
    parser.add_argument("--base_dir", type=str, default='./logs/', 
                        help='where to store ckpts and logs')

    # dataset options
    parser.add_argument("--data_dir", type=str, required=True, 
                        help='input data directory')
    parser.add_argument("--calib_npy", type=str,  
                        help='path to calibrated camera parameters')
    parser.add_argument("--near", type=float, required=True, 
                        help='manually defined min depth bound')
    parser.add_argument("--far", type=float, required=True, 
                        help='manually defined max depth bound')
    parser.add_argument("--ior_totem", type=float, default=1.52,
                        help='totem index of refraction, generally use 1.52 unless known')
    parser.add_argument("--totem_radius", type=float, required=True, 
                        help='totem radius in unit of meters')

    # training options
    parser.add_argument("--optimize_totems", action='store_true', 
                        help='if True, jointly optimize NeRF and totem pose; if False, only optimize NeRF')

    parser.add_argument("--n_iters", type=int, default=50000, 
                        help='Number of training iterations')
    parser.add_argument("--n_iters_freeze_totem", type=int, default=100, 
                        help='number of iterations to freeze totem pose before jointly optimizing with nerf')

    parser.add_argument("--lrate", type=float, default=5e-4, 
                        help='learning rate')
    parser.add_argument("--totem_lrate", type=float, default=1e-5, 
                        help='totem pose residual learning rate')

    parser.add_argument("--lrate_decay", type=int, default=500, 
                        help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--totem_gamma", type=float, default=0.99, 
                        help='gamma for totem pose lr decay')

    parser.add_argument("--img_loss_weight", type=float, default=10.0, 
                        help='weight to be multiplied to the iou loss')
    parser.add_argument("--iou_loss_weight", type=float, default=1.0, 
                        help='weight to be multiplied to the iou loss')

    parser.add_argument("--n_rand", type=int, default=1024, 
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--n_samples", type=int, default=64, 
                        help='number of coarse samples per ray')
    parser.add_argument("--n_importance", type=int, default=128,
                        help='number of additional fine samples per ray')

    parser.add_argument("--use_viewdirs", action='store_true', 
                        help='use full 5D input instead of 3D, turns on view-dependency')
    parser.add_argument("--netdepth", type=int, default=8, 
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256, 
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int, default=8, 
                        help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256, 
                        help='channels per layer in fine network')
    parser.add_argument("--chunk", type=int, default=1024*32, 
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024*64, 
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_reload", action='store_true', 
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--i_embed", type=int, default=0, 
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10, 
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4, 
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=1e0, 
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')

    # # logging/saving options
    parser.add_argument("--i_print",   type=int, default=100, 
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_weights", type=int, default=10000, 
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_test", type=int, default=5000, 
                        help='frequency of rendering camera view')

    # rendering options
    parser.add_argument("--render_only", action='store_true', 
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_totem", action='store_true', 
                        help='render totem views')
    parser.add_argument("--render_cam", action='store_true', 
                        help='render camera view')

    # detection options
    parser.add_argument("--detect_only", action='store_true',
                        help='do not optimize, reload weights and save detection results')

    return parser


def train(args):

    # Load data and commonly used arguments
    data = load_real_data(args)
    n_totems = data['n_totems']
    n_iters = args.n_iters
    n_iters_freeze_totem = args.n_iters_freeze_totem
    totem_idxs = list(range(n_totems))
    test_data = load_real_data(args, downsample=8)
    test_H, test_W = test_data['H'], test_data['W']

    # mask to filter out totem pixels when computing L2 loss of camera view, 0 = totem pixels, 1 = image pixels
    test_mask = (test_data['test_mask']).astype('float32')
    test_mask = torch.from_numpy(test_mask).to(device)
    
    # Create log dir and copy the config file
    base_dir = args.base_dir
    exp_name = args.exp_name
    os.makedirs(os.path.join(base_dir, exp_name), exist_ok=True)
    f = os.path.join(base_dir, exp_name, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(base_dir, exp_name, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    # Create nerf model
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args)
    render_kwargs_train['data'] = data
    render_kwargs_test['data'] = test_data

    # Prepare totem training or precomputation
    initial_totem_pos = np.load(os.path.join(args.data_dir, 'initial_totem_pose.npy')) # Unit = meters

    ## If optimizing totems, create totem pose parameters
    if args.optimize_totems:
        from torch.optim.lr_scheduler import MultiStepLR

        if 'totem_pos' in render_kwargs_train.keys():
            totem_pos_residual = render_kwargs_train['totem_pos'] - initial_totem_pos
        else:
            totem_pos_residual = np.zeros((n_totems, 3))

        # Two stage training -> freezing totem pose in the first stage
        totem_pos_net = LearnTotemPos(initial_totem_pos, totem_pos_residual, req_grad=False, device=device).to(device)
        opt_totem_pos = torch.optim.Adam(totem_pos_net.parameters(), lr=args.totem_lrate)
        scheduler_totem_pos = MultiStepLR(opt_totem_pos, milestones=list(range(0, n_iters - n_iters_freeze_totem, 100)), gamma=args.totem_gamma)

    ## If not optimizing totems, precompute totem rays before training
    else:
        all_precomputed_rays = []
        for totem_idx in range(n_totems):
            rays_o, rays_d, ys, xs, target_rgbs = get_totem_rays_numpy(args, data, totem_idx, initial_totem_pos[totem_idx], n_rays=None)
            rays_np = np.stack([rays_o, rays_d], axis=0)
            precomputed_rays = {'rays_np': rays_np, 'target_rgbs_np': target_rgbs}
            all_precomputed_rays.append(precomputed_rays)

    # Training begins here
    loss_txt = os.path.join(base_dir, exp_name, 'loss.txt')
    f = open(loss_txt, 'w+')
    f.close()
    writer = SummaryWriter(os.path.join(base_dir, 'summaries', exp_name))

    global_step = start # 0, 1, ..., n_iters-1
    start += 1
    n_iters += 1
    print(f'Begin training from iteration {start}---------------------------------------------------------')
    for i in trange(start, n_iters): # Iterations 1, 2, ..., n_iters

        # Start training totem poses
        if args.optimize_totems and i > n_iters_freeze_totem and (not totem_pos_net.totem_pos_residual.requires_grad):
            totem_pos_net.requires_grad_()
            totem_pos_net.train()

        # Every n totems shuffle the order
        if global_step % n_totems == 0:
            np.random.shuffle(totem_idxs)
        totem_idx = totem_idxs[global_step % n_totems]

        if args.optimize_totems:
            totem_pos = totem_pos_net(totem_idx) # learnable parameters
            totem_pos_np = totem_pos.detach().cpu().numpy()
            
            # Preselect valid totem rays in numpy
            rays_o_np, rays_d_np, ys, xs, target_rgbs = get_totem_rays_numpy(args, data, totem_idx, totem_pos_np, n_rays=args.n_rand)

            # Compute again in pytorch
            rays_o, rays_d, ys, xs, target_rgbs = get_totem_rays_torch(args, ys, xs, device, data, totem_idx, totem_pos, n_rays=args.n_rand)
            batch_rays = torch.stack([rays_o, rays_d], 0)
            batch_rays = shift_and_normalize(batch_rays)

            # Convert to float and move to device
            target_rgbs = target_rgbs.astype('float32') / 255
            target_rgbs = torch.Tensor(target_rgbs).to(device)

        else:
            n_rays_total = len(all_precomputed_rays[totem_idx]['target_rgbs_np'])
            batch_ids = np.arange(n_rays_total)
            np.random.shuffle(batch_ids)
            batch_ids = batch_ids[:args.n_rand]
            batch_rays = all_precomputed_rays[totem_idx]['rays_np'][:, batch_ids]
            target_rgbs = all_precomputed_rays[totem_idx]['target_rgbs_np'][batch_ids]

            # Moving to gpu
            batch_rays = torch.from_numpy(batch_rays).to(device)
            batch_rays = shift_and_normalize(batch_rays)
            target_rgbs = target_rgbs.astype('float32') / 255
            target_rgbs = torch.Tensor(target_rgbs).to(device)

        # Forward pass
        optimizer.zero_grad()
        if args.optimize_totems and totem_pos_net.totem_pos_residual.requires_grad:
            opt_totem_pos.zero_grad()
        rgb, disp, acc, extras = render(rays=batch_rays, chunk=args.chunk, **render_kwargs_train)

        # compute image losses
        img_loss = img2mse(rgb, target_rgbs)
        if args.optimize_totems and totem_pos_net.totem_pos_residual.requires_grad:
            iou_loss = totem_pose_iou_loss(args, data, totem_idx, totem_pos)
            loss = img_loss * args.img_loss_weight + iou_loss * args.iou_loss_weight
        else:
            loss = img_loss
        psnr = mse2psnr(img_loss)

        if 'rgb0' in extras:
            img_loss0 = img2mse(extras['rgb0'], target_rgbs)
            loss = loss + img_loss0

        loss.backward()
        optimizer.step()
        if args.optimize_totems and totem_pos_net.totem_pos_residual.requires_grad:
            opt_totem_pos.step()

        ###   update learning rate   ###
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate
        if args.optimize_totems and totem_pos_net.totem_pos_residual.requires_grad:
            scheduler_totem_pos.step()

        # Logging
        if args.optimize_totems:
            learned_totem_pos_torch = torch.stack([totem_pos_net(totem_idx) for totem_idx in range(n_totems)])
            learned_totem_pos_numpy = learned_totem_pos_torch.detach().cpu().numpy()

        if i%args.i_weights==0:

            path = os.path.join(base_dir, exp_name, '{:06d}.tar'.format(i))
            out = {
                'global_step': global_step,
                'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                'network_fine_state_dict': render_kwargs_train['network_fine'].state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }
            if args.optimize_totems:
                out['totem_pos'] = learned_totem_pos_torch
            torch.save(out, path)
            print('Saved checkpoints at', path)

        if i%args.i_print==0:
            if args.optimize_totems:
                iou_loss_print = np.mean([totem_pose_iou_loss(args, data, totem_idx, totem_pos).item() for totem_idx in range(n_totems)])
                tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item():0.8f} MSE:{img_loss.item():0.8f} IoU: {iou_loss_print}  PSNR: {psnr.item():0.8f}")
                for totem_idx in range(n_totems):
                    tqdm.write(f"[TRAIN] Iter: {i} Totem{totem_idx+1}: {learned_totem_pos_numpy[totem_idx]}")
            else:
                tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item():0.8f} MSE:{img_loss.item():0.8f} PSNR: {psnr.item():0.8f}")

            with open(loss_txt, 'a') as f:
                if args.optimize_totems:
                    f.write(f"[TRAIN] Iter: {i} Loss: {loss.item():0.8f} MSE:{img_loss.item():0.8f} IoU: {iou_loss_print}  PSNR: {psnr.item():0.8f}\n")
                    for n in range(n_totems):
                        f.write(f"[TRAIN] Iter: {i} Totem{n+1}: {learned_totem_pos_numpy[n]}\n")
                else:
                    f.write(f"[TRAIN] Iter: {i} Loss: {loss.item():0.8f} MSE:{img_loss.item():0.8f} PSNR: {psnr.item():0.8f}\n")

            # summary writer: write loss and learning rate 
            writer.add_scalar('losses/loss', loss.item(), i)
            writer.add_scalar('losses/mse', img_loss.item(), i)
            writer.add_scalar('losses/psnr', psnr.item(), i)
            opt_lr = optimizer.param_groups[0]['lr']
            writer.add_scalar('learning_rates/optim', opt_lr, i)
            if args.optimize_totems:
                writer.add_scalar('losses/iou', iou_loss_print, i)
                opt_totem_lr = scheduler_totem_pos.get_last_lr()[0]
                writer.add_scalar('learning_rates/optim_totem', opt_totem_lr, i)

        if i % args.i_test==0:

            cam_rays_o, cam_rays_d = test_data['cam_rays_o'], test_data['cam_rays_d']
            rays_o = cam_rays_o.reshape(-1, 3)
            rays_d = cam_rays_d.reshape(-1, 3)
            rays = np.stack([rays_o, rays_d], axis=0)
            rays = torch.from_numpy(rays).to(device)
            rays = shift_and_normalize(rays)

            target_rgbs = test_data['image']
            target_rgbs = target_rgbs.astype('float32') / 255
            target_rgbs = torch.Tensor(target_rgbs).to(device)

            with torch.no_grad():
                rgb, disp, acc, extras = render(rays=rays, chunk=args.chunk, **render_kwargs_test)
                rgb = rgb.view(test_H, test_W, -1)
                disp = disp.view(test_H, test_W)
                acc = acc.view(test_H, test_W)
                img_loss = img2mse(rgb*torch.unsqueeze(test_mask, dim=2), target_rgbs*torch.unsqueeze(test_mask, dim=2))
                psnr = mse2psnr(img_loss)
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
        
        global_step += 1


def test(args, downsample=4):
    '''
        Use the following flags to choose what to render from a trained NeRF model:

            render_totem: render totems only (pixels used for training)
            render_cam: render the camera view for detection stage

    '''

    # Load commonly used variables
    data = load_real_data(args, downsample)
    H, W, K = data['H'], data['W'], data['K']
    n_totems = data['n_totems']

    # Load NeRF model
    _, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args)
    render_kwargs_test['data'] = data
    
    # Create output folder
    render_dir = os.path.join(args.base_dir, args.exp_name, 'render_{:06d}'.format(start))
    os.makedirs(render_dir, exist_ok=True)

    # Render totem views
    if args.render_totem:

        # Logging
        print('Rendering totem views --------------------------------------------------')

        # Create output folder
        save_dir = os.path.join(render_dir, 'totem_views')
        os.makedirs(save_dir, exist_ok=True)

        if args.optimize_totems:
            all_totem_pos = render_kwargs_test['totem_pos']
        else:
            all_totem_pos = np.load(os.path.join(args.data_dir, 'initial_totem_pose.npy'))

        for totem_idx in range(n_totems):
            start_time = time.time()
            rays_o, rays_d, ys, xs, target_rgbs = get_totem_rays_numpy(args, data, totem_idx, all_totem_pos[totem_idx], n_rays=None)
            rays_np = np.stack([rays_o, rays_d], axis=0)
            rays = torch.from_numpy(rays_np).to(device)
            rays = shift_and_normalize(rays)

            with torch.no_grad():
                rgb, disp, acc, extras = render(rays=rays, chunk=args.chunk, **render_kwargs_test)
                rgb8 = to8b(rgb.cpu().numpy())
                render_path = os.path.join(save_dir, 'totem_%03d.png' % totem_idx)
                out = np.zeros(data['image'].shape, dtype='uint8')
                out[ys, xs] = rgb8
                imageio.imwrite(render_path, out)
                print(f'Totem view #%02d rendered in %.4f seconds: {render_path}' % (totem_idx, time.time()-start_time))

            # Crop and save rendered totems (and the absolute difference from ground truth, and the groundtruth)
            compare_dir = os.path.join(save_dir,'compare')
            os.makedirs(compare_dir, exist_ok=True)
            render_path = os.path.join(compare_dir, 'totem_%03d.png' % totem_idx)

            # Find bounding box
            margin = 16
            up, down, left, right = np.min(ys), np.max(ys), np.min(xs), np.max(xs)
            up, down, left, right = max(0, up-margin), min(H, down+margin), max(0, left-margin), min(W, right+margin)

            # Totem crop
            crop = out[up:down, left:right]

            # Difference crop
            diff = np.zeros((H, W, 3), dtype='uint8')
            diff[ys, xs] = (np.abs(rgb8.astype(int) - target_rgbs.astype(int))).astype('uint8')
            diff_crop = diff[up:down, left:right]

            # Ground truth crop
            gt = np.zeros((H, W, 3), dtype='uint8')
            gt[ys, xs] = target_rgbs
            gt_crop = gt[up:down, left:right]
            
            # Save difference crop
            concat = np.concatenate([crop, diff_crop, gt_crop], axis=1)
            imageio.imwrite(render_path, concat)
            
    # Render camera view
    if args.render_cam:

        # Logging
        print('Rendering camera view --------------------------------------------------')

        start_time = time.time()
        cam_rays_o, cam_rays_d = get_cam_rays(W, H, K)

        rays_o = cam_rays_o.reshape(-1, 3)
        rays_d = cam_rays_d.reshape(-1, 3)
        rays = np.stack([rays_o, rays_d], axis=0)
        rays = torch.from_numpy(rays).to(device)
        rays = shift_and_normalize(rays)

        with torch.no_grad():
            rgb, disp, acc, extras = render(rays=rays, chunk=args.chunk, **render_kwargs_test)
            rgb = rgb.view(H, W, -1)
            rgb8 = to8b(rgb.cpu().numpy())
            render_path = os.path.join(render_dir, 'camera_view.png')
            imageio.imwrite(render_path, rgb8)
            print(f'Camera view rendered in %.4f seconds: {render_path}' % (time.time()-start_time))


def detect(args, downsample=4):
    '''
        First write intermediate files
        Then detect manipulations
    '''

    # Load commonly used variables
    data = load_real_data(args)
    n_totems = data['n_totems']
    data_ds = load_real_data(args, downsample)
    H, W, K = data_ds['H'], data_ds['W'], data_ds['K']

    # Load NeRF model
    _, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args)
    render_kwargs_test['data'] = data

    # Create output folder
    render_dir = os.path.join(args.base_dir, args.exp_name, 'render_{:06d}'.format(start))
    detect_dir = os.path.join(args.base_dir, args.exp_name, 'detect_{:06d}'.format(start))
    intmd_dir = os.path.join(detect_dir, 'intermediate')
    results_dir = os.path.join(detect_dir, 'results')
    os.makedirs(intmd_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # A. Save intermediate files needed for detection

    ## 1. Save intermediate protect mask files
    protect_dir = os.path.join(intmd_dir, 'protect_mask')
    if os.path.exists(protect_dir):
        print("Folder exists, skipping protect mask generation-----------------------")
    else:
        print('Saving protect mask --------------------------------------------------')
        os.makedirs(protect_dir, exist_ok=True)

        ### [Part1] Saving totem coverage binary map (original resolution)
        ### a. Find the max weight sample on each totem ray
        ### b. Project the 3D position onto the 2D image (downsampled, so we can get denser points)
        ### b. Fill in 255 (white) for these pixels

        if args.optimize_totems:
            all_totem_pos = render_kwargs_test['totem_pos']
        else:
            all_totem_pos = np.load(os.path.join(args.data_dir, 'initial_totem_pose.npy'))
            
        out = np.zeros((H, W))
        for totem_idx in range(n_totems):
            start_time = time.time()

            # Process NeRF outputs - extract max_weight samples
            rays_o, rays_d, ys, xs, target_rgbs = get_totem_rays_numpy(args, data, totem_idx, all_totem_pos[totem_idx], n_rays=None)
            rays_np = np.stack([rays_o, rays_d], axis=0)
            rays = torch.from_numpy(rays_np).to(device)
            rays = shift_and_normalize(rays)

            with torch.no_grad():
                rgb, disp, acc, extras = render(rays=rays, chunk=args.chunk, **render_kwargs_test)
                                                  
                # Project to 2D
                wmax_pts = extras['wmax_pts']
                wmax_pts_2D_xs, wmax_pts_2D_ys = project_3D_pts_to_2D(wmax_pts, W, H, K)
                wmax_pts_2D_xs = wmax_pts_2D_xs.cpu().numpy()
                wmax_pts_2D_ys = wmax_pts_2D_ys.cpu().numpy()

                # store binary mask for this totem's scene coverage
                mask = np.zeros((H, W))
                mask[wmax_pts_2D_ys, wmax_pts_2D_xs] = 1
                out += mask
                out_path = os.path.join(protect_dir, 'protect_coverage_totem_%03d.png' % totem_idx)
                cv2.imwrite(out_path, (mask * 255).astype('uint8'))
                print('Totem #%02d scene coverage computed in %.4f seconds' % (totem_idx, time.time()-start_time))

        out_path = os.path.join(protect_dir, 'protect_coverage.png')
        out = (out == n_totems).astype('uint8') * 255
        cv2.imwrite(out_path, out)

        ### [Part2] Saving the convex hull protect mask (downsampled resolution)
        start_time = time.time()
        from scipy.spatial import ConvexHull
        import scipy.interpolate

        def smooth_hull(points):
            '''
                points: size (N, 2)
            '''
            xs = points[:, 0]
            ys = points[:, 1]

            nt = np.linspace(0, 1, 500)
            t = np.zeros(xs.shape)
            t[1:] = np.sqrt((xs[1:] - xs[:-1])**2 + (ys[1:] - ys[:-1])**2)
            t = np.cumsum(t)
            t /= t[-1]
            method = 'quadratic' # 'cubic' 
            interpolator = scipy.interpolate.interp1d(t, points, kind=method, axis=0)
            new_points = interpolator(nt)
            return new_points

        thres = 10          # 10% of the pixels inside sliding window is white
        win_size = 30
        density = 1.5       # [1.0, inf) how densely to sample windows
        n_patches_w = int((W // win_size) * density)
        n_patches_h = int((H // win_size) * density)
        coverage = out.astype(float)
        coverage_filtered = np.zeros((H, W))

        # Sliding window - filter out pixels inside sparse windows
        for i_h in range(n_patches_h):
            start_h = ((H-win_size)//n_patches_h) * i_h
            end_h = start_h + win_size
            for i_w in range(n_patches_w):
                start_w = ((W-win_size)//n_patches_w) * i_w
                end_w = start_w + win_size
                not_sparse = np.mean(coverage[start_h:end_h, start_w:end_w]) > thres
                if not_sparse:
                    coverage_filtered[start_h:end_h, start_w:end_w] += coverage[start_h:end_h, start_w:end_w]

        ys, xs = np.where(coverage_filtered > 0)
        points = np.stack([xs, ys], axis=1)
        hull = ConvexHull(points)
        hull_points = points[hull.vertices]
        smooth_hull_points = smooth_hull(hull_points)
        smooth_hull_points = (np.round(smooth_hull_points)).astype(int)

        out = np.zeros((H, W), dtype='uint8')
        _ = cv2.fillPoly(out, [smooth_hull_points], 255)
        out_path = os.path.join(protect_dir, 'protect_mask.png')
        cv2.imwrite(out_path, out)
        print('Protect mask computed in %.4f seconds' % (time.time()-start_time))

    ## 2. Save 4 files: image.png, recon.png, totem_mask.png, protect_mask.png
    print('Saving detection intermediate files --------------------------------------------------')
    
    # Load crop factors
    x, y, w, h = data_ds['roi']

    # Paths
    recon_src_path = os.path.join(render_dir, 'camera_view.png')
    protect_mask_src_path = os.path.join(protect_dir, 'protect_mask.png')

    # Early exit if paths don't exist
    if not os.path.exists(recon_src_path):
        print('[Error]: camera view does not exist')
        print('Run with flag --render_only and --render_cam to save camera view rendering')
        import sys; sys.exit()

    if not os.path.exists(protect_mask_src_path):
        print('[Error]: protect mask does not exist')
        print(f"Try removing {protect_dir} and rerun")
        import sys; sys.exit()

    # Process the src files (i.e. crop black pixels due to undistortion)
    recon = imageio.imread(recon_src_path)[y:y+h, x:x+w]
    image = data_ds['image'][y:y+h, x:x+w]
    protect_mask = imageio.imread(protect_mask_src_path)[y:y+h, x:x+w]
    totem_mask = 255 - (data_ds['test_mask']).astype('uint8') * 255
    totem_mask = totem_mask[y:y+h, x:x+w]
    
    ## Write files
    image_dst_path = os.path.join(intmd_dir, 'image.png')
    recon_dst_path = os.path.join(intmd_dir, 'recon.png')
    protect_mask_dst_path = os.path.join(intmd_dir, 'protect_mask.png')
    totem_mask_dst_path = os.path.join(intmd_dir, 'totem_mask.png')
    imageio.imwrite(image_dst_path, image)
    imageio.imwrite(recon_dst_path, recon)
    imageio.imwrite(protect_mask_dst_path, protect_mask)
    imageio.imwrite(totem_mask_dst_path, totem_mask)

    # B. Run detection
    run_detect(results_dir, image, recon, protect_mask, totem_mask)


if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    parser = config_parser()
    args = parser.parse_args()

    if args.render_only:
        test(args)
    elif args.detect_only:
        detect(args)
    else:
        train(args)
