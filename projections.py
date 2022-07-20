import numpy as np
import torch

# euclidean projection = standard 3d space

# ndc projection = maps view frustum to [-1, 1] unit cube, with z linear in
# disparity. this is used in NERF for real scenes, due to the large depth
# range (it samples more densely closer to the camera)
# and also helps to even out sampling within a cube
# see detailed discussion here: https://github.com/bmild/nerf/issues/18

# cube projection = projects x and y to [-1, 1] by dividing by z, but keeps the
# z axis linear in depth. I couldn't figure out an elegant way of converting
# this from euclidean ray o+td to projected ray o'+t'd' (like they did in ndc)
# so i just do it on the 3d points directly

def ndc_to_euc(cam_sensor_h, cam_sensor_w, focal, near, far, pts_np):
    # numpy function for plotting / visualization
    assert(np.ndim(pts_np) == 2) # Nx3
    # similar triangles based on image plane
    # as long as cam_sensor_w and focal are in the same units, it's ok
    right = near * cam_sensor_w / 2 / focal
    top = near * cam_sensor_h / 2 / focal
    ax = -near/right
    ay = -near/top
    az = (far+near)/(far-near)
    bz = 2*far*near/(far-near)
    
    ray_ndc = pts_np
    Z = bz / (ray_ndc[:, 2] - az)  
    X  = ray_ndc[:, 0] * Z / ax
    Y = ray_ndc[:, 1] * Z / ay
    return np.stack([X, Y, Z], axis=1)

def euc_to_cube(cam_sensor_h, cam_sensor_w, focal, near, far, pts):
    # torch function
    # can be any dimension, e.g. N_rays x N_samples x 3
    right = near * cam_sensor_w / 2 / focal
    top = near * cam_sensor_h / 2 / focal
    cube_x = near/right * pts[..., 0] / pts[..., 2]
    cube_y = near/top * pts[..., 1] / pts[..., 2]
    cube_z = 2 / (far-near) * pts[..., 2] + 1 - 2*far/(far-near)
    pts_cube = torch.stack([cube_x, cube_y, cube_z], axis=-1)
    return pts_cube

def cube_to_euc(cam_sensor_h, cam_sensor_w, focal, near, far, pts_np):
    # numpy function for plotting / visualization
    assert(np.ndim(pts_np) == 2) # Nx3
    right = near * cam_sensor_w / 2 / focal
    top = near * cam_sensor_h / 2 / focal
    Z = (pts_np[:, 2] - 1 + 2*far/(far-near)) * (far-near) / 2
    X = pts_np[:, 0] * right / near * Z
    Y = pts_np[:, 1] * top / near * Z
    return np.stack([X, Y, Z], axis=1)

def my_ndc_rays(cam_sensor_h, cam_sensor_w, focal, near, far, rays_o, rays_d, thresh=2.0):
    # find the nerf implementation unintuitive
    # this is directly taken from the formulas in the nerf supplemental
    # similar triangles based on image plane
    # as long as cam_sensor_w and focal are in the same units, it's ok
    right = near * cam_sensor_w / 2 / focal
    top = near * cam_sensor_h / 2 / focal
    ax = -near/right
    ay = -near/top
    az = (far+near)/(far-near)
    bz = 2*far*near/(far-near)

    # shift the origin of the NDC ray -- note that the origin must be shifted
    # away from zero to avoid divison by zero later
    o = rays_o
    d = rays_d
    t0  = -(near + o[:, [2]]) / d[:, [2]]
    o = o + t0 * d

    # NDC ray origin
    ox, oy, oz = o[:, [0]], o[:, [1]], o[:, [2]]
    o_ndc_analytic = torch.cat([ax*ox/oz, ay*oy/oz, az+bz/oz], axis=1)

    # NDC ray direction
    dx, dy, dz = d[:, [0]], d[:, [1]], d[:, [2]]
    d_ndc = torch.cat([ax*(dx/dz-ox/oz), ay*(dy/dz-oy/oz), -bz/oz], axis=1)

    rays_o_ndc = o_ndc_analytic
    rays_d_ndc = d_ndc

    # identify rays that fall outside view frustum
    rays_thresholded = 1 - torch.logical_or(torch.abs(o_ndc_analytic[:, 0]) > thresh,
                                         torch.abs(o_ndc_analytic[:, 1]) > thresh).long()

    return rays_o_ndc, rays_d_ndc, rays_thresholded


def my_ndc_rays_np(cam_sensor_h, cam_sensor_w, focal, near, far, rays_o,
                   rays_d, thresh=2.0):
    # numpy version of the above function
    # similar triangles based on image plane
    # as long as cam_sensor_w and focal are in the same units, it's ok
    right = near * cam_sensor_w / 2 / focal
    top = near * cam_sensor_h / 2 / focal
    ax = -near/right
    ay = -near/top
    az = (far+near)/(far-near)
    bz = 2*far*near/(far-near)

    # shift the origin of the NDC ray -- note that the origin must be shifted
    # away from zero to avoid divison by zero later
    o = rays_o
    d = rays_d
    t0  = -(near + o[:, [2]]) / d[:, [2]]
    o = o + t0 * d

    # NDC ray origin
    ox, oy, oz = o[:, [0]], o[:, [1]], o[:, [2]]
    o_ndc_analytic = np.concatenate([ax*ox/oz, ay*oy/oz, az+bz/oz], axis=1)

    # NDC ray direction
    dx, dy, dz = d[:, [0]], d[:, [1]], d[:, [2]]
    d_ndc = np.concatenate([ax*(dx/dz-ox/oz), ay*(dy/dz-oy/oz), -bz/oz], axis=1)

    rays_o_ndc = o_ndc_analytic
    rays_d_ndc = d_ndc

    # identify rays that fall outside view frustum
    rays_thresholded = 1 - np.logical_or(np.abs(o_ndc_analytic[:, 0]) > thresh,
                                         np.abs(o_ndc_analytic[:, 1]) > thresh)

    return rays_o_ndc, rays_d_ndc, rays_thresholded
