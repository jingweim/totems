import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy.linalg as LA
import os
import time

def line_sphere_intersection(totem_pos, totem_radius, vec, pos, use_min=True):
    '''
        Calculate camera ray intersection with totem
        Equation from here
        https://stackoverflow.com/questions/32571063/specific-location-of-intersection-between-sphere-and-3d-line-segment
        vec = (X01, Y01, Z01)  (N,3)
        pos = (X0, Y0, Z0)  (N,3)
    '''

    # Formula form would just be quadratic formula 
    pos_shifted = pos - totem_pos #(N, 3)
    a = np.sum(vec**2, axis=1) #(N,)
    b = 2 * np.sum(pos_shifted*vec, axis=1) #(N,)
    c = np.sum(pos_shifted ** 2, axis=1) - totem_radius ** 2 #(N,)
    root1 = (-b+np.sqrt(b**2-4*a*c))/(2*a)
    root2 = (-b-np.sqrt(b**2-4*a*c))/(2*a)

    # Filter out no-intersection and one-intersection rays
    notequal = np.equal(root1, root2)==False # (N,)
    notnan1 = np.isnan(root1)==False
    notnan2 = np.isnan(root2)==False
    notnan = notnan1 * notnan2 # (N,)
    valid_idx = np.where((notequal * notnan) == True)[0]
    root1 = root1[valid_idx]
    root2 = root2[valid_idx]
    roots = [root1, root2]

    # If there are two solutions, use the smaller one or the bigger one
    # (intersection closer to or farther from pinhole)
    if use_min:
        root = np.min(roots, axis=0) # (N,)
    else:
        root = np.max(roots, axis=0)

    sol = pos[valid_idx] + root[:, None] * vec[valid_idx]
    return sol, valid_idx

def line_sphere_intersection_v2(totem_pos, totem_radius, vec, pos, use_min=True):
    '''
        [This version gets rid of the ugly waning messages]
        Calculate camera ray intersection with totem
        Equation from here
        https://stackoverflow.com/questions/32571063/specific-location-of-intersection-between-sphere-and-3d-line-segment
        vec = (X01, Y01, Z01)  (N,3)
        pos = (X0, Y0, Z0)  (N,3)
    '''

    # Formula form would just be quadratic formula 
    pos_shifted = pos - totem_pos #(N, 3)
    a = np.sum(vec**2, axis=1) #(N,)
    b = 2 * np.sum(pos_shifted*vec, axis=1) #(N,)
    c = np.sum(pos_shifted ** 2, axis=1) - totem_radius ** 2 #(N,)
    sqrt_inner = b**2-4*a*c
    valid_idx = np.where(sqrt_inner > 0)[0]
    sqrt_inner = sqrt_inner[valid_idx]
    a = a[valid_idx]
    b = b[valid_idx]
    root1 = (-b+np.sqrt(sqrt_inner))/(2*a)
    root2 = (-b-np.sqrt(sqrt_inner))/(2*a)

    # If there are two solutions, use the smaller one or the bigger one
    # (intersection closer to or farther from pinhole, assumes z -> pos away from cam)
    roots = [root1, root2]
    if use_min:
        root = np.min(roots, axis=0) # (N,)
    else:
        root = np.max(roots, axis=0)

    sol = pos[valid_idx] + root[:, None] * vec[valid_idx]
    return sol, valid_idx

def get_refracted_ray(S1, N, n1, n2):
    return n1/n2 * np.cross(N, np.cross(-N, S1)) - N * np.sqrt(1-n1**2/n2**2 * np.sum(np.cross(N, S1) * np.cross(N, S1), axis=1))[:, None]


class LearnTotemPos(nn.Module):
    def __init__(self, init_totem_pos, totem_pos_residual, req_grad, device):
        super(LearnTotemPos, self).__init__()
        self.totem_pos_residual = nn.Parameter(torch.from_numpy(totem_pos_residual.astype('float32')), requires_grad=req_grad)
        self.init_totem_pos = torch.from_numpy(init_totem_pos.astype('float32')).to(device)

    def forward(self, totem_id):
        totem_pos = self.init_totem_pos[totem_id] + self.totem_pos_residual[totem_id]
        return totem_pos


def uv2cam_parallel_calib(xs, ys, mtx):
    cx, cy = mtx[0,2], mtx[1,2]
    fx, fy = mtx[0,0], mtx[1,1]
    cam_vecs = np.stack([(xs-cx)/fx, (ys-cy)/fy, np.ones(xs.shape)], axis=-1)
    # fx = Wf/cam_w
    # fy = Hf/cam_h
    # x = (u - u0) / W / f * cam_w
    # y = (v - v0) / H / f * cam_h
    # z = 1
    return cam_vecs


def uv2cam_parallel(xs, ys, mtx):
    cx, cy = mtx[0,2], mtx[1,2]
    fx, fy = mtx[0,0], mtx[1,1]
    cam_vecs = np.stack([(xs-cx)/fx, (ys-cy)/fy, np.ones(xs.shape)], axis=1)
    return cam_vecs


def line_sphere_intersection_torch(totem_pos, totem_radius, cam_ray_ds, cam_ray_o, use_min=True, use_filter=False):
    # Equation from here
    # https://stackoverflow.com/questions/32571063/specific-location-of-intersection-between-sphere-and-3d-line-segment
    # vec = (X01, Y01, Z01)  (N,3)
    # pos = (X0, Y0, Z0)  (N,3)

    if use_filter:
        totem_pos = totem_pos.detach()
    # Formula form would just be quadratic formula
    pos_shifted = cam_ray_o - totem_pos #(3,)
    a = torch.sum(cam_ray_ds**2, dim=1) #(N,)
    b = 2 * torch.sum(pos_shifted*cam_ray_ds, dim=1) #(N,)
    c = torch.sum(pos_shifted ** 2, dim=-1) - totem_radius ** 2 #(1,)
    # valid_idx = torch.squeeze(((b**2-4*a*c) > 0).nonzero())
    # root1 = (-b+torch.sqrt(b**2-4*a*c))/torch.unsqueeze(2*a, dim=1) #(N,)
    # root2 = (-b-torch.sqrt(b**2-4*a*c))/torch.unsqueeze(2*a, dim=1) #(N,)
    root1 = (-b+torch.sqrt(b**2-4*a*c))/(2*a) #(N,)
    root2 = (-b-torch.sqrt(b**2-4*a*c))/(2*a) #(N,)
    # import pdb; pdb.set_trace()

    # Filter out no-intersection and one-intersection rays
    if use_filter:
        notequal = root1 != root2
        notnan1 = ~torch.isnan(root1)
        notnan2 = ~torch.isnan(root2)
        valid_idx = torch.squeeze((notequal * notnan1 * notnan2).nonzero())
        cam_ray_ds = cam_ray_ds[valid_idx]
        root1 = root1[valid_idx]
        root2 = root2[valid_idx]
    # valid_idx = torch.squeeze(torch.nonzero(notequal * notnan1 * notnan2))

    # root1 = torch.squeeze(root1[valid_idx])
    # root2 = torch.squeeze(root2[valid_idx])
    roots = torch.stack([root1, root2], dim=0)
    # cam_ray_ds_filtered = torch.squeeze(cam_ray_ds[valid_idx])

    # If there are two solutions, use the smaller one or the bigger one
    # (intersection closer to or farther from pinhole)
    if use_min:
        root, _ = torch.min(roots, dim=0) # (N,)
    else:
        root, _ = torch.max(roots, dim=0)

    # sol = cam_ray_o + torch.unsqueeze(root, dim=1) * cam_ray_ds_filtered
    sol = cam_ray_o + torch.unsqueeze(root, dim=1) * cam_ray_ds
    if use_filter:
       return sol, valid_idx

    return sol


def get_refracted_ray_torch(S1, N, n1, n2):
    A = torch.cross(-N, S1)
    B = torch.cross(N, A)
    C = torch.cross(N, S1)
    if C.isnan().any():
        import pdb; pdb.set_trace()
    D = torch.sum(C * C, dim=1)
    E = 1-n1**2/n2**2 * torch.unsqueeze(D , dim=1)
    F = torch.sqrt(E)
    G = N * F
    return n1/n2 * B - G

def save_totem_vis_v2(totem_radius, all_totem_pos, iter_idx, out_folder, colors):
    u = np.linspace(0, np.pi, 30)
    v = np.linspace(0, 2 * np.pi, 30)
    radius = totem_radius * 100 # 3cm
    elev = -37.8371173821
    azim = -89.9612903226

    # Set axis size s.t. x,y,z lengths are visually the same
    fig = plt.figure(figsize=(10,10))
    ax = plt.axes(projection='3d')
    size = 15 # arbitrary, totem depth around 10
    ax.set_xlim3d(-1*size/2, size/2)
    ax.set_ylim3d(-1*size/2, size/2)
    ax.set_zlim3d(0, size)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    # Draw optical axis
    x = [0, 0]
    y = [0, 0]
    z = [0, size]
    ax.plot3D(x, y, z)
    ax.scatter3D([0], [0], [0], s=10)

    # Set viewing pose
    # https://stackoverflow.com/questions/47610614/get-viewing-camera-angles-in-matplotlib-3d-plot
    ax.view_init(elev, azim)

    # Draw spherical totems
    all_totem_pos *= 100
    for i, totem_pos in enumerate(all_totem_pos):
        x = np.outer(np.sin(u), np.sin(v)) * radius + totem_pos[0]
        y = np.outer(np.sin(u), np.cos(v)) * radius + totem_pos[1]
        z = np.outer(np.cos(u), np.ones_like(v)) * radius + totem_pos[2]
        ax.plot_wireframe(x, y, z, color=colors[i])

    fname = 'Iter%06d' % iter_idx
    plt.title(fname, y=-0.01)
    out_path = os.path.join(out_folder, fname+'.png')
    plt.savefig(out_path)
    plt.close()


def save_totem_vis(totem_radius, all_totem_pos, iter_idx, out_folder):
    colors = [(1.0, 0.0, 0.0, 0.5), (0.0, 1.0, 0.0, 0.5), (0.0, 0.0, 1.0, 0.5), (1.0, 0.0, 1.0, 0.5)]
    u = np.linspace(0, np.pi, 30)
    v = np.linspace(0, 2 * np.pi, 30)
    radius = totem_radius * 100 # 3cm
    elev = -37.8371173821
    azim = -89.9612903226

    # Set axis size s.t. x,y,z lengths are visually the same
    fig = plt.figure(figsize=(10,10))
    ax = plt.axes(projection='3d')
    size = 15 # arbitrary, totem depth around 10
    ax.set_xlim3d(-1*size/2, size/2)
    ax.set_ylim3d(-1*size/2, size/2)
    ax.set_zlim3d(0, size)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    # Draw optical axis
    x = [0, 0]
    y = [0, 0]
    z = [0, size]
    ax.plot3D(x, y, z)
    ax.scatter3D([0], [0], [0], s=10)

    # Set viewing pose
    # https://stackoverflow.com/questions/47610614/get-viewing-camera-angles-in-matplotlib-3d-plot
    ax.view_init(elev, azim)

    # Draw spherical totems
    all_totem_pos *= 100
    for i, totem_pos in enumerate(all_totem_pos):
        x = np.outer(np.sin(u), np.sin(v)) * radius + totem_pos[0]
        y = np.outer(np.sin(u), np.cos(v)) * radius + totem_pos[1]
        z = np.outer(np.cos(u), np.ones_like(v)) * radius + totem_pos[2]
        ax.plot_wireframe(x, y, z, color=colors[i])

    fname = 'Iter%06d' % iter_idx
    plt.title(fname, y=-0.01)
    out_path = os.path.join(out_folder, fname+'.png')
    plt.savefig(out_path)
    plt.close()


### [Keypoint in image] Helper functions ###
def get_cam_ray_ds_calib(W, H, mtx, downsample=1):
    x = np.linspace(0, W-1, W//downsample)
    y = np.linspace(0, H-1, H//downsample)
    xs, ys = np.meshgrid(x, y)
    cam_vecs = uv2cam_parallel_calib(xs, ys, mtx) # unnormalized
    cam_ray_ds = cam_vecs / LA.norm(cam_vecs, axis=2)[..., None]
    return cam_vecs, cam_ray_ds

def get_cam_ray_ds(W, H, is_mitsuba, cam_sensor_w, cam_sensor_h, f, downsample=1):
    '''
        Return a HxWx3 array representing camera ray directions 
        of all pixel coordinates
        
        Mitsuba coordinate system
        x: left to right = pos to neg
        y: top to bottom = pos to neg
        z: out to in = neg to pos
        Real image coordinate system
        x: left to right = neg to pos
        y: top to bottom = neg to pos
        z: out to in = neg to pos
    '''
    # Step 1: Get pixel coordinates' corresponding world position
    # on the virtual image plane. We assume the virtual image plane
    # to be one focal length away from the pinhole in the pos z direction 
    # and the same size as the sensor.

    # initialize x, y coordinates and recenter
    x = np.linspace(0, W-1, W//downsample)
    y = np.linspace(0, H-1, H//downsample)
    if is_mitsuba:
        x = x[::-1]
        y = y[::-1]
    x -= (W-1)/2
    y -= (H-1)/2

    # convert pixel unit to distance unit
    if is_mitsuba:
        x *= cam_sensor_h / H
    else:
        x *= cam_sensor_w / W
    y *= cam_sensor_h / H

    # Create meshgrid of converted xys and set zs to -f
    pos_map_xy = np.meshgrid(x, y)
    pos_map_z = np.ones([H//downsample, W//downsample]) * f
    pos_map = np.stack([pos_map_xy[0], pos_map_xy[1], pos_map_z], axis=-1)
    cam_ray_ds = pos_map / LA.norm(pos_map, axis=2)[..., None]
    return cam_ray_ds


def bb_intersection_over_union_torch(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = torch.max(torch.tensor([boxA[0], boxB[0]], dtype=torch.float32))
    yA = torch.max(torch.tensor([boxA[1], boxB[1]], dtype=torch.float32))
    xB = torch.min(torch.tensor([boxA[2], boxB[2]], dtype=torch.float32))
    yB = torch.min(torch.tensor([boxA[3], boxB[3]], dtype=torch.float32))
    # compute the area of intersection rectangle
    interArea = torch.max(torch.tensor([0, xB - xA + 1], dtype=torch.float32)) * torch.max(torch.tensor([0, yB - yA + 1], dtype=torch.float32))
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / (boxAArea + boxBArea - interArea)
    # import pdb; pdb.set_trace()
    # return the intersection over union value
    return iou


def totem_pose_proj_loss_torch(totem_pos, totem_radius, contour_pts, f):
    # https://math.stackexchange.com/questions/1184038/what-is-the-equation-of-a-general-circle-in-3-d-space
    # Find the equation of the plane cone base resides in
    C = totem_pos
    R = totem_radius
    # First, find the cone side (side) and base radius (R_base)
    hyp = torch.norm(C)
    side = torch.sqrt(hyp**2-R**2)
    R_base = side/hyp*R
    # Then, find the cone base center
    disp = R/hyp*R # cone base center displacement from totem center
    n = C/hyp # unit vec in cone axis direction, also normal vec for plane, pointing away from cam
    C_base = C - disp*n # 3D coords of cone base center


    # Project points around totem contour onto cone base plane
    # [Important] not actual, orthogonal projection, more like camera ray intersection with the cone base plane
    # (1) Plane equation - A(x - x0) + B(y - y0) + C(z - z0)=0, ABC = n, x0y0z0 = C_base
    # (2) Contour point X, Y, Z = m * (x, y, z)
    # Plug (2) into (1) to solve for the m's: (AX+BY+CZ)/m = Ax0+By0+Cz0
    ms = torch.unsqueeze(torch.dot(n, C_base)/torch.sum(contour_pts * n, dim=1), dim=1)
    proj_pts = contour_pts * ms
    # Distance from these projected points to cone base center should be close to R_base
    avg_dist = torch.mean(torch.norm(proj_pts-C_base, dim=1))
    return torch.abs(avg_dist-R_base)

def totem_pose_proj_loss(totem_pos, totem_radius, contour_pts, f):
    # https://math.stackexchange.com/questions/1184038/what-is-the-equation-of-a-general-circle-in-3-d-space
    # Find the equation of the plane cone base resides in
    C = totem_pos
    R = totem_radius
    # First, find the cone side (side) and base radius (R_base)
    hyp = torch.norm(C)
    side = torch.sqrt(hyp**2-R**2)
    R_base = side/hyp*R
    # Then, find the cone base center
    disp = R/hyp*R # cone base center displacement from totem center
    n = C/hyp # unit vec in cone axis direction, also normal vec for plane, pointing away from cam
    C_base = C - disp*n # 3D coords of cone base center


    # Project points around totem contour onto cone base plane
    # [Important] not actual, orthogonal projection, more like camera ray intersection with the cone base plane
    # (1) Plane equation - A(x - x0) + B(y - y0) + C(z - z0)=0, ABC = n, x0y0z0 = C_base
    # (2) Contour point X, Y, Z = m * (x, y, z)
    # Plug (2) into (1) to solve for the m's: (AX+BY+CZ)/m = Ax0+By0+Cz0
    ms = torch.unsqueeze(torch.dot(n, C_base)/torch.sum(contour_pts * n, dim=1), dim=1)
    proj_pts = contour_pts * ms
    # Distance from these projected points to cone base center should be close to R_base
    avg_dist = torch.mean(torch.norm(proj_pts-C_base, dim=1))
    return torch.abs(avg_dist-R_base).item()


def totem_pose_iou_loss_calib(totem_pos, totem_radius, cx, cy, scene_params, boxB):
    C = totem_pos
    R = totem_radius

    hyp = torch.norm(C)
    side = torch.sqrt(hyp**2-R**2)
    R_base = side/hyp*R
    disp = R/hyp*R
    n = C/hyp # unit vec in cone axis direction, also normal vec for plane
    C_base = C - disp*n
    # https://math.stackexchange.com/questions/1184038/what-is-the-equation-of-a-general-circle-in-3-d-space
    # the parametric one, t = (0, 2pi)
    N_samples = 1000
    ts = torch.linspace(0, 2*np.pi, N_samples)
    ts = ts.view(N_samples, 1)
    # Equation of the plane the cone base resides on
    # A(x - x0) + B(y - y0) + C(z - z0)=0, ABC = n, x0y0z0 = C_base
    # assume x,y = 1, solve for z, then normalize vector
    x, y = 1,1
    z = (n[0] * (x - C_base[0]) + n[1] * (y - C_base[1])) / (-1*n[2]) + C_base[2]
    v1 = torch.tensor([x, y, z], dtype=torch.float32)-C_base
    v1_norm = v1/torch.norm(v1)
    v2 = torch.cross(n, v1) # cross product
    v2_norm = v2/torch.norm(v2)
    v1_norm = v1_norm.view(1, 3)
    v2_norm = v2_norm.view(1, 3)
    # (N, 3)
    samples_3d = C_base + R_base * torch.matmul(torch.cos(ts),v1_norm) + R_base * torch.matmul(torch.sin(ts),v2_norm)
    # Project 3D points onto the image plane, reversing the uv2cam method
    samples_3d_norm = samples_3d/torch.unsqueeze(samples_3d[:, 2], dim=1) # scale s.t. z=1
    samples_2d = samples_3d_norm[:, :2] * scene_params['focal'] # slides points to the focal plane
    samples_2d[:, 0] *= (scene_params['image_width'] / scene_params['cam_sensor_w']) # scale points from real world units to pixel units
    samples_2d[:, 1] *= (scene_params['image_height'] / scene_params['cam_sensor_h'])
    samples_2d[:, 0] += cx # shift points s.t. optical center is at (0,0)
    samples_2d[:, 1] += cy

    # Compute from totem pose
    boxA = [min(samples_2d[:, 0]), min(samples_2d[:, 1]), max(samples_2d[:, 0]), max(samples_2d[:, 1])]
    iou = bb_intersection_over_union_torch(boxA, boxB)
    return 1-iou


def totem_pose_iou_loss(is_mitsuba, totem_pos, totem_radius, f, H, W, 
    cam_sensor_h, cam_sensor_w, boxB):
    C = totem_pos
    R = totem_radius

    hyp = torch.norm(C)
    side = torch.sqrt(hyp**2-R**2)
    R_base = side/hyp*R
    disp = R/hyp*R
    n = C/hyp # unit vec in cone axis direction, also normal vec for plane
    C_base = C - disp*n
    # https://math.stackexchange.com/questions/1184038/what-is-the-equation-of-a-general-circle-in-3-d-space
    # the parametric one, t = (0, 2pi)
    N_samples = 1000
    ts = torch.linspace(0, 2*np.pi, N_samples)
    ts = ts.view(N_samples, 1)
    # Equation of the plane the cone base resides on
    # A(x - x0) + B(y - y0) + C(z - z0)=0, ABC = n, x0y0z0 = C_base
    # assume x,y = 1, solve for z, then normalize vector
    x, y = 1,1
    z = (n[0] * (x - C_base[0]) + n[1] * (y - C_base[1])) / (-1*n[2]) + C_base[2]
    v1 = torch.tensor([x, y, z], dtype=torch.float32)-C_base
    v1_norm = v1/torch.norm(v1)
    v2 = torch.cross(n, v1) # cross product
    v2_norm = v2/torch.norm(v2)
    v1_norm = v1_norm.view(1, 3)
    v2_norm = v2_norm.view(1, 3)
    # (N, 3)
    samples_3d = C_base + R_base * torch.matmul(torch.cos(ts),v1_norm) + R_base * torch.matmul(torch.sin(ts),v2_norm)
    # Project 3D points onto the image plane, reversing the uv2cam method
    samples_3d_norm = samples_3d/torch.unsqueeze(samples_3d[:, 2], dim=1)
    samples_2d = samples_3d_norm[:, :2] * f 
    samples_2d *= H / cam_sensor_h
    samples_2d[:, 0] += (W-1)/2
    samples_2d[:, 1] += (H-1)/2
    if is_mitsuba:
        samples_2d[:, 0] = (W-1) - samples_2d[:, 0] 
        samples_2d[:, 1] = (H-1) - samples_2d[:, 1]  
    # # removing round cuz not differentiable
    # samples_2d = np.round(samples_2d).astype(int)

    # Compute from totem pose
    boxA = [min(samples_2d[:, 0]), min(samples_2d[:, 1]), max(samples_2d[:, 0]), max(samples_2d[:, 1])]
    iou = bb_intersection_over_union_torch(boxA, boxB)
    return 1-iou

def totem_pose_iou_loss_v2(is_mitsuba, totem_pos, totem_radius, f, H, W,
    cam_sensor_h, cam_sensor_w, boxB):
    start1 = time.time()
    C = totem_pos # Different at each iteration
    R = totem_radius

    # Bunch of calculations that depends on totem_pos
    hyp = torch.norm(C)
    side = torch.sqrt(hyp**2-R**2)
    R_base = side/hyp*R
    disp = R/hyp*R
    n = C/hyp # unit vec in cone axis direction, also normal vec for plane
    C_base = C - disp*n
    end1 = '%.4f seconds' % (time.time()-start1)
    print('Iou block1: %s' % end1)

    # https://math.stackexchange.com/questions/1184038/what-is-the-equation-of-a-general-circle-in-3-d-space
    # the parametric one, t = (0, 2pi)
    start2 = time.time()
    N_samples = 1000
    ts = torch.linspace(0, 2*np.pi, N_samples)
    ts = ts.view(N_samples, 1)
    # Equation of the plane the cone base resides on
    # A(x - x0) + B(y - y0) + C(z - z0)=0, ABC = n, x0y0z0 = C_base
    # assume x,y = 1, solve for z, then normalize vector
    x, y = 1,1
    z = (n[0] * (x - C_base[0]) + n[1] * (y - C_base[1])) / (-1*n[2]) + C_base[2]
    v1 = torch.tensor([x, y, z], dtype=torch.float32)-C_base
    v1_norm = v1/torch.norm(v1)
    v2 = torch.cross(n, v1) # cross product
    v2_norm = v2/torch.norm(v2)
    v1_norm = v1_norm.view(1, 3)
    v2_norm = v2_norm.view(1, 3)
    # (N, 3)
    samples_3d = C_base + R_base * torch.matmul(torch.cos(ts),v1_norm) + R_base * torch.matmul(torch.sin(ts),v2_norm)
    # Project 3D points onto the image plane, reversing the uv2cam method
    samples_3d_norm = samples_3d/torch.unsqueeze(samples_3d[:, 2], dim=1)
    samples_2d = samples_3d_norm[:, :2] * f
    samples_2d *= H / cam_sensor_h
    samples_2d[:, 0] += (W-1)/2
    samples_2d[:, 1] += (H-1)/2
    if is_mitsuba:
        samples_2d[:, 0] = (W-1) - samples_2d[:, 0]
        samples_2d[:, 1] = (H-1) - samples_2d[:, 1]
    end2 = '%.4f seconds' % (time.time()-start2)
    print('Iou block2: %s' % end2)

    start3 = time.time()
    # Compute from totem pose
    boxA = [min(samples_2d[:, 0]), min(samples_2d[:, 1]), max(samples_2d[:, 0]), max(samples_2d[:, 1])]
    iou = bb_intersection_over_union_torch(boxA, boxB)
    end3 = '%.4f seconds' % (time.time()-start3)
    print('Iou block3: %s' % end3)
    return 1-iou
