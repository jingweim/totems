import os
import numpy as np
import numpy.linalg as LA
import imageio
import cv2
import torch
import torch.nn as nn

# ---------------------------- NumPy methods --------------------------------------------


def get_valid_rays_view_frustum_projection(W, H, K, near, rays_o, rays_d, thresh=2.0):
    '''
        Project ray origins onto a [-1, 1] xy view frustum at near clipping plane
        (using NDC formulas in NeRF supplementary).
        Filter out those that fall outside the view frustum.

        Args:
            W, H: image width and height
            K: 3x3 camera intrinsic matrix
            near: near clipping plane (in meters)
            rays_o, rays_d: ray origins and directions in euclidean space
            thresh: side length of a square boundary on the near clipping plane;
                    rays with origins within this square are kept

        Return:
            valid_idx: the indices of remaining valid rays

    '''

    # Paragraph under eq.26, shift the origin of the NDC ray to near plane.
    # Note that the origin must be shifted away from zero to avoid divison by zero later
    o = rays_o
    d = rays_d
    t0  = -(near + o[:, [2]]) / d[:, [2]]
    o = o + t0 * d
   
    # Eq 21, 22
    ax = -K[0,0]/(W/2)
    ay = -K[1,1]/(H/2)

    # Eq 25, NDC ray origin
    ox, oy, oz = o[:, [0]], o[:, [1]], o[:, [2]]
    o_ndc_analytic = np.concatenate([ax*ox/oz, ay*oy/oz], axis=1)

    # Identify ray origins that fall outside view frustum
    rays_thresholded = 1 - np.logical_or(np.abs(o_ndc_analytic[:, 0]) > thresh,
                                         np.abs(o_ndc_analytic[:, 1]) > thresh)
    valid_idx = np.where(rays_thresholded == 1)[0]

    return valid_idx


def euc_to_cube(W, H, K, near, far, pts):
    '''
        Projects euclidean 3D points into a cube space [-1, 1]^3, 
        z is linear in depth

        Args:
            W, H: image width and height
            K: 3x3 camera intrinsic matrix
            near, far: near and far clipping plane (in meters)
            pts: 3D points in euclidean space, torch tensor

        Returns:
            pts_cube: 3D points in cube space, torch tensor
    '''

    right = near * (W/2) / K[0,0]
    top = near * (H/2) / K[1,1]
    cube_x = near/right * pts[..., 0] / pts[..., 2]
    cube_y = near/top * pts[..., 1] / pts[..., 2]
    cube_z = 2 / (far-near) * pts[..., 2] + 1 - 2*far/(far-near)
    pts_cube = torch.stack([cube_x, cube_y, cube_z], axis=-1)
    return pts_cube


def get_cam_rays(W, H, K):
    '''
        Given the camera intrinsics matrix, compute camera ray directions.
        z = 1 for all rays.

        Args:
            W, H: image width and height
            K: 3x3 camera intrinsic matrix

        Returns:
            cam_rays_d: camera ray origins, (w, h, 3)
            cam_rays_d: camera ray directions, (w, h, 3)
    '''

    # Get 2D pixels (x,y)
    x = np.linspace(0, W-1, W)
    y = np.linspace(0, H-1, H)
    xs, ys = np.meshgrid(x, y)

    # Convert xys to ray directions
    cx, cy = K[0,2], K[1,2]
    fx, fy = K[0,0], K[1,1]
    cam_rays_d = np.stack([(xs-cx)/fx, (ys-cy)/fy, np.ones(xs.shape)], axis=-1)
    cam_rays_d = cam_rays_d / cam_rays_d[..., 2][..., None] # Normalize

    cam_ray_o = np.array([0,0,0])
    cam_rays_o = np.broadcast_to(cam_ray_o, cam_rays_d.shape)
    return cam_rays_o, cam_rays_d


def load_real_data(args, downsample=1):
    '''
        One-time loading of data and parameters into a dictionary.

        Args:
            downsample: downsample factor (for rendering, if image size is too big)

        Structure of the dictionary returned
            data: 
                image: the totem-protected image (undistorted)
                H, W: image height and width
                K: the 3x3 camera intrinsic matrix
                n_totems: number of totems in the image
                totem_000:
                    mask: the annotated totem mask (also undistorted)
                    ys, xs: the x, y coordinates of totem pixels
                    n_totem_pixs: number of totem pixels
                totem_001:
                ...

    '''

    data = dict()

    # Commonly used arguments
    data_dir = args.data_dir

    # Paths
    img_path = os.path.join(data_dir, 'image.png')
    mask_folder = os.path.join(data_dir, 'totem_masks')

    # Image specific
    img = imageio.imread(img_path)
    H, W, _ = img.shape
    K = np.load(args.calib_npy, allow_pickle=True).item()['mtx']

    # Downsample
    if downsample != 1:
        H, W = H // downsample, W // downsample
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        K[0,0] /= downsample
        K[0,2] /= downsample
        K[1,1] /= downsample
        K[1,2] /= downsample
    cam_rays_o, cam_rays_d = get_cam_rays(W, H, K)

    # Load into data
    data['image'] = img
    data['H'] = H
    data['W'] = W
    data['K'] = K
    data['cam_rays_o'] = cam_rays_o
    data['cam_rays_d'] = cam_rays_d

    # Totem specific 
    mask_fnames = sorted(os.listdir(mask_folder))
    data['n_totems'] = len(mask_fnames)

    for mask_fname in mask_fnames:
        totem_name = mask_fname.split('.')[0]
        totem_data = dict()

        # Load totem mask (downsampled) and totem pixels
        mask_path = os.path.join(mask_folder, mask_fname)
        mask = imageio.imread(mask_path, pilmode='L')
        if downsample != 1:
            mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
        ys, xs = np.where(mask)
        totem_data['mask'] = mask
        totem_data['ys'] = ys
        totem_data['xs'] = xs
        totem_data['n_totem_pixs'] = len(ys)
        totem_data['bbox'] = [min(xs), min(ys), max(xs), max(ys)]

        # Add totem data to dictionary
        data[totem_name] = totem_data

    return data


def line_sphere_intersection_numpy(totem_pos, totem_radius, rays_d, rays_o, use_min=True):
    '''
        Calculate the camera ray intersection with a totem.
            Equation from here
            https://stackoverflow.com/questions/32571063/specific-location-of-intersection-between-sphere-and-3d-line-segment
            
        Args:
            totem_pos: 3D position of the spherical totem's center
            totem_radius: totem radius in centimeters
            rays_d: (X01, Y01, Z01) in the above link, size (N,3)
            rays_o: (X0, Y0, Z0) in the above link, size (N,3)
            use_min: True if choosing the intersection closer to camera, False if otherwise

        Returns:
            pts: 3D position of the intersections
            valid_idx: indices of the remaining rays (the ones not intersecting get filtered out)

    '''

    # Quadratic formula 
    shift = rays_o-totem_pos #(N, 3)
    a = np.sum(rays_d**2, axis=1) #(N,)
    b = 2 * np.sum(shift*rays_d, axis=1) #(N,)
    c = np.sum(shift**2, axis=1) - totem_radius**2 #(N,)
    sqrt_inner = b**2 - 4*a*c

    # Filter invalid rays (not intersecting or one intersection)
    valid_idx = np.where(sqrt_inner > 0)[0]
    sqrt_inner = sqrt_inner[valid_idx]
    a = a[valid_idx]
    b = b[valid_idx]

    # Select one of the intersections (ts, distance along rays_d)
    t1 = (-b+np.sqrt(sqrt_inner))/(2*a)
    t2 = (-b-np.sqrt(sqrt_inner))/(2*a)
    ts = [t1, t2]
    if use_min: # first refraction, intersection closer to camera
        t = np.min(ts, axis=0) # (N,)
    else: # second refraction, intersection farther from camera
        t = np.max(ts, axis=0)

    # Compute the 3D position of intersections
    pts = rays_o[valid_idx] + t[:, None] * rays_d[valid_idx]
    return pts, valid_idx


def get_refracted_ray_numpy(S1, N, n1, n2):
    '''
        Compute the ray direction after refraction.
            Formula from here: http://www.starkeffects.com/snells-law-vector.shtml

        Args:
            S1: incoming ray direction, unit vector
            N: surface normal, unit vector
            n1: refraction index of the medium the ray came from
            n2: refraction index the ray is going into

        Returns:
            the refracted ray direction, array size (N, 3)
    '''
    return n1/n2 * np.cross(N, np.cross(-N, S1)) - N * np.sqrt(1-n1**2/n2**2 * np.sum(np.cross(N, S1) * np.cross(N, S1), axis=1))[:, None]


def cam_rays_to_totem_rays_numpy(args, cam_rays_o, cam_rays_d, totem_pos, W, H, K, ior_totem, ior_air=1.0):
    '''
        Figure 3 in paper.
        Convert camera rays to totem rays (two refractions).
        Filter out rays that don't intersect with the totem in 3D space.

        Args:
            cam_rays_d: camera ray directions
            totem_pos: 3D position of the spherical totem's center, numpy
            totem_radius: totem radius in centimeters
            W, H: image width and height
            K: 3x3 camera intrinsic matrix
            ior_totem: totem's index of refraction
            ior_air: air's index of refraction


        Returns:
            totem_rays_o:
            totem_rays_d:
            valid_idx_1: valid indices after the 1st refraction
            valid_idx_2: valid indices (of valid_idx_1) after the 2nd refraction
            valid_idx_3: valid indices (of valid_idx_2) after the view frustum projection
    '''

    cam_ray_o = cam_rays_o[0]
    totem_radius = args.totem_radius

    # The first refraction
    D, valid_idx_1 = line_sphere_intersection_numpy(totem_pos, totem_radius, cam_rays_d, cam_rays_o)
    OD = (D-cam_ray_o)/LA.norm(D-cam_ray_o, axis=1)[:, None]
    AD = (D-totem_pos)/LA.norm(D-totem_pos, axis=1)[:, None]
    DE= get_refracted_ray_numpy(OD, AD, ior_air, ior_totem)

    # The second refraction
    E, valid_idx_2 = line_sphere_intersection_numpy(totem_pos, totem_radius, DE, D, use_min=False)
    EA = (totem_pos-E)/LA.norm(totem_pos-E, axis=1)[:, None]
    direction = get_refracted_ray_numpy(DE, EA, ior_totem, ior_air)
    totem_rays_o = E
    totem_rays_d = direction

    # Filter out totem rays that are outside of view frustum
    valid_idx_3 = get_valid_rays_view_frustum_projection(W, H, K, args.near, totem_rays_o, totem_rays_d)
    totem_rays_o = totem_rays_o[valid_idx_3]
    totem_rays_d = totem_rays_d[valid_idx_3]
    
    return totem_rays_o, totem_rays_d, valid_idx_1, valid_idx_2, valid_idx_3


def get_totem_rays_numpy(args, data, totem_idx, totem_pos, n_rays=None):

    '''
        Compute totem ray origins and directions for training or rendering

        Args:
            totem_idx: totem index (0 to n-1 = left to right in image)
            n_rays: if specified (train), randomly sample n rays, otherwise (render), use all rays
        
        Returns: A torch tensor [rays_o, rays_d], size (2, n_rays, 3)
            
    '''

    # Load commonly used data
    ior_totem = args.ior_totem
    W, H, K = data['W'], data['H'], data['K']
    totem_data = data['totem_%03d' % totem_idx]
    n_totem_pixs = totem_data['n_totem_pixs']
    ys, xs = totem_data['ys'], totem_data['xs']
    cam_rays_o, cam_rays_d = data['cam_rays_o'], data['cam_rays_d']
    img = data['image']

    # Sample n rays for training
    if n_rays:

        # Shuffle totem pixel indices
        pix_ids = np.arange(n_totem_pixs)
        np.random.shuffle(pix_ids)

        # Divide shuffled cam rays into chunks
        ys_input, xs_input = [],[]
        rays_o, rays_d = [],[]
        chunk_size = n_rays * 2
        n_chunks = n_totem_pixs // chunk_size + 1

        for chunk_idx in range(n_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min((chunk_idx+1)*chunk_size, n_totem_pixs)
            pix_ids_chunk = pix_ids[start_idx:end_idx]
            ys_chunk = ys[pix_ids_chunk]
            xs_chunk = xs[pix_ids_chunk]
            cam_rays_d_chunk = cam_rays_d[ys_chunk, xs_chunk]
            cam_rays_o_chunk = cam_rays_o[ys_chunk, xs_chunk]
            totem_rays_o_chunk, totem_rays_d_chunk, valid_idx_1, valid_idx_2, valid_idx_3 = cam_rays_to_totem_rays_numpy(args, cam_rays_o_chunk, cam_rays_d_chunk, totem_pos, W, H, K, ior_totem)
            ys_chunk = ys_chunk[valid_idx_1][valid_idx_2][valid_idx_3]
            xs_chunk = xs_chunk[valid_idx_1][valid_idx_2][valid_idx_3]

            # Merge chunk with previous data
            if len(ys_input): # not empty
                ys_input = np.concatenate([ys_input, ys_chunk], axis=0)
                xs_input = np.concatenate([xs, xs_chunk], axis=0)
                rays_o = np.concatenate([rays_o, totem_rays_o_chunk], axis=0)
                rays_d = np.concatenate([rays_d, totem_rays_d_chunk], axis=0)
            else:
                ys_input = ys_chunk
                xs_input = xs_chunk
                rays_o = totem_rays_o_chunk
                rays_d = totem_rays_d_chunk

            # If accumulated > N_rand rays, exit loop
            if len(ys_input) > n_rays:
                ys_input = ys_input[:n_rays]
                xs_input = xs_input[:n_rays]
                rays_o = rays_o[:n_rays]
                rays_d = rays_d[:n_rays]
                break

        assert len(ys_input) >= n_rays, "Totem has gone too far from its position in the mask, not enough rays to sample"

    else:
        ys_input, xs_input = ys.copy(), xs.copy()
        cam_rays_d_input = cam_rays_d[ys_input, xs_input]
        cam_rays_o_input = cam_rays_o[ys_input, xs_input]
        totem_rays_o_input, totem_rays_d_input, valid_idx_1, valid_idx_2, valid_idx_3 = cam_rays_to_totem_rays_numpy(args, cam_rays_o_input, cam_rays_d_input, totem_pos, W, H, K, ior_totem)
        rays_o = totem_rays_o_input
        rays_d = totem_rays_d_input
        ys_input = ys_input[valid_idx_1][valid_idx_2][valid_idx_3]
        xs_input = xs_input[valid_idx_1][valid_idx_2][valid_idx_3]

    target_rgbs = img[ys_input, xs_input]
    return rays_o, rays_d, ys_input, xs_input, target_rgbs


# ---------------------------- Pytorch methods --------------------------------------------


def shift_and_normalize(rays):
    rays_o, rays_d = rays

    # Scale ray directions to dz = 1
    d_scale = 1/rays_d[:, [2]]
    rays_d = rays_d * d_scale

    # Shift ray origins to z = 0
    t_shift = -rays_o[:, [2]] / rays_d[:, [2]]
    rays_o = rays_o + t_shift * rays_d

    return torch.stack([rays_o, rays_d], dim=0)


def project_3D_pts_to_2D(pts_3D, W, H, K):
    pts_2D_x = pts_3D[:, 0] / pts_3D[:, 2] * K[0,0] + K[0,2]
    pts_2D_y = pts_3D[:, 1] / pts_3D[:, 2] * K[1,1] + K[1,2]
    pts_2D_x = (torch.round(pts_2D_x)).int()
    pts_2D_y = (torch.round(pts_2D_y)).int()

    # Filter points that fall outside the image frame
    filter = torch.nonzero((pts_2D_x >= 0) * (pts_2D_x < W) * (pts_2D_y >= 0) * (pts_2D_y < H))
    pts_2D_x = pts_2D_x[filter]
    pts_2D_y = pts_2D_y[filter]

    return pts_2D_x, pts_2D_y


def line_sphere_intersection_torch(totem_pos, totem_radius, rays_d, rays_o, use_min=True):
    '''
        Pytorch version of line_sphere_intersection_numpy

    '''

    # Quadratic formula 
    shift = rays_o-totem_pos #(N, 3)
    a = torch.sum(rays_d**2, dim=1) #(N,)
    b = 2 * torch.sum(shift*rays_d, dim=1) #(N,)
    c = torch.sum(shift**2, dim=1) - totem_radius**2 #(N,)

    # Filter invalid rays (not intersecting or one intersection)
    t1 = (-b+torch.sqrt(b**2-4*a*c))/(2*a) #(N,)
    t2 = (-b-torch.sqrt(b**2-4*a*c))/(2*a) #(N,)
    # notequal = t1 != t2 # True = two intersections, False = one intersection
    # notnan1 = ~torch.isnan(t1)
    # notnan2 = ~torch.isnan(t2)
    # valid_idx = torch.squeeze(torch.nonzero(notequal * notnan1 * notnan2))
    # t1 = t1[valid_idx]
    # t2 = t2[valid_idx]
    # rays_o = rays_o[valid_idx]
    # rays_d = rays_d[valid_idx]
    ts = torch.stack([t1, t2], dim=0)

    # Select one of the intersections (ts, distance along rays_d)
    if use_min: # first refraction, intersection closer to camera
        t, _ = torch.min(ts, dim=0) # (N,)
    else: # second refraction, intersection farther from camera
        t, _ = torch.max(ts, dim=0)

    # Compute the 3D position of intersections
    pts = rays_o + torch.unsqueeze(t, dim=1) * rays_d
    return pts
    # return pts, valid_idx


def get_refracted_ray_torch(S1, N, n1, n2):
    '''
        Pytorch version of get_refracted_ray_numpy
    '''
    return n1/n2 * torch.cross(N, torch.cross(-N, S1)) - N * torch.sqrt(1-n1**2/n2**2 * torch.unsqueeze(torch.sum(torch.cross(N, S1) * torch.cross(N, S1), dim=1), dim=1))


def cam_rays_to_totem_rays_torch(args, cam_rays_o, cam_rays_d, totem_pos, W, H, K, ior_totem, ior_air=1.0):
    '''
        Pytorch version of cam_rays_to_totem_rays_numpy
    '''

    cam_ray_o = cam_rays_o[0]
    totem_radius = args.totem_radius

    # The first refraction
    D = line_sphere_intersection_torch(totem_pos, totem_radius, cam_rays_d, cam_rays_o)
    OD = (D-cam_ray_o)/torch.unsqueeze(torch.norm(D-cam_ray_o, dim=1), dim=1)
    AD = (D-totem_pos)/torch.unsqueeze(torch.norm(D-totem_pos, dim=1), dim=1)
    DE= get_refracted_ray_torch(OD, AD, ior_air, ior_totem)

    # The second refraction
    E = line_sphere_intersection_torch(totem_pos, totem_radius, DE, D, use_min=False)
    EA = (totem_pos-E)/torch.unsqueeze(torch.norm(totem_pos-E, dim=1), dim=1)
    direction = get_refracted_ray_torch(DE, EA, ior_totem, ior_air)
    totem_rays_o = E
    totem_rays_d = direction
    
    return totem_rays_o, totem_rays_d


def get_totem_rays_torch(args, ys_input, xs_input, device, data, totem_idx, totem_pos, n_rays=None):

    '''
        Pytorch version of get_totem_rays_numpy

        Args:
            ys_input, xs_input: n_ray number of rays sampled using get_totem_rays_numpy
    '''

    # Load commonly used data
    ior_totem = args.ior_totem
    W, H, K = data['W'], data['H'], data['K']
    cam_rays_o, cam_rays_d = data['cam_rays_o'], data['cam_rays_d']
    img = data['image']
    n_rays = len(ys_input)

    # Load selected rays and move to pytorch
    cam_rays_d_input = cam_rays_d[ys_input, xs_input]
    cam_rays_o_input = cam_rays_o[ys_input, xs_input]
    cam_rays_d_input = torch.from_numpy(cam_rays_d_input.astype('float32')).to(device)
    cam_rays_o_input = torch.from_numpy(cam_rays_o_input.astype('float32')).to(device)

    # Refraction into totem rays
    totem_rays_o_input, totem_rays_d_input = cam_rays_to_totem_rays_torch(args, cam_rays_o_input, cam_rays_d_input, totem_pos, W, H, K, ior_totem)
    rays_o = totem_rays_o_input
    rays_d = totem_rays_d_input
    target_rgbs = img[ys_input, xs_input]
    assert len(rays_o) == n_rays

    return rays_o, rays_d, ys_input, xs_input, target_rgbs


# ---------------------------- Totem pose joint optimization --------------------------------------------


class LearnTotemPos(nn.Module):
    def __init__(self, initial_totem_pos, totem_pos_residual, req_grad, device):
        super(LearnTotemPos, self).__init__()
        self.totem_pos_residual = nn.Parameter(torch.from_numpy(totem_pos_residual.astype('float32')), requires_grad=req_grad)
        self.initial_totem_pos = torch.from_numpy(initial_totem_pos.astype('float32')).to(device)

    def forward(self, totem_id):
        totem_pos = self.initial_totem_pos[totem_id] + self.totem_pos_residual[totem_id]
        return totem_pos


def bb_intersection_over_union_torch(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = torch.max(torch.tensor([boxA[0], boxB[0]], dtype=torch.float32)).item()
    yA = torch.max(torch.tensor([boxA[1], boxB[1]], dtype=torch.float32)).item()
    xB = torch.min(torch.tensor([boxA[2], boxB[2]], dtype=torch.float32)).item()
    yB = torch.min(torch.tensor([boxA[3], boxB[3]], dtype=torch.float32)).item()
    # compute the area of intersection rectangle
    interArea = torch.max(torch.tensor([0, xB - xA + 1], dtype=torch.float32)).item() * torch.max(torch.tensor([0, yB - yA + 1], dtype=torch.float32)).item()
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


def totem_pose_iou_loss(args, data, totem_idx, totem_pos):
    W, H, K = data['W'], data['H'], data['K']
    boxB = data['totem_%03d' % totem_idx]['bbox']
    totem_radius = args.totem_radius

    # https://math.stackexchange.com/questions/1184038/what-is-the-equation-of-a-general-circle-in-3-d-space
    # the parametric one formula in the top answer, t = (0, 2pi)
    n_samples = 1000
    ts = torch.linspace(0, 2*np.pi, n_samples)
    ts = ts.view(n_samples, 1)

    # Figure 8 in supplementary
    P = totem_pos
    R = totem_radius

    OP = torch.norm(P)
    OA = torch.sqrt(OP**2-R**2)
    CA = OA*(R/OP) # R:OP = AC:OA
    d_c = P/OP # unit vector in cone axis direction
    phi = (R/OP)*R # phi:R = R:OP
    C = P - d_c * phi

    # Equation of the plane the cone base resides on
    # A(x - x0) + B(y - y0) + C(z - z0)=0
    # ABC = d_c, normal vector to the plane
    # x0y0z0 = C
    # xyz, a point on the plane
    # assume x,y = 1, solve for z
    # v1, v2 = two orthogonal vectors on the plane
    x,y = 1,1
    z = (d_c[0] * (x - C[0]) + d_c[1] * (y - C[1])) / (-1*d_c[2]) + C[2]
    v1 = torch.tensor([x, y, z], dtype=torch.float32)-C
    v1_norm = v1/torch.norm(v1)
    v2 = torch.cross(d_c, v1_norm) # cross product, orthogonal to d_c and v1
    v2_norm = v2/torch.norm(v2)
    v1_norm = v1_norm.view(1, 3)
    v2_norm = v2_norm.view(1, 3)
    samples_3d = C + CA * torch.matmul(torch.cos(ts),v1_norm) + CA * torch.matmul(torch.sin(ts),v2_norm)
    samples_2d_x, samples_2d_y = project_3D_pts_to_2D(samples_3d, W, H, K)

    # Compute iou loss by comparing the 2D projection with annotated 2D totem mask
    boxA = [min(samples_2d_x).item(), min(samples_2d_y).item(), max(samples_2d_x).item(), max(samples_2d_y).item()]
    iou = bb_intersection_over_union_torch(boxA, boxB)
    return 1-iou