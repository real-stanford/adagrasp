import collections
import os
import queue
import shutil
import threading

import cv2
import dominate
import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch

from models.model_utils import rotate_tensor2d, rotate_tensor3d


# Get Tableau color palette (10 colors) https://www.tableau.com/
# Output:
#   palette - 10x3 uint8 array of color values in range 0-255 (each row is a color)
def get_tableau_palette():
    palette = np.array([[ 78,121,167], # blue
                        [255, 87, 89], # red
                        [ 89,169, 79], # green
                        [242,142, 43], # orange
                        [237,201, 72], # yellow
                        [176,122,161], # purple
                        [255,157,167], # pink 
                        [118,183,178], # cyan
                        [156,117, 95], # brown
                        [186,176,172]  # gray
                        ],dtype=np.uint8)
    return palette


# Get 3D pointcloud from RGB-D image
# Input:
#   color_img - HxWx3 uint8 array of color values in range 0-255
#   depth_img - HxW float array of depth values in meters aligned with color_img
#   segmentation_mask - HxW int array of segmentation instance
#   cam_intr  - 3x3 float array of camera intrinsic parameters
# Output:
#   cam_pts   - Nx3 float array of 3D points in camera coordinates
#   color_pts - Nx3 uint8 array of color values in range 0-255 corresponding to cam_pts
#   segmentation_pts - Nx1 int array of segmentation instance corresponding to cam_pts
def get_pointcloud(color_img, depth_img, segmentation_mask, cam_intr):

    img_h = depth_img.shape[0]
    img_w = depth_img.shape[1]

    # Project depth into 3D pointcloud in camera coordinates
    pixel_x,pixel_y = np.meshgrid(np.linspace(0,img_w-1,img_w),
                                  np.linspace(0,img_h-1,img_h))
    cam_pts_x = np.multiply(pixel_x-cam_intr[0,2],depth_img/cam_intr[0,0])
    cam_pts_y = np.multiply(pixel_y-cam_intr[1,2],depth_img/cam_intr[1,1])
    cam_pts_z = depth_img
    cam_pts = np.array([cam_pts_x,cam_pts_y,cam_pts_z]).transpose(1,2,0).reshape(-1,3)

    color_pts = None if color_img is None else color_img.reshape(-1,3)
    segmentation_pts = None if segmentation_mask is None else segmentation_mask.reshape(-1, 1)

    # Remove invalid 3D points (where depth == 0)
    valid_depth_ind = np.where(depth_img.flatten() > 0)[0]
    cam_pts = cam_pts[valid_depth_ind,:]
    color_pts = None if color_img is None else color_pts[valid_depth_ind,:]
    segmentation_pts = None if segmentation_mask is None else segmentation_pts[valid_depth_ind,:]

    return cam_pts, color_pts, segmentation_pts


# Apply rigid transformation to 3D pointcloud
# Input:
#   xyz_pts      - Nx3 float array of 3D points
#   rigid_transform - 3x4 or 4x4 float array defining a rigid transformation (rotation and translation)
# Output:
#   xyz_pts      - Nx3 float array of transformed 3D points
def transform_pointcloud(xyz_pts,rigid_transform):
    xyz_pts = np.dot(rigid_transform[:3,:3],xyz_pts.T) # apply rotation
    xyz_pts = xyz_pts+np.tile(rigid_transform[:3,3].reshape(3,1),(1,xyz_pts.shape[1])) # apply translation
    return xyz_pts.T


# Get top-down (along z-axis) orthographic heightmap image from 3D pointcloud
# Input:
#   cam_pts          - Nx3 float array of 3D points in world coordinates
#   color_pts        - Nx3 uint8 array of color values in range 0-255 corresponding to cam_pts
#   segmentation_pts - Nx1 int array of segmentation instance corresponding to cam_pts
#   view_bounds      - 3x2 float array of values (rows: X,Y,Z; columns: min,max) defining region in 3D space of heightmap in world coordinates
#   heightmap_pix_sz - float value defining size of each pixel in meters (determines heightmap resolution)
#   zero_level       - float value defining z coordinate of zero level (i.e. bottom) of heightmap 
# Output:
#   depth_heightmap  - HxW float array of height values (from zero level) in meters
#   color_heightmap  - HxWx3 uint8 array of backprojected color values in range 0-255 aligned with depth_heightmap
#   segmentation_heightmap - HxW int array of segmentation instance aligned with depth_heightmap
def get_heightmap(cam_pts,color_pts,segmentation_pts,view_bounds,heightmap_pix_sz,zero_level):

    heightmap_size = np.round(((view_bounds[1,1]-view_bounds[1,0])/heightmap_pix_sz,
                               (view_bounds[0,1]-view_bounds[0,0])/heightmap_pix_sz)).astype(int)

    # Remove points outside workspace bounds
    heightmap_valid_ind = np.logical_and(np.logical_and(
                          np.logical_and(np.logical_and(cam_pts[:,0] >= view_bounds[0,0],
                                                        cam_pts[:,0] <  view_bounds[0,1]),
                                                        cam_pts[:,1] >= view_bounds[1,0]),
                                                        cam_pts[:,1] <  view_bounds[1,1]),
                                                        cam_pts[:,2] <  view_bounds[2,1])
    cam_pts = cam_pts[heightmap_valid_ind]
    color_pts = color_pts[heightmap_valid_ind]
    segmentation_pts = segmentation_pts[heightmap_valid_ind]

    # Sort points by z value (works in tandem with array assignment to ensure heightmap uses points with highest z values)
    sort_z_ind = np.argsort(cam_pts[:,2])
    cam_pts = cam_pts[sort_z_ind]
    color_pts = color_pts[sort_z_ind]
    segmentation_pts = segmentation_pts[sort_z_ind]

    # Backproject 3D pointcloud onto heightmap
    heightmap_pix_x = np.floor((cam_pts[:,0]-view_bounds[0,0])/heightmap_pix_sz).astype(int)
    heightmap_pix_y = np.floor((cam_pts[:,1]-view_bounds[1,0])/heightmap_pix_sz).astype(int)

    # Get height values from z values minus zero level
    depth_heightmap = np.zeros(heightmap_size)
    depth_heightmap[heightmap_pix_y,heightmap_pix_x] = cam_pts[:,2]
    depth_heightmap = depth_heightmap-zero_level
    depth_heightmap[depth_heightmap < 0] = 0
    depth_heightmap[depth_heightmap == -zero_level] = 0

    # Map colors
    color_heightmap = np.zeros((heightmap_size[0],heightmap_size[1],3),dtype=np.uint8)
    for c in range(3):
        color_heightmap[heightmap_pix_y,heightmap_pix_x,c] = color_pts[:,c]
    
    # Map segmentations
    segmentation_heightmap = np.zeros((heightmap_size[0],heightmap_size[1]),dtype=np.int)
    segmentation_heightmap[heightmap_pix_y,heightmap_pix_x] = segmentation_pts[:, 0]

    return color_heightmap, depth_heightmap, segmentation_heightmap


def mkdir(path, clean=False):
    """
    Make directory
    :param path: path of the target directory
    :param clean: If there exist such directory, remove the original one or not
    
    :return image of new dtype 
    """
    if clean and os.path.exists(path):
        shutil.rmtree(path)
    if not os.path.exists(path):
        os.makedirs(path)


def imretype(im, dtype):
    """
    Image retype
    :param im: original image. dtype support: float, float16, float32, float64, uint8, uint16
    :param dtype: target dtype. dtype support: float, float16, float32, float64, uint8, uint16
    
    :return image of new dtype
    """
    im = np.array(im)

    if im.dtype in ['float', 'float16', 'float32', 'float64']:
        im = im.astype(np.float)
    elif im.dtype == 'uint8':
        im = im.astype(np.float) / 255.
    elif im.dtype == 'uint16':
        im = im.astype(np.float) / 65535.
    else:
        raise NotImplementedError('unsupported source dtype: {0}'.format(im.dtype))
    try:
        assert np.min(im) >= 0 and np.max(im) <= 1
    except:
        im = np.clip(im, 0, 1.0)

    if dtype in ['float', 'float16', 'float32', 'float64']:
        im = im.astype(dtype)
    elif dtype == 'uint8':
        im = (im * 255.).astype(dtype)
    elif dtype == 'uint16':
        im = (im * 65535.).astype(dtype)
    else:
        raise NotImplementedError('unsupported target dtype: {0}'.format(dtype))

    return im


def imwrite(path, obj):
    """
    Save Image
    :param path: path to save the image. Suffix support: png or jpg or gif
    :param image: array or list of array(list of image --> save as gif). Shape support: WxHx3 or WxHx1 or 3xWxH or 1xWxH
    """
    if not isinstance(obj, (collections.Sequence, collections.UserList)):
        obj = [obj]
    writer = imageio.get_writer(path)
    for im in obj:
        im = imretype(im, dtype='uint8').squeeze()
        if len(im.shape) == 3 and im.shape[0] == 3:
            im = np.transpose(im, (1, 2, 0))
        writer.append_data(im)
    writer.close()


def draw_grasp(img, grasp_pix, grasp_angle, vis_pts, success=None, color=(186,176,172)):
    """
    Draw grasp icon on image 

    :param img: color heightmap, [W, H, 3]
    :param grasp_pix: coordinate of the gripper center. (x, y)
    :param vis_pts: list or contact points respect to grasp_pix, [(x1, y1), (x2, y2), ...]
    """
    if success is not None:
        if success:
            color = (89, 169, 79)
        else:
            color = (255, 87, 89)
            
    img = img.copy()
    grasp_icon_pix = vis_pts
    grasp_icon_pix = np.transpose(np.dot(np.asarray([[np.cos(-grasp_angle),-np.sin(-grasp_angle)],
                                                     [np.sin(-grasp_angle), np.cos(-grasp_angle)]]),np.transpose(grasp_icon_pix)))
    grasp_icon_pix = (grasp_icon_pix+grasp_pix).astype(int)
    img = cv2.circle(img, (int(grasp_pix[1]), int(grasp_pix[0])), 1, color, 1)
    for pix in grasp_icon_pix:
        img = cv2.circle(img, (int(pix[1]), int(pix[0])), 1, color, 2)
        img = cv2.line(img, (int(grasp_pix[1]), int(grasp_pix[0])), (int(pix[1]), int(pix[0])), color, 2)
    return img


def multithreading_exec(num, q, fun, blocking=True):
    """
    Multi-threading Execution
    
    :param num: number of threadings
    :param q: queue of args
    :param fun: function to be executed
    :param blocking: blocking or not (default True)
    """
    class Worker(threading.Thread):
        def __init__(self, q, fun):
            super().__init__()
            self.q = q
            self.fun = fun
            self.start()

        def run(self):
            while True:
                try:
                    args = self.q.get(block=False)
                    self.fun(*args)
                    self.q.task_done()
                except queue.Empty:
                    break
    thread_list = [Worker(q, fun) for i in range(num)]
    if blocking:
        for t in thread_list:
            if t.is_alive():
                t.join()

                
def data_augmentation(data, device, batch=True):
    def pos_uv2xy(pos, img_shape):
        x = pos[1] - img_shape[1] / 2
        y = img_shape[0] / 2 - pos[0]
        return np.array([x, y])

    def pos_xy2uv(pos, img_shape):
        u = img_shape[0] / 2 - pos[1]
        v = pos[0] + img_shape[1] / 2
        return np.array([u, v])

    def pos_transform(pos, img_shape, theta, offset=[0, 0]):
        x, y = pos_uv2xy(pos, img_shape)
        rot_mat = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        pos_rot = rot_mat @ np.array([x, y])
        return pos_xy2uv(pos_rot, img_shape) + np.array(offset)

    ############ start ##############

    # this value should be consistent with sim.py
    workspace_bnd_pix = np.array([[16, 176], [16, 176]])

    if not batch:
        for k in data:
            data[k] = [data[k]]
    
    data_num = len(data['color_heightmap'])
    img_shape = data['color_heightmap'][0].shape[:2]

    if 'grasp_pixel' in data:
        offsets, rotate_thetas = list(), list()
        for i in range(data_num):
            pos_old = np.array(data['grasp_pixel'][i])
            final_theta = 0
            for _ in range(20):
                rotate_theta = np.random.rand() * 2 * np.pi
                pos_rot = np.array(pos_transform(pos_old, img_shape, rotate_theta))
                if pos_rot[0] >= workspace_bnd_pix[0, 0] and pos_rot[0] <= workspace_bnd_pix[0, 1] and \
                   pos_rot[1] >= workspace_bnd_pix[1, 0] and pos_rot[1] <= workspace_bnd_pix[1, 1]:
                   final_theta = rotate_theta
                   break
            rotate_theta = final_theta
            pos_rot = np.array(pos_transform(pos_old, img_shape, rotate_theta))
            

            offset = np.random.rand(2) * (workspace_bnd_pix[:, 1] - workspace_bnd_pix[:, 0]) - pos_rot + workspace_bnd_pix[:, 0]
            rotate_thetas.append(rotate_theta)
            offsets.append(offset)
            data['grasp_pixel'][i] = pos_rot + offset
            data['grasp_angle'][i] = (data['grasp_angle'][i] - rotate_theta) % (2 * np.pi)
        rotate_thetas = np.array(rotate_thetas)
        offsets = np.array(offsets).T / np.max(img_shape) * 2
        offsets = -np.asarray([offsets[1], offsets[0]])
    else:
        offsets = (np.random.rand(2, 1) * 2 - 1) * 0.2
        rotate_thetas = np.random.rand(1) * 2 * np.pi

    # color_heightmap
    tensor_old = torch.tensor(np.array(data['color_heightmap']).transpose([0, 3, 1, 2]).astype(np.float32), device=device)
    tensor_new = rotate_tensor2d(tensor_old, rotate_thetas, offsets, padding_mode='border', pre_padding=0)
    data['color_heightmap'] = tensor_new.cpu().numpy().transpose([0, 2, 3, 1]).astype(np.uint8)

    # target_heightmap
    tensor_old = torch.tensor(np.array(data['target_heightmap'])[:, np.newaxis].astype(np.float32), device=device)
    tensor_new = rotate_tensor2d(tensor_old, rotate_thetas, offsets, padding_mode='border', pre_padding=0)
    data['target_heightmap'] = np.round(tensor_new.cpu().numpy()[:, 0]).astype(np.uint8)

    # obstacle_vol
    tensor_old = torch.tensor(np.array(data['obstacle_vol'])[:, np.newaxis].astype(np.float32), device=device)
    tensor_new = rotate_tensor3d(tensor_old, rotate_thetas, offsets, padding_mode='border', pre_padding=0)
    data['obstacle_vol'] = np.round(tensor_new.cpu().numpy()[:, 0]).astype(np.uint8)

    # scene_tsdf
    tensor_old = torch.tensor(np.array(data['scene_tsdf'])[:, np.newaxis].astype(np.float32), device=device)
    tensor_new = rotate_tensor3d(tensor_old, rotate_thetas, offsets, padding_mode='border', pre_padding=1)
    data['scene_tsdf'] = tensor_new.cpu().numpy()[:, 0]


    if not batch:
        for k in data:
            data[k] = data[k][0]

    return data


def html_visualize(web_path, data, ids, cols, others=[], title='visualization', threading_num=10):
    """
    :param web_path: (str) directory to save webpage. It will clear the old data!
    :param data: (dict of data). 
        key: {id}_{col}. 
        value: figure or text
            - figure: ndarray --> .png or [ndarrays,] --> .gif
            - text: str or [str,]
    :param ids: (list of str) name of each row
    :param cols: (list of str) name of each column
    :param others: (list of dict) other figures
        'name': str, name of the data, visualize using h2()
        'data': string or ndarray(image)
        'height': int, height of the image (default 256)
    :param title: (str) title of the webpage
    :param threading_num: number of threadings for imwrite (default 10)
    """
    figure_path = os.path.join(web_path, 'figures')
    mkdir(web_path, clean=True)
    mkdir(figure_path, clean=True)
    q = queue.Queue()
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            q.put((os.path.join(figure_path, key + '.png'), value))
        elif not isinstance(value, list) and isinstance(value[0], np.ndarray):
            q.put((os.path.join(figure_path, key + '.gif'), value))
    multithreading_exec(threading_num, q, imwrite)

    with dominate.document(title=title) as web:
        dominate.tags.h1(title)
        with dominate.tags.table(border=1, style='table-layout: fixed;'):
            with dominate.tags.tr():
                with dominate.tags.td(style='word-wrap: break-word;', halign='center', align='center', width='64px'):
                    dominate.tags.p('id')
                for col in cols:
                    with dominate.tags.td(style='word-wrap: break-word;', halign='center', align='center', ):
                        dominate.tags.p(col)
            for id in ids:
                with dominate.tags.tr():
                    bgcolor = 'F1C073' if id.startswith('train') else 'C5F173'
                    with dominate.tags.td(style='word-wrap: break-word;', halign='center', align='center', bgcolor=bgcolor):
                        for part in id.split('_'):
                            dominate.tags.p(part)
                    for col in cols:
                        with dominate.tags.td(style='word-wrap: break-word;', halign='center', align='top'):
                            value = data[f'{id}_{col}']
                            if isinstance(value, str):
                                dominate.tags.p(value)
                            elif isinstance(value, list) and isinstance(value[0], str):
                                for v in value:
                                    dominate.tags.p(v)
                            else:
                                dominate.tags.img(style='height:128px', src=os.path.join('figures', '{}_{}.png'.format(id, col)))
        for idx, other in enumerate(others):
            dominate.tags.h2(other['name'])
            if isinstance(other['data'], str):
                dominate.tags.p(other['data'])
            else:
                imwrite(os.path.join(figure_path, '_{}_{}.png'.format(idx, other['name'])), other['data'])
                dominate.tags.img(style='height:{}px'.format(other.get('height', 256)),
                    src=os.path.join('figures', '_{}_{}.png'.format(idx, other['name'])))
    with open(os.path.join(web_path, 'vis.html'), 'w') as fp:
        fp.write(web.render())


def visualization(vis_data, num_env, seq_len, num_open_scales, num_rotations, num_vis, vis_path):
    num_vis = min(num_vis, num_env)
    data = {}
    ids = [f'{x}_{y}_{z}' for x in range(num_vis) for y in range(seq_len) for z in range(num_open_scales)]
    cols = ['target_heightmap', 'grasping_vis', 'action']
    for i in range(num_rotations):
        cols.append(f'affordance_map-{i}')

    heightmap_pix_size = 0.002
    palette = get_tableau_palette()

    for (rank, step), sample_data in vis_data.items():
        if rank >= num_vis:
            continue
        color_heightmap = sample_data['color_heightmap']

        target_heightmap = np.zeros_like(color_heightmap)
        target_heightmap[sample_data['target_heightmap'] == 1] = palette[1] # target use red
        target_heightmap[sample_data['target_heightmap'] == 2] = palette[2] # obstacle use green

        y, x = sample_data['grasp_pixel']
        angle = sample_data['grasp_angle']
        angle_idx = int(angle / (2 * np.pi) * num_rotations)
        open_scale_idx = sample_data['open_scale_idx']

        grasping_vis = draw_grasp(
            img=color_heightmap,
            grasp_pix=sample_data['grasp_pixel'],
            grasp_angle=sample_data['grasp_angle'],
            vis_pts=np.flip(sample_data['vis_pts'][open_scale_idx], 1) / heightmap_pix_size,
            success=sample_data['reward'] == 1
        )
        
        for j in range(num_open_scales):
            target_heightmap_umat = cv2.UMat(target_heightmap)
            gripper_type = sample_data['gripper_type'] if isinstance(sample_data['gripper_type'], str) else sample_data['gripper_type'][j]
            cv2.putText(target_heightmap_umat, gripper_type, (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, [int(x) for x in palette[9]], 2)
            # cv2.putText(target_heightmap_umat, '%.3f' % sample_data['open_scales'][j], (5, target_heightmap.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, [int(x) for x in palette[9]], 3)
            data[f'{rank}_{step}_{j}_target_heightmap'] = target_heightmap_umat.get()
            affordance_map = sample_data['affordance_map'][j]
            best_grasp_ind = np.array(np.unravel_index(np.argmax(affordance_map), affordance_map.shape))
            grasp_angle = best_grasp_ind[0] * (2*np.pi / num_rotations)
            grasp_pixel = best_grasp_ind[1:] # (y, x)
            score = affordance_map[best_grasp_ind[0], grasp_pixel[0], grasp_pixel[1]]

            grasping_vis = draw_grasp(
                img=color_heightmap,
                grasp_pix=grasp_pixel,
                grasp_angle=grasp_angle,
                vis_pts=np.flip(sample_data['vis_pts'][j], 1) / heightmap_pix_size,
                color=(120, 120, 120)
            )
            if j == open_scale_idx:
                grasping_vis = draw_grasp(
                    img=grasping_vis,
                    grasp_pix=sample_data['grasp_pixel'],
                    grasp_angle=sample_data['grasp_angle'],
                    vis_pts=np.flip(sample_data['vis_pts'][open_scale_idx], 1) / heightmap_pix_size,
                    success=sample_data['reward'] == 1
                )
            data[f'{rank}_{step}_{j}_grasping_vis'] = grasping_vis

            if j == open_scale_idx:
                data[f'{rank}_{step}_{j}_action'] = [f'{y},{x}/{angle_idx}', 'score=%.3f' % sample_data['score'], 'best=%.3f' % score]
            else:
                data[f'{rank}_{step}_{j}_action'] = 'best=%.3f' % score

        # visualize affordance maps
        cmap = plt.get_cmap('jet')
        tsdf_topdown = np.sum((sample_data['scene_tsdf'] < 0).astype(np.float), axis=2)
        norm_tsdf_topdown = tsdf_topdown / max(np.max(tsdf_topdown), 1e-5)
        for j, affordance_maps in enumerate(sample_data['affordance_map']):
            for i, affordance_map in enumerate(affordance_maps):
                vis_affordance_map = cmap(affordance_map)[:, :, :3] * 0.7 + norm_tsdf_topdown[:, :, np.newaxis] * 0.3
                vis_affordance_map = imretype(vis_affordance_map, 'uint8')
                grasp_pixel = np.array(np.unravel_index(np.argmax(affordance_map), affordance_map.shape))
                grasp_angle = i * (2*np.pi / num_rotations)
                score = affordance_map[grasp_pixel[0], grasp_pixel[1]]
                if score > 0.5:
                    vis_affordance_map = draw_grasp(
                        img=vis_affordance_map,
                        grasp_pix=grasp_pixel,
                        grasp_angle=grasp_angle,
                        vis_pts=np.flip(sample_data['vis_pts'][j], 1) / heightmap_pix_size
                    )

                # put the score on the image (top-left corner)
                vis_affordance_map_umat = cv2.UMat(vis_affordance_map)
                color = int(score * 255)
                cv2.putText(vis_affordance_map_umat, '%.3f' % score, (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (color, color, color),3)
                cv2.putText(vis_affordance_map_umat, f'{i}', (5, vis_affordance_map.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 3)
                vis_affordance_map = vis_affordance_map_umat.get()

                data[f'{rank}_{step}_{j}_affordance_map-{i}'] = vis_affordance_map

    html_visualize(vis_path, data, ids, cols)
