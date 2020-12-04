import math

import numpy as np
import pybullet
import pybullet_data
import pybullet_utils.bullet_client as bc

import utils
from fusion import TSDFVolume
from gripper import Gripper
from misc.basic_object_loader import load_dexnet


class PybulletSim:
    def __init__(self, gui_enabled, gripper_selection=False, num_cam=1):
        # Defines where robot end effector can move to in world coordinates
        self._workspace_bounds = np.array([[-0.128, 0.128], # 3x2 rows: x,y,z cols: min,max
                                           [-0.128, 0.128],
                                           [ 0.000, 0.128]])

        # Defines where robot can see in world coordinates (3D bounds of heightmaps)
        self._view_bounds = np.array([[-0.192, 0.192], # 3x2 rows: x,y,z cols: min,max
                                      [-0.192, 0.192],
                                      [ 0.000, 0.128]])
        
        self._valid_bounds = np.array([[-0.16, 0.16], # 3x2 rows: x,y,z cols: min,max
                                       [-0.16, 0.16],
                                       [ 0.00, 0.128]])

        self._heightmap_pix_size = 0.002

        self._volume_bounds = self._view_bounds
        self._voxel_size = 0.002

        if gui_enabled:
            self.bullet_client = bc.BulletClient(connection_mode=pybullet.GUI)
        else:
            self.bullet_client = bc.BulletClient(connection_mode=pybullet.DIRECT)
        
        self.bullet_client.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.bullet_client.setGravity(0, 0, -9.8)

        self._plane_id = self.bullet_client.loadURDF("plane.urdf")
        self.bullet_client.changeDynamics(self._plane_id, -1, lateralFriction=1.0)

        # Add RGB-D camera (mimic RealSense D415) overlooking scene
        self._scene_cam_lookat = self._view_bounds.mean(axis=1)
        self._scene_cam_position = [self._scene_cam_lookat[0], self._scene_cam_lookat[1], 0.5]
        self._scene_cam_up_direction = [0, 1, 0]
        self._scene_cam_image_size = (480, 640)
        self._scene_cam_z_near = 0.01
        self._scene_cam_z_far = 10.0
        self._scene_cam_fov_w = 69.40
        self._scene_cam_focal_length = (float(self._scene_cam_image_size[1])/2)/np.tan((np.pi*self._scene_cam_fov_w/180)/2)
        self._scene_cam_fov_h = (math.atan((float(self._scene_cam_image_size[0])/2)/self._scene_cam_focal_length)*2/np.pi)*180
        self._scene_cam_projection_matrix = self.bullet_client.computeProjectionMatrixFOV(
            fov=self._scene_cam_fov_h,
            aspect=float(self._scene_cam_image_size[1])/float(self._scene_cam_image_size[0]),
            nearVal=self._scene_cam_z_near, farVal=self._scene_cam_z_far
        )  # notes: 1) FOV is vertical FOV 2) aspect must be float
        self._scene_cam_intrinsics = np.array([[self._scene_cam_focal_length, 0, float(self._scene_cam_image_size[1])/2],
                                             [0, self._scene_cam_focal_length, float(self._scene_cam_image_size[0])/2],
                                             [0, 0, 1]])

        self._palette = utils.get_tableau_palette()

        # gripper and list of object ids
        self._object_ids = list()
        self._gripper = None
        self._gripper_home_position = [-0.2, 0, 0.3]

        # gripper selection flag. If true, gripper will be load in step, instead of reset
        self.gripper_selection = gripper_selection

        self._gripper_record_dict = dict()
        self._num_cam = num_cam


    # Get latest RGB-D image from scene camera
    def get_scene_cam_data(self, cam_position=None, cam_lookat=None, cam_up_direction=None):
        if cam_position is None:
            cam_position = self._scene_cam_position
        if cam_lookat is None:
            cam_lookat = self._scene_cam_lookat
        if cam_up_direction is None:
            cam_up_direction = self._scene_cam_up_direction
        cam_view_matrix = self.bullet_client.computeViewMatrix(cam_position, cam_lookat, cam_up_direction)
        cam_pose_matrix = np.linalg.inv(np.array(cam_view_matrix).reshape(4, 4).T)
        cam_pose_matrix[:, 1:3] = -cam_pose_matrix[:, 1:3]

        camera_data = self.bullet_client.getCameraImage(
            self._scene_cam_image_size[1],
            self._scene_cam_image_size[0],
            cam_view_matrix,
            self._scene_cam_projection_matrix,
            shadow=1,
            flags=self.bullet_client.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
            renderer=self.bullet_client.ER_BULLET_HARDWARE_OPENGL
        )
        rgb_pixels = np.array(camera_data[2]).reshape((self._scene_cam_image_size[0], self._scene_cam_image_size[1], 4))
        color_image = rgb_pixels[:,:,:3] # remove alpha channel
        z_buffer = np.array(camera_data[3]).reshape((self._scene_cam_image_size[0], self._scene_cam_image_size[1]))
        segmentation_mask = np.array(camera_data[4], np.int) # - not implemented yet with renderer=p.ER_BULLET_HARDWARE_OPENGL
        depth_image = (2.0*self._scene_cam_z_near*self._scene_cam_z_far)/(self._scene_cam_z_far+self._scene_cam_z_near-(2.0*z_buffer-1.0)*(self._scene_cam_z_far-self._scene_cam_z_near))
        return color_image, depth_image, segmentation_mask, cam_pose_matrix


    # Get RGB-D heightmap from RGB-D image from bin camera
    def get_scene_heightmap(self, color_image, depth_image, segmentation_mask, cam_pose):
        camera_points,color_points,segmentation_points = utils.get_pointcloud(color_image, depth_image, segmentation_mask, self._scene_cam_intrinsics)
        camera_points = utils.transform_pointcloud(camera_points, cam_pose)
        color_heightmap,depth_heightmap,segmentation_heightmap = utils.get_heightmap(camera_points,color_points,segmentation_points,self._view_bounds,self._heightmap_pix_size,zero_level=self._view_bounds[2,0])
        self._depth_heightmap = depth_heightmap
        return color_heightmap, depth_heightmap, segmentation_heightmap


    def reset_objects(self, category_name='train', target_num=1, obstacle_num=0, instance_id=None, **kwargs):
        # reset obejects randomly, all of them are targets.
        self._target_ids, self._obstacle_ids, self._object_ids = [], [], []

        wksp_center = self._workspace_bounds.mean(1)
        r = 0.1
        position_bound = np.array([
            [wksp_center[0]-r, wksp_center[0]+r],
            [wksp_center[1]-r, wksp_center[1]+r],
            [wksp_center[2]-r, wksp_center[2]+r]
        ])

        for obj_id in range(target_num+obstacle_num):

            for _ in range(5):
                # random position and rotation angle
                x = np.random.rand() * np.diff(position_bound[0])[0] + position_bound[0, 0]
                y = np.random.rand() * np.diff(position_bound[1])[0] + position_bound[1, 0]

                body_id = load_dexnet(
                    category_name=category_name,
                    bullet_client=self.bullet_client,
                    scaling=1,
                    position=[x, y, 0.08],
                    orientation=self.bullet_client.getQuaternionFromEuler([np.pi/2, 0, np.random.rand() * 2 * np.pi]), 
                    fixed_base=False,
                    instance_id=instance_id
                )
                for _ in range(240 * 1):
                    self.bullet_client.stepSimulation()
                color_image, depth_image, segmentation_mask, cam_pose_matrix = self.get_scene_cam_data()
                color_heightmap, depth_heightmap, segmentation_heightmap = self.get_scene_heightmap(color_image, depth_image, segmentation_mask, cam_pose_matrix)
                object_depth = depth_heightmap * (segmentation_heightmap==body_id)
                if object_depth.max() > self._valid_bounds[2, 1] - 0.005: # object is too high, reset again
                    self.bullet_client.removeBody(body_id)
                    body_id = None
                else:
                    break

            if body_id is not None:
                self._object_ids.append(body_id)

        np.random.shuffle(self._object_ids)
        self._target_ids = self._object_ids[:target_num]
        self._obstacle_ids = self._object_ids[target_num:]


    def _get_scene_observation(self):
        # take image of the scene
        if self._gripper is not None:
            self._gripper.move(self._gripper_home_position, 0)

        self._scene_tsdf = TSDFVolume(self._volume_bounds, voxel_size=self._voxel_size)

        if self._num_cam > 1:
            scene_center = self._view_bounds.mean(axis=1)
            for cam_id in range(self._num_cam):
                theta = cam_id / self._num_cam * 2 * np.pi
                cam_position = [scene_center[0] + 0.3 * math.cos(theta), scene_center[1] + 0.3 * math.sin(theta), 0.3]
                cam_lookat = [scene_center[0], scene_center[1], 0]
                cam_up_direction = [0, 0, 1]
                color_image, depth_image, segmentation_mask, cam_pose_matrix = self.get_scene_cam_data(cam_position, cam_lookat, cam_up_direction)
                obstacle_mask = np.zeros_like(color_image)
                for obj_id in self._obstacle_ids:
                    obstacle_mask[segmentation_mask == obj_id] = [1, 0, 0]
                self._scene_tsdf.integrate(obstacle_mask, depth_image, self._scene_cam_intrinsics, cam_pose_matrix, obs_weight=1.)
        elif self._num_cam == 1:
            color_image, depth_image, segmentation_mask, cam_pose_matrix = self.get_scene_cam_data()
            obstacle_mask = np.zeros_like(color_image)
            for obj_id in self._obstacle_ids:
                obstacle_mask[segmentation_mask == obj_id] = [1, 0, 0]
            self._scene_tsdf.integrate(obstacle_mask, depth_image, self._scene_cam_intrinsics, cam_pose_matrix, obs_weight=1.)
            
        
        # get scene_tsdf(WxHxD) and obstacle_vol(WxHxDx3)
        scene_tsdf, obstacle_vol = self._scene_tsdf.get_volume()

        # make the empty space 0 in obstacle_vol
        obstacle_vol *= (scene_tsdf < 0).astype(np.int)
        
        scene_tsdf = np.transpose(scene_tsdf, [1, 0, 2]) # swap x-axis and y-axis to make it consitent with heightmap
        obstacle_vol = np.transpose(obstacle_vol, [1, 0, 2])
        
        # get target_heightmap. 0: background, 1: target, 2: obstacle
        color_heightmap, depth_heightmap, segmentation_heightmap = self.get_scene_heightmap(color_image, depth_image, segmentation_mask, cam_pose_matrix)
        target_heightmap = np.zeros_like(segmentation_heightmap, dtype=np.uint8)
        for obj_id in self._target_ids:
            target_heightmap[segmentation_heightmap == obj_id] = 1
        for obj_id in self._obstacle_ids:
            target_heightmap[segmentation_heightmap == obj_id] = 2

        scene_observation = {
            'color_heightmap': color_heightmap,
            'target_heightmap': target_heightmap,
            'scene_tsdf': scene_tsdf,
            'obstacle_vol': obstacle_vol,
            'valid_pix_heightmap': self.get_valid_pix_heightmap(depth_heightmap),
            'n_target_left': len(self._target_ids)
        }
        return scene_observation
    

    def load_gripper(self, gripper_type, gripper_size, open_scales=None, gripper_final_state=False, remove=False, **kwargs):
        # reset gripper & get gripper_observation
        # balance 2f and 3f grippers when they are added together
        if gripper_type == "train":
            # load training grippers; half 2f half 3f
            if np.random.rand() > 0.5:
                gripper_type = ["wsg_32", "sawyer", "franka", "robotiq_2f_140", "ezgripper"]
            else:
                gripper_type = ["robotiq_3f", "kinova_3f"]

        elif gripper_type == "test":
            # load testing grippers; half 2f half 3f
            if np.random.rand() > 0.5:
                gripper_type = ["wsg_50", "rg2", "robotiq_2f_85"]
            else:
                gripper_type = ["barrett_hand"]
        self._gripper_type = gripper_type if isinstance(gripper_type, str) else np.random.choice(gripper_type)

        gripper_kwargs = dict()
        if remove:
            if self._gripper_type == 'barrett_hand':
                self._gripper_record_dict[(self._gripper_type, kwargs['palm_joint'])] = gripper_kwargs
            else:
                self._gripper_record_dict[self._gripper_type] = gripper_kwargs

        self._open_scales = [] if open_scales is None else open_scales
        self._gripper_size = gripper_size
        self._gripper_orientation = [0, 0, 0]
        gripper_kwargs['gripper_size'] = self._gripper_size

        if self._gripper_type == 'barrett_hand':
            gripper_kwargs['palm_joint'] = 0.25 * np.random.rand() if kwargs['palm_joint'] is None else kwargs['palm_joint']

        self._gripper = Gripper(
            gripper_type=self._gripper_type,
            bullet_client=self.bullet_client,
            home_position=self._gripper_home_position,
            num_side_images=8,
            **gripper_kwargs
        )

        gripper_tsdf = np.array([self._gripper.get_tsdf(open_scale) for open_scale in self._open_scales])
        gripper_close_tsdf = self._gripper.get_tsdf(0) if gripper_final_state else None
        vis_pts = [self._gripper.get_vis_pts(open_scale) for open_scale in self._open_scales]

        gripper_observation = {
            'gripper_tsdf': gripper_tsdf,
            'gripper_close_tsdf': gripper_close_tsdf,
            'vis_pts': vis_pts,
            'open_scales': self._open_scales,
            'gripper_type': self._gripper_type,
            'gripper_size': self._gripper_size,
        }
        if remove:
            self._gripper.remove()
            self._gripper = None
        else:
            self._gripper.move(self._gripper_home_position, 0)
        return gripper_observation


    def reset(self, gripper_type, gripper_size, open_scales=None, gripper_final_state=False, **kwargs):
        """Reset scene: 1. generate scene randomly. 2. generate a gripper randomly.

        Args:
            gripper_type: str / [str]. Name(s) of gripper.
            gripper_size: float. Size of gripper. The number shoule be in (0, 1]. None means random choose from [0.4, 1]
            open_scales: [float]: List of open scales. The number should be in (0, 1]
            kwargs: arguments of objects

        Returns:
            observation (e.g. depth heightmap) and the gripper info.
        """
        # remove outdated obejcts and gripper
        for obj_id in self._object_ids:
            self.bullet_client.removeBody(obj_id)
        self._object_ids = []
        if self._gripper is not None:
            self._gripper.remove()
            self._gripper = None
        assert self.bullet_client.getNumBodies()==1
        
        # load fence
        fence_center = np.mean(self._workspace_bounds, axis=1)
        fence_center[-1] = 0
        fence_id = self.bullet_client.loadURDF(
            'assets/fence/fence.urdf',
            basePosition=fence_center,
            useFixedBase=True
        )

        # reset objects & get scene_observation
        self.reset_objects(**kwargs)
        
        # change physics (mass, friction)
        for obj_id in self._object_ids:
            self.bullet_client.changeDynamics(obj_id, -1, lateralFriction=0.6, spinningFriction=0.003, mass=0.1)

        # remove fence
        for _ in range(240):
            self.bullet_client.stepSimulation()
        self.bullet_client.removeBody(fence_id)
        
        try:
            self.wait_till_stable(10)
        except:
            for _ in range(240*2):
                self.bullet_client.stepSimulation()

        if kwargs['num_cam']:
            self._num_cam = kwargs['num_cam']
        scene_observation = self._get_scene_observation()

        if self.gripper_selection:
            # don't load gripper
            self._gripper_observation = dict()
        else:
            self._gripper_observation=self.load_gripper(gripper_type, gripper_size, open_scales, gripper_final_state, **kwargs)

        # return observation (gripper + scene)
        observation = {
            **self._gripper_observation,
            **scene_observation
        }
        return observation


    def wait_till_stable(self, wait_iter=20):
        """Wait till objects are settled.
        """
        last_states = [self.bullet_client.getBasePositionAndOrientation(id) for id in self._object_ids]
        for i in range(wait_iter):
            for _ in range(60):
                self.bullet_client.stepSimulation()
            current_states = [self.bullet_client.getBasePositionAndOrientation(id) for id in self._object_ids]
            stable_flag = True
            for last_state, current_state in zip(last_states, current_states):
                max_diff_position = np.max(np.abs(np.array(last_state[0]) - np.array(current_state[0])))
                max_diff_orientation = np.max(np.abs(np.array(last_state[1]) - np.array(current_state[1])))
                if max_diff_position > 5e-5 and max_diff_orientation > 5e-5:
                    stable_flag = False
                    break
            if stable_flag:
                break
            last_states = current_states


    def step(self, action, gripper_type=None, is_position=False, **kwargs):
        """Execute action and return reward and next observation.

        Args:
            action: tuple or list (y_pix, x_pix, angle, open_scale).
                y_pix is the first channel of the heightmap.
                x_pix is the second channel of the heightmap.
                angle is the rotation angle (radians) around z axis.

        Returns:
            reward: float number
            observation: dict, scene observation + gripper observation
        """
        if self.gripper_selection:
            if len(kwargs) == 0:
                if gripper_type.startswith('barrett_hand-'):
                    palm_joint = float(gripper_type.split('-')[-1])
                    gripper_type = 'barrett_hand'
                    gripper_record = self._gripper_record_dict[(gripper_type, palm_joint)]
                else:
                    gripper_record = self._gripper_record_dict[gripper_type]
            else:
                gripper_record = kwargs
            
            self._gripper = Gripper(
                gripper_type=gripper_type,
                bullet_client=self.bullet_client,
                home_position=self._gripper_home_position,
                num_side_images=8,
                **gripper_record
            )
            self._gripper.move(self._gripper_home_position, 0)

        a_y, a_x, angle, open_scale = action
        if is_position:
            y, x = a_y, a_x
            y_pix = int((y-self._view_bounds[1,0]) / self._heightmap_pix_size)
            x_pix = int((x-self._view_bounds[1,0]) / self._heightmap_pix_size)
        else:
            y_pix, x_pix = a_y, a_x
            x = x_pix * self._heightmap_pix_size + self._view_bounds[0,0]
            y = y_pix * self._heightmap_pix_size + self._view_bounds[1,0]

        grasp_pix = np.array([y_pix, x_pix])
        valid_pix = np.array(np.where(self._depth_heightmap > 0)).T # y,x
        dist_to_valid = np.sqrt(np.sum((valid_pix-grasp_pix)**2,axis=1))
        closest_valid_pix = grasp_pix
        if np.min(dist_to_valid) < 2: # get nearest non-zero pixel less than 2 pixels away
            closest_valid_pix = valid_pix[np.argmin(dist_to_valid),:]

        z = self._depth_heightmap[closest_valid_pix[0],closest_valid_pix[1]] - 0.05
        z = max(z, 0)
        grasp_position = np.array([x, y, z])
        grasp_position = np.clip(grasp_position,self._workspace_bounds[:,0],self._workspace_bounds[:,1]) # clamp grasp position w.r.t. workspace bounds
        self._gripper.primitive_grasping(grasp_position, angle % (2 * np.pi), open_scale=open_scale, stop_at_contact=True)
        try:
            self.wait_till_stable()
        except:
            for _ in range(240*2):
                self.bullet_client.stepSimulation()

        objects_lifted = 0
        target_ids_copy = list(self._target_ids)
        for target_id in target_ids_copy:
            pos = self.bullet_client.getBasePositionAndOrientation(target_id)[0]
            if pos[-1] > 0.2:
                self.bullet_client.resetBasePositionAndOrientation(target_id,[1, target_id, 0.3],[0,0,0,1])
                self.bullet_client.stepSimulation()
                objects_lifted += 1
                self._target_ids.remove(target_id)
            elif np.any(pos<self._workspace_bounds[:, 0]) or np.any(pos>self._workspace_bounds[:, 1]):
                self.bullet_client.resetBasePositionAndOrientation(target_id,[1, target_id, 0.3],[0,0,0,1])
                self.bullet_client.stepSimulation()
                self._target_ids.remove(target_id)

        # should not lift obstacles
        for target_id in self._obstacle_ids:
            pos = self.bullet_client.getBasePositionAndOrientation(target_id)[0]
            if pos[-1] > 0.2:
                self.bullet_client.resetBasePositionAndOrientation(target_id,[1, target_id, 0.3],[0,0,0,1])
                self.bullet_client.stepSimulation()
                objects_lifted = -1

        reward = objects_lifted==1

        scene_observation = self._get_scene_observation()
        observation = {**self._gripper_observation, **scene_observation}

        if self.gripper_selection:
            self._gripper.remove()
            self._gripper = None

        return reward, observation


    def get_valid_pix_heightmap(self, depth_heightmap):
        """Returns logical map with same size as heightmap where 1 - inside workspace bounds and 0 - outside workspace bounds

        Args:
            depth_heightmap: float [W, H]

        Returns:
            valid_pix_heightmap: binary [W, H]
        """
        heightmap_width = depth_heightmap.shape[1]
        heightmap_height = depth_heightmap.shape[0]

        # Get 3D locations of each heightmap pixel in world coordinates
        heightmap_pix_x,heightmap_pix_y = np.meshgrid(np.linspace(0,heightmap_width-1,heightmap_width),
                                                          np.linspace(0,heightmap_height-1,heightmap_height))
        heightmap_points = np.array([heightmap_pix_x*self._heightmap_pix_size+self._view_bounds[0,0],
                                     heightmap_pix_y*self._heightmap_pix_size+self._view_bounds[1,0],
                                     depth_heightmap+self._view_bounds[2,0]]).transpose(1,2,0)

        valid_pix_heightmap = np.logical_and(heightmap_points[:,:,0] >= self._valid_bounds[0,0],
                                  np.logical_and(heightmap_points[:,:,0] <= self._valid_bounds[0,1],
                                  np.logical_and(heightmap_points[:,:,1] >= self._valid_bounds[1,0],
                                  np.logical_and(heightmap_points[:,:,1] <= self._valid_bounds[1,1],
                                  np.logical_and(heightmap_points[:,:,2] >= self._valid_bounds[2,0],
                                                 heightmap_points[:,:,2] <= self._valid_bounds[2,1])))))
        return valid_pix_heightmap
