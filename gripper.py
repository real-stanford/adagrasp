import math
import os
import time

import numpy as np
import pybullet as p
import pybullet_utils.bullet_client as bc

from gripper_module import load_gripper
from misc.urdf_editor import UrdfEditor
import utils
from fusion import TSDFVolume


class Gripper(object):

    """
    A moving mount and a gripper.
    the mount has 4 joints:
        0: prismatic x;
        1: prismatic y;
        2: prismatic z;
        3: revolute z;
    the gripper is defined by the `gripper_type`.
    """

    def __init__(self, gripper_type, bullet_client, home_position, num_side_images, voxel_size=0.004, trunc_margin_scale=5, **kwargs):
        self._bullet_client = bullet_client
        self._gripper_type = gripper_type
        self._gripper_size = kwargs['gripper_size']
        self._home_position = home_position
        self._default_orientation = [0,0,0]
        self._num_side_images = num_side_images

        # load gripper
        self._gripper = load_gripper(gripper_type)(self._bullet_client, **kwargs)
        gripper_body_id = self._gripper.load(self._home_position)

        # load mount
        mount_urdf = 'assets/gripper/mount.urdf'
        mount_body_id = self._bullet_client.loadURDF(
            mount_urdf,
            basePosition=self._home_position,
            useFixedBase=True
        )

        # combine mount and gripper by a joint
        ed_mount = UrdfEditor()
        ed_mount.initializeFromBulletBody(mount_body_id, self._bullet_client._client)
        ed_gripper = UrdfEditor()
        ed_gripper.initializeFromBulletBody(gripper_body_id, self._bullet_client._client)

        self._gripper_parent_index = 4
        newjoint = ed_mount.joinUrdf(
            childEditor=ed_gripper,
            parentLinkIndex=self._gripper_parent_index,
            jointPivotXYZInParent=self._gripper.get_pos_offset(),
            jointPivotRPYInParent=self._bullet_client.getEulerFromQuaternion(self._gripper.get_orn_offset()),
            jointPivotXYZInChild=[0, 0, 0],
            jointPivotRPYInChild=[0, 0, 0],
            parentPhysicsClientId=self._bullet_client._client,
            childPhysicsClientId=self._bullet_client._client
        )
        newjoint.joint_type = self._bullet_client.JOINT_FIXED
        newjoint.joint_name = "joint_mount_gripper"
        urdfname = f".tmp_combined_{self._gripper_type}_{self._gripper_size:.4f}_{np.random.random():.10f}_{time.time():.10f}.urdf"
        ed_mount.saveUrdf(urdfname)
        # remove mount and gripper bodies
        self._bullet_client.removeBody(mount_body_id)
        self._bullet_client.removeBody(gripper_body_id)

        self._body_id = self._bullet_client.loadURDF(
            urdfname,
            useFixedBase=True,
            basePosition=self._home_position,
            baseOrientation=self._bullet_client.getQuaternionFromEuler([0, 0, 0])
        )
        
        # remove the combined URDF
        os.remove(urdfname)
        
        # configure the gripper (e.g. friction)
        self._gripper.configure(self._body_id, self._gripper_parent_index+1)

        # define force and speed (movement of mount)
        self._force = 10000
        self._speed = 0.005

        self._tsdf_size = [64, 64, 32]
        self._voxel_size = voxel_size
        self._trunc_margin_scale = trunc_margin_scale
        bond = np.array(self._tsdf_size) * self._voxel_size
        self._vol_bnds = np.array([[-bond[0]/2, bond[0]/2],
                                   [-bond[1]/2, bond[1]/2],
                                   [0, bond[2]]])
        self._vol_bnds += np.array(self._home_position).reshape(3, -1)

        # Add RGB-D camera (mimic RealSense D415) for gripper
        self._gripper_cam_lookat = self._vol_bnds.mean(1)
        self._gripper_cam_image_size = (512, 512)
        self._gripper_cam_z_near = 0.01
        self._gripper_cam_z_far = 10.0
        self._gripper_cam_fov_w = 69.40
        self._gripper_cam_focal_length = (float(self._gripper_cam_image_size[1])/2)/np.tan((np.pi*self._gripper_cam_fov_w/180)/2)
        self._gripper_cam_fov_h = (math.atan((float(self._gripper_cam_image_size[0])/2)/self._gripper_cam_focal_length)*2/np.pi)*180
        self._gripper_cam_projection_matrix = self._bullet_client.computeProjectionMatrixFOV(
            fov=self._gripper_cam_fov_h,
            aspect=float(self._gripper_cam_image_size[1])/float(self._gripper_cam_image_size[0]),
            nearVal=self._gripper_cam_z_near,
            farVal=self._gripper_cam_z_far
        )  # notes: 1) FOV is vertical FOV 2) aspect must be float
        self._gripper_cam_intrinsics = np.array([[self._gripper_cam_focal_length, 0, float(self._gripper_cam_image_size[1])/2],
                                                [0, self._gripper_cam_focal_length, float(self._gripper_cam_image_size[0])/2],
                                                [0, 0, 1]])

        self.fix_joints(range(self._bullet_client.getNumJoints(self._body_id)))

    def get_gripper_cam_data(self, cam_position, cam_lookat, cam_up_direction):
        cam_view_matrix = self._bullet_client.computeViewMatrix(cam_position, cam_lookat, cam_up_direction)
        cam_pose_matrix = np.linalg.inv(np.array(cam_view_matrix).reshape(4, 4).T)
        # TODO: fix flipped up and forward vectors (quick hack)
        cam_pose_matrix[:, 1:3] = -cam_pose_matrix[:, 1:3]

        camera_data = self._bullet_client.getCameraImage(self._gripper_cam_image_size[1],self._gripper_cam_image_size[0],
                                       cam_view_matrix,self._gripper_cam_projection_matrix,
                                       shadow=1,flags=self._bullet_client.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
                                       renderer=self._bullet_client.ER_BULLET_HARDWARE_OPENGL)
        rgb_pixels = np.array(camera_data[2]).reshape((self._gripper_cam_image_size[0], self._gripper_cam_image_size[1], 4))
        color_image = rgb_pixels[:,:,:3] # remove alpha channel
        z_buffer = np.array(camera_data[3]).reshape((self._gripper_cam_image_size[0], self._gripper_cam_image_size[1]))
        segmentation_mask = None # camera_data[4] - not implemented yet with renderer=p.ER_BULLET_HARDWARE_OPENGL
        depth_image = (2.0*self._gripper_cam_z_near*self._gripper_cam_z_far)/(self._gripper_cam_z_far+self._gripper_cam_z_near-(2.0*z_buffer-1.0)*(self._gripper_cam_z_far-self._gripper_cam_z_near))
        return color_image, depth_image, segmentation_mask, cam_pose_matrix

    
    def get_tsdf(self, open_scale):
        self.move(self._home_position, 0)
        self.close()
        self.open(open_scale=open_scale)

        self._gripper_tsdf = TSDFVolume(self._vol_bnds, voxel_size=self._voxel_size)

        # take side images
        cam_up_direction = [0, 0, 1]
        side_look_directions = np.linspace(0, 2*np.pi, num=self._num_side_images, endpoint=False)
        cam_distance = 1
        for direction in side_look_directions:
            cam_position = [
                self._home_position[0] + cam_distance * np.cos(direction),
                self._home_position[1] + cam_distance * np.sin(direction),
                self._home_position[2]
            ]
            color_image, depth_image, _, cam_pose_matrix = self.get_gripper_cam_data(cam_position, self._gripper_cam_lookat, cam_up_direction)
            self._gripper_tsdf.integrate(color_image, depth_image, self._gripper_cam_intrinsics, cam_pose_matrix, obs_weight=1.)

        # take image from top
        color_image, depth_image, _, cam_pose_matrix = self.get_gripper_cam_data([0, 0, 2], self._gripper_cam_lookat, [1, 0, 0])
        self._gripper_tsdf.integrate(color_image, depth_image, self._gripper_cam_intrinsics, cam_pose_matrix, obs_weight=2.)

        # take image from bottom
        color_image, depth_image, _, cam_pose_matrix = self.get_gripper_cam_data([0, 0, 0], self._gripper_cam_lookat, [1, 0, 0])
        self._gripper_tsdf.integrate(color_image, depth_image, self._gripper_cam_intrinsics, cam_pose_matrix, obs_weight=2.)
        tsdf_vol_cpu, _ = self._gripper_tsdf.get_volume()
        tsdf_vol_cpu = np.transpose(tsdf_vol_cpu, [1, 0, 2]) # swap x-axis and y-axis to make it consitent with scene_tsdf

        return tsdf_vol_cpu


    def open(self, open_scale):
        self._gripper.open(self._body_id, self._gripper_parent_index+1, open_scale=open_scale)


    def close(self):
        self._gripper.close(self._body_id, self._gripper_parent_index+1)

    
    def move(self, target_position, rotation_angle, stop_at_contact=False):
        """
        :param target_position: (x, y, z). the position of the bottom center, not the base!
        :param rotation_angle: rotation in z axis \in [0, 2 * \pi]. For 2-finger gripper, angle=0 --> parallel to x-axis
        """

        target_position = np.array(target_position) - np.array(self._home_position)
        joint_ids = [0, 1, 2, 3]
        target_states = [target_position[0], target_position[1], target_position[2], rotation_angle%(2*np.pi)]

        self._bullet_client.setJointMotorControlArray(
            self._body_id,
            joint_ids,
            self._bullet_client.POSITION_CONTROL,
            targetPositions=target_states,
            forces=[self._force] * len(joint_ids),
            positionGains=[self._speed] * len(joint_ids)
        )
        
        for i in range(240 * 6):
            current_states = np.array([self._bullet_client.getJointState(self._body_id, joint_id)[0] for joint_id in joint_ids])
            states_diff = np.abs(target_states - current_states)
            # stop moving gripper if gripper collide with other objects
            if stop_at_contact:
                is_in_contact = False
                points = self._bullet_client.getContactPoints(bodyA=self._body_id)
                if len(points) > 0:
                    for p in points:
                        if p[9] > 0:
                            is_in_contact = True
                            break
                if is_in_contact:
                    break
            if np.all(states_diff < 1e-4):
                break
            self._gripper.step_constraints(self._body_id, self._gripper_parent_index+1)
            self._bullet_client.stepSimulation()

        self.fix_joints(joint_ids)

    
    def fix_joints(self, joint_ids):
        current_states = np.array([self._bullet_client.getJointState(self._body_id, joint_id)[0] for joint_id in joint_ids])
        self._bullet_client.setJointMotorControlArray(
            self._body_id,
            joint_ids,
            self._bullet_client.POSITION_CONTROL,
            targetPositions=current_states,
            forces=[self._force] * len(joint_ids),
            positionGains=[self._speed] * len(joint_ids)
        )
    

    def primitive_grasping(self, target_position, rotation_angle, open_scale=1.0, stop_at_contact=False):
        """
        :param target_position: (x, y, z). the position of the bottom center, not the base!
        :param rotation_angle: rotation in z axis \in  [0, 2 * \pi]

        :return successs or not (True/False)
        """
        self.move([target_position[0], target_position[1], self._home_position[2]], rotation_angle)
        self.open(open_scale)
        self.move(target_position, rotation_angle, stop_at_contact=stop_at_contact)
        self.close()
        self.move([target_position[0], target_position[1], self._home_position[2]], rotation_angle)


    def remove(self):
        self._bullet_client.removeBody(self._body_id)


    def get_vis_pts(self, open_scale):
        pts = self._gripper.get_vis_pts(open_scale)
        angle = self._default_orientation[-1] # only add rotation around z axis
        rotated_pts = np.transpose(np.dot(np.asarray(
            [[np.cos(angle),-np.sin(angle)],
             [np.sin(angle), np.cos(angle)]]),np.transpose(pts)))
        return rotated_pts