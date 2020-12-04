import numpy as np
from gripper_module.gripper_base import GripperBase
import time


class GripperSawyer(GripperBase):
    def __init__(self, bullet_client, gripper_size):
        r""" Initialization of Sawyer
        specific args for Sawyer:
            - gripper_size: global scaling of the gripper when loading URDF
        """
        super().__init__()

        self._bullet_client = bullet_client

        self._gripper_size = gripper_size

        # offset the gripper to a down facing pose for grasping
        self._pos_offset = np.array([0, 0, 0.121 * self._gripper_size]) # offset from base to center of grasping
        self._orn_offset = self._bullet_client.getQuaternionFromEuler([np.pi, 0, np.pi/2])

        self._driver_joint_id = 0
        self._follower_joint_id = 1
        self._upper_limit = 0.044 * self._gripper_size
        self._lower_limit = 0
        
        # define force and speed (grasping)
        self._force = 300
        self._grasp_speed = 0.1

        
    def load(self, basePosition):
        gripper_urdf = "assets/gripper/sawyer/model.urdf"
        body_id = self._bullet_client.loadURDF(
            gripper_urdf, flags=self._bullet_client.URDF_USE_SELF_COLLISION,
            globalScaling=self._gripper_size,
            basePosition=basePosition
        )
        # change color
        for link_id in range(-1, self._bullet_client.getNumJoints(body_id)):
           self._bullet_client.changeVisualShape(body_id, link_id, rgbaColor=[0.5, 0.5, 0.5, 1])
        return body_id
        
    
    def configure(self, mount_gripper_id, n_links_before):
        # Set friction coefficients for gripper fingers
        for i in range(n_links_before, self._bullet_client.getNumJoints(mount_gripper_id)):
            self._bullet_client.changeDynamics(mount_gripper_id,i,lateralFriction=1.0,spinningFriction=1.0,rollingFriction=0.0001,frictionAnchor=True)


    def step_constraints(self, mount_gripper_id, n_joints_before):
        pos = self._bullet_client.getJointState(mount_gripper_id, self._driver_joint_id+n_joints_before)[0]
        self._bullet_client.setJointMotorControl2(
            mount_gripper_id,
            self._follower_joint_id+n_joints_before,
            self._bullet_client.POSITION_CONTROL,
            targetPosition=self._upper_limit-pos,
            force=self._force,
            positionGain=2*self._grasp_speed
        )
        return pos


    def open(self, mount_gripper_id, n_joints_before, open_scale):
        target_state = self._upper_limit * (1 - open_scale)
        
        self._bullet_client.setJointMotorControl2(
            mount_gripper_id,
            self._driver_joint_id+n_joints_before,
            self._bullet_client.POSITION_CONTROL,
            targetPosition=target_state, 
            force=self._force
        )

        for i in range(240 * 2):
            self.step_constraints(mount_gripper_id, n_joints_before)
            self._bullet_client.stepSimulation()

    
    def close(self, mount_gripper_id, n_joints_before):
        
        self._bullet_client.setJointMotorControl2(
            mount_gripper_id,
            self._driver_joint_id+n_joints_before,
            self._bullet_client.VELOCITY_CONTROL,
            targetVelocity=self._grasp_speed,
            force=self._force,
        )

        for i in range(240 * 2):
            pos = self.step_constraints(mount_gripper_id, n_joints_before)
            self._bullet_client.stepSimulation()

    
    def get_pos_offset(self):
        return self._pos_offset

    
    def get_orn_offset(self):
        return self._orn_offset


    def get_vis_pts(self, open_scale):
        return np.array([
            [self._upper_limit * open_scale, 0],
            [-self._upper_limit * open_scale, 0]
        ])