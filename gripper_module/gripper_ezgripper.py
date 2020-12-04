import numpy as np
from gripper_module.gripper_base import GripperBase
import time
import threading


class GripperEZGripper(GripperBase):
    def __init__(self, bullet_client, gripper_size):
        r""" Initialization of EZgripper
        specific args for EZgripper:
            - gripper_size: global scaling of the gripper when loading URDF
        """
        super().__init__()

        self._bullet_client = bullet_client
        self._gripper_size = gripper_size

        # offset the gripper to a down facing pose for grasping
        self._pos_offset = np.array([0, 0, 0.1805 * self._gripper_size]) # offset from base to center of grasping
        self._orn_offset = self._bullet_client.getQuaternionFromEuler([np.pi / 2, np.pi / 2, 0])
        
        # define force and speed (grasping)
        self._force = 500
        self._grasp_speed = 1.5

        finger1_joint_ids = [2, 3]
        finger2_joint_ids = [4, 5]
        self._finger_joint_ids = [finger1_joint_ids, finger2_joint_ids]
        self._driver_joint_id = 2
        self._follower_joint_ids = [3, 4, 5]
        self._joint_ids = [self._driver_joint_id] + self._follower_joint_ids

        self._joint_lower = 0.8
        self._joint_upper = 1.9


    def load(self, basePosition):
        gripper_urdf = "assets/gripper/ezgripper/model.urdf"
        body_id = self._bullet_client.loadURDF(
            gripper_urdf,
            flags=self._bullet_client.URDF_USE_SELF_COLLISION,
            globalScaling=self._gripper_size,
            basePosition=basePosition
        )
        return body_id


    def configure(self, mount_gripper_id, n_links_before):
        # Set friction coefficients for gripper fingers
        for i in range(n_links_before, self._bullet_client.getNumJoints(mount_gripper_id)):
            self._bullet_client.changeDynamics(mount_gripper_id,i,lateralFriction=1.0,spinningFriction=1.0,rollingFriction=0.0001,frictionAnchor=True)
        

    def step_constraints(self, mount_gripper_id, n_joints_before):
        pos = self._bullet_client.getJointState(mount_gripper_id, self._driver_joint_id+n_joints_before)[0]
        self._bullet_client.setJointMotorControlArray(
            mount_gripper_id,
            [id+n_joints_before for id in self._follower_joint_ids],
            self._bullet_client.POSITION_CONTROL,
            targetPositions=[1.9-pos, pos, 1.9-pos],
            forces=[self._force]*len(self._follower_joint_ids),
            positionGains=[1.2]*len(self._follower_joint_ids)
        )
        return pos


    def open(self, mount_gripper_id, n_joints_before, open_scale):
        open_scale = np.clip(open_scale, 0.1, 1.0)
        open_pos = open_scale*self._joint_lower + (1-open_scale)*self._joint_upper  # recalculate scale because larger joint position corresponds to smaller open width
        self._bullet_client.setJointMotorControl2(
            mount_gripper_id,
            self._driver_joint_id+n_joints_before,
            self._bullet_client.POSITION_CONTROL,
            targetPosition=open_pos,
            force=self._force
        )
        for i in range(240 * 2):
            pos = self.step_constraints(mount_gripper_id, n_joints_before)
            if np.abs(open_pos-pos)<1e-5:
                break
            self._bullet_client.stepSimulation()

    
    def close(self, mount_gripper_id, n_joints_before):
        self._bullet_client.setJointMotorControl2(
            mount_gripper_id,
            self._driver_joint_id+n_joints_before,
            self._bullet_client.VELOCITY_CONTROL,
            targetVelocity=self._grasp_speed,
            force=self._force
        )
        for i in range(240 * 4):
            pos = self.step_constraints(mount_gripper_id, n_joints_before)
            if pos>self._joint_upper:
                break
            self._bullet_client.stepSimulation()

    
    def get_pos_offset(self):
        return self._pos_offset

    
    def get_orn_offset(self):
        return self._orn_offset


    def get_vis_pts(self, open_scale):
        width = 0.09 * np.sin(open_scale)
        return self._gripper_size * np.array([
            [-width, 0],
            [width, 0]
        ])