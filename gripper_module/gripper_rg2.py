import numpy as np
from gripper_module.gripper_base import GripperBase
import time
import threading


class GripperRG2(GripperBase):
    def __init__(self, bullet_client, gripper_size):
        r""" Initialization of RG2 gripper
        specific args for RG2:
            - gripper_size: global scaling of the gripper when loading URDF
        """
        super().__init__()

        self._bullet_client = bullet_client
        self._gripper_size = gripper_size

        # offset the gripper to a down facing pose for grasping
        self._pos_offset = np.array([0, 0, 0.163 * self._gripper_size]) # offset from base to center of grasping
        self._orn_offset = self._bullet_client.getQuaternionFromEuler([np.pi, 0, 0])
        
        # define force and speed (grasping)
        self._force = 500
        self._grasp_speed = 0.5

        # define driver joint; the follower joints need to satisfy constraints when grasping
        self._driver_joint_id = 1
        self._driver_joint_lower = 0
        self._driver_joint_upper = 0.86
        self._follower_joint_ids = [0, 2, 3, 4, 5]
        self._follower_joint_sign = [1, -1, -1, -1, 1]


    def load(self, basePosition):
        gripper_urdf = "assets/gripper/rg2/model.urdf"
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
        targets = pos * np.array(self._follower_joint_sign)
        self._bullet_client.setJointMotorControlArray(
            mount_gripper_id,
            [(joint_id+n_joints_before) for joint_id in self._follower_joint_ids],
            self._bullet_client.POSITION_CONTROL,
            targetPositions=targets,
            forces=[self._force] * len(self._follower_joint_ids),
            positionGains=[1] * len(self._follower_joint_ids)
        )
        return pos


    def open(self, mount_gripper_id, n_joints_before, open_scale):
        open_scale = np.clip(open_scale, 0.1, 1.0)
        target_pos = open_scale*self._driver_joint_lower + (1-open_scale)*self._driver_joint_upper  # recalculate scale because larger joint position corresponds to smaller open width
        self._bullet_client.setJointMotorControl2(
            mount_gripper_id,
            self._driver_joint_id+n_joints_before,
            self._bullet_client.POSITION_CONTROL,
            targetPosition=target_pos,
            force=self._force
        )
        for i in range(240 * 2):
            driver_pos = self.step_constraints(mount_gripper_id, n_joints_before)
            if np.abs(driver_pos - target_pos)<1e-5:
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
        for i in range(240 * 2):
            pos = self.step_constraints(mount_gripper_id, n_joints_before)
            if pos>self._driver_joint_upper:
                break
            self._bullet_client.stepSimulation()
    
    def get_pos_offset(self):
        return self._pos_offset

    
    def get_orn_offset(self):
        return self._orn_offset


    def get_vis_pts(self, open_scale):
        width = 0.05 * np.sin(open_scale)
        return self._gripper_size * np.array([
            [-width, 0],
            [width, 0]
        ])