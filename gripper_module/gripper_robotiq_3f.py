import numpy as np
from gripper_module.gripper_base import GripperBase
import time
import threading


class GripperRobotiq3F(GripperBase):
    def __init__(self, bullet_client, gripper_size):
        r""" Initialization of robotiq-3f gripper
        specific args for robotiq-3f:
            - gripper_size: global scaling of the gripper when loading URDF
        """
        super().__init__()

        self._bullet_client = bullet_client
        self._gripper_size = 0.8 * gripper_size
        self._pos_offset = np.array([0, 0, 0.163 * self._gripper_size]) # offset from base to center of grasping
        self._orn_offset = self._bullet_client.getQuaternionFromEuler([-np.pi/2, 0, 0])
        
        # define force and speed (grasping)
        self._force = 100
        self._grasp_speed = 0.5

        self._palm_joint_ids = [0, 4] # the joints that link the palm and fingers. not moved in grasping
        finger1_joint_ids = [1, 2, 3]
        finger2_joint_ids = [5, 6, 7]
        finger3_joint_ids = [9, 10, 11]
        # self._finger_joint_ids = [finger1_joint_ids, finger2_joint_ids, finger3_joint_ids]
        self._finger_joint_ids = finger1_joint_ids+finger2_joint_ids+finger3_joint_ids
        self._driver_joint_id = self._finger_joint_ids[0]
        self._follower_joint_ids = self._finger_joint_ids[1:]
        # joint limits
        self._joint_lower = 0.1
        self._joint_upper = 0.4


    def load(self, basePosition):
        gripper_urdf = "assets/gripper/robotiq_3f/model.urdf"
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
        # fix 2 fingers in 0 position
        self._bullet_client.setJointMotorControlArray(
            mount_gripper_id,
            [id+n_joints_before for id in self._palm_joint_ids],
            self._bullet_client.POSITION_CONTROL,
            targetPositions=[0.0] * 2,
            forces=[self._force] * 2,
            positionGains=[1.6] * 2
        )
        pos = self._bullet_client.getJointState(mount_gripper_id, self._driver_joint_id+n_joints_before)[0]
        self._bullet_client.setJointMotorControlArray(
            mount_gripper_id,
            [id+n_joints_before for id in self._follower_joint_ids],
            self._bullet_client.POSITION_CONTROL,
            targetPositions=[pos, pos-0.5, pos, pos, pos-0.5, pos, pos, pos-0.5],
            forces=[self._force]*len(self._follower_joint_ids),
            positionGains=[1.2]*len(self._follower_joint_ids)
        )
        return pos


    def open(self, mount_gripper_id, n_joints_before, open_scale):
        target_pos = open_scale*self._joint_lower + (1-open_scale)*self._joint_upper  # recalculate scale because larger joint position corresponds to smaller open width
        self._bullet_client.setJointMotorControl2(
            mount_gripper_id,
            self._driver_joint_id+n_joints_before,
            self._bullet_client.POSITION_CONTROL,
            targetPosition=target_pos,
            force=self._force
        )
        for i in range(240 * 2):
            pos = self.step_constraints(mount_gripper_id, n_joints_before)
            if np.abs(target_pos-pos)<1e-5:
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
        x = 0.0455 + 0.072 * np.sin(2*open_scale - 1.1418)
        return self._gripper_size * np.array([
            [-x, 0.04],
            [-x, -0.04],
            [x, 0]
        ])