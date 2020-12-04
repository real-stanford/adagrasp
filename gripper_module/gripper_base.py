import abc


class GripperBase(abc.ABC):
    r"""Base class for all grippers.
    Any gripper should subclass this class.
    You have to implement the following class method:
        - load(): load URDF and return the body_id
        - configure(): configure the gripper (e.g. friction)
        - open(): open gripper
        - close(): close gripper
        - get_pos_offset(): return [x, y, z], the coordinate of the grasping center relative to the base
        - get_orn_offset(): the base orientation (in quaternion) when loading the gripper
        - get_vis_pts(open_scale): [(x0, y0), (x1, y1), (x2, y2s), ...], contact points for visualization (in world coordinate)
    """
    
    @abc.abstractmethod
    def load(self):
        pass


    @abc.abstractmethod
    def configure(self):
        pass


    @abc.abstractmethod
    def open(self):
        pass


    @abc.abstractmethod
    def close(self):
        pass


    @abc.abstractmethod
    def get_pos_offset(self):
        pass


    @abc.abstractmethod
    def get_orn_offset(self):
        pass


    @abc.abstractmethod
    def get_vis_pts(self, open_scale):
        pass