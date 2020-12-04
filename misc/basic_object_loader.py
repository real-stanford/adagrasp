import os
import time

import numpy as np


def load_obj(bullet_client, obj_path, scaling, position, orientation, fixed_base=False, visual_path=None):
    template = """<?xml version="1.0" encoding="UTF-8"?>
<robot name="obj.urdf">
    <link name="baseLink">
        <contact>
            <lateral_friction value="1.0"/>
            <rolling_friction value="0.0001"/>
            <inertia_scaling value="3.0"/>
        </contact>
        <inertial>
            <origin rpy="0 0 0" xyz="0 0 0"/>
            <mass value=".1"/>
            <inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/>
        </inertial>
        <visual>
            <origin rpy="0 0 0" xyz="0 0 0"/>
            <geometry>
                <mesh filename="{0}" scale="1 1 1"/>
            </geometry>
        </visual>
        <collision>
            <origin rpy="0 0 0" xyz="0 0 0"/>
            <geometry>
                <mesh filename="{1}" scale="1 1 1"/>
            </geometry>
        </collision>
    </link>
</robot>"""
    urdf_path = '.tmp_my_obj_%.8f%.8f.urdf' % (time.time(), np.random.rand())
    with open(urdf_path, "w") as f:
        f.write(template.format(obj_path, obj_path))
    body_id = bullet_client.loadURDF(
        fileName=urdf_path,
        basePosition=position,
        baseOrientation=orientation,
        globalScaling=scaling,
        useFixedBase=fixed_base
    )
    os.remove(urdf_path)

    return body_id


def load_dexnet(category_name=None, instance_id=None, *args, **kwargs):
    """
    Load DexNet Object based on category_name. Instance will be random chosen if category_id is None.

    :param category_name: camegory name. choices=['adversarial', 'kit', '3dnet']
    :param instance_id: Unique instance id. Will randomly choose one if it is None.
    :param others: refer to 'load_obj()'

    :return: body_id
    """
    dexnet_path = 'data/dexnet'
    assert category_name in os.listdir(dexnet_path)
    category_path = os.path.join(dexnet_path, category_name)

    # Choose instance
    if instance_id is None:
        # Randomly choose an instance
        instance_id = np.random.choice(sorted(os.listdir(category_path)))
    else:
        # Check whether the instance_id is valid
        if instance_id not in os.listdir(category_path):
            raise ValueError(f'instance_id \'{instance_id}\' can not be found in {category_name}')
    obj_path = os.path.join(category_path, instance_id, 'obj_com_vhacd.obj')
    visual_path = os.path.join(category_path, instance_id, 'obj_com.obj')

    return load_obj(obj_path=obj_path, visual_path=visual_path, *args, **kwargs)