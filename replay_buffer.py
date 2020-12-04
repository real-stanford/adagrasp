import multiprocessing as mp
import os
import pickle
import shutil

import h5py
import numpy as np

import utils


class ReplayBuffer(object):
    def __init__(self, root_dir, max_length):
        self.tensor_keys = ['target_heightmap', 'color_heightmap', 'scene_tsdf', 'obstacle_vol', 'gripper_tsdf', 'gripper_close_tsdf']
        self.scalar_keys = ['grasp_pixel', 'grasp_angle', 'reward', 'score', 'vis_pts', 'gripper_type', 'open_scale_idx']

        self.root_dir = root_dir
        utils.mkdir(self.root_dir)
        self.tensor_dir = os.path.join(root_dir, 'tensor_data')
        utils.mkdir(self.tensor_dir)

        self.scalar_data = {key: [] for key in self.scalar_keys}

        self.length = 0
        self.max_length = max_length
        self.point = 0
        self.scalar_data['length'] = 0
        self.scalar_data['max_length'] = max_length
        self.scalar_data['point'] = 0

    def load(self, other_root_dir):
        # load scalar data (including length)
        with open(os.path.join(other_root_dir, 'scalar_data.pkl'), 'rb') as f:
            self.scalar_data = pickle.load(f)
        if self.max_length != self.scalar_data['max_length']:
            raise ValueError('max length should be cosistent when loading replay buffer')

        self.length = self.scalar_data['length']
        self.point = self.scalar_data['point']

        # load tensor data
        if other_root_dir != self.root_dir:
            shutil.rmtree(self.tensor_dir)
            shutil.copytree(os.path.join(other_root_dir, 'tensor_data'), self.tensor_dir)


    def dump(self):
        # dump scalar data (including length)
        with open(os.path.join(self.root_dir, 'scalar_data.pkl'), 'wb') as f:
            pickle.dump(self.scalar_data, f)


    def save_data(self, data_dict):
        # check the data everything required by replay buffer
        for key in self.tensor_keys + self.scalar_keys:
            if not key in data_dict.keys():
                raise KeyError(f'Can not find {key} in data')
        
        f = h5py.File(os.path.join(self.tensor_dir, 'tensor_data-%d.hdf5' % (self.point)), 'w')
        for key, data in data_dict.items():
            if data is None:
                continue # skip some data
            if key in self.tensor_keys:
                f.create_dataset(name=key, data=data, compression="gzip", compression_opts=4)
            elif key in self.scalar_keys:
                if self.point < self.length:
                    self.scalar_data[key][self.point] = data
                else:
                    self.scalar_data[key].append(data)
        f.close()

        self.length = min(self.length + 1, self.max_length)
        self.point = (self.point + 1) % self.max_length

        self.scalar_data['length'] = self.length
        self.scalar_data['point'] = self.point

    
    def fetch_data(self, indexes, exclude=None):
        return_data = {x: [] for x in self.tensor_keys}
        for idx in indexes:
            f = h5py.File(os.path.join(self.tensor_dir, 'tensor_data-%d.hdf5' % (idx)), 'r')
            for key in self.tensor_keys:
                if exclude and (key in exclude):
                    continue
                return_data[key].append(np.array(f[key]))
            f.close()
        for key in self.scalar_keys:
            return_data[key] = np.array(self.scalar_data[key])[indexes]
        return return_data

    
    def update_data(self, key, indexes, data):
        if not key in self.scalar_keys:
            raise KeyError(f'{key} is not a scalar key')
        for idx, d in zip(indexes, data):
            self.scalar_data[key][idx] = d
