from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.model_utils import (ConvBlock2D, ConvBlock3D, ResBlock2D,
                                 rotate_tensor3d, rotate_tensor2d)


class GraspingModelSceneOnly(nn.Module):
    def __init__(self, num_rotations=16, downsample_num=3, input_channel=2, **kwargs):
        super().__init__()
        
        self.num_rotations = num_rotations

        # scene encoder. [B, 2, W, H, 32] --> [B, 64, W/4, H/4, 1]
        self._scene_encoder_3d = nn.Sequential(OrderedDict([
            ('scn_enc-conv0', ConvBlock3D(2, 16, stride=2, norm=True)),
            ('scn_enc-conv1', ConvBlock3D(16, 32, stride=2, norm=True)),
            ('scn_enc-conv2', ConvBlock3D(32, 64, kernel_size=(1, 1, 16), padding=0, norm=True)),
        ]))
        self._scene_encoder_2d = nn.Sequential(OrderedDict([
            ('scn_enc-conv3', ResBlock2D(64, 64)),
        ]))

        # scene decoder. [B, 64, W/4, H/4] --> [B, 2, W, H]
        self._decoder = nn.Sequential(OrderedDict([
            ('dec-resn0', ResBlock2D(64, 64)),
            ('dec-conv1', ConvBlock2D(64, 32, norm=True, upsm=True)),
            ('dec-resn2', ResBlock2D(32, 32)),
            ('dec-conv3', ConvBlock2D(32, 16, norm=True, upsm=True)),
            ('dec-conv4', nn.Conv2d(16, 2, kernel_size=1, bias=False))
        ]))

        # Initialize random weights
        for m in self.named_modules():
            if isinstance(m[1], nn.Conv2d) or isinstance(m[1], nn.Conv3d):
                nn.init.kaiming_normal_(m[1].weight.data)
            elif isinstance(m[1], nn.BatchNorm2d) or isinstance(m[1], nn.BatchNorm3d):
                m[1].weight.data.fill_(1)
                m[1].bias.data.zero_()
    

    # Forward pass
    def forward(self, scene_input, gripper_input, rotation_angle=None, **kwargs):
        '''
        input:
            - scene_input: [B, 2, W, H, 32]
            - gripper_input: [B, S, 64, 64, 32] Not used in the model!
        output:
            - output_tensors: [S, K, B, 2, W, H]
        '''
        # If rotation angle (in radians) is not specified, do forward pass with all rotations
        if rotation_angle is None:
            rotation_angle = np.arange(self.num_rotations)*(2*np.pi/self.num_rotations)
            rotation_angle = np.array([[x] * scene_input.size(0) for x in rotation_angle])

        output_tensors = []
        for _ in range(1):
            rotated_scenes = [rotate_tensor3d(scene_input, - angles) for angles in rotation_angle] # rotate scene tensor clockwise <=> rotate scene counter-clockwise
            
            outputs = []
            for scene in rotated_scenes: # scene: [B, 2, 128, 128, 32]
                scene_embedding = self._scene_encoder_3d(scene).squeeze(-1) # [B, 64, W/4, H/4]
                scene_embedding = self._scene_encoder_2d(scene_embedding) 
                scene_decoded = self._decoder(scene_embedding) # [B, 2, W, H]
                outputs.append(scene_decoded)

            outputs_rotated_back = [rotate_tensor2d(scene, theta) for scene, theta in zip(outputs, rotation_angle)] # rotate scene tensor counter-clockwise <=> rotate scene clockwise
            output_tensors.append(outputs_rotated_back)

        return output_tensors # [1, K, B, 2, W, H]