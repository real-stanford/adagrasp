from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.model_utils import (ConvBlock2D, ConvBlock3D, ResBlock2D,
                                 cross_conv2d, rotate_tensor2d,
                                 rotate_tensor3d)


class GraspingModel(nn.Module):
    def __init__(self, num_rotations=16, gripper_final_state=False, **kwargs):
        super().__init__()

        self.num_rotations = num_rotations
        self._gripper_final_state = gripper_final_state
        self._input_gripper_channel = 2 if self._gripper_final_state else 1


        # scene encoder. [B, 2, W, H, 32] --> [B, 16, W/4, H/4, 1]
        self._scene_encoder_3d = nn.Sequential(OrderedDict([
            ('scn_enc-conv0', ConvBlock3D(2, 16, stride=1, norm=True, relu=True)),
            ('scn_enc-conv1', ConvBlock3D(16, 32, stride=2, norm=True, relu=True)),
            ('scn_enc-conv2', ConvBlock3D(32, 64, stride=2, norm=True, relu=True)),
            ('scn_enc-conv3', ConvBlock3D(64, 64, kernel_size=(1, 1, 16), padding=0, norm=True, relu=True)),
        ]))
        self._scene_encoder_2d = nn.Sequential(OrderedDict([
            ('scn_enc-conv4', ResBlock2D(64, 64)),
            ('scn_enc-conv5', ResBlock2D(64, 64)),
            ('scn_enc-conv6', ConvBlock2D(64, 16, relu=True)),
        ]))

        # gripper encoder. [B, 1, 64, 64, 32] --> [B, 32, 32, 32, 1]
        self._gripper_encoder = nn.Sequential(OrderedDict([
            ('grp_enc-conv0', ConvBlock3D(self._input_gripper_channel, 16, stride=1, norm=True, relu=True)),
            ('grp_enc-conv1', ConvBlock3D(16, 64, stride=2, norm=True, relu=True)),
            ('grp_enc-conv2', ConvBlock3D(64, 64, kernel_size=(1, 1, 16), padding=0)),
        ]))
        self._gripper_encoder_2d = nn.Sequential(OrderedDict([
            ('grp_enc-conv3', ResBlock2D(64, 64)),
            ('grp_enc-conv4', ResBlock2D(64, 64)),
            ('grp_enc-conv5', ConvBlock2D(64, 16, relu=True)),
        ]))

        # scene decoder. [B, 64, W/8, H/8] --> [B, 2, W, H]
        self._decoder = nn.Sequential(OrderedDict([
            ('dec-conv0', ConvBlock2D(16, 64, relu=True)),
            ('dec-resn1', ResBlock2D(64, 64)),
            ('dec-conv2', ConvBlock2D(64, 64, norm=True, relu=True, upsm=True)),
            ('dec-resn3', ResBlock2D(64, 64)),
            ('dec-conv4', ConvBlock2D(64, 32, norm=True, relu=True, upsm=True)),
            ('dec-resn5', ResBlock2D(32, 32)),
            ('dec-conv6', nn.Conv2d(32, 2, kernel_size=1, bias=False))
        ]))

        # Initialize random weights
        for m in self.named_modules():
            if isinstance(m[1], nn.Conv2d) or isinstance(m[1], nn.Conv3d):
                nn.init.kaiming_normal_(m[1].weight.data)
            elif isinstance(m[1], nn.BatchNorm2d) or isinstance(m[1], nn.BatchNorm3d):
                m[1].weight.data.fill_(1)
                m[1].bias.data.zero_()
    

    # Forward pass
    def forward(self, scene_input, gripper_input, rotation_angle=None, gripper_close_input=None, **kwargs):
        '''
        input:
            - scene_input: [B, 2, W, H, 32]
            - gripper_input: [B, S, 64, 64, 32]. tsdf of different open scales
        output:
            - output_tensors: [S, K, B, 2, W, H]
        '''
        # If rotation angle (in radians) is not specified, do forward pass with all rotations
        if rotation_angle is None:
            rotation_angle = np.arange(self.num_rotations)*(2*np.pi/self.num_rotations)
            rotation_angle = np.array([[x] * scene_input.size(0) for x in rotation_angle])

        device = scene_input.device

        # Define forward pass
        def forward_pass(scene_input, kernels):
            """
            :param scene_input: [B, 2, W, H, 32]
            :param scene_embedding: [B, C, 1, S, S]
            """
            scene_embedding = self._scene_encoder_3d(scene_input).squeeze(-1)
            scene_embedding = self._scene_encoder_2d(scene_embedding)
            # print(np.mean(scene_input.cpu().numpy()), np.mean(scene_embedding.cpu().numpy()))
            WW, HH = kernels.size(-2), kernels.size(-1)
            scene_embedding_2d = cross_conv2d(
                inputs=scene_embedding,
                weights=kernels,
                padding=(WW//2-1, HH//2-1),
                groups=kernels.size(1)
            )
            # print(np.mean(kernels.cpu().numpy()), np.mean(scene_embedding_2d.cpu().numpy()))
            scene_embedding_2d_pad = nn.ZeroPad2d((1, 0, 1, 0))(scene_embedding_2d) # kernel_size is even! need padding to make the dimension aligned
            output = self._decoder(scene_embedding_2d_pad)
            return output

        # Pad scene_input such that width == height
        lr_pad_w = int((np.max(scene_input.shape[2:4])-scene_input.shape[3])/2)
        ud_pad_h = int((np.max(scene_input.shape[2:4])-scene_input.shape[2])/2)
        add_pad = nn.ConstantPad3d((0, 0, lr_pad_w,lr_pad_w,ud_pad_h,ud_pad_h), 1).to(device)
        pad_scene_input = add_pad(scene_input)

        output_tensors = []
        gripper_size_num = gripper_input.size(1)

        for s in range(gripper_size_num): # gripper_input.size(1) : S
            current_gripper_input = gripper_input[:, s:s+1] # (B, 1, 64, 64, 32)
            if self._gripper_final_state:
                assert gripper_close_input != None
                current_gripper_input = torch.stack([current_gripper_input.squeeze(1), gripper_close_input], 1) # (B, 2, 64, 64, 32)
            kernels = self._gripper_encoder(current_gripper_input).squeeze(-1)
            kernels = self._gripper_encoder_2d(kernels).unsqueeze(2)  #[B, C, 1, S, S]
            tmp_output_tensors = list()
            for rotate_theta in rotation_angle:
                rotated_pad_scene_input = rotate_tensor3d(pad_scene_input, rotate_theta, padding_mode="border")
                rotated_pad_output_tensor = forward_pass(rotated_pad_scene_input, kernels)
                pad_output_tensor = rotate_tensor2d(rotated_pad_output_tensor, -rotate_theta, padding_mode="border")
                output_tensor = pad_output_tensor[:,:,ud_pad_h:(pad_output_tensor.shape[2]-ud_pad_h),
                                                      lr_pad_w:(pad_output_tensor.shape[3]-lr_pad_w)]
                tmp_output_tensors.append(output_tensor)
            output_tensors.append(tmp_output_tensors)
        return output_tensors # [S, K, B, 2, W, H]