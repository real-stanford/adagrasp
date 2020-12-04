import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

__all__ = [
    'ConvBlock2D', 'cross_conv2d', 'rotate_tensor2d', 'ResBlock2D',
    'ConvBlock3D', 'cross_conv3d', 'rotate_tensor3d'
]


class ConvBlock2D(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, padding=None, stride=1, dilation=1, norm=False, relu=False, pool=False, upsm=False):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2 if padding is None else padding
        self.conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=not norm)
        self.norm = nn.BatchNorm2d(planes) if norm else None
        self.relu = nn.ReLU(inplace=True) if relu else None
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) if pool else None
        self.upsm = upsm

    def forward(self, x):
        out = self.conv(x)
        out = out if self.norm is None else self.norm(out)
        out = out if self.relu is None else self.relu(out)
        out = out if self.pool is None else self.pool(out)
        out = out if not self.upsm else F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=True)

        return out


class ConvBlock3D(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, padding=None, stride=1, dilation=1, norm=False, relu=False, pool=False, upsm=False):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2 if padding is None else padding
        self.conv = nn.Conv3d(inplanes, planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=not norm)
        self.norm = nn.BatchNorm3d(planes) if norm else None
        self.relu = nn.ReLU(inplace=True) if relu else None
        self.pool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1) if pool else None
        self.upsm = upsm

    def forward(self, x):
        out = self.conv(x)

        out = out if self.norm is None else self.norm(out)
        out = out if self.relu is None else self.relu(out)
        out = out if self.pool is None else self.pool(out)
        out = out if not self.upsm else F.interpolate(out, scale_factor=2, mode='trilinear', align_corners=True)

        return out


class ResBlock2D(nn.Module):
    def __init__(self, inplanes, planes, downsample=None, last_activation=True):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.activation1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.activation2 = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.activation1(out)

        return out


def cross_conv2d(inputs, weights, groups=1, bias=None, stride=1, padding=0, dilation=1):
    outputs = []
    for input, weight in zip(inputs, weights):
        output = F.conv2d(
            input=input.unsqueeze(0),
            weight=weight,
            bias=bias,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups
        )
        outputs.append(output)
    outputs = torch.cat(outputs, dim=0)
    return outputs


def cross_conv3d(inputs, weights, groups=1, bias=None, stride=1, padding=0, dilation=1):
    outputs = []
    for input, weight in zip(inputs, weights):
        output = F.conv3d(
            input=input.unsqueeze(0),
            weight=weight,
            bias=bias,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups
        )
        outputs.append(output)
    outputs = torch.cat(outputs, dim=0)
    return outputs


def rotate_tensor2d(inputs, rotate_theta, offset=None, padding_mode='zeros', pre_padding=None):
    """rotate 2D tensor counter-clockwise

    Args:
        inputs: torch tensor, [N, C, W, H]
        rotate_theta: ndarray,[N]
        offset: None or ndarray, [2, N]
        padding_mode: "zeros" or "border"
        pre_padding: None of float. the valud used for pre-padding such that width == height
    Retudn:
        outputs: rotated tensor
    """
    device = inputs.device

    if pre_padding is not None:
        lr_pad_w = int((np.max(inputs.shape[2:])-inputs.shape[3])/2)
        ud_pad_h = int((np.max(inputs.shape[2:])-inputs.shape[2])/2)
        add_pad = nn.ConstantPad2d((lr_pad_w,lr_pad_w,ud_pad_h,ud_pad_h),0.0).to(device)
        inputs = add_pad(inputs)

    const_zeros = np.zeros(len(rotate_theta))
    affine = np.asarray([[np.cos(rotate_theta), -np.sin(rotate_theta), const_zeros],
                         [np.sin(rotate_theta), np.cos(rotate_theta), const_zeros]])
    affine = torch.from_numpy(affine).permute(2, 0, 1).float().to(device)
    flow_grid = F.affine_grid(affine, inputs.size(), align_corners=True).to(device)
    outputs = F.grid_sample(inputs, flow_grid, padding_mode=padding_mode, align_corners=True)

    if offset is not None:
        const_ones = np.ones(len(rotate_theta))
        affine = np.asarray([[const_ones, const_zeros, offset[0]],
                            [const_zeros, const_ones, offset[1]]])
        affine = torch.from_numpy(affine).permute(2, 0, 1).float().to(device)
        flow_grid = F.affine_grid(affine, inputs.size(), align_corners=True).to(device)
        outputs = F.grid_sample(outputs, flow_grid, padding_mode=padding_mode, align_corners=True)
    if pre_padding is not None:
        outputs = outputs[:,:,ud_pad_h:(outputs.shape[2]-ud_pad_h),
                              lr_pad_w:(outputs.shape[3]-lr_pad_w)]
    return outputs


def rotate_tensor3d(inputs, rotate_theta, offset=None, padding_mode='zeros', pre_padding=None):
    """rotate 3D tensor counter-clockwise in z-axis 

    Args:
        inputs: torch tensor, [N, C, W, H, D]
        rotate_theta: ndarray,[N]
        offset: None or ndarray, [2, N]
        padding_mode: "zeros" or "border"
        pre_padding: None of float. the valud used for pre-padding such that width == height
    
    Retudn:
        outputs: rotated tensor
    """
    device = inputs.device
    
    if pre_padding is not None:
        lr_pad_w = int((np.max(inputs.shape[2:4])-inputs.shape[3])/2)
        ud_pad_h = int((np.max(inputs.shape[2:4])-inputs.shape[2])/2)
        add_pad = nn.ConstantPad3d((0, 0, lr_pad_w,lr_pad_w,ud_pad_h,ud_pad_h), pre_padding).to(device)
        inputs = add_pad(inputs)

    const_zeros = np.zeros(len(rotate_theta))
    const_ones = np.ones(len(rotate_theta))
    affine = np.asarray([[const_ones,  const_zeros,          const_zeros,           const_zeros],
                         [const_zeros, np.cos(rotate_theta), -np.sin(rotate_theta), const_zeros],
                         [const_zeros, np.sin(rotate_theta), np.cos(rotate_theta),  const_zeros]])
    
    affine = torch.from_numpy(affine).permute(2, 0, 1).float().to(device)
    flow_grid = F.affine_grid(affine, inputs.size(),align_corners=True).to(device)
    outputs = F.grid_sample(inputs, flow_grid, padding_mode=padding_mode, align_corners=True)
    if offset is not None:
        affine = np.asarray([[const_ones,  const_zeros, const_zeros, const_zeros],
                             [const_zeros, const_ones,  const_zeros, offset[0]],
                             [const_zeros, const_zeros, const_ones,  offset[1]]])
        affine = torch.from_numpy(affine).permute(2, 0, 1).float().to(device)
        flow_grid = F.affine_grid(affine, inputs.size(), align_corners=True).to(device)
        outputs = F.grid_sample(outputs, flow_grid, padding_mode=padding_mode, align_corners=True)
    if pre_padding is not None:
        outputs = outputs[:,:,ud_pad_h:(outputs.shape[2]-ud_pad_h),
                              lr_pad_w:(outputs.shape[3]-lr_pad_w)]
    return outputs


# Get grasping confidence scores (i.e., grasping affordances) from output tensors
def process_output(output_tensors):
    grasp_affordance_maps = []
    for tensors in output_tensors: # tensors: [K, B, 2, W, H]]
        tmp = [F.softmax(x, dim=1).data.cpu().numpy()[:, 1, :, :] for x in tensors] # list of [B, W, H]
        grasp_affordance_maps.append(np.stack(tmp, 1)) # [B, K, W, H]
    grasp_affordance_maps = np.stack(grasp_affordance_maps, 1)
    return grasp_affordance_maps # [B, S, K, W, H]


def get_action(affordance_map, epsilon, open_scales):
    """Get action based on affordance_map.

    Args:
        affordance_map: [S, K, W, H]
        epsilon: random aciton based on prob VS choose best aciton. If epsilon < 0: get action with best score
        open_scales: list of open_scales [S]

    Returns:
        action: (y, x, grasp_angle, open_scale)
        score: float
        others: dict, including 'open_scale_idx', 'grasp_angle_idx'
    """
    # select action using probability equal to score prediction at probability epsilon
    if np.random.rand()<epsilon:
        if np.all(affordance_map==0):
            print('Warning: model predicts 0 for all states! Might want to check visualization.')
            prob = np.ones_like(affordance_map) * (1/affordance_map.size)
        else:
            prob = affordance_map / np.sum(affordance_map)
        idx = np.random.choice(affordance_map.size, p=prob.ravel())
    else:
        idx = np.argmax(affordance_map)

    best_grasp_ind = np.array(np.unravel_index(idx, affordance_map.shape))
        
    num_rotations = affordance_map.shape[1]
    open_scale_idx = best_grasp_ind[0]
    open_scale = open_scales[open_scale_idx]
    grasp_angle = best_grasp_ind[1] * (2*np.pi / num_rotations)
    grasp_pixel = best_grasp_ind[2:] # (y, x)
    score = affordance_map[best_grasp_ind[0], best_grasp_ind[1], grasp_pixel[0], grasp_pixel[1]]

    action = (grasp_pixel[0], grasp_pixel[1], grasp_angle, open_scale)

    others = {
        'open_scale_idx': best_grasp_ind[0],
        'grasp_angle_idx': best_grasp_ind[1]
    }
    return action, score, others


def get_affordance(observations, model, device, rotation_angle=None, torch_tensor=False, gripper_final_state=False):
    """Get affordance maps.

    Args:
        observations: list of dict / dict of list
            - scene_tsdfs: [W, H, D]
            - obstacle_vols: [W, H, D]
            - gripper_tsdfs: [S, W, H, D]. tsdf of different open scales
        rotation_angle: None / list of float. Specific rotation angle or not.
        torch_tensor: Whether the retuen value is torch tensor (default is numpy array). torch tensor is used for training.
        gripper_final_state: gripper TSDF when it's at final state.
    Return:
        if torch_tensor is False: (for data collection)
            affordance_maps: numpy array, [B, S, K, W, H]
        if torch_tensor is True: (for training)
            output_tensors: torch tensor, [S, K, B, 2, W, H]
        (S: open_scale; K: rotation)
        
    """
    if isinstance(observations, list):
        scene_inputs = []
        gripper_inputs = []
        gripper_close_inputs = []
        for observation in observations:
            scene_tsdf = observation['scene_tsdf'].astype(np.float32)
            obstacle_vol = observation['obstacle_vol'].astype(np.float32)
            gripper_tsdf = observation['gripper_tsdf'].astype(np.float32)

            scene_inputs.append(np.stack([scene_tsdf, obstacle_vol], axis=0))
            gripper_inputs.append(gripper_tsdf)
            if gripper_final_state:
                gripper_close_tsdf = observation['gripper_close_tsdf'].astype(np.float32)
                assert gripper_close_tsdf.shape==gripper_tsdf.shape[1:]
                gripper_close_inputs.append(gripper_close_tsdf)

        scene_input_tensor = torch.from_numpy(np.stack(scene_inputs))
        gripper_input_tensor = torch.from_numpy(np.stack(gripper_inputs))
        if gripper_final_state:
            gripper_close_input_tensor = torch.from_numpy(np.stack(gripper_close_inputs))
    else:
        scene_tsdfs = np.array(observations['scene_tsdf'], dtype=np.float32)
        obstacle_vols = np.array(observations['obstacle_vol'], dtype=np.float32)
        scene_input_tensor = torch.from_numpy(np.stack([scene_tsdfs, obstacle_vols], axis=1))
        gripper_input_tensor = torch.from_numpy(np.array(observations['gripper_tsdf'], dtype=np.float32))
        if gripper_final_state:
            gripper_close_input_tensor = torch.from_numpy(np.array(observations['gripper_close_tsdf'], dtype=np.float32))

    gripper_close_input_tensor = gripper_close_input_tensor.to(device) if gripper_final_state else None
    # Get grasp affordance maps
    output_tensors = model(scene_input_tensor.to(device), gripper_input_tensor.to(device), rotation_angle, gripper_close_input=gripper_close_input_tensor)
    # print(np.mean(output_tensors[0][0].cpu().numpy()))
    if torch_tensor:
        return output_tensors
    else:
        affordance_maps = process_output(output_tensors)
        return affordance_maps
    