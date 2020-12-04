import argparse
import os

import numpy as np
import torch
import tqdm

import utils
from models.grasping_model import GraspingModel
from models.grasping_model_scene_only import GraspingModelSceneOnly
from models.model_utils import get_action, get_affordance
from sim import PybulletSim

parser = argparse.ArgumentParser()

# global
parser.add_argument('--mode', default='vis', choices=['fixed_gripper', 'select_gripper'], help='test mode')
parser.add_argument('--save', default='test_vis', type=str, help='where to save the visualizations')
parser.add_argument('--gui', action="store_true", help='use GUI')
parser.add_argument('--gpu', default='0', type=str, help='GPU device ID. -1 means cpu')
parser.add_argument('--seed', default=0, type=int, help='random seed of pytorch and numpy')
parser.add_argument('--n_test', default=100, type=int, help='number of envs, each env has a process')
parser.add_argument('--n_vis_stored', default=20, type=int, help='number of sequences stored for visualization')

# environment
parser.add_argument('--gripper_types', default=None, type=str, nargs='+', help='list of gripper_name to be used, separated by space')
parser.add_argument('--num_open_scale', default=5, type=int, help='number of open scales')
parser.add_argument('--min_open_scale', default=0.5, type=float, help='minimal open scale')
parser.add_argument('--max_open_scale', default=1, type=float, help='maximum open scale')	
parser.add_argument('--random_open_scale', action="store_true", help='if only choose 1 open scale')
parser.add_argument('--target_num', default=1, type=int, help='number of target objects')
parser.add_argument('--obstacle_num', default=0, type=int, help='number of obstacle objects')
parser.add_argument('--seq_len', default=None, type=int, help='number of steps for each sequence. None means target_num')
parser.add_argument('--category_name', default=None, type=str, help='category of dexnet objects')
parser.add_argument('--num_cam', default=1, type=int, help='number of camera for capturing scene')

# model
parser.add_argument('--model_type', default='adagrasp', type=str,choices=['adagrasp', 'adagrasp_init_only', 'scene_only'], help='the type of grasping model to test')
parser.add_argument('--num_rotations', default=16, type=int, help='number of rotations')
parser.add_argument('--load_checkpoint', default=None, type=str, help='path of model checkpoint (suffix is .pth)')


def main():
    args = parser.parse_args()

    # Seq sequence length & visualization_num
    args.seq_len = args.target_num if args.seq_len is None else args.seq_len
    
    # Reset random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Set device
    device = torch.device('cpu') if args.gpu == '-1' else torch.device(f'cuda:{args.gpu}')

    # Set model & load checkpoint
    if args.model_type=='adagrasp':
        model = GraspingModel(num_rotations=args.num_rotations, gripper_final_state=True)
    elif args.model_type=='adagrasp_init_only':
        model = GraspingModel(num_rotations=args.num_rotations, gripper_final_state=False)
    elif args.model_type=='scene_only':
        model = GraspingModelSceneOnly(num_rotations=args.num_rotations, gripper_final_state=True)
    else:
        raise NotImplementedError(f'Does not support {args.model_type}')
    model = model.to(device)
    checkpoint = torch.load(args.load_checkpoint, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    print(f'==> Loaded checkpoint from {args.load_checkpoint}')
    model.eval()
    torch.set_grad_enabled(False)

    if args.mode == 'fixed_gripper':
        return test_fixed_gripper(model, device, args)
    elif args.mode == 'select_gripper':
        return test_select_gripper(model, device, args)


def test_select_gripper(model, device, args):
    env = PybulletSim(gui_enabled=args.gui, gripper_selection=True, num_cam=args.num_cam)
    open_scales_pre = np.linspace(args.min_open_scale, args.max_open_scale, args.num_open_scale, True)
    random_open_scale_list = np.random.choice(open_scales_pre, args.n_test * args.seq_len)

    gripper_type_list = args.gripper_types
    gripper_observation_list = list()
    for gripper_type in gripper_type_list:
        gripper_kwargs = dict() if not gripper_type.startswith('barrett_hand-') else {'palm_joint': float(gripper_type.split('-')[-1])}
        gripper_observation_list.append(
            env.load_gripper(
                gripper_type='barrett_hand' if gripper_type.startswith('barrett_hand-') else gripper_type,
                gripper_size=1,
                open_scales=open_scales_pre,
                gripper_final_state=(args.model_type=='adagrasp'),
                remove=True,
                **gripper_kwargs
            )
        )

    data = dict()
    rewards, scores = list(), list()
    n_vis_stored = min(args.n_test, args.n_vis_stored)

    reset_args = {
        'gripper_type': args.gripper_types,
        'target_num': args.target_num,
        'obstacle_num': args.obstacle_num,
        'gripper_size': 1,
        'category_name': args.category_name,
        'gripper_final_state': args.model_type=='adagrasp',
        'num_cam': args.num_cam
    }
    
    for rank in tqdm.trange(args.n_test):
        open_scales = [random_open_scale_list[rank]] if args.random_open_scale else open_scales_pre
        
        np.random.seed(rank)
        observation = env.reset(
            open_scales=open_scales,
            **reset_args
        )

        for step in range(args.seq_len):
            observations = list()
            for gripper_observation in gripper_observation_list:
                observations.append(
                    {**gripper_observation, **observation}
                )
            affordance_maps = get_affordance(observations, model, device, gripper_final_state=(args.model_type=='adagrasp'))

            # Zero out predicted action values for all pixels outside workspace
            valid = observation['valid_pix_heightmap']
            s = affordance_maps.shape
            affordance_maps[np.logical_not(np.tile(valid,(s[0],s[1],s[2],1,1)))] = 0

            action_list, score_list, others_list = list(), list(), list()

            for affordance_map in affordance_maps:
                cur_action, cur_score, cur_others = get_action(affordance_map, 0, open_scales_pre)
                action_list.append(cur_action)
                score_list.append(cur_score)
                others_list.append(cur_others)
            best_idx = np.argmax(score_list)

            action = action_list[best_idx]
            gripper_type = gripper_type_list[best_idx]
            score = score_list[best_idx]
            other = others_list[best_idx]

            # store data for visualization
            if rank < n_vis_stored:
                gripper_observation = gripper_observation_list[best_idx]
                data[(rank, step)] = {**gripper_observation, **observation}

            reward, observation = env.step(action=action, gripper_type=gripper_type)
            scores.append(score)
            rewards.append(reward)

            # store data for visualization
            if rank < n_vis_stored:
                data[(rank, step)]['affordance_map'] = np.concatenate([affordance_maps[i] for i in range(len(gripper_type_list))])
                data[(rank, step)]['open_scale_idx'] = best_idx * len(open_scales_pre) + other['open_scale_idx']
                data[(rank, step)]['grasp_pixel'] = np.array(action[:2])
                data[(rank, step)]['grasp_angle'] = action[2]
                data[(rank, step)]['score'] = score
                data[(rank, step)]['reward'] = reward
                vis_pts = []
                for gripper_observation in gripper_observation_list:
                    vis_pts += gripper_observation['vis_pts']
                data[(rank, step)]['vis_pts'] = vis_pts
                data[(rank, step)]['gripper_type'] = list()
                for gripper_type in gripper_type_list:
                    data[(rank, step)]['gripper_type'] += [gripper_type] * len(open_scales_pre)
            
    # print result
    success_rate = np.mean(rewards)
    print(f'{args.gripper_types} test success rate: {success_rate}\n')

    # visualization
    num_open_scale = 1 if args.random_open_scale else args.num_open_scale
    utils.visualization(data, n_vis_stored, args.seq_len, num_open_scale * len(gripper_type_list), args.num_rotations, n_vis_stored, args.save)


def test_fixed_gripper(model, device, args):
    assert len(args.gripper_types) == 1
    env = PybulletSim(gui_enabled=args.gui, num_cam=args.num_cam)
    data = dict()
    rewards, scores = list(), list()
    n_vis_stored = min(args.n_test, args.n_vis_stored)

    reset_args = {
        'gripper_type': args.gripper_types,
        'gripper_size': 1,
        'gripper_final_state': args.model_type=='adagrasp',
        'target_num': args.target_num,
        'obstacle_num': args.obstacle_num,
        'category_name': args.category_name,
        'num_cam': args.num_cam
    }
    if args.gripper_types[0].startswith('barrett_hand-'):
        reset_args['gripper_type'] = 'barrett_hand'
        reset_args['palm_joint'] = float(args.gripper_types[0].split('-')[-1])
    open_scales_pre = np.linspace(args.min_open_scale, args.max_open_scale, args.num_open_scale, True)
    random_open_scale_list = np.random.choice(open_scales_pre, args.n_test*args.seq_len)
    for rank in tqdm.trange(args.n_test):
        np.random.seed(rank)
        open_scales = [random_open_scale_list[rank]] if args.random_open_scale else open_scales_pre
        observation = env.reset(open_scales=open_scales, **reset_args)
        for step in range(args.seq_len):
            affordance_maps = get_affordance([observation], model, device, gripper_final_state=(args.model_type=='adagrasp'))

            # store data for visualization
            if rank < n_vis_stored:
                data[(rank, step)] = observation

            # Zero out predicted action values for all pixels outside workspace
            valid = observation['valid_pix_heightmap']
            s = affordance_maps.shape
            affordance_maps[np.logical_not(np.tile(valid,(s[0],s[1],s[2],1,1)))] = 0
            action, score, others = get_action(affordance_maps[0], 0, observation['open_scales'])
            reward, observation = env.step(action)
            rewards.append(reward)
            scores.append(score)
            
            # store data for visualization
            if rank < n_vis_stored:
                data[(rank, step)]['affordance_map'] = affordance_maps[0]
                data[(rank, step)]['open_scale_idx'] = others['open_scale_idx']
                data[(rank, step)]['grasp_pixel'] = np.array(action[:2])
                data[(rank, step)]['grasp_angle'] = action[2]
                data[(rank, step)]['score'] = score
                data[(rank, step)]['reward'] = reward

    # print result
    success_rate = np.mean(rewards)
    print(f'{args.gripper_types} test success rate: {success_rate}\n')

    # visualization
    num_open_scale = 1 if args.random_open_scale else args.num_open_scale
    utils.visualization(data, n_vis_stored, args.seq_len, num_open_scale, args.num_rotations, n_vis_stored, args.save)


if __name__=='__main__':
    main()
