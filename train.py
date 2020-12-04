import argparse
import multiprocessing as mp
import os
import shutil
import signal
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

import utils
from models.grasping_model import GraspingModel
from models.grasping_model_scene_only import GraspingModelSceneOnly
from models.model_utils import get_action, get_affordance
from replay_buffer import ReplayBuffer
from sim import PybulletSim

parser = argparse.ArgumentParser()

# global
parser.add_argument('--exp', default='exp', type=str, help='name of experiment. The directory to save data is exp/[exp]')
parser.add_argument('--gpu', default='0', type=str, help='GPU device ID. -1 means cpu')
parser.add_argument('--seed', default=0, type=int, help='random seed of pytorch and numpy')
parser.add_argument('--snapshot_gap', default=50, type=int, help='Frequence of saving the snapshot (e.g. visualization, model, optimizer)')
parser.add_argument('--num_envs', default=8, type=int, help='number of envs, each env has a process')
parser.add_argument('--num_vis', default=2, type=int, help='numer of visualization sequences, None means num_envs')
parser.add_argument('--gui', action="store_true", help='use GUI')

# environment
parser.add_argument('--num_open_scale', default=5, type=int, help='number of open scales')
parser.add_argument('--min_open_scale', default=0.5, type=float, help='minimal open scale')
parser.add_argument('--max_open_scale', default=1, type=float, help='maximum open scale')
parser.add_argument('--target_num', default=5, type=int, help='number of target objects')
parser.add_argument('--obstacle_num', default=0, type=int, help='number of obstacle objects')
parser.add_argument('--seq_len', default=None, type=int, help='number of steps for each sequence. None means target_num')
parser.add_argument('--num_cam', default=None, type=int, help='number of camera for capturing scene, default uses random number from 1 to 4')

# model
parser.add_argument('--model_type', default='adagrasp', type=str,choices=['adagrasp', 'adagrasp_init_only', 'scene_only'], help='the type of grasping model to train.')
parser.add_argument('--num_rotations', default=16, type=int, help='number of rotations')
parser.add_argument('--load_checkpoint', default=None, type=str, help='exp name or a directpry of ckpt (suffix is .pth). Load the the checkpoint from another training exp')

# policy
parser.add_argument('--min_epsilon', default=0.2, type=float, help='minimal epsilon in data collection')
parser.add_argument('--max_epsilon', default=0.8, type=float, help='maximal epsilon in data collection')
parser.add_argument('--exploration_epoch', default=2000, type=int, help='how many epoched to decay from 1 to min_epsilon')

# training
parser.add_argument('--learning_rate', default=5e-4, type=float, help='learning rate of the optimizer')
parser.add_argument('--epoch', default=5000, type=int, help='How many training epochs')
parser.add_argument('--iter_per_epoch', default=32, type=int, help='numer of traininig iterations per epoch')
parser.add_argument('--batch_size', default=8, type=int, help='batch size for training')
parser.add_argument('--data_augmentation', action="store_true", help='use data_augmentation or not')

# replay_buffer
parser.add_argument('--load_replay_buffer', default=None, type=str, help='exp name. Load the replay buffer from another training exp')
parser.add_argument('--replay_buffer_size', default=12000, type=int, help='maximum size of replay buffer')


def main():
    args = parser.parse_args()

    # Seq sequence length & visualization_num
    args.seq_len = args.target_num if args.seq_len is None else args.seq_len
    args.visualization_dir = os.path.join('exp', args.exp, 'visualization')
    utils.mkdir(args.visualization_dir)

    # Set exp directory and tensorboard writer
    writer_dir = os.path.join('exp', args.exp)
    utils.mkdir(writer_dir)
    writer = SummaryWriter(writer_dir)

    # Save arguments
    str_list = []
    for key in vars(args):
        print('[{0}] = {1}'.format(key, getattr(args, key)))
        str_list.append('--{0}={1} \\'.format(key, getattr(args, key)))
    with open(os.path.join('exp', args.exp, 'args.txt'), 'w+') as f:
        f.write('\n'.join(str_list))

    # Set directory. e.g. replay buffer, visualization, model snapshot
    args.replay_buffer_dir = os.path.join('exp', args.exp, 'replay_buffer')
    args.model_dir = os.path.join('exp', args.exp, 'models')
    utils.mkdir(args.model_dir)

    # Reset random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Set device
    device = torch.device('cpu') if args.gpu == '-1' else torch.device(f'cuda:{args.gpu}')

    # Set replay buffer
    replay_buffer = ReplayBuffer(args.replay_buffer_dir, args.replay_buffer_size)
    if args.load_replay_buffer is not None:
        print(f'==> Loading replay buffer from {args.load_replay_buffer}')
        replay_buffer.load(os.path.join('exp', args.load_replay_buffer, 'replay_buffer'))
        print(f'==> Loaded replay buffer from {args.load_replay_buffer} [size = {replay_buffer.length}]')

    # Set model and optimizer
    if args.model_type=='adagrasp':
        model = GraspingModel(num_rotations=args.num_rotations, gripper_final_state=True)
    elif args.model_type=='adagrasp_init_only':
        model = GraspingModel(num_rotations=args.num_rotations, gripper_final_state=False)
    elif args.model_type=='scene_only':
        model = GraspingModelSceneOnly(num_rotations=args.num_rotations, gripper_final_state=True)
    else:
        raise NotImplementedError(f'Does not support {args.model_type}')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.95))
    model = model.to(device)

    #check cuda memory allocation
    if args.gpu != '-1':
        bytes_allocated = torch.cuda.memory_allocated(device)
        print("Model size: {:.3f} MB".format(bytes_allocated/(1024**2)))

    # load checkpoint
    if args.load_checkpoint is not None:
        print(f'==> Loading checkpoint from {args.load_checkpoint}')
        if args.load_checkpoint.endswith('.pth'):
            checkpoint = torch.load(args.load_checkpoint, map_location=device)
        else:
            checkpoint = torch.load(os.path.join('exp', args.load_checkpoint, 'models', 'latest.pth'), map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] if args.load_replay_buffer is not None else 0
        print(f'==> Loaded checkpoint from {args.load_checkpoint}')
    else:
        start_epoch = 0

    # launch processes for each env
    args.num_envs = 1 if args.gui else args.num_envs
    processes, conns = [], []
    ctx = mp.get_context('spawn')
    for rank in range(args.num_envs):
        conn_main, conn_env = ctx.Pipe()
        reset_args = {
            'num_open_scale': args.num_open_scale,
            'max_open_scale': args.max_open_scale,
            'min_open_scale': args.min_open_scale,
            'gripper_final_state': args.model_type=='adagrasp',
            'target_num': args.target_num,
            'obstacle_num': args.obstacle_num
        }
        p = ctx.Process(target=env_process,
                        args=(rank, start_epoch+args.seed+rank, conn_env, args.gui, args.num_cam, args.seq_len, reset_args))
        p.daemon=True
        p.start()
        processes.append(p)
        conns.append(conn_main)

    # Initialize exit signal handler (for graceful exits)
    def save_and_exit(signal,frame):
        print('Warning: keyboard interrupt! Cleaning up...')
        for p in processes:
            p.terminate()
        replay_buffer.dump()
        writer.close()
        print('Finished. Now exiting gracefully.')
        sys.exit(0)
    signal.signal(signal.SIGINT, save_and_exit)


    for epoch in range(start_epoch, args.epoch):
        print(f'---------- epoch-{epoch + 1} ----------')
        timestamp = time.time()

        assert args.min_epsilon <= args.max_epsilon
        m1, m2 = args.min_epsilon, args.max_epsilon
        epsilon = max(m1, m2-(m2-m1) * epoch/args.exploration_epoch)
        # Data collection
        data = collect_data(
            conns, model, device,
            n_steps=1,
            epsilon=epsilon,
            gripper_final_state=(args.model_type=='adagrasp')
        )
        
        for d in data.values():
            replay_buffer.save_data(d)

        average_reward = np.mean([d['reward'] for d in data.values()])
        average_score = np.mean([d['score'] for d in data.values()])
        print(f'Mean reward = {average_reward:.3f}, Mean score = {average_score:.3f}')
        writer.add_scalar('Data Collection/Reward', average_reward, epoch + 1)
        writer.add_scalar('Data Collection/Score', average_score, epoch + 1)

        time_data_collection = time.time() - timestamp

        # Replay buffer statistic
        reward_data = np.array(replay_buffer.scalar_data['reward'])
        print(f'Replay buffer size = {len(reward_data)} (positive = {len(np.argwhere(reward_data == 1))}, negative = {len(np.argwhere(reward_data == 0))})')

        # Policy training
        model.train()
        torch.set_grad_enabled(True)
        sum_loss = 0
        score_statics = {'positive': list(), 'negative': list()}
        for _ in range(args.iter_per_epoch):
            iter_loss, iter_score_statics = train(model, device, replay_buffer, optimizer, args.batch_size, gripper_final_state=(args.model_type=='adagrasp'))
            sum_loss += iter_loss
            score_statics['positive'].append(iter_score_statics[1])
            score_statics['negative'].append(iter_score_statics[0])
        average_loss = sum_loss / args.iter_per_epoch
        positive_score_prediction = np.mean(score_statics['positive'])
        negative_score_prediction = np.mean(score_statics['negative'])
        print(f'Training loss = {average_loss:.5f}, positive_mean = {positive_score_prediction:.3f}, negative_mean = {negative_score_prediction:.3f}')
        writer.add_scalar('Policy Training/Loss', average_loss, epoch + 1)
        writer.add_scalar('Policy Training/Positive Score Prediction', positive_score_prediction, epoch + 1)
        writer.add_scalar('Policy Training/Negative Score Prediction', negative_score_prediction, epoch + 1)

        # Save model and optimizer
        if (epoch + 1) % args.snapshot_gap == 0:
            model.eval()
            torch.set_grad_enabled(False)

            # Visualization
            [conn.send("reset") for conn in conns]
            data = collect_data(
                conns, model, device,
                n_steps=args.seq_len,
                epsilon=0,
                gripper_final_state=(args.model_type=='adagrasp')
            )

            vis_path = os.path.join(args.visualization_dir, 'epoch_%06d' % (epoch + 1))
            utils.visualization(data, args.num_envs, args.seq_len, args.num_open_scale, args.num_rotations, args.num_vis, vis_path)

            save_state = {
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch + 1
            }
            torch.save(save_state, os.path.join(args.model_dir, 'latest.pth'))
            shutil.copyfile(
                os.path.join(args.model_dir, 'latest.pth'),
                os.path.join(args.model_dir, 'epoch_%06d.pth' % (epoch + 1))
            )

            # Save replay buffer
            replay_buffer.dump()

        # Print elapsed time for an epoch
        time_all = time.time() - timestamp
        time_training = time_all - time_data_collection
        print(f'Elapsed time = {time_all:.2f}: (collect) {time_data_collection:.2f} + (train) {time_training:.2f}')

    save_and_exit(None, None)


def env_process(rank, seed, conn, use_gui, num_cam, seq_len, reset_args):
    # set random
    np.random.seed(seed)
    
    env = PybulletSim(gui_enabled=use_gui, num_cam=np.random.randint(4) if num_cam is None else num_cam)
    while True:
        min_opening = reset_args['min_open_scale']
        max_opening = reset_args['max_open_scale']
        num_opening = reset_args['num_open_scale']
        open_scales_pre = np.linspace(min_opening, max_opening, num_opening, False)
        assert num_opening>0
        random_step = np.random.rand(num_opening) * (max_opening - min_opening)/num_opening
        open_scales = open_scales_pre + random_step
        gripper_size = 0.4 * np.random.rand() + 0.8
        observation = env.reset(
            gripper_type='train',
            gripper_size=gripper_size,
            open_scales=open_scales,
            gripper_final_state=reset_args['gripper_final_state'],
            category_name='train',
            target_num=reset_args['target_num'],
            obstacle_num=reset_args['obstacle_num'],
            num_cam=np.random.randint(4) if num_cam is None else num_cam)
        for _ in range(seq_len):
            if observation['n_target_left']==0:
                break
            message = conn.recv()
            if message == "reset":
                break
            elif message == "step":
                conn.send(observation)
                kwargs = conn.recv()
                action, score, others = get_action(kwargs['affordance_map'], kwargs['epsilon'], kwargs['open_scales'])
                reward, observation = env.step(action)
                conn.send((action, score, others, reward))
            else:
                raise ValueError


def collect_data(conns, model, device, n_steps, epsilon, gripper_final_state):

    num_envs = len(conns)

    model.eval()
    torch.set_grad_enabled(False)

    data = dict()

    for step in range(n_steps):
        for conn in conns:
            conn.send("step")
        observations = [conn.recv() for conn in conns]
        affordance_maps = get_affordance(observations, model, device, gripper_final_state=gripper_final_state)
        for rank in range(num_envs):
            affordance_map = affordance_maps[rank]
            # Zero out predicted action values for all pixels outside workspace
            valid = observations[rank]['valid_pix_heightmap']
            s = affordance_map.shape
            affordance_map[np.logical_not(np.tile(valid,(s[0],s[1],1,1)))] = 0
            
            conns[rank].send({
                'message':'step',
                'affordance_map': affordance_map,
                'epsilon': epsilon,
                'open_scales': observations[rank]['open_scales']
            })
            data[(rank, step)] = observations[rank]
            data[(rank, step)]['affordance_map'] = affordance_map

        for rank in range(num_envs):
            (action, score, others, reward) = conns[rank].recv()

            data[(rank, step)]['open_scale_idx'] = others['open_scale_idx']
            data[(rank, step)]['grasp_pixel'] = np.array(action[:2])
            data[(rank, step)]['grasp_angle'] = action[2]
            data[(rank, step)]['score'] = score
            data[(rank, step)]['reward'] = reward

    return data


def train(model, device, replay_buffer, optimizer, batch_size, gripper_final_state=False):
    train_ind = np.arange(replay_buffer.length)
    reward_data = np.array(replay_buffer.scalar_data['reward'])[train_ind]
    score_data = np.array(replay_buffer.scalar_data['score'])[train_ind]
    
    replay_iter = []
    batch_split = {
        0: batch_size // 2,
        1: batch_size // 2
    }

    for grasp_label in [0, 1]:
        # Fetch previous training samples from the replay buffer
        sample_ind = np.argwhere(reward_data == grasp_label)[:, 0]
        if sample_ind.shape[0] == 0:
            print('[Warning] Data is not balanced')
            continue

        # add data with high suprise value
        sample_surprise_values = np.abs(grasp_label - score_data[sample_ind])
        sorted_surprise_ind = np.argsort(sample_surprise_values)
        sorted_sample_ind = sample_ind[sorted_surprise_ind]
        prob = range(1, len(sorted_sample_ind)+1)
        prob = prob / np.sum(prob)
        replay_iter.append(np.random.choice(
            sorted_sample_ind,
            min(len(sorted_sample_ind), batch_split[grasp_label]),
            replace=False,
            p=prob
        ))

    replay_iter = np.concatenate(replay_iter)
    np.random.shuffle(replay_iter)

    # fetch data from replay buffer
    exclude=['gripper_close_tsdf'] if not gripper_final_state else None
    sample_data = replay_buffer.fetch_data(train_ind[replay_iter], exclude=exclude)
    # data augmentation
    if np.random.rand() < 0.7:
        sample_data = utils.data_augmentation(sample_data, device)

    open_scale_idx = np.array(sample_data['open_scale_idx'])
    gripper_tsdfs = np.array(sample_data['gripper_tsdf']) # [B, S, G, G, G]; G: gripper tsdf size (64)
    gripper_tsdfs = gripper_tsdfs[np.arange(len(open_scale_idx)), open_scale_idx, np.newaxis] # [B, 1, G, G, G]
    scene_tsdfs = sample_data['scene_tsdf']
    obstacle_vols = sample_data['obstacle_vol']

    grasp_label = sample_data['reward']
    grasp_pixel = sample_data['grasp_pixel']
    rotation_angle = sample_data['grasp_angle']
    gripper_close_tsdf = sample_data['gripper_close_tsdf'] if gripper_final_state else None

    observations = {
        'scene_tsdf': scene_tsdfs,
        'obstacle_vol': obstacle_vols,
        'gripper_tsdf': gripper_tsdfs,
        'gripper_close_tsdf': gripper_close_tsdf
    }

    output_tensors = get_affordance(
        observations, model, device,
        rotation_angle=rotation_angle.reshape(1, -1),
        torch_tensor=True,
        gripper_final_state=gripper_final_state
    ) # [S, K, B, 2, W, H]

    # update score
    affordance_map = nn.functional.softmax(output_tensors[0][0], dim=1).data.cpu().numpy()[:, 1, :, :] # [B, W, H]
    latest_score = affordance_map[np.arange(grasp_pixel.shape[0]), grasp_pixel[:, 0], grasp_pixel[:, 1]]
    replay_buffer.update_data('score', replay_iter, latest_score)
    score_statics = dict()
    for label in [0, 1]:
        idx_mask = (grasp_label == label).astype(np.int)
        score_statics[label] = 0 if np.sum(idx_mask) == 0 else np.sum(idx_mask * latest_score) / np.sum(idx_mask)

    # Compute loss and gradients. output_tensors: [S, K, B, 2, W, H]
    optimizer.zero_grad()
    criterion = nn.CrossEntropyLoss()
    loss = criterion(
        output_tensors[0][0][np.arange(grasp_pixel.shape[0]), :, grasp_pixel[:, 0], grasp_pixel[:, 1]],
        torch.from_numpy(grasp_label).long().to(device)
    )
    loss.backward()
    optimizer.step()
    
    return loss.item(), score_statics


if __name__ == '__main__':
    main()
