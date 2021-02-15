import os
import logging
import sys
import socket
import numpy as np
import rospy
import torch
import torch.nn as nn
import argparse
from mpi4py import MPI

from gym import spaces

from torch.optim import Adam
import datetime
from torch.utils.tensorboard import SummaryWriter
from collections import deque

from model.net import QNetwork, ValueNetwork, GaussianPolicy, DeterministicPolicy
from stage_test_world import StageWorld
from model.sac import SAC
from model.replay_memory import ReplayMemory


parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--env-name', default="Stage",
                    help='Environment name (default: Stage)')

parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')

parser.add_argument('--eval', type=bool, default=True,
                    help='Evaluates a policy a policy every 10 episode (default: True)')

parser.add_argument('--num_steps', type=int, default=10, metavar='N',
                    help='maximum number of steps (default: 1000000)')

parser.add_argument('--cuda', action="store_true",
                    help='run on CUDA (default: False)')

parser.add_argument('--laser_beam', type=int, default=512,
                    help='the number of Lidar scan [observation] (default: 512)')
parser.add_argument('--num_env', type=int, default=1,
                    help='the number of environment (default: 1)')

parser.add_argument('--laser_hist', type=int, default=3,
                    help='the number of laser history (default: 3)')

parser.add_argument('--act_size', type=int, default=2,
                    help='Action size (default: 2, translation, rotation velocity)')

parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                    help='hidden size (default: 256)')                    
args = parser.parse_args()


def run(comm, env, agent, policy_path, args):

    # Training Loop
    total_numsteps = 0

    # world reset
    if env.index == 0: # step
        env.reset_world()

        #Tesnorboard
        writer = SummaryWriter('test_runs/' + policy_path)
        
    for i_episode in range(args.num_steps):

        episode_reward = 0
        episode_steps = 0
        done = False        

        # Env reset
        env.reset_pose_test()
        # generate goal
        env.generate_goal_point_test()
        
        # Get initial state
        frame = env.get_laser_observation()
        frame_stack = deque([frame, frame, frame])
        goal = np.asarray(env.get_local_goal())
        speed = np.asarray(env.get_self_speed())
        state = [frame_stack, goal, speed]

        # Episode start
        while not done and not rospy.is_shutdown():    
            state_list = comm.gather(state, root=0)

            if env.index == 0:

                action = agent.select_action(state_list, evaluate=True)
            else:
                action = None

            # Execute actions
            #-------------------------------------------------------------------------            
        
            real_action = comm.scatter(action, root=0)    

            env.control_vel(real_action)
            rospy.sleep(0.001)

            ## Get reward and terminal state
            reward, done, result = env.get_reward_and_terminate(episode_steps)
            episode_steps += 1
            total_numsteps += 1
            episode_reward += reward

            # Get next state
            next_frame = env.get_laser_observation()
            left = frame_stack.popleft()
            
            frame_stack.append(next_frame)
            next_goal = np.asarray(env.get_local_goal())
            next_speed = np.asarray(env.get_self_speed())
            next_state = [frame_stack, next_goal, next_speed]

            r_list = comm.gather(reward, root=0)
            done_list = comm.gather(done, root=0)
            next_state_list = comm.gather(next_state, root=0)

            state = next_state  

        if total_numsteps > args.num_steps:
            break
        
        if env.index == 0:
            writer.add_scalar('reward/train', episode_reward, i_episode)

        distance = np.sqrt((env.goal_point[0] - env.init_pose[0])**2 + (env.goal_point[1]-env.init_pose[1])**2)
        print("Env: {}, Goal: ({} , {}), Episode: {}, steps: {}, Reward: {}, Distance: {}, {}".format(env.index, round(env.goal_point[0],2), round(env.goal_point[1],2), i_episode+1, episode_steps, round(episode_reward, 2), round(distance, 2), result))

if __name__ == '__main__':
    comm = MPI.COMM_WORLD # There is one special communicator that exists when an MPI program starts, that contains all the processes in the MPI program. This communicator is called MPI.COMM_WORLD
    size = comm.Get_size() # The first of these is called Get_size(), and this returns the total number of processes contained in the communicator (the size of the communicator).
    rank = comm.Get_rank() # The second of these is called Get_rank(), and this returns the rank of the calling process within the communicator. Note that Get_rank() will return a different value for every process in the MPI program.
    print("MPI size=%d, rank=%d" % (size, rank))

    # Environment
    env = StageWorld(beam_num=args.laser_beam, index=rank, num_env=args.num_env)
    print("Ready to environment")
    
    reward = None
    if rank == 0:
        policy_path = 'a10_epi_0'
        #board_path = 'runs/r2_epi_0'
        # Agent num_frame_obs, num_goal_obs, num_vel_obs, action_space, args
        action_bound = [[0, 1], [-1, 1]] #### Action maximum, minimum values
        action_bound = spaces.Box(-1, +1, (2,), dtype=np.float32)
        agent = SAC(num_frame_obs=args.laser_hist, num_goal_obs=2, num_vel_obs=2, action_space=action_bound, args=args)

        if not os.path.exists(policy_path):
            os.makedirs(policy_path)

        file_policy = policy_path + '/policy_epi_1000.pth'
        file_critic_1 = policy_path + '/critic_1_epi_1000.pth'
        file_critic_2 = policy_path + '/critic_2_epi_1000.pth'

        if os.path.exists(file_policy):
            logger.info('###########################################')
            logger.info('############Loading Policy Model###########')
            logger.info('###########################################')
            state_dict = torch.load(file_policy)
            agent.policy.load_state_dict(state_dict)
        else:
            logger.info('###########################################')
            logger.info('############Start policy Training###########')
            logger.info('###########################################')

        if os.path.exists(file_critic_1):
            logger.info('###########################################')
            logger.info('############Loading critic_1 Model###########')
            logger.info('###########################################')
            state_dict = torch.load(file_critic_1)
            agent.critic_1.load_state_dict(state_dict)
        else:
            logger.info('###########################################')
            logger.info('############Start critic_1 Training###########')
            logger.info('###########################################')
    
        if os.path.exists(file_critic_2):
            logger.info('###########################################')
            logger.info('############Loading critic_2 Model###########')
            logger.info('###########################################')
            state_dict = torch.load(file_critic_2)
            agent.critic_2.load_state_dict(state_dict)
        else:
            logger.info('###########################################')
            logger.info('############Start critic_2 Training###########')
            logger.info('###########################################')    

    else:
        agent = None
        policy_path = None
        
    try:
        run(comm=comm, env=env, agent=agent, policy_path=policy_path, args=args)
    except KeyboardInterrupt:
        pass