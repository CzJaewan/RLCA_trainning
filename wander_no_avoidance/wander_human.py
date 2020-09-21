import os
import logging
import sys
import socket
import numpy as np
import rospy

from mpi4py import MPI

from collections import deque

from gym_stage_human import StageWorld

MAX_EPISODES = 1000000
LASER_BEAM = 512
LASER_HIST = 3
HORIZON = 128
GAMMA = 0.99
LAMDA = 0.95
BATCH_SIZE = 1024
EPOCH = 2
COEFF_ENTROPY = 5e-4
CLIP_VALUE = 0.1
NUM_ENV = 12
OBS_SIZE = 8
ACT_SIZE = 2
LEARNING_RATE = 5e-5

def run(comm, env):

    if env.index == 0:
        env.reset_world()
        


    for id in range(MAX_EPISODES):

        env.reset_pose()
        
        scaled_action = [[1,0],[1,0],[1,0]]

        rospy.sleep(0.1)

        terminate = False
    
        while not terminate and not rospy.is_shutdown():
            
            real_action = comm.scatter(scaled_action, root=0)

            env.control_vel(real_action)

            rospy.sleep(0.1)

            terminate = env.step()
        


if __name__ == '__main__':

    comm = MPI.COMM_WORLD

    rank = comm.Get_rank()
    size = comm.Get_size()

    print("go")

    env = StageWorld(8, index=rank, num_env=NUM_ENV)
    
    print("ENV")
 
    # torch.manual_seed(1)
    # np.random.seed(1)
    if rank == 0:

        print('####################################')
        print('############wander Start########$###')
        print('####################################')
        
    try:
        run(comm=comm, env=env)
    except KeyboardInterrupt:
        pass
