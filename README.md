# RLCA_trainning
## Folder description
- main branch : multiagent
- PPO_R : origin observation ( lidar, goal, speed ) + radius, lidar data ( lidar_(t-2), lidar_(t-1), lidar_(t)
- PPO_R_lidar : PPO_R_new + lidar data ( lidar_(t-10), lidar_(t-5), lidar_(t)
- PPO_R_new : PPO_R + collision lidar add
- PPO_no_update : Do not policy update (Purpose to use as a wander)
- wander : wander code, collision avoidance used scan data
- wander_no_avoidance : wander code
- worlds : stage model file

## Setting
### Stage & stageros & ppo train py
```
  sudo apt-get install ros-kinetic-stage
  cd /catkin_ws/src
  git clone https://github.com/Geonhee-LEE/rl-collision-avoidance
  git clone https://github.com/CzJaewan/RLCA_trainning
  cd /catkin_ws
  catkin_make
  cd /catkin_ws/src/rl-collision-avoidance
  git branch stageros_w
```

## Run
### stage_run
```
  roscore
  rosrun stage_ros_add_pose_and_crash stageros_w -u -n 6 /home/nscl/rl_ws/src/RLCA_trainning/worlds/servingbot_single_agent_world
```
### train
```
  mpiexec -np 1 python ppo_stage.py
```
