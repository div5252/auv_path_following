# AUV Path Following using DQN based controller
Build on the UUV Simulator - https://github.com/uuvsimulator/uuv_simulator

## Requirements & Installation
1. Ubuntu machine, version 18.04
2. Python version 2.7
3. ROS Melodic 1.14 (http://wiki.ros.org/melodic/Installation/Ubuntu)
4. Gazebo 9
5. Torch 1.4.0

## Instruction for running
1. Clone https://github.com/div5252/auv_path_following and https://github.com/div5252/uuv_simulator in ```src``` folder of catkin workspace.
2. Build catkin workspace.
```sh
catkin init
catkin build
source devel/setup.bash
```
3. For running RL controller -
```sh
cd src/auv_path_following/rl_controller
bash run.sh
```

## Features
- PID (Proportional-Integral-Derivative) based controller
- DQN (Deep Q-Network) based controller
- Addition of current velocity
- Noise in IMU and DVL sensor measurements
- Plot of AUV's path

## Demonstration
The video demonstration of the controllers can be seen [here](https://drive.google.com/drive/folders/1iW5KxLySI7HdzcDCg64KdeqeSTThlZ3k?usp=sharing).
