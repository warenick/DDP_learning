# DDP_learning


## Installation ##

### Environment Setup ###

We tested code on ubuntu 20.04 with ROS Noetic, but we didn't use specific features excluding Python3, and you can try other versions
First, you need to install Ubuntu 20.04 and ROS Noetic. You can find detailed instructions for installation on sites:

https://old-releases.ubuntu.com/releases/20.04/

http://wiki.ros.org/noetic/Installation/Ubuntu

Also we use Pytorch with CPU calculations 

```
pip3 install torch
```

### Cloning ###

This is ROS pkg. Because of this, you need to clone the project into the catkin workspace:
```
cd ~/catkin_ws/src
git clone https://github.com/warenick/DDP_learning.git
cd ~/catkin_ws ; catkin_make ; source ./devel/setup.bash
```

<!-- ### Additional steps ### -->

## Quickstart ##

We have a launch file for starting preconfigured rviz with the demo of DDP.
```
roslaunch ddp_learning ddp.launch
```
After running this command, you will see something like this
![ddp](https://user-images.githubusercontent.com/7687321/138083649-c75b1a58-0373-4277-b804-a382ec4b3672.gif)


### Configuration and Development ###

We have two main runs - `run_solo.py` for debugging one agent and `run_crowd.py` for debugging several agents(crowd) in one scene. We have some prepared scenes for debugging different cases that are kept in `scripts/configs/` folder. 
Prepared scenes include Agents and optimizer descriptions. For shortness and readability, they have default parameters that may be set to Agents and optimizers.

Generally, for development, we use `roslaunch ddp_learning ddp.launch` that include rviz, some static_tf, map_server for map reading, and move_base for creating inflated costmap. But main run such `run_solo.py` or `run_crowd.py` we start by hands in the iterative development process.

Now we have several Agent options for trajectory optimisation - linear, ddp, social-ddp, costmap-ddp, social-costmap-ddp, but costmap cases still in developing.
