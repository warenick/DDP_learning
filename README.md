# DDP_learning


## Installation ##

### Environment Setup ###

We tested code on ubuntu 20.04 with ROS Noetic, but we didn't use specific features excluding Python3, and you can try other versions
First, you need to install Ubuntu 20.04 and ROS Noetic. You can find detailed instructions for installation on sites:

https://old-releases.ubuntu.com/releases/20.04/

http://wiki.ros.org/noetic/Installation/Ubuntu

### Cloning ###

This is ROS pkg. Because of this, you need to clone the project into the catkin workspace:
```
cd ~/catkin_ws/src
git clone https://github.com/warenick/DDP_learning.git
```

### Additional steps ###

The project uses only Pytho3. Because of this, you don't need to build it, but don't forget to make a source on catkin environment.
```
source ~/catkin_ws/devel/setup.bash
```
Also, you need to install additional Python3 packages via pip3
```
pip3 install numpy autograd
```
