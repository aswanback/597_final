#!/bin/bash 

echo "Start running final project ..."

source /opt/ros/noetic/setup.bash
source /final_project_ws/devel/setup.bash
export TURTLEBOT3_MODEL=waffle

roslaunch final_project task_1_all.launch

echo "Final project finished!"
