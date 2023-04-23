# ME597_Final_Project_S23

## Clone and build
git clone this repo to your workspace's src folder, then do `catkin build` or `catkin_make` in workspace directory.

## Set environment variable of the robot model we use for this project
`export TURTLEBOT3_MODEL=waffle`

## Source the package in the workspace directory
`source devel/setup.bash`

## Task 1 - Autonomous exploration
`roslaunch final_project task_1.launch`

After you have mapped the whole environment, save map in another terminal by 
`rosrun map_server map_saver -f map`

## Task 2 - Navigation in static environment
`roslaunch final_project task_2.launch`

## Task 3 - Navigation in environment with dynamic obstacles
`roslaunch final_project task_3.launch`

