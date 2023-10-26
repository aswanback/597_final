# ME597_Final_Project_S23
This project had three parts, all implemented in Python using ROS. 
## Task 1 - Autonomous Exploration
The robot (in simulation) is dropped into a random, closed environment (not infinite) and must map as much of it as possible in 10 minutes. This was accomplished using an occupancy grid and frontier exploration using K-means, then using A* to route to the most promising frontier. This achieved a performance of 4 minutes on the test map. 
## Task 2 - Navigation (static)
The robot, using the map it just created, needs to navigate to any arbitrary point on that map. This was implemented using A* as well, with much of the same architecture as part 1. Unfortunately due to grading requirements the task files could not share code so duplication was necessary. 

## Task 3 - Navigation (dynamic obstacles)
The robot now must navigate around the map where there are moving obstacles in its path. These were avoided by using a LIDAR and calculating a gradient away from such obstacles, effectively making "peaks" in the control input topography where obstacles are and "valleys" where the safe areas are, so the robot would naturally fall into the valleys, even when the path it was navigating was not in those valleys. This allowed it to dynamically veer off course when necessary and return to the path when possible.



# Given instructions for the assignment below:

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

