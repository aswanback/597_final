#!/usr/bin/env python3

import sys
import numpy as np
import time
import rospy
from tf.transformations import euler_from_quaternion
from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Twist
from geometry_msgs.msg import TransformStamped
import tf2_ros
import rospkg

import heapq
from typing import Dict, List, Optional, Tuple
from graphviz import Graph
from PIL import Image, ImageOps 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import yaml
import pandas as pd
from copy import copy
import rospy
from collections import deque
import cv2

import cv2
import numpy as np
import yaml

class Map():
    def __init__(self, map_name):
        with open(map_name + '.yaml', 'r') as f:
            map_dict = yaml.safe_load(f)
            self.thresh = map_dict['occupied_thresh'] * 255
        img = cv2.imread(map_name+'.pgm', cv2.IMREAD_GRAYSCALE)
        # self.map = cv2.resize(img, (200, 200), interpolation=cv2.INTER_LINEAR)
        _, self.map = cv2.threshold(img, self.thresh, 255, cv2.THRESH_BINARY_INV)
    
    def get_map(self):
        return self.map

class MapProcessor():
    def __init__(self, map:np.ndarray):
        self.map = map
        self.inf_map = None
        
    def dilate(self, size):
        kernel = np.ones((size, size), dtype=np.uint8)
        self.inf_map = cv2.dilate(self.map.astype(np.uint8), kernel, iterations=1)
        return self.inf_map
    
    def is_valid(self, coord):
        x, y = coord
        return 0 <= x < self.inf_map.shape[0] and 0 <= y < self.inf_map.shape[1] and self.inf_map[x, y] == 0

    def find_closest_valid_point(self, goal_coord):
        # Check if the goal coordinate is already a valid coordinate
        if self.is_valid(goal_coord):
            return tuple(goal_coord)

        # Use a BFS approach to find the closest valid pixel
        queue = deque([goal_coord])
        visited = set()
        moves = [(0, 1), (1, 0), (0, -1), (-1, 0), (-1, -1), (1, 1), (-1, 1), (1, -1)]

        while queue:
            coord = queue.popleft()
            visited.add(coord)

            for move in moves:
                new_coord = coord[0] + move[0], coord[1] + move[1]
                if new_coord not in visited and self.is_valid(new_coord):
                    return new_coord
                if new_coord not in visited:
                    visited.add(new_coord)
                    queue.append(new_coord)
        return None

    def get_map(self):
        return self.inf_map if self.inf_map is not None else self.map
    
    def display(self,path=None):
        m = self.inf_map if self.inf_map is not None else self.map
        if path:
            path_array = copy(m)
            for tup in path:
                path_array[tup] = 0.5
            path_array = np.rot90(path_array, k=3, axes=(0, 1))
            plt.imshow(path_array)
        else:
            plt.imshow(np.rot90(m, k=3, axes=(0, 1)))
        plt.colorbar()
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()

class AStar():
    def __init__(self,map:Image, start:Tuple[int,int], end:Tuple[int,int]):
        self.map:Image = map
        self.q:List[Tuple(int,int)] = []
        self.dist = {}                  
        self.h = {}                     
        self.via = {}
        self.end = end
        self.start = start

    def __get_f_score(self, node:Tuple[int,int]) -> float:
        if node not in self.dist:
            self.dist[node] = np.Inf
        if node not in self.h:
            self.h[node] = (self.end[0]-node[0])**2 + (self.end[1]-node[1])**2
        return self.dist[node]**2 + self.h[node], id(node) # A-star heuristic, distance + h (defined in __init__)

    def get_neighbors(self, coord:Tuple[int,int]) -> List[Tuple[int,int]]:
        sqrt2 = np.sqrt(2)
        weights = np.array([sqrt2, 1, sqrt2, 1, 1, sqrt2, 1, sqrt2])
        # Define the relative coordinates of the 8 neighboring pixels
        relative_coords = np.array([
            [-1, -1],
            [-1,  0],
            [-1,  1],
            [ 0, -1],
            [ 0,  1],
            [ 1, -1],
            [ 1,  0],
            [ 1,  1]
        ])
        # Calculate the absolute coordinates of the neighboring pixels
        neighbors = np.array(coord) + relative_coords
        # Check if the coordinates are within the image bounds
        in_bounds = np.all((neighbors >= 0) & (neighbors < np.array(self.map.shape)), axis=1)
        # Extract the in-bound neighboring pixel coordinates
        valid_neighbors = neighbors[in_bounds]
        # Get the values of the in-bound neighboring pixels in the binary image
        neighbor_values = self.map[valid_neighbors[:, 0], valid_neighbors[:, 1]]
        # Filter out the non-zero neighbors
        non_zero_mask = neighbor_values == 0
        # Return the valid neighbors and their corresponding weights
        return [tuple(item) for item in valid_neighbors[non_zero_mask]], weights[in_bounds][non_zero_mask]
    
    def solve(self):
        sn = self.start
        en = self.end
        self.dist[sn] = 0                       # set initial dist to zero
        heapq.heappush(self.q, (self.__get_f_score(sn), sn))   # add start node to priority queue
        while len(self.q) > 0:                    # while there are nodes left to be searched in the queue:
            u:Tuple[int,int] = heapq.heappop(self.q)[1]          # get node with the lowest f score from priority queue
            if u[0] == en[0] and u[1] == en[1]:                 # if it's the end node, done
                break
            children,weights = self.get_neighbors(u)  # get all children of the node
            for c, w in zip(children,weights):      # for each connected child of the node:
                if u not in self.dist:
                    self.dist[u] = np.Inf
                if c not in self.dist:
                    self.dist[c] = np.Inf
                new_dist = self.dist[u] + w  # new distance including this child node in path
                if new_dist < self.dist[c]:  # if the new distance is better than the old one:
                    self.dist[c] = new_dist  # add new dist of c to the dictionary
                    self.via[c] = u     # add new node c with parent reference u
                    heapq.heappush(self.q, (self.__get_f_score(c), c))   # add c to the priority queue with new f score
                    
    def reconstruct_path(self):
        sn = self.start
        u = en = self.end                # end key index to start rewind at
        path = [u]                  # initial (backwards) path starts at u
        while not (u[0] == sn[0] and u[1] == sn[1]):       # while not back to start:   
            u = self.via[u]         # go back one step in the chain
            path.insert(0, u)          # add to the list of nodes
        return path, self.dist[en]            # return optimal path and least dist
    
    def collapse_path(self, path):
        ''' Collapses a path by removing redundant waypoints'''
        if len(path) < 3:
            return path
        
        result = [path[0], path[1]]
        
        for i in range(2, len(path)):
            prev = result[-1]
            prev2 = result[-2]
            curr = path[i]
            
            if ((prev[0] - prev2[0]) * (curr[1] - prev[1])) == ((curr[0] - prev[0]) * (prev[1] - prev2[1])):
                result[-1] = curr
            else:
                result.append(curr)
        return result
    
    def run(self):
        self.solve()
        try:
            path, dist = self.reconstruct_path()
        except KeyError:
            rospy.loginfo('No path found, outside bounds')
            return None, np.Inf
        path = self.collapse_path(path)
        return path,dist

class Navigation:
    def __init__(self, node_name='Navigation'):
        # ROS node initialization
        rospy.init_node(node_name, anonymous=True)
        self.rate = rospy.Rate(50)
        # Subscribers
        rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.__goal_pose_cbk, queue_size=1)
        rospy.Subscriber('/amcl_pose', PoseWithCovarianceStamped, self.__ttbot_pose_cbk, queue_size=10)
        rospy.Subscriber('/odom', Odometry, self.__odom_cbk)
        # Publishers
        self.path_pub = rospy.Publisher('global_plan', Path, queue_size=2)
        self.cmd_vel_pub = rospy.Publisher('cmd_vel', Twist, queue_size=2)
        self.odom_set_pub = rospy.Publisher('/initialpose',PoseWithCovarianceStamped, queue_size=2)
        self.pose_pub = rospy.Publisher('est_ttbot_pose', PoseStamped, queue_size=2)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        self.my_path_pub = rospy.Publisher('/path_topic',Path,queue_size=2)
        p = Path()
        p.header.frame_id = 'map'
        self.my_path_pub.publish(p)
        # Map and tree creation
        rospack = rospkg.RosPack()
        pkgpath = rospack.get_path("final_project")
        
        t = time.time_ns()
        map = Map(f'{pkgpath}/maps/map').get_map()
        self.mp = MapProcessor(map)
        self.mp.dilate(11)
        # self.mp.display()
        t2 = time.time_ns()
        rospy.loginfo(f'Created map in {(t2-t)/1e6} ms')
        # self.mp.display()
         
        # Create path planning variables
        self.path = Path()
        self.goal_pose:PoseStamped = None
        
        self.ttbot_pose:PoseStamped = PoseStamped()
        self.ttbot_pose.header.frame_id = 'map'
        
        self.heading_pid = PIDController(3,0,0.3, [-2,2])
        self.distance_pid = PIDController(0.5,0,0.1,[-1,1])
        self.heading_tolerance = 10
        
        self.linear = 0
        self.angular = 0

        self.confident = False
        self.currIdx = 0
        self.last_time = None
        self.dt = None

    def __odom_cbk(self, data:Odometry):
        ''' Callback to catch the position of the vehicle from odom.'''
        self.ttbot_pose.pose = data.pose.pose
        self.pose_pub.publish(self.ttbot_pose)
    def __goal_pose_cbk(self, data:PoseStamped):
        """! Callback to catch the goal pose.
        @param  data    PoseStamped object from RVIZ.
        @return None.
        """
        self.goal_pose = data
        gp = self.goal_pose.pose.position
        rospy.loginfo(f'new goal pose:({gp.x:.2f},{gp.y:.2f}), pix:{self._model_to_screen(gp.x,gp.y)}')
    def __ttbot_pose_cbk(self, data:PoseWithCovarianceStamped):
        """! Callback to catch the position of the vehicle.
        @param  data    PoseWithCovarianceStamped object from amcl.
        @return None.
        """
        self.ttbot_pose.pose = data.pose.pose
        # set confidence true if every cov value is < 0.01
        cov = np.array(data.pose.covariance)
        # self.confident = np.any(cov[abs(cov) > 0.01])
        self.confident = True
        for i in cov:
            if abs(i) > 0.01:
                self.confident = False
                break

        if not self.confident:
            rospy.loginfo(f'Locating...')
        else:
            # rospy.loginfo(f'pose update')

            # Update base_link to odom transform instead of publishing to /initialpose
            t = TransformStamped()
            t.header.stamp = rospy.Time.now()
            t.header.frame_id = "base_link"
            t.child_frame_id = "odom"
            t.transform.translation.x = -data.pose.pose.position.x
            t.transform.translation.y = -data.pose.pose.position.y
            t.transform.translation.z = -data.pose.pose.position.z
            t.transform.rotation = data.pose.pose.orientation

            self.tf_broadcaster.sendTransform(t)
    
    def get_lin_params(self):
        x_pix_min = 72
        x_pix_max = 119
        y_pix_min = 80
        y_pix_max = 127

        x_min = -2.29
        x_max = 2.36
        y_min = -2.28
        y_max = 2.30

        x_const = (x_pix_min + x_pix_max) / 2
        y_const = (y_pix_min + y_pix_max) / 2
        x_mult = -(x_pix_max - x_pix_min) / (x_max - x_min)
        y_mult = (y_pix_max - y_pix_min) / (y_max - y_min)

        return x_mult, x_const, y_mult, y_const
    def _model_to_screen(self, x, y):
        # linear transformation with y and x switched for orientation
        x_mult, x_const, y_mult, y_const = self.get_lin_params()
        return int(x_mult * y + x_const), int(y_mult * x + y_const)
    def _screen_to_model(self, x_pix, y_pix):
        # linear de-transformation of _model_to_screen
        x_mult, x_const, y_mult, y_const = self.get_lin_params()
        return (y_pix - y_const) / y_mult, (x_pix - x_const) / x_mult
    def find_closest_valid_point(self, valid_coords, goal_coord)->str:
        # Convert the coords from 'x,y' strings to tuples
        valid_coords = [tuple(map(int, coord.split(','))) for coord in valid_coords]
        goal_coord = tuple(map(int, goal_coord.split(',')))
        valid_set = set(valid_coords)
        
        # Check if the goal coordinate is already a valid coordinate
        if goal_coord in valid_set:
            return goal_coord
        
        # Convert the valid coordinates to a NumPy array
        valid_coords_array = np.array(valid_coords)
        rospy.loginfo(f'goal:{goal_coord}')
        # Compute the squared Euclidean distances from the goal coordinate to each valid coordinate
        distances = np.sum((valid_coords_array - goal_coord)**2, axis=1)
        # Find the index of the minimum distance
        min_distance_idx = np.argmin(distances)
        # Return the closest valid coordinate
        return f'{valid_coords[min_distance_idx][0]},{valid_coords[min_distance_idx][1]}'

    
    def a_star_path_planner(self, start_pose:PoseStamped, end_pose:PoseStamped):
        """! A Star path planner.
        @param  start_pose    PoseStamped object containing the start of the path to be created.
        @param  end_pose      PoseStamped object containing the end of the path to be created.
        @return path          Path object containing the sequence of waypoints of the created path.
        """
        start_time = time.time_ns()
        # convert from PoseStamped to screen space A* format
        start = self._model_to_screen(start_pose.pose.position.x, start_pose.pose.position.y)
        end = self._model_to_screen(end_pose.pose.position.x, end_pose.pose.position.y)
        
        # solve and reconstruct astar
        start = self.mp.find_closest_valid_point(start)
        if start is None: return None
        new_end = self.mp.find_closest_valid_point(end)
        if new_end is None: return None
        if new_end != end:
            rospy.loginfo(f'A* unable to solve with invalid end point, using closest valid point instead: {new_end}')
            end = new_end
            gp = self.goal_pose.pose.position
            gp.x, gp.y = self._screen_to_model(*end)
        
        raw_path, dist = AStar(self.mp.inf_map, start, end).run()
        if raw_path is None:
            return None
        rospy.loginfo(f'{raw_path}')
        self.mp.display(raw_path)
        end_time = time.time_ns()
        
        # create Path object and fill, converted from A* solver format
        path = Path()
        path.header.frame_id = 'map'
        for coord in raw_path:
            x_pix, y_pix = coord
            p = PoseStamped()
            x, y = self._screen_to_model(x_pix, y_pix)
            p.pose.position.x = x
            p.pose.position.y = y
            path.poses.append(p)
        
        # publish path to RVIZ
        self.my_path_pub.publish(path)
        
        rospy.loginfo(f'A* solved:{start_pose.pose.position.x:.3f},{start_pose.pose.position.y:.3f} > end:{end_pose.pose.position.x:.3f},{end_pose.pose.position.y:.3f} in {(end_time - start_time)/1e6:.2f}ms')
        # self.mp.draw_path(raw_path)
        return path

    def get_path_idx(self, path:Path, vehicle_pose:PoseStamped, currIdx:int):
        """! Path follower.
        @param  path                  Path object containing the sequence of waypoints of the created path.
        @param  vehicle_pose          PoseStamped object containing the current vehicle position.
        @return idx                   Position int the path pointing to the next goal pose to follow.
        """
        vp = vehicle_pose.pose.position
        p = path.poses[currIdx].pose.position
        sqdist = (p.x - vp.x) ** 2 + (p.y - vp.y) ** 2   
        if sqdist < 0.1**2:
            return min(currIdx+1, len(path.poses) - 1)
        return currIdx

    def pid_controller(self, current_pose: PoseStamped, goal_pose: PoseStamped):
        '''Return linear and angular velocity'''

        # Calculate distance and heading errors
        cp = current_pose.pose.position
        co = current_pose.pose.orientation
        gp = goal_pose.pose.position
        dist_error = np.sqrt((gp.x - cp.x)**2 + (gp.y - cp.y)**2)
        desired_heading = np.arctan2(gp.y - cp.y, gp.x - cp.x)
        heading_error = desired_heading - euler_from_quaternion([co.x, co.y, co.z, co.w])[2]

        # Check if at goal
        sgp = self.goal_pose.pose.position
        sgo = self.goal_pose.pose.orientation
        dist = np.sqrt((sgp.x - cp.x)**2 + (sgp.y - cp.y)**2)
        if dist < 0.15:
            rospy.loginfo('At goal, waiting')
            # Adjust to goal heading
            dist_error = 0
            heading_error = euler_from_quaternion([sgo.x, sgo.y, sgo.z, sgo.w])[2] - euler_from_quaternion([co.x, co.y, co.z, co.w])[2]
        
        heading_error = (heading_error + np.pi) % (2 * np.pi) - np.pi
        dt = rospy.Time().now().to_sec() - self.last_time if self.last_time else None

        heading_control = self.heading_pid.update(heading_error, dt)
        if abs(heading_error) > self.heading_tolerance/180*np.pi:
            return 0, heading_control
        # PID controller for distance
        distance_control = self.distance_pid.update(dist_error, dt)

        # Update last time and return velocity commands
        self.last_time = rospy.Time().now().to_sec()
        return distance_control, heading_control
    
    def move_ttbot(self, linear, angular):
        msg = Twist()
        msg.linear.x = linear
        msg.angular.z = angular
        self.cmd_vel_pub.publish(msg)

    def run(self):
        """! Main loop of the node. You need to wait until a new pose is published, create a path and then
        drive the vehicle towards the final pose.
        @return none
        """
        path = None
        old_goal_pose = None
        start_time = time.time()
        while not rospy.is_shutdown():
            self.rate.sleep()

            # if not sure of localization, spin in a circle
            if not self.confident:
                self.move_ttbot(0, 1.5)
                continue
            
            # if the goal_pose has changed, replan A*
            if self.goal_pose != old_goal_pose:
                self.currIdx = 0
                path = self.a_star_path_planner(self.ttbot_pose, self.goal_pose)
                
            old_goal_pose = self.goal_pose

            # if no path is returned, stop moving
            if path is None:
                self.move_ttbot(0, 0)
                continue

            # find next point to go towards
            self.currIdx = self.get_path_idx(path, self.ttbot_pose, self.currIdx)
            current_goal = path.poses[self.currIdx]
            # rospy.loginfo(f'idx: {self.currIdx+1}/{len(path.poses)}')
            
            self.linear, self.angular = self.pid_controller(self.ttbot_pose, current_goal)
            # rospy.loginfo(f'linear:{self.linear:.3f}\tangular:{self.angular:.3f}')

            # move robot
            self.move_ttbot(self.linear, self.angular)

            # timeout of 10 minutes in case stuck
            if time.time() > start_time + (10 * 60) * 1000:  # timeout after 10 minutes
                rospy.signal_shutdown("Navigation timeout")

        rospy.signal_shutdown("Finished Cleanly")

class PIDController:
    def __init__(self, kp, ki, kd, output_limits=None):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limits = output_limits
        self.last_error = 0.0
        self.integral = 0.0

    def update(self, error, dt):
        derivative = (error - self.last_error) / dt if (dt is not None and dt != 0) else 0
        self.integral += error * dt if dt is not None else 0
        self.last_error = error

        output = self.kp * error + self.ki * self.integral + self.kd * derivative

        if self.output_limits:
            output = np.clip(output, self.output_limits[0], self.output_limits[1])
            if self.integral > 0 and output == self.output_limits[1]:
                self.integral = 0
            elif self.integral < 0 and output == self.output_limits[0]:
                self.integral = 0
        return output

if __name__ == "__main__":
    nav = Navigation(node_name='Navigation')
    try:
        nav.run()
    except rospy.ROSInterruptException:
        print("program interrupted before completion", file=sys.stderr)