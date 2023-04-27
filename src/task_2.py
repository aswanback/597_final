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
from typing import Dict, List, Optional, Tuple, Union
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
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Twist, Pose
from nav_msgs.msg import OccupancyGrid, Path
import cv2
import numpy as np
import yaml
class Map():
    def __init__(self, grid:Union[OccupancyGrid,str], dilate_size=3):
        if isinstance(grid, OccupancyGrid):
            self.origin = grid.info.origin.position.x, grid.info.origin.position.y, grid.info.origin.orientation.w
            self.resolution = grid.info.resolution
            self.map = np.array(grid.data).reshape((grid.info.height, grid.info.width))
            self.map[self.map < 0] = 50  
            self.map = self.map.astype(np.uint8)
        elif isinstance(grid, str):
            self.map = None
            self.__open_map(grid)
        else:
            raise Exception("Map.__init__: invalid map type")
        unknown_erode_size = 3
        occupied_dilate_size = dilate_size
        free_erode_size = 3
        free_dilate_size = 3
        occupied_mask = cv2.inRange(self.map, 100, 100)
        unknown_mask = cv2.inRange(self.map, 50, 50)
        free_mask = cv2.inRange(self.map, 0, 0)
        unknown_mask = cv2.erode(unknown_mask, self.kernel(unknown_erode_size), iterations=1)
        unknown_mask = cv2.dilate(unknown_mask, self.kernel(unknown_erode_size), iterations=2)
        occupied_mask = cv2.dilate(occupied_mask, self.kernel(occupied_dilate_size), iterations=1)
        free_mask = cv2.erode(free_mask, self.kernel(free_erode_size), iterations=1)
        free_mask = cv2.dilate(free_mask, self.kernel(free_dilate_size), iterations=1)
        modified_map = np.zeros_like(self.map)
        modified_map[free_mask > 0] = 0
        modified_map[unknown_mask > 0] = 50
        modified_map[occupied_mask > 0] = 100
        self.map = modified_map
        self.downsize_factor = 1
    def kernel(self, size):
        return np.ones((size, size), dtype=np.uint8)
    def downsize(self, downsize_factor):
        if downsize_factor > min(self.map.shape[0], self.map.shape[1]):
            downsize_factor = min(self.map.shape[0], self.map.shape[1])
        if downsize_factor != 1:
            self.map = cv2.resize(self.map, (self.map.shape[0]//downsize_factor, self.map.shape[1]//downsize_factor), interpolation=cv2.INTER_NEAREST)
        self.downsize_factor = downsize_factor
    def pixel_to_world(self, x: int, y: int) -> PoseStamped:
        cell_size = self.resolution * self.downsize_factor
        x_center = (x + 0.5) * cell_size
        y_center = (y + 0.5) * cell_size
        x_offset, y_offset, w = self.origin
        theta = np.arccos(w) * 2  
        x_center_world = x_center * np.cos(theta) - y_center * np.sin(theta) - x_offset
        y_center_world = x_center * np.sin(theta) + y_center * np.cos(theta) - y_offset
        p = PoseStamped()
        p.pose.position.x = -y_center_world
        p.pose.position.y = -x_center_world
        p.pose.orientation.w = 1
        return p
    def world_to_pixel(self, pose:PoseStamped) -> tuple:
        origin_x, origin_y, w = self.origin
        resolution = self.resolution * self.downsize_factor
        pixel_x = int((pose.pose.position.x - origin_x) / resolution)
        pixel_y = int((pose.pose.position.y - origin_y) / resolution)
        return (pixel_y, pixel_x)
    def get_map(self):
        return self.map
    def __open_map(self, map_name):
        with open(map_name + '.yaml', 'r') as f:
            map_dict = yaml.load(f)
            self.thresh = 250
            self.origin = map_dict['origin']
            self.resolution = map_dict['resolution']
        self.map = cv2.imread(map_name+'.pgm', cv2.IMREAD_UNCHANGED)
        cv2.threshold(self.map, self.thresh, 100, cv2.THRESH_BINARY_INV, dst=self.map)
        self.map = cv2.flip(self.map, 0)
    def __is_valid(self, coord):
        x, y = coord
        return 0 <= x < self.map.shape[0] and 0 <= y < self.map.shape[1] and self.map[x, y] == 0
    def find_closest_valid_point(self, goal_coord):
        if self.__is_valid(goal_coord):
            return tuple(goal_coord)
        queue = deque([goal_coord])
        visited = set()
        moves = [(0, 1), (1, 0), (0, -1), (-1, 0), (-1, -1), (1, 1), (-1, 1), (1, -1)]
        while queue:
            coord = queue.popleft()
            visited.add(coord)
            for move in moves:
                new_coord = coord[0] + move[0], coord[1] + move[1]
                if new_coord not in visited and self.__is_valid(new_coord):
                    return new_coord
                if new_coord not in visited:
                    visited.add(new_coord)
                    queue.append(new_coord)
        rospy.logerr("map.find_closest_valid_point: no valid point found")
        return None
    def display(self,path:Path=None):
        if self.map is None:
            raise Exception("Map.display: map is None")
        fig, ax = plt.subplots()
        if path is not None:
            path_array = copy(self.map)
            for p in path.poses:
                tup = self.world_to_pixel(p)
                path_array[tup] = 200
            data = path_array
        else:
            data = self.map
        data = np.rot90(data, k=3, axes=(0, 1))
        ax.imshow(data)
        nonzero_indices = np.nonzero(data)
        b = 5
        x_min, x_max = np.min(nonzero_indices[1] - b), np.max(nonzero_indices[1] + b)
        y_min, y_max = np.min(nonzero_indices[0] - b), np.max(nonzero_indices[0] + b)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        fig.colorbar(ax.get_images()[0], ax=ax)
        plt.show()
class AStar():
    def __init__(self, mp:Map, start:PoseStamped, end:Union[PoseStamped, Tuple[int,int]]):
        self.mp:Map = mp
        self.q:List[Tuple(int,int)] = []
        self.dist = {}                  
        self.h = {}                     
        self.via = {}
        self.map_shape = np.array(self.mp.map.shape)
        start = self.mp.world_to_pixel(start)
        self.mp.map[start] = 150
        if isinstance(end, PoseStamped):
            end = self.mp.world_to_pixel(end)
            self.mp.map[end] = 250
        self.start = self.mp.find_closest_valid_point(start)
        self.end = self.mp.find_closest_valid_point(end)
        sqrt2 = 17
        one = 12
        sqrt5 = 27
        self.dirs = np.array([
            [1, 1, sqrt2], [1, -1, sqrt2], [-1, 1, sqrt2], [-1, -1, sqrt2],
            [1, 0, one], [0, 1, one], [-1, 0, one], [0, -1, one],
            [1,2,sqrt5],[2,1,sqrt5],[-1,2,sqrt5],[-2,1,sqrt5],[1,-2,sqrt5],[2,-1,sqrt5],[-1,-2,sqrt5],[-2,-1,sqrt5]
        ]).astype(int)
        self.dir_tuples = [tuple(d) for d in self.dirs]
    def __get_f_score(self, node:Tuple[int,int], parent_direction:Optional[np.array] = None) -> float:
        if node not in self.dist:
            self.dist[node] = np.Inf
        if node not in self.h:
            self.h[node] = (self.end[0]-node[0])**2 + (self.end[1]-node[1])**2
        if parent_direction is not None:
            current_direction = np.array(node) - np.array(self.via[node])
            current_direction = current_direction / np.linalg.norm(current_direction)
            direction_penalty = (1 - np.dot(parent_direction, current_direction)) * 0.5  
        else:
            direction_penalty = 0
        direction_penalty *= 1*self.dist[node]
        return self.dist[node] ** 2 + direction_penalty + self.h[node], id(node) 
    def get_children(self, coord: Tuple[int, int]) -> List[Tuple[int, int]]:
        coords = np.array(coord + (0,)) + self.dirs
        in_bounds_mask = np.all((coords[:, :2] >= 0) & (coords[:, :2] < self.map_shape), axis=1)
        zero_neighbors = self.mp.map[coords[in_bounds_mask, 0], coords[in_bounds_mask, 1]] == 0
        return coords[in_bounds_mask][zero_neighbors]
    def solve(self):
        sn = self.start
        en = self.end
        self.dist[sn] = 0                       
        heapq.heappush(self.q, (self.__get_f_score(sn), sn))   
        rospy.loginfo('Astar.solve: running..')
        while len(self.q) > 0:                    
            u:Tuple[int,int] = heapq.heappop(self.q)[1]          
            if u[0] == en[0] and u[1] == en[1]:                 
                break
            for (cx,cy,w) in self.get_children(u):
                c = (cx,cy)
                if u not in self.dist:
                    self.dist[u] = np.Inf
                if c not in self.dist:
                    self.dist[c] = np.Inf
                new_dist = self.dist[u] + w/12  
                if new_dist < self.dist[c]:  
                    self.dist[c] = new_dist  
                    self.via[c] = u     
                    parent_direction = np.array(u) - np.array(self.via[u]) if u in self.via else None
                    heapq.heappush(self.q, (self.__get_f_score(c, parent_direction), c))   
    def reconstruct_path(self):
        sn = self.start
        u = en = self.end                
        path = [u]                  
        while not (u[0] == sn[0] and u[1] == sn[1]):       
            u = self.via[u]         
            path.insert(0, u)          
        return path, self.dist[en]            
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
    def make_poses(self,raw_path):
        if raw_path is None:
            return
        path = Path()
        path.header.frame_id = 'map'
        path.poses = [self.mp.pixel_to_world(*coord) for coord in raw_path]
        return path
    def run(self):
        self.solve()
        try:
            path, dist = self.reconstruct_path()
        except KeyError as e:
            rospy.loginfo(f'astar.run: no path found, outside bounds ({e})')
            return None, np.Inf
        path = self.collapse_path(path)
        poses = self.make_poses(path)
        return poses, dist*np.sqrt(2)
class Navigation:
    def __init__(self, node_name='navigation'):
        rospy.init_node(node_name, anonymous=True)
        self.rate = rospy.Rate(50)
        rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.__goal_pose_cbk, queue_size=1)
        rospy.Subscriber('/amcl_pose', PoseWithCovarianceStamped, self.__ttbot_pose_cbk, queue_size=10)
        rospy.Subscriber('/odom', Odometry, self.__odom_cbk)
        self.path_pub = rospy.Publisher('global_plan', Path, queue_size=2)
        self.cmd_vel_pub = rospy.Publisher('cmd_vel', Twist, queue_size=2)
        self.odom_set_pub = rospy.Publisher('/initialpose',PoseWithCovarianceStamped, queue_size=2)
        self.pose_pub = rospy.Publisher('est_ttbot_pose', PoseStamped, queue_size=2)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        self.my_path_pub = rospy.Publisher('/path_topic',Path,queue_size=2)
        p = Path()
        p.header.frame_id = 'map'
        self.my_path_pub.publish(p)
        rospack = rospkg.RosPack()
        pkgpath = rospack.get_path("final_project")
        t = time.time_ns()
        self.mp = Map(f'{pkgpath}/maps/map', 14)
        t2 = time.time_ns()
        rospy.loginfo(f'Created map in {(t2-t)/1e6} ms')
        self.path = Path()
        self.goal_pose:PoseStamped = None
        self.ttbot_pose:PoseStamped = PoseStamped()
        self.ttbot_pose.header.frame_id = 'map'
        self.heading_pid = PIDController(3,0,0.3, [-2,2])
        self.distance_pid = PIDController(0.5,0,0.1,[-0.5,0.5], 0.2)
        self.heading_tolerance = 10
        self.linear = 0
        self.angular = 0
        self.confident = False
        self.currIdx = 0
        self.last_time = None
        self.dt = None
        self.last_conf_time = None
    def __odom_cbk(self, data:Odometry):
        ''' Callback to catch the position of the vehicle from odom.'''
        if not self.ttbot_pose:
            self.ttbot_pose = PoseStamped()
        self.ttbot_pose.pose = data.pose.pose
        self.pose_pub.publish(self.ttbot_pose)
    def __goal_pose_cbk(self, data:PoseStamped):
        """! Callback to catch the goal pose.
        @param  data    PoseStamped object from RVIZ.
        @return None.
        """
        self.goal_pose = data
        gp = self.goal_pose.pose.position
        rospy.loginfo(f'new goal pose:({gp.x:.2f},{gp.y:.2f}), pix:{self.mp.world_to_pixel(self.goal_pose)}')
    def __ttbot_pose_cbk(self, data:PoseWithCovarianceStamped):
        """! Callback to catch the position of the vehicle.
        @param  data    PoseWithCovarianceStamped object from amcl.
        @return None.
        """
        self.ttbot_pose.pose = data.pose.pose
        cov = np.array(data.pose.covariance)
        self.confident = True
        for i in cov:
            if abs(i) > 0.01:
                self.confident = False
                break
            self.last_conf_time = rospy.Time.now()
        if not self.confident:
            if self.last_conf_time and rospy.Time.now() - self.last_conf_time > rospy.Duration(5):
                self.move_ttbot(np.random.random(), np.random.random())
                time.sleep(2)
                self.move_ttbot(0,0)
            rospy.loginfo(f'Locating...')
        else:
            t = TransformStamped()
            t.header.stamp = rospy.Time.now()
            t.header.frame_id = "base_link"
            t.child_frame_id = "odom"
            t.header.stamp = rospy.Time.now()
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
        x_mult, x_const, y_mult, y_const = self.get_lin_params()
        return int(x_mult * y + x_const), int(y_mult * x + y_const)
    def _screen_to_model(self, x_pix, y_pix):
        x_mult, x_const, y_mult, y_const = self.get_lin_params()
        return (y_pix - y_const) / y_mult, (x_pix - x_const) / x_mult
    def a_star_path_planner(self, start_pose:PoseStamped, end_pose:PoseStamped):
        """! A Star path planner.
        @param  start_pose    PoseStamped object containing the start of the path to be created.
        @param  end_pose      PoseStamped object containing the end of the path to be created.
        @return path          Path object containing the sequence of waypoints of the created path.
        """
        start_time = time.time_ns()
        path, dist = AStar(self.mp, start_pose, end_pose).run()
        if path is None:
            rospy.logerr('A* failed to find a path')
        else:
            self.my_path_pub.publish(path)
            rospy.loginfo(f'A* solved:{start_pose.pose.position.x:.3f},{start_pose.pose.position.y:.3f} ({self.mp.world_to_pixel(start_pose)}) > end:{end_pose.pose.position.x:.3f},{end_pose.pose.position.y:.3f} ({self.mp.world_to_pixel(end_pose)}) in {(time.time_ns() - start_time)/1e6:.2f}ms')
        return path
    def get_path_idx(self, path:Path, vehicle_pose:PoseStamped, currIdx:int):
        """! Path follower.
        @param  path                  Path object containing the sequence of waypoints of the created path.
        @param  vehicle_pose          PoseStamped object containing the current vehicle position.
        @return idx                   Position int the path pointing to the next goal pose to follow.
        """
        vp = vehicle_pose.pose.position
        p = path.poses[currIdx].pose.position
        gg = path.poses[-1].pose.position
        sqdist = (p.x - vp.x) ** 2 + (p.y - vp.y) ** 2   
        ggsqdist = (gg.x - vp.x) ** 2 + (gg.y - vp.y) ** 2
        if currIdx == 0 and len(path.poses) > 1:
            p_next = path.poses[currIdx + 1].pose.position
            sqdist_next = (p_next.x - vp.x) ** 2 + (p_next.y - vp.y) ** 2
            if sqdist_next < sqdist:
                currIdx = 1
        if sqdist < 0.1**2:
            return min(currIdx+1, len(path.poses) - 1)
        if ggsqdist < 0.4**2:
            return -1
        return currIdx
    def pid_controller(self, current_pose: PoseStamped, goal_pose: PoseStamped):
        '''Return linear and angular velocity'''
        cp = current_pose.pose.position
        co = current_pose.pose.orientation
        gp = goal_pose.pose.position
        dist_error = np.sqrt((gp.x - cp.x)**2 + (gp.y - cp.y)**2)
        desired_heading = np.arctan2(gp.y - cp.y, gp.x - cp.x)
        heading_error = desired_heading - euler_from_quaternion([co.x, co.y, co.z, co.w])[2]
        sgp = self.goal_pose.pose.position
        sgo = self.goal_pose.pose.orientation
        dist = np.sqrt((sgp.x - cp.x)**2 + (sgp.y - cp.y)**2)
        if dist < 0.15:
            rospy.loginfo('At goal, waiting')
            dist_error = 0
            heading_error = euler_from_quaternion([sgo.x, sgo.y, sgo.z, sgo.w])[2] - euler_from_quaternion([co.x, co.y, co.z, co.w])[2]
        heading_error = (heading_error + np.pi) % (2 * np.pi) - np.pi
        dt = rospy.Time().now().to_sec() - self.last_time if self.last_time else None
        heading_control = self.heading_pid.update(heading_error, dt)
        if abs(heading_error) > self.heading_tolerance/180*np.pi:
            return 0, heading_control
        distance_control = self.distance_pid.update(dist_error, dt)
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
            if not self.confident:
                self.move_ttbot(0, 1.5)
                continue
            if self.goal_pose != old_goal_pose:
                self.currIdx = 0
                path = self.a_star_path_planner(self.ttbot_pose, self.goal_pose)
            old_goal_pose = self.goal_pose
            if path is None:
                self.move_ttbot(0, 0)
                continue
            self.currIdx = self.get_path_idx(path, self.ttbot_pose, self.currIdx)
            current_goal = path.poses[self.currIdx]
            self.linear, self.angular = self.pid_controller(self.ttbot_pose, current_goal)
            self.move_ttbot(self.linear, self.angular)
            if time.time() > start_time + (10 * 60) * 1000:  
                rospy.signal_shutdown("Navigation timeout")
        rospy.signal_shutdown("Finished Cleanly")
class PIDController:
    def __init__(self, kp, ki, kd, output_limits=None, min_output=None):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limits = output_limits
        self.min_output = min_output
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
        if self.min_output:
            if output > 0:
                output = max(output, self.min_output)
            elif output < 0:
                output = min(output, -self.min_output)
        return output
if __name__ == "__main__":
    nav = Navigation(node_name='Navigation')
    try:
        nav.run()
    except rospy.ROSInterruptException:
        print("program interrupted before completion", file=sys.stderr)