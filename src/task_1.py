#!/usr/bin/env python3

import cProfile
import time
from typing import List, Tuple
from tf.transformations import euler_from_quaternion

from matplotlib import pyplot as plt
import rospy
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Twist, Pose
import numpy as np
from visualization_msgs.msg import MarkerArray, Marker
# from a_star2 import Map, AStar
import tf
import cv2
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
            self.map[self.map < 0] = 50  # set all unknown cells to 255
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

        # Erode and dilate the unknown_mask
        unknown_mask = cv2.erode(unknown_mask, self.kernel(unknown_erode_size), iterations=1)
        unknown_mask = cv2.dilate(unknown_mask, self.kernel(unknown_erode_size), iterations=2)
        occupied_mask = cv2.dilate(occupied_mask, self.kernel(occupied_dilate_size), iterations=1)

        # Erode and dilate the free_mask
        free_mask = cv2.erode(free_mask, self.kernel(free_erode_size), iterations=1)
        free_mask = cv2.dilate(free_mask, self.kernel(free_dilate_size), iterations=1)

        # Combine the modified masks back into the original image format
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
        # Compute the coordinates of the center of the cell at (x, y)
        cell_size = self.resolution * self.downsize_factor
        x_center = (x + 0.5) * cell_size
        y_center = (y + 0.5) * cell_size
        
        # Compute the coordinates of the center of the grid in the world frame
        x_offset, y_offset, w = self.origin
        theta = np.arccos(w) * 2  # Convert quaternion to angle
        x_center_world = x_center * np.cos(theta) - y_center * np.sin(theta) + x_offset
        y_center_world = x_center * np.sin(theta) + y_center * np.cos(theta) + y_offset
        
        p = PoseStamped()
        p.pose.position.x = y_center_world
        p.pose.position.y = x_center_world
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
            self.thresh = map_dict['occupied_thresh'] * 255
            self.origin = map_dict['origin']
            self.resolution = map_dict['resolution']
        self.map = cv2.imread(map_name+'.pgm', cv2.IMREAD_GRAYSCALE)
        # self.map = cv2.resize(self.map, (200, 200), interpolation=cv2.INTER_AREA)
        cv2.threshold(self.map, self.thresh, 100, cv2.THRESH_BINARY_INV, dst=self.map)
        
    
    def __is_valid(self, coord):
        x, y = coord
        return 0 <= x < self.map.shape[0] and 0 <= y < self.map.shape[1] and self.map[x, y] == 0

    def find_closest_valid_point(self, goal_coord):
        # Check if the goal coordinate is already a valid coordinate
        if self.__is_valid(goal_coord):
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
                if new_coord not in visited and self.__is_valid(new_coord):
                    return new_coord
                if new_coord not in visited:
                    visited.add(new_coord)
                    queue.append(new_coord)
        rospy.logerr("map.find_closest_valid_point: no valid point found")
        return None
    # def find_closest_valid_point(self, goal_coord):
    #     # Check if the goal coordinate is already a valid coordinate
    #     if self.__is_valid(goal_coord):
    #         return tuple(goal_coord)

    #     # Use a DFS approach to find the closest valid pixel
    #     stack = [goal_coord]
    #     visited = set()
    #     moves = [(0, 1), (1, 0), (0, -1), (-1, 0), (-1, -1), (1, 1), (-1, 1), (1, -1)]

    #     while stack:
    #         coord = stack.pop()
    #         visited.add(coord)

    #         for move in moves:
    #             new_coord = coord[0] + move[0], coord[1] + move[1]
    #             if new_coord not in visited and self.__is_valid(new_coord):
    #                 return new_coord
    #             if new_coord not in visited:
    #                 visited.add(new_coord)
    #                 stack.append(new_coord)
    #     rospy.logerr("map.find_closest_valid_point: no valid point found")
    #     return None
    

    def display(self,path:Path=None):
        if self.map is None:
            raise Exception("Map.display: map is None")
        fig, ax = plt.subplots()
        if path is not None:
            path_array = copy(self.map)
            for p in path.poses:
                tup = self.world_to_pixel(p)
                path_array[tup] = 30
            data = path_array
        else:
            data = self.map
        
        data = np.rot90(data, k=3, axes=(0, 1))
        # data = np.fliplr(data)
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
        if isinstance(end, PoseStamped):
            end = self.mp.world_to_pixel(end)
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

    # def __get_f_score(self, node, parent=None) -> float:
    #     if node not in self.dist:
    #         self.dist[node] = np.Inf
    #     if node not in self.h:
    #         self.h[node] = (self.end[0]-node[0])**2 + (self.end[1]-node[1])**2
    #     # return self.dist[node]**2 + self.h[node], id(node) # A-star heuristic, distance + h (defined in __init__)
    #     penalty = 0
    #     # if parent:
    #     #     dir = self.dir_tuples[node[2]]
    #     #     if len(parent) == 3:
    #     #         parent_dir = self.dir_tuples[parent[2]]
    #     #         penalty = 1 - (parent_dir[0]*dir[0] + parent_dir[1]*dir[1]) / (parent_dir[2]*dir[2])
    #     # penalty *= 0.5*self.dist[node]
        
    #     # if parent is not None:
    #     #     current_direction = (node[0] - parent[0])/dir, (node[1]-parent[1])/dir
    #     #     direction_penalty = (1 - (parent_dir[0]*current_direction[0] + parent_dir[1]*current_direction[1]))) * 0.5  # Calculate penalty based on direction change
    #     # direction_penalty *= 1*self.dist[node]
    #     return self.dist[node] ** 2 + penalty + self.h[node], id(node) # A-star heuristic, distance + h (defined in __init__)

    # def get_children(self, coord: Tuple[int, int]) -> List[Tuple[int, int]]:
    #     # Calculate the absolute coordinates of the neighboring pixels
    #     coords = np.array(coord + (0,)) + self.dirs
    #     # Check if the coordinates are within the image bounds
    #     in_bounds_mask = np.all((coords[:, :2] >= 0) & (coords[:, :2] < self.map_shape), axis=1)
    #     # Filter out the non-zero neighbors and apply the in_bounds_mask
    #     zero_neighbors = self.mp.map[coords[in_bounds_mask, 0], coords[in_bounds_mask, 1]] == 0
    #     return coords[in_bounds_mask][zero_neighbors]
        
    #     # z = self.dirs[in_bounds_mask][zero_neighbors]
    #     # return [self.dir_tuples.index(tuple(f)) for f in z]
    
    # def get_children(self, coord: Tuple[int, int]) -> List[Tuple[int, int]]:
    #     # Calculate the absolute coordinates of the neighboring pixels
    #     coords = np.array(coord + (0,)) + self.dirs
    #     # Check if the coordinates are within the image bounds
    #     in_bounds_mask = np.all((coords[:, :2] >= 0) & (coords[:, :2] < self.map_shape), axis=1)

    #     # Get the indices of the in_bounds elements
    #     in_bounds_indices = np.where(in_bounds_mask)[0]

    #     # Filter out the non-zero neighbors and apply the in_bounds_mask
    #     zero_neighbors = self.mp.map[coords[in_bounds_mask, 0], coords[in_bounds_mask, 1]] == 0
        
    #     # Get the indices of the valid children
    #     valid_children_indices = in_bounds_indices[zero_neighbors]

    #     # Return the indices of the valid children in self.dirs
        
    #     return valid_children_indices

    # def solve(self):
    #     sn = self.start
    #     en = self.end
    #     self.dist[sn] = 0                       # set initial dist to zero
    #     heapq.heappush(self.q, (self.__get_f_score(sn), sn))   # add start node to priority queue
    #     rospy.loginfo(f"start: {sn} end: {en}")
    #     while len(self.q) > 0:                    # while there are nodes left to be searched in the queue:
    #         u:Tuple[int,int,int] = heapq.heappop(self.q)[1]          # get node with the lowest f score from priority queue
    #         if u[0] == en[0] and u[1] == en[1]:                 # if it's the end node, done
    #             break
    #         for (cx,cy,w) in self.get_children((u[0],u[1])):
    #             # c = tuple(idx[0],idx[1]), 0
    #             # w = idx[2]
    #             # c = u[0]+self.dir_tuples[idx][0], u[1]+ self.dir_tuples[idx][1], idx
    #             # w = self.dir_tuples[idx][2]
    #             c = (cx,cy)
    #             if u not in self.dist:
    #                 self.dist[u] = np.Inf
    #             if c not in self.dist:
    #                 self.dist[c] = np.Inf
    #             new_dist = self.dist[u] + w/12  # new distance including this child node in path
    #             if new_dist < self.dist[c]:  # if the new distance is better than the old one:
    #                 self.dist[c] = new_dist  # add new dist of c to the dictionary
    #                 self.via[c] = u     # add new node c with parent reference u
    #                 # heapq.heappush(self.q, (self.__get_f_score(c), c))   # add c to the priority queue with new f score
    #                 heapq.heappush(self.q, (self.__get_f_score(c, u), c))   # add c to the priority queue with new f score
    def __get_f_score(self, node:Tuple[int,int], parent_direction:Optional[np.array] = None) -> float:
        if node not in self.dist:
            self.dist[node] = np.Inf
        if node not in self.h:
            self.h[node] = (self.end[0]-node[0])**2 + (self.end[1]-node[1])**2
        # return self.dist[node]**2 + self.h[node], id(node) # A-star heuristic, distance + h (defined in __init__)
        if parent_direction is not None:
            current_direction = np.array(node) - np.array(self.via[node])
            current_direction = current_direction / np.linalg.norm(current_direction)
            direction_penalty = (1 - np.dot(parent_direction, current_direction)) * 0.5  # Calculate penalty based on direction change
        else:
            direction_penalty = 0
        direction_penalty *= 1*self.dist[node]
        return self.dist[node] ** 2 + direction_penalty + self.h[node], id(node) # A-star heuristic, distance + h (defined in __init__)

    def get_children(self, coord: Tuple[int, int]) -> List[Tuple[int, int]]:
        # Calculate the absolute coordinates of the neighboring pixels
        coords = np.array(coord + (0,)) + self.dirs
        # Check if the coordinates are within the image bounds
        in_bounds_mask = np.all((coords[:, :2] >= 0) & (coords[:, :2] < self.map_shape), axis=1)
        # Filter out the non-zero neighbors and apply the in_bounds_mask
        zero_neighbors = self.mp.map[coords[in_bounds_mask, 0], coords[in_bounds_mask, 1]] == 0
        return coords[in_bounds_mask][zero_neighbors]
    
    def solve(self):
        sn = self.start
        en = self.end
        self.dist[sn] = 0                       # set initial dist to zero
        heapq.heappush(self.q, (self.__get_f_score(sn), sn))   # add start node to priority queue
        rospy.loginfo('Astar.solve: running..')
        while len(self.q) > 0:                    # while there are nodes left to be searched in the queue:
            u:Tuple[int,int] = heapq.heappop(self.q)[1]          # get node with the lowest f score from priority queue
            if u[0] == en[0] and u[1] == en[1]:                 # if it's the end node, done
                break
            for (cx,cy,w) in self.get_children(u):
                c = (cx,cy)
                if u not in self.dist:
                    self.dist[u] = np.Inf
                if c not in self.dist:
                    self.dist[c] = np.Inf
                new_dist = self.dist[u] + w/12  # new distance including this child node in path
                if new_dist < self.dist[c]:  # if the new distance is better than the old one:
                    self.dist[c] = new_dist  # add new dist of c to the dictionary
                    self.via[c] = u     # add new node c with parent reference u
                    # heapq.heappush(self.q, (self.__get_f_score(c), c))   # add c to the priority queue with new f score
                    parent_direction = np.array(u) - np.array(self.via[u]) if u in self.via else None
                    heapq.heappush(self.q, (self.__get_f_score(c, parent_direction), c))   # add c to the priority queue with new f score
    
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


class Task1Node:
    def __init__(self, node_name):
        rospy.init_node(node_name, anonymous=True)
        self.rate = rospy.Rate(50)
        self.listener = tf.TransformListener()
        rospy.Timer(rospy.Duration(0.01), self.__timer_cbk)
        rospy.Subscriber("/map", OccupancyGrid, self.__grid_cb)
        # rospy.Subscriber('/odom', Odometry, self.__odom_cbk)
        
        self.selected_frontier_pub = rospy.Publisher('/selected_frontier',Marker,queue_size=2)
        self.frontiers_pub = rospy.Publisher("/frontiers", MarkerArray, queue_size=2)
        self.raw_frontiers_pub = rospy.Publisher("/raw_frontiers", MarkerArray, queue_size=2)
        self.path_pub = rospy.Publisher("/path", Path, queue_size=2)
        self.cmd_vel_pub = rospy.Publisher('cmd_vel', Twist, queue_size=2)
        self.ttbot_pub = rospy.Publisher('ttbot_pose', PoseStamped, queue_size=2)
        
        self.grid:OccupancyGrid = None
        self.resolution:float = None
        self.origin:Pose = None
        self.path:Path = None
        self.frontier = None
        self.ttbot_pose:PoseStamped = PoseStamped()
        self.ttbot_pose.header.frame_id = 'map'
        self.ttbot_pose.pose.orientation.w = 1.0
        self.ttbot_pose_is_none = True
        
        # self.heading_pid = PIDController(3,0,0.3, [-2,2])
        # self.distance_pid = PIDController(0.5,0,0.1,[-0.5,0.5], 0.2)
        self.heading_pid = PIDController(2.5,0,7, [-5,5])
        self.distance_pid = PIDController(0.8,0.1,0.4,[-1.1,1.1], 0.3)
        self.heading_tolerance = 10 # degrees
        self.currIdx = 0
        self.last_time = None
        
        self.k = 4 # kmeans
        self.frontier_downsample = 1
        self.replan_downsample = 2
        self.dilate_size = 13
        self.map:Map = None
        self.last = (None, None)
    
    # def __odom_cbk(self, data:Odometry):
    #     self.ttbot_pose.pose = data.pose.pose
        
    def __timer_cbk(self, event):
        if self.grid is None:
            # rospy.loginfo('Waiting for map or odom')
            return
        try:
            # get the transform from map to odom
            (position, heading) = self.listener.lookupTransform('/map', '/base_link', rospy.Time(0))
            self.ttbot_pose.pose.position.x = position[0]
            self.ttbot_pose.pose.position.y = position[1]
            self.ttbot_pose.pose.orientation.x = heading[0]
            self.ttbot_pose.pose.orientation.y = heading[1]
            self.ttbot_pose.pose.orientation.z = heading[2]
            self.ttbot_pose.pose.orientation.w = heading[3]
            self.ttbot_pub.publish(self.ttbot_pose)
            self.ttbot_pose_is_none = False
        except Exception as e:
            rospy.logerr(f'__timer_cbk: {e}')
            pass
    
    def __grid_cb(self, data:OccupancyGrid):
        self.grid = data
        self.map = Map(data, self.dilate_size)
        # self.map.downsize(self.replan_downsample)
        # self.map.display(delay=2)
        # self.replan()
   
    def find_frontiers(self, map):
        # Create a binary mask for free space (0) and unexplored cells (-1)
        free_space_mask = cv2.inRange(map, 0, 0)
        unexplored_mask = cv2.inRange(map, 50, 50)

        # Define the kernel for checking adjacency
        kernel = np.array([[1, 1, 1],
                        [1, 0, 1],
                        [1, 1, 1]], dtype=np.uint8)

        # Filter the free_space_mask with the kernel
        adjacent_free_space_mask = cv2.filter2D(free_space_mask, -1, kernel)

        # Threshold the adjacent_free_space_mask to binary values
        _, adjacent_free_space_binary = cv2.threshold(adjacent_free_space_mask, 1, 255, cv2.THRESH_BINARY)

        # Perform a bitwise AND operation between the adjacent free space mask and the unexplored cells mask
        frontier_mask = cv2.bitwise_and(adjacent_free_space_binary, unexplored_mask)
        
        # frontier_points = cv2.findNonZero(frontier_mask)
        # frontier_points = np.squeeze(frontier_points, axis=1)
        Y, X = np.where(frontier_mask > 0)
        # return np.flip(frontier_points, axis=1)
        return np.column_stack((Y,X))
   
    def kmeans(self,points:np.ndarray, k):
        # Set criteria and apply k-means
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100000, 10)
        _, labels, (centers) = cv2.kmeans(points.astype(np.float32), k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        # Calculate the size of each cluster as a fraction of the total samples
        unique_labels, counts = np.unique(labels, return_counts=True)
        centers = centers.astype(np.uint16)
        return [tuple(c) for c in centers], counts / np.sum(counts)
   
    def select_frontier(self, mp:Map, frontiers:List[Tuple[int,int]], cluster_sizes:List[int], current_position:PoseStamped):
        best_score = np.Inf
        best_frontier = None
        for cluster_size,frontier in zip(cluster_sizes,frontiers):
            t = time.time_ns()
            # path, dist = AStar(mp, current_position, frontier, self.frontier_downsample).run()
            # mp.display(path)
            # rospy.loginfo(f'select_frontier astar done in {(time.time_ns()-t)/1e9:.2f}s ')
            # if path is None:
            #     rospy.logerr('select_frontier: no path found')
            #     continue
            dist = np.linalg.norm(frontier - np.array([current_position.pose.position.x, current_position.pose.position.y]))
            if dist < best_score:
                best_score = dist
                best_frontier = frontier
        return best_frontier

    def get_frontier(self):
        t = time.time_ns()
        if self.ttbot_pose_is_none or self.grid is None:
            rospy.loginfo('solve: waiting for pose and grid')
            return None
        mp = Map(self.grid, self.dilate_size)
        raw_frontiers = self.find_frontiers(mp.map)
        if len(raw_frontiers) < self.k:
            return None
        rf = [tuple(x) for x in raw_frontiers]
        self.publish_raw_frontiers([mp.pixel_to_world(x, y) for x, y in rf])
        
        frontiers, sizes = self.kmeans(raw_frontiers, self.k)
        self.publish_frontiers([mp.pixel_to_world(x, y) for x, y in frontiers])
        
        frontier = self.select_frontier(mp, frontiers, sizes, self.ttbot_pose)
        if frontier is None:
            rospy.logerr('node.get_frontier: no frontier found')
            return None
        world_frontier = mp.pixel_to_world(frontier[0], frontier[1])
        self.selected_frontier_pub.publish(self.make_marker(world_frontier,0,rgb=(1,0,0)))
        
        # rospy.loginfo(f"Found frontier in {(time.time_ns() - t)/1e9:.1f}s")
        return frontier

    def replan(self):
        t = time.time_ns()
        if self.frontier is None:
            rospy.logerr('node.replan: no frontier')
            return
        if self.ttbot_pose_is_none:
            rospy.logerr('node.replan: no pose')
            return
        if self.map is None:
            rospy.logerr('node.replan: no map')
            return
        
        start = self.ttbot_pose # if self.path is None else self.path.poses[self.currIdx]
        # self.map.display()
        path, dist = AStar(self.map, start, self.frontier).run()
        if path is None:
            rospy.logerr('node.replan: no path found')
            return
        
        self.path = path
        self.currIdx = 0
        self.path_pub.publish(self.path)
        rospy.loginfo(f'replan: planned path in {(time.time_ns() - t)/1e9:.1f}s')
        
    def run(self):
        # time.sleep(10)
        while not rospy.is_shutdown():
            t = time.time_ns()
            if self.map is None or self.ttbot_pose_is_none:
                # rospy.loginfo('Waiting for grid and pose')
                continue
            
            if self.path is None or self.currIdx == -1:
                rospy.loginfo('Frontier reached')
                self.move_ttbot(0, 0)
                self.frontier = self.get_frontier()
                if self.frontier is None:
                    rospy.logerr('node.run: no frontiers')
                    continue
                self.replan()
            
            self.currIdx = self.get_path_idx(self.path, self.ttbot_pose, self.currIdx)
            if self.currIdx == -1:
                continue
            linear, angular = self.pid_controller(self.ttbot_pose, self.path.poses[self.currIdx], self.path.poses[-1])
            self.move_ttbot(linear, angular)
            
            self.rate.sleep()

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
    def pid_controller(self, current_pose: PoseStamped, goal_pose: PoseStamped, global_goal: PoseStamped):
        '''Return linear and angular velocity'''

        # Calculate distance and heading errors
        cp = current_pose.pose.position
        co = current_pose.pose.orientation
        gp = goal_pose.pose.position
        dist_error = np.sqrt((gp.x - cp.x)**2 + (gp.y - cp.y)**2)
        desired_heading = np.arctan2(gp.y - cp.y, gp.x - cp.x)
        heading_error = desired_heading - euler_from_quaternion([co.x, co.y, co.z, co.w])[2]

        # Check if at goal
        sgp = global_goal.pose.position
        # sgo = self.frontier.pose.orientation
        dist = np.sqrt((sgp.x - cp.x)**2 + (sgp.y - cp.y)**2)
        if dist < 0.15:
            rospy.loginfo('At goal, waiting')
            # Adjust to goal heading
            dist_error = 0
            # heading_error = euler_from_quaternion([sgo.x, sgo.y, sgo.z, sgo.w])[2] - euler_from_quaternion([co.x, co.y, co.z, co.w])[2]
        
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
        # rospy.loginfo(f'linear: {linear:.2f}, angular: {angular:.2f}')

    def make_marker(self, pose:PoseStamped, id=0, size=0.15, rgb=(0,0,0)):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.id = id
        marker.type = marker.SPHERE
        marker.action = marker.ADD
        marker.pose.position.x = pose.pose.position.x
        marker.pose.position.y = pose.pose.position.y
        marker.pose.position.z = 0
        marker.pose.orientation.w = 1.0
        marker.scale.x = marker.scale.y = marker.scale.z = size
        marker.color.a = 1.0
        marker.color.r = rgb[0]
        marker.color.g = rgb[1]
        marker.color.b = rgb[2]
        return marker
    def publish_frontiers(self, frontiers):
        marker_array = MarkerArray()
        marker_array.markers = [self.make_marker(frontier,idx,rgb=(0,0,1)) for idx,frontier in enumerate(frontiers)]
        self.frontiers_pub.publish(marker_array)  
    def publish_raw_frontiers(self, frontiers):
        marker_array = MarkerArray()
        marker_array.markers = [self.make_marker(frontier,idx,rgb=(0,1,1),size=0.05) for idx,frontier in enumerate(frontiers)]
        self.raw_frontiers_pub.publish(marker_array)       

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
        # rospy.loginfo(f'error: {error:.2f}, integral: {self.integral:.2f}, derivative: {derivative:.2f}, output: {output:.2f}')

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
    node = Task1Node(node_name='explore_node')
    try:
        node.run()
    except rospy.ROSInterruptException:
        rospy.logerr("Node interrupted")
