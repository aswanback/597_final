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
        # rospy.loginfo(f'origin: {self.origin}')
        x_offset, y_offset, w = self.origin
        theta = np.arccos(w) * 2  # Convert quaternion to angle
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
            self.thresh = 250# map_dict['occupied_thresh'] * 255
            self.origin = map_dict['origin']
            self.resolution = map_dict['resolution']
        self.map = cv2.imread(map_name+'.pgm', cv2.IMREAD_UNCHANGED)
        # self.map = cv2.resize(self.map, (200, 200), interpolation=cv2.INTER_AREA)
        cv2.threshold(self.map, self.thresh, 100, cv2.THRESH_BINARY_INV, dst=self.map)
        # self.map = cv2.rotate(self.map, cv2.ROTATE_90_COUNTERCLOCKWISE)
        # self.map = cv2.flip(self.map, 1)
        self.map = cv2.flip(self.map, 0)
        
        
    
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
                path_array[tup] = 200
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
        # rospy.loginfo("AStar: start: %s, end: %s", start, end)
        
        self.map_shape = np.array(self.mp.map.shape)
        start = self.mp.world_to_pixel(start)
        self.mp.map[start] = 150
        if isinstance(end, PoseStamped):
            end = self.mp.world_to_pixel(end)
            self.mp.map[end] = 250
        self.start = self.mp.find_closest_valid_point(start)
        self.end = self.mp.find_closest_valid_point(end)
        
        # rospy.loginfo(f'astar: {self.start}, {self.end}')
        
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
        # rospy.loginfo(f'astar path: {path}')
        poses = self.make_poses(path)
        return poses, dist*np.sqrt(2)