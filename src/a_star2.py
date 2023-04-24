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
from nav_msgs.msg import OccupancyGrid
import cv2
import numpy as np
import yaml

class Map():
    def __init__(self, map:Union[np.ndarray,str], dilate_size:int=7):
        if isinstance(map, np.ndarray):
            self.map = np.clip(map, 0, None).astype(np.uint8)
        elif isinstance(map, str):
            self.map = None
            self.__open_map(map)
        else:
            raise Exception("Map.__init__: invalid map type")
        self.dilate(dilate_size)
        self.downsize_factor = 1
    
    def downsize(self, downsize_factor):
        if downsize_factor != 1:
            self.map = cv2.resize(self.map, (self.map.shape[0]//downsize_factor, self.map.shape[1]//downsize_factor), interpolation=cv2.INTER_NEAREST)
        self.downsize_factor = downsize_factor
        
    def get_map(self):
        return self.map
        
    def dilate(self, size):
        kernel = np.ones((size, size), dtype=np.uint8)
        self.map = cv2.dilate(self.map, kernel, iterations=1)
    
    def erode(self, size):
        kernel = np.ones((size, size), dtype=np.uint8)
        self.map = cv2.erode(self.map, kernel, iterations=1)
    
    def __open_map(self, map_name):
        with open(map_name + '.yaml', 'r') as f:
            map_dict = yaml.safe_load(f)
            self.thresh = map_dict['occupied_thresh'] * 255
        self.map = cv2.imread(map_name+'.pgm', cv2.IMREAD_GRAYSCALE)
        self.map = cv2.resize(self.map, (200, 200), interpolation=cv2.INTER_AREA)
        cv2.threshold(self.map, self.thresh, 255, cv2.THRESH_BINARY_INV, dst=self.map)
    
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
        rospy.logerr("No valid point found")
        return None
    
    def display(self,path=None):
        if self.map is None:
            raise Exception("Map.display: map is None")
        fig, ax = plt.subplots()
        if path is not None:
            path_array = copy(self.map)
            for tup in path:
                path_array[tup] = 100
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
        plt.show()

class AStar():
    def __init__(self, map:Map, start:Tuple[int,int], end:Tuple[int,int], downsize_factor:int=1):
        self.m:Map = map
        self.map:np.ndarray = map.get_map()
        self.q:List[Tuple(int,int)] = []
        self.dist = {}                  
        self.h = {}                     
        self.via = {}
        if downsize_factor != 1:
            self.m.downsize(downsize_factor)
        self.end = self.m.find_closest_valid_point(self.downsize(end))
        self.start = self.m.find_closest_valid_point(self.downsize(start))
        if not (isinstance(self.start[0], (int, np.integer)) and isinstance(self.start[1], (int, np.integer)) and isinstance(self.end[0], (int, np.integer)) and isinstance(self.end[1], (int, np.integer))):
            raise Exception(f"AStar.__init__: start or end is not an int tuple. start: {start}  end:{end}")
        sqrt2 = 17
        one = 12
        sqrt5 = 27
        self.dirs = np.array([
            [1, 1, sqrt2], [1, -1, sqrt2], [-1, 1, sqrt2], [-1, -1, sqrt2],
            [1, 0, one], [0, 1, one], [-1, 0, one], [0, -1, one],
            [1,2,sqrt5],[2,1,sqrt5],[-1,2,sqrt5],[-2,1,sqrt5],[1,-2,sqrt5],[2,-1,sqrt5],[-1,-2,sqrt5],[-2,-1,sqrt5]
        ]).astype(int)
        self.map_shape = np.array(self.map.shape)
    
    def downsize(self, point):
        return tuple(np.array(point) // self.m.downsize_factor)
    
    def upsize(self, point):
        return tuple(np.array(point) * self.m.downsize_factor)

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
        direction_penalty *= 0.2*self.dist[node]
        return (self.dist[node] + direction_penalty) ** 2 + self.h[node], id(node) # A-star heuristic, distance + h (defined in __init__)

    
    def get_children(self, coord: Tuple[int, int]) -> List[Tuple[int, int]]:
        # Calculate the absolute coordinates of the neighboring pixels
        coords = np.array(coord + (0,)) + self.dirs
        # Check if the coordinates are within the image bounds
        in_bounds_mask = np.all((coords[:, :2] >= 0) & (coords[:, :2] < self.map_shape), axis=1)
        # Filter out the non-zero neighbors and apply the in_bounds_mask
        zero_neighbors = self.map[coords[in_bounds_mask, 0], coords[in_bounds_mask, 1]] == 0
        return coords[in_bounds_mask][zero_neighbors]

    def solve(self):
        sn = self.start
        en = self.end
        self.dist[sn] = 0                       # set initial dist to zero
        heapq.heappush(self.q, (self.__get_f_score(sn), sn))   # add start node to priority queue
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
    
    def run(self):
        self.solve()
        try:
            path, dist = self.reconstruct_path()
        except KeyError as e:
            rospy.loginfo(f'astar.run: no path found, outside bounds ({e})')
            return None, np.Inf
        path = self.collapse_path(path)
        if self.m.downsize_factor is not None:
            path = np.array(path) * self.m.downsize_factor
        return path, dist