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
        self.map = cv2.resize(img, (200, 200), interpolation=cv2.INTER_LINEAR)
        _, self.map = cv2.threshold(self.map, self.thresh, 255, cv2.THRESH_BINARY_INV)
    
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
        plt.ylim(65,140)
        plt.xlim(125+15,60+15)
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