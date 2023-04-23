#!/usr/bin/env python3

import time
from typing import List, Tuple
import rospy
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Twist, Pose
import numpy as np
from visualization_msgs.msg import MarkerArray, Marker
from a_star2 import MapProcessor, AStar
import tf

class Task1Node:
    def __init__(self, node_name):
        rospy.init_node(node_name, anonymous=True)
        self.rate = rospy.Rate(50)
        self.k = 5 #kmeans
        # rospy.Subscriber('/odom', Odometry, self.__odom_cbk)
        rospy.Timer(rospy.Duration(0.05), self.__timer_cbk)
        rospy.Subscriber("/map", OccupancyGrid, self.__grid_cb)
        self.listener = tf.TransformListener()
        self.frontier_pub = rospy.Publisher("/frontiers", MarkerArray, queue_size=1)
        self.path_pub = rospy.Publisher("/path", Path, queue_size=self.k)
        self.goal_pub = rospy.Publisher('/goal_frontier',Marker,queue_size=1)
        
        self.grid:np.ndarray = None
        self.resolution:float = None
        self.origin:Pose = None
        self.frontiers = None
        self.ttbot_pose:PoseStamped = PoseStamped()
        self.ttbot_pose.header.frame_id = 'map'
        self.ttbot_pose.pose.orientation.w = 1.0
    
    def __timer_cbk(self, event):
        try:
            # get the transform from map to odom
            (position, heading) = self.listener.lookupTransform('/map', '/odom', rospy.Time(0))
            self.ttbot_pose.pose.position.x = position[0]
            self.ttbot_pose.pose.position.y = position[1]
            self.ttbot_pose.pose.orientation.x = heading[0]
            self.ttbot_pose.pose.orientation.y = heading[1]
            self.ttbot_pose.pose.orientation.z = heading[2]
            self.ttbot_pose.pose.orientation.w = heading[3]
        except Exception as e:
            rospy.loginfo(f'Exception in tf odom: {e}')
            pass
    
    def __grid_cb(self, data:OccupancyGrid):
        t = time.time_ns()
        self.grid = np.array(data.data).reshape((data.info.height, data.info.width))
        self.origin = data.info.origin
        self.resolution = data.info.resolution
        frontiers = self.get_frontiers()
        num = len(frontiers)
        frontiers = self.kmeans(frontiers, self.k)
        self.frontiers = [self.pixel_to_world(x, y) for x, y in frontiers]
        yeet = self.select_frontier(self.frontiers, (self.ttbot_pose.pose.position.x, self.ttbot_pose.pose.position.y))
        
        
        rospy.loginfo(f"Found {num} frontiers ({self.k} clusters) in {(time.time_ns() - t)/1e6:.1f}ms")
        self.publish_frontiers(self.frontiers)
        
    def make_marker(self, x, y, id=0, size=0.15, rgb=(0,0,0)):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.id = id
        marker.type = marker.SPHERE
        marker.action = marker.ADD
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = 0
        marker.pose.orientation.w = 1.0
        marker.scale.x = marker.scale.y = marker.scale.z = size
        marker.color.a = 1.0
        marker.color.r = rgb[0]
        marker.color.g = rgb[1]
        marker.color.b = rgb[2]
    def publish_path(self,raw_path):
        if raw_path is None:
            return
        path = Path()
        path.header.frame_id = 'map'
        for coord in raw_path:
            p = PoseStamped()
            x, y = self.pixel_to_world(*coord)
            p.pose.position.x = x
            p.pose.position.y = y
            path.poses.append(p)
        
        # publish path to RVIZ
        self.path_pub.publish(path)
    def publish_frontiers(self, frontiers):
        marker_array = MarkerArray()
        marker_array.markers = [self.make_marker(frontier[0],frontier[1],idx) for idx,frontier in enumerate(frontiers)]
        self.frontier_pub.publish(marker_array)       
    
    def pixel_to_world(self, x: int, y: int) -> tuple:
        # Compute the coordinates of the center of the cell at (x, y)
        cell_size = self.resolution
        x_center = (x + 0.5) * cell_size
        y_center = (y + 0.5) * cell_size
        
        # Compute the coordinates of the center of the grid in the world frame
        x_offset = self.origin.position.x
        y_offset = self.origin.position.y
        theta = np.arccos(self.origin.orientation.w) * 2  # Convert quaternion to angle
        x_center_world = x_center * np.cos(theta) - y_center * np.sin(theta) + x_offset
        y_center_world = x_center * np.sin(theta) + y_center * np.cos(theta) + y_offset
        
        return y_center_world, x_center_world
    def world_to_pixel(self, world_x, world_y):
        origin_x = self.origin.position.x
        origin_y = self.origin.position.y
        resolution = self.resolution

        pixel_x = int((world_x - origin_x) / resolution)
        pixel_y = int((world_y - origin_y) / resolution)

        return (pixel_x, pixel_y)

    def get_frontiers(self):
        # Find unexplored cells
        unexplored_cells = (self.grid == -1).astype(int)

        # Create a sliding window view for unexplored cells
        window_shape = (3, 3)
        sliding_window = np.lib.stride_tricks.sliding_window_view(unexplored_cells, window_shape)

        # Sum the window view along the last two axes to simulate a convolution with a 3x3 kernel
        convoluted_unexplored = sliding_window.sum(axis=(-1, -2))

        # Find cells with at least two unexplored neighbors
        regions_of_interest = (convoluted_unexplored > 1)

        # Pad the regions_of_interest array to match the grid's shape
        pad_height = self.grid.shape[0] - regions_of_interest.shape[0]
        pad_width = self.grid.shape[1] - regions_of_interest.shape[1]
        regions_of_interest_padded = np.pad(regions_of_interest, ((0, pad_height), (0, pad_width)))

        # Create a mask to find the frontiers by combining the regions_of_interest and free cells
        mask = (self.grid == 0) & regions_of_interest_padded

        # Get the indices of the frontier cells
        frontiers = np.argwhere(mask)

        return [tuple(frontier) for frontier in frontiers]
    def kmeans(self, points, k):
        # Convert the list of points to a NumPy array
        points = np.array(points)

        # Randomly initialize k cluster centroids
        centroids = points[np.random.choice(len(points), k, replace=False), :]

        # Initialize cluster assignments for each point
        labels = np.zeros(len(points))

        while True:
            # Compute the squared Euclidean distance between each point and each centroid
            sqdistances = np.sum((points[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2, axis=-1)

            # Assign each point to the closest centroid
            new_labels = np.argmin(sqdistances, axis=1)

            # If no points have changed cluster assignments, terminate
            if np.array_equal(labels, new_labels):
                break

            # Update cluster assignments and centroids
            labels = new_labels
            for i in range(k):
                centroids[i, :] = np.mean(points[labels == i, :], axis=0)
        return [tuple(c) for c in centroids]
    def select_frontier(self, frontiers:List[Tuple[int,int]], current_position:Tuple[int,int]):
        start = self.world_to_pixel(*current_position)
        self.mp = MapProcessor(self.grid)
        self.mp.dilate(5)
        
        ranked_frontiers = []
        for frontier in frontiers:
            end = self.world_to_pixel(*frontier)
            raw_path, dist = AStar(self.mp.inf_map, start, end).run()
            self.publish_path(raw_path)
            ranked_frontiers.append((frontier, dist, raw_path))
        ranked_frontiers.sort(key=lambda x: x[1])
        if len(ranked_frontiers) == 0:
            return None
        return ranked_frontiers

    def run(self):
        while not rospy.is_shutdown():
            if self.frontiers is None or self.grid is None:
                # rospy.loginfo('No map yet')
                continue
            # goal_frontier = self.select_frontier(self.frontiers)
            # self.goal_pub.publish(self.make_marker(goal_frontier[0], goal_frontier[1],size=0.2,rgb=(1,1,0)))
            
            
            
            self.rate.sleep()
            
if __name__ == "__main__":
    node = Task1Node(node_name='explore_node')
    try:
        node.run()
    except rospy.ROSInterruptException:
        rospy.logerr("Node interrupted")
