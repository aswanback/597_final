#!/usr/bin/env python3

import rospy
from nav_msgs.msg import OccupancyGrid
import numpy as np
from visualization_msgs.msg import MarkerArray, Marker


class Task1Node:
    def __init__(self, node_name):
        rospy.init_node(node_name, anonymous=True)
        self.rate = rospy.Rate(50)
        rospy.Subscriber("/map", OccupancyGrid, self.grid_cb)
        self.grid = None
    
    def grid_cb(self, data:OccupancyGrid):
        self.grid = np.array(data.data).reshape((data.info.height, data.info.width))
        frontiers = self.get_frontiers(self.grid)
        frontiers = [self.pixel_to_world(data, x, y) for x, y in frontiers]
        pub = rospy.Publisher("/frontiers", MarkerArray, queue_size=1)
        # turn frontiers into markers
        markers = []
        for idx,frontier in enumerate(frontiers):
            marker = Marker()
            marker.header.frame_id = "map"
            marker.id = idx
            marker.type = marker.SPHERE
            marker.action = marker.ADD
            marker.pose.position.x = frontier[0]
            marker.pose.position.y = frontier[1]
            marker.pose.position.z = 0
            marker.pose.orientation.w = 1.0
            marker.scale.x = marker.scale.y = marker.scale.z = 0.05
            marker.color.a = 1.0
            marker.color.r = 0.0
            marker.color.g = 0.0
            marker.color.b = 1.0
            markers.append(marker)
        marker_array = MarkerArray()
        marker_array.markers = markers
        pub.publish(marker_array)
        
        
    
    def pixel_to_world(self, grid: OccupancyGrid, x: int, y: int) -> tuple:
        # Compute the coordinates of the center of the cell at (x, y)
        cell_size = grid.info.resolution
        x_center = (x + 0.5) * cell_size
        y_center = (y + 0.5) * cell_size
        
        # Compute the coordinates of the center of the grid in the world frame
        x_offset = grid.info.origin.position.x
        y_offset = grid.info.origin.position.y
        theta = np.arccos(grid.info.origin.orientation.w) * 2  # Convert quaternion to angle
        x_center_world = x_center * np.cos(theta) - y_center * np.sin(theta) + x_offset
        y_center_world = x_center * np.sin(theta) + y_center * np.cos(theta) + y_offset
        
        return y_center_world, x_center_world
    
    # Step 2: Locate frontiers
    def get_frontiers(self, grid):
        frontiers = []
        for row in range(len(grid)):
            for col in range(len(grid[0])):
                if self.is_frontier(row, col, grid):
                    frontiers.append((row, col))
        return frontiers

    def is_frontier(self,row, col, grid):
        if grid[row][col] == -1:
            neighbors = self.get_neighbors(row, col, grid)
            for neighbor in neighbors:
                if grid[neighbor[0]][neighbor[1]] == 0:
                    return True
        return False

    def get_neighbors(self,row, col, grid):
        neighbors = []
        for r_offset, c_offset in [(-1, 0), (1,1), (1, 0), (-1,-1), (0, -1), (1,-1), (0, 1), (-1,1)]:
            new_row, new_col = row + r_offset, col + c_offset
            if (0 <= new_row < len(grid)) and (0 <= new_col < len(grid[0])):
                neighbors.append((new_row, new_col))
        return neighbors
    
    def kmeans(points, k):
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

        return centroids, labels

    # def select_frontier(self, frontiers, current_position, a_star_algorithm):
    #     ranked_frontiers = []
    #     for frontier in frontiers:
    #         cost = a_star_algorithm(current_position, frontier)
    #         ranked_frontiers.append((frontier, cost))
    #     ranked_frontiers.sort(key=lambda x: x[1])
    #     if len(ranked_frontiers) == 0:
    #         return None
    #     return ranked_frontiers[0][0]

    # # Step 4: Select the best frontier
    # def select_best_frontier(self, ranked_frontiers):
    #     if not ranked_frontiers:
    #         return None
    #     return ranked_frontiers[0][0]
    
    
    def run(self):
        while not rospy.is_shutdown():
            # frontiers = self.get_frontiers(self.grid)
            # ranked_frontiers = self.rank_frontiers(frontiers, current_position, a_star_algorithm)  # Replace `a_star_algorithm` with your A* algorithm
            # best_frontier = select_best_frontier(ranked_frontiers)
            
            self.rate.sleep()
            
if __name__ == "__main__":
    node = Task1Node(node_name='explore_node')
    try:
        node.run()
    except rospy.ROSInterruptException:
        rospy.logerr("Node interrupted")
