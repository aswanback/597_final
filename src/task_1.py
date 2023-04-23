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
from a_star2 import Map, AStar
import tf

class Task1Node:
    def __init__(self, node_name):
        rospy.init_node(node_name, anonymous=True)
        self.rate = rospy.Rate(50)
        self.k = 3 #kmeans
        # rospy.Subscriber('/odom', Odometry, self.__odom_cbk)
        self.listener = tf.TransformListener()
        rospy.Timer(rospy.Duration(0.05), self.__timer_cbk)
        rospy.Subscriber("/map", OccupancyGrid, self.__grid_cb)
        # rospy.wait_for_message('/map', OccupancyGrid)
        
        self.frontier_pub = rospy.Publisher("/frontiers", MarkerArray, queue_size=1)
        self.path_pub = rospy.Publisher("/path", Path, queue_size=self.k)
        self.goal_pub = rospy.Publisher('/goal_frontier',Marker,queue_size=1)
        self.selected_pub = rospy.Publisher('/selected_frontier',Marker,queue_size=1)
        self.cmd_vel_pub = rospy.Publisher('cmd_vel', Twist, queue_size=2)
        self.ttbot_pub = rospy.Publisher('ttbot_pose', PoseStamped, queue_size=2)
        
        self.grid:np.ndarray = None
        self.resolution:float = None
        self.origin:Pose = None
        self.path:Path = None
        # self.frontiers = None
        self.ttbot_pose:PoseStamped = PoseStamped()
        self.ttbot_pose.header.frame_id = 'map'
        self.ttbot_pose.pose.orientation.w = 1.0
        
        self.heading_pid = PIDController(3,0,0.3, [-2,2])
        self.distance_pid = PIDController(0.5,0,0.1,[-1,1])
        self.heading_tolerance = 10
        
        self.linear = 0
        self.angular = 0
        
        self.currIdx = 0
        self.last_time = None

    
    def __timer_cbk(self, event):
        if self.ttbot_pose is None or self.grid is None:
            rospy.loginfo('Waiting for map or odom')
            return
        tt = self.ttbot_pose.pose.position
        to = self.ttbot_pose.pose.orientation
        rospy.loginfo(f'ttbot_pose before update: {tt.x}, {tt.y}, {euler_from_quaternion((to.x, to.y, to.z, to.w))[2]}')
        try:
            # get the transform from map to odom
            if tf.frameExists("/odom") and tf.frameExists("/map"):
                (position, heading) = self.listener.lookupTransform('/odom', '/map', rospy.Time.now())
                self.ttbot_pose.pose.position.x = position[0]
                self.ttbot_pose.pose.position.y = position[1]
                self.ttbot_pose.pose.orientation.x = heading[0]
                self.ttbot_pose.pose.orientation.y = heading[1]
                self.ttbot_pose.pose.orientation.z = heading[2]
                self.ttbot_pose.pose.orientation.w = heading[3]
                self.ttbot_pub.publish(self.ttbot_pose)
                rospy.loginfo(f'ttbot_pose before update: {tt.x}, {tt.y}, {euler_from_quaternion((to.x, to.y, to.z, to.w))[2]}')
            
        except Exception as e:
            rospy.loginfo(f'Exception in tf odom: {e}')
            pass
    
    def __grid_cb(self, data:OccupancyGrid):
        self.grid = np.array(data.data, dtype=np.int8).reshape((data.info.height, data.info.width)).astype(np.int8)
        self.origin = data.info.origin
        self.resolution = data.info.resolution
        self.solve()  

    def solve(self):
        if self.ttbot_pose is None or self.grid is None:
            return
        
        t = time.time_ns()
        self.mp = Map(self.grid, 11, None)
        # self.mp.display()
        frontiers = self.get_frontiers()
        num = len(frontiers)
        frontiers = self.kmeans(frontiers, self.k)
        start = (self.ttbot_pose.pose.position.x, self.ttbot_pose.pose.position.y)
        ret = self.select_frontier([frontiers[0]], start)
        if ret is None:
            rospy.loginfo('No frontier found')
            return
        front, dist, raw_path = ret
        self.frontier = self.pixel_to_world(*front)
        self.path = self.make_path(raw_path)
        
        # logging
        rospy.loginfo(f"Found {num} frontiers ({self.k} clusters) in {(time.time_ns() - t)/1e6:.1f}ms")
        self.publish_frontiers([self.pixel_to_world(x, y) for x, y in frontiers])
        self.selected_pub.publish(self.make_marker(self.frontier[0],self.frontier[1],0,rgb=(1,0,0)))
        self.path_pub.publish(self.path)
        
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
        return marker
    
    def make_path(self,raw_path):
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
        return path
  
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
        
        ranked_frontiers = []
        for frontier in frontiers:
            t = time.time_ns()
            raw_path, dist = AStar(self.mp, start, frontier).run()
            t2 = time.time_ns()
            if raw_path is None:
                continue
            # rospy.loginfo(f'Astar: {(t2-t)/1e6:.2f} ms ')
            ranked_frontiers.append((frontier, dist, raw_path))
        ranked_frontiers.sort(key=lambda x: x[1])
        if len(ranked_frontiers) == 0:
            return None
        return ranked_frontiers[0]

    def run(self):
        
        robot_path:Path = None
        while not rospy.is_shutdown():
            if self.grid is None:
                # rospy.loginfo('No map yet')
                continue
            
            if robot_path is None:
                robot_path = self.path
                self.move_ttbot(0, 0)
                continue
            
            # rospy.loginfo(f'robot_path: {self.currIdx}/{len(robot_path.poses)}')
            self.currIdx = self.get_path_idx(robot_path, self.ttbot_pose, self.currIdx)
            if self.currIdx == len(robot_path.poses) - 1:
                rospy.loginfo('Reached goal')
                robot_path = self.path
                continue
            
            self.linear, self.angular = self.pid_controller(self.ttbot_pose, robot_path.poses[self.currIdx])

            self.move_ttbot(self.linear, self.angular)
            
            self.rate.sleep()
    
    
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
        sgp = self.frontier
        # sgo = self.frontier.pose.orientation
        dist = np.sqrt((sgp[0] - cp.x)**2 + (sgp[1] - cp.y)**2)
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
    node = Task1Node(node_name='explore_node')
    try:
        node.run()
    except rospy.ROSInterruptException:
        rospy.logerr("Node interrupted")
