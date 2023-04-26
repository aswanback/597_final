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
import cv2



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
