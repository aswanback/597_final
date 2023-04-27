from visualization_msgs.msg import MarkerArray, Marker
import sys
import numpy as np
import time
import rospy
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import tf
from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Twist
from geometry_msgs.msg import TransformStamped
import tf2_ros
import rospkg
import heapq
from typing import Dict, List, Optional, Tuple, Union
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
from sensor_msgs.msg import LaserScan
class Navigation:
    def __init__(self, node_name='navigation'):
        rospy.init_node(node_name, anonymous=True)
        self.rate = rospy.Rate(50)
        rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.__goal_pose_cbk, queue_size=1)
        rospy.Subscriber('/amcl_pose', PoseWithCovarianceStamped, self.__ttbot_pose_cbk, queue_size=10)
        rospy.Subscriber('/odom', Odometry, self.__odom_cbk)
        rospy.Subscriber('/scan', LaserScan, self.__scan_cbk)
        rospy.Timer(rospy.Duration(0.01), self.__timer_cbk)
        rospy.Timer(rospy.Duration(0.01), self.__timer2_cbk)
        self.path_pub = rospy.Publisher('global_plan', Path, queue_size=2)
        self.cmd_vel_pub = rospy.Publisher('cmd_vel', Twist, queue_size=2)
        self.odom_set_pub = rospy.Publisher('/initialpose',PoseWithCovarianceStamped, queue_size=2)
        self.pose_pub = rospy.Publisher('est_ttbot_pose', PoseStamped, queue_size=2)
        self.listener = tf.TransformListener()
        self.my_path_pub = rospy.Publisher('/path_topic',Path,queue_size=2)
        self.points_pub = rospy.Publisher('/points',MarkerArray,queue_size=2)
        self.ttbot_pub = rospy.Publisher('ttbot_pose', PoseStamped, queue_size=2)
        self.vector_pub = rospy.Publisher('/vector',PoseStamped,queue_size=2)
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
        self.heading_pid = PIDController(2.5,0, 7, [-2.5,2.5])
        self.distance_pid = PIDController(0.8,0.1,0.4,[-1.1,1.1], 0.3)
        self.avoid_heading_pid = PIDController(2.5,0, 0.01, [-2.5,2.5])
        self.avoid_velocity_pid = PIDController(0.8,0,0,[-5,5]) 
        self.heading_tolerance = 10
        self.linear = 0
        self.angular = 0
        self.confident = False
        self.currIdx = 0
        self.last_time = None
        self.dt = None
        self.last_conf_time = None
        self.i = 0
        self.laser_scan = None
        self.avoid_angle = 0
        self.avoid_mag = 0
        self.last_avoid_time = None
    def __scan_cbk(self, laser_scan:LaserScan):
        self.laser_scan = laser_scan
    def __timer2_cbk(self, event):
        if self.laser_scan is None:
            return
        laser_scan = self.laser_scan
        ranges = np.array(laser_scan.ranges)
        angles = np.array([laser_scan.angle_min + i * laser_scan.angle_increment for i in range(len(laser_scan.ranges))])
        valid_indices = np.logical_and(np.isfinite(ranges), ranges <= 2, ranges >= 0.25)
        valid_ranges = ranges[valid_indices]
        valid_angles = angles[valid_indices]
        sorted_indices = np.argsort(valid_ranges)[:30]
        closest_ranges = valid_ranges[sorted_indices]
        closest_angles = valid_angles[sorted_indices]
        x_rel = closest_ranges * np.cos(closest_angles)
        y_rel = closest_ranges * np.sin(closest_angles)
        tt = self.ttbot_pose.pose.orientation
        curr = euler_from_quaternion([tt.x, tt.y, tt.z, tt.w])[2]
        smx = np.sum(x_rel / (closest_ranges))
        smy = np.sum(y_rel / (closest_ranges))
        direction_angle = np.arctan2(smy, smx) + np.pi
        magnitude = np.sqrt(smx**2 + smy**2)
        rot = quaternion_from_euler(0, 0, direction_angle + curr)
        p = PoseStamped()
        p.pose.position.x = self.ttbot_pose.pose.position.x
        p.pose.position.y = self.ttbot_pose.pose.position.y
        p.pose.orientation.x = rot[0]
        p.pose.orientation.y = rot[1]
        p.pose.orientation.z = rot[2]
        p.pose.orientation.w = rot[3]
        p.header.frame_id = "map"
        self.vector_pub.publish(p)
        tt = self.ttbot_pose.pose.position
        self.avoid_mag = magnitude if self.make_cond(self.ttbot_pose) else 0
        self.avoid_angle = direction_angle if self.make_cond(self.ttbot_pose) else 0
    def make_marker(self, x, y, id=0, size=0.15, rgb=(0,0,0), w=1.0):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.id = id
        marker.type = marker.SPHERE
        marker.action = marker.ADD
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = 0
        marker.pose.orientation.w = w
        marker.scale.x = marker.scale.y = marker.scale.z = size
        marker.color.a = 1.0
        marker.color.r = rgb[0]
        marker.color.g = rgb[1]
        marker.color.b = rgb[2]
        return marker
    def __odom_cbk(self, data:Odometry):
        ''' Callback to catch the position of the vehicle from odom.'''
        if not self.ttbot_pose:
            self.ttbot_pose = PoseStamped()
        self.ttbot_pose.pose = data.pose.pose
        self.pose_pub.publish(self.ttbot_pose)
    def cond2(self, tt):
        return -1 < tt.y < 4 
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
            pass
    def __timer_cbk(self, event):
        try:
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
            pass
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
    def cond1(self, tt):
        return -1 < tt.y < 4 
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
            dist_error = None
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
    def avoid_pid(self, magnitude, angle):
        dt = rospy.Time().now().to_sec() - self.last_avoid_time if self.last_avoid_time else None
        tt = self.ttbot_pose.pose.orientation
        heading_error = angle - euler_from_quaternion([tt.x, tt.y, tt.z, tt.w])[2]
        heading_error = (heading_error + np.pi) % (2 * np.pi) - np.pi
        avoid_head_control = self.avoid_heading_pid.update(heading_error, dt)
        if abs(heading_error) > 20/180*np.pi:
            return 0, avoid_head_control
        avoid_vel_control = self.avoid_velocity_pid.update(magnitude / 10, dt)
        self.last_time = rospy.Time().now().to_sec()
        return avoid_vel_control, avoid_head_control
    def make_cond(self, ttbot_pose):
        tt = ttbot_pose.pose.position
        return tt.x < 5.5 and tt.x > 4.4 and self.cond1(tt) or tt.x < -2 and tt.x > -3 and self.cond2(tt)
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
            if self.avoid_mag > 20:
                self.linear, self.angular = self.avoid_pid(self.avoid_mag, self.avoid_angle)
                rospy.loginfo(f'avoiding: {self.linear:.3f}\t{self.angular:.3f}')
            else:
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
        if error is None:
            return 0
        derivative = (error - self.last_error) / dt if (dt is not None and dt != 0) else 0
        self.integral += error * dt if dt is not None else 0
        self.last_error = error
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        rospy.loginfo(f'error: {error:.2f}, integral: {self.integral:.2f}, derivative: {derivative:.2f}, output: {output:.2f}')
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