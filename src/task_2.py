#!/usr/bin/env python3

#!/usr/bin/env python3

import sys
import numpy as np
import time
import rospy
from tf.transformations import euler_from_quaternion
from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Twist
from a_star2 import AStar, Map
from geometry_msgs.msg import TransformStamped
import tf2_ros
import rospkg

class Navigation:
    def __init__(self, node_name='navigation'):
        # ROS node initialization
        rospy.init_node(node_name, anonymous=True)
        self.rate = rospy.Rate(50)
        # Subscribers
        rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.__goal_pose_cbk, queue_size=1)
        rospy.Subscriber('/amcl_pose', PoseWithCovarianceStamped, self.__ttbot_pose_cbk, queue_size=10)
        rospy.Subscriber('/odom', Odometry, self.__odom_cbk)
        # Publishers
        self.path_pub = rospy.Publisher('global_plan', Path, queue_size=2)
        self.cmd_vel_pub = rospy.Publisher('cmd_vel', Twist, queue_size=2)
        self.odom_set_pub = rospy.Publisher('/initialpose',PoseWithCovarianceStamped, queue_size=2)
        self.pose_pub = rospy.Publisher('est_ttbot_pose', PoseStamped, queue_size=2)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        self.my_path_pub = rospy.Publisher('/path_topic',Path,queue_size=2)
        p = Path()
        p.header.frame_id = 'map'
        self.my_path_pub.publish(p)
        # Map and tree creation
        rospack = rospkg.RosPack()
        pkgpath = rospack.get_path("lab_4_pkg")
        
        t = time.time_ns()
        self.mp = Map(f'{pkgpath}/maps/map2')
        t2 = time.time_ns()
        rospy.loginfo(f'Created map in {(t2-t)/1e6} ms')
        # self.mp.display()
         
        # Create path planning variables
        self.path = Path()
        self.goal_pose:PoseStamped = None
        
        self.ttbot_pose:PoseStamped = PoseStamped()
        self.ttbot_pose.header.frame_id = 'map'
        
        self.heading_pid = PIDController(3,0,0.3, [-2,2])
        self.distance_pid = PIDController(0.5,0,0.1,[-1,1])
        self.heading_tolerance = 10
        
        self.linear = 0
        self.angular = 0

        self.confident = False
        self.currIdx = 0
        self.last_time = None
        self.dt = None

    def __odom_cbk(self, data:Odometry):
        ''' Callback to catch the position of the vehicle from odom.'''
        self.ttbot_pose.pose = data.pose.pose
        self.pose_pub.publish(self.ttbot_pose)
    
    def __goal_pose_cbk(self, data:PoseStamped):
        """! Callback to catch the goal pose.
        @param  data    PoseStamped object from RVIZ.
        @return None.
        """
        self.goal_pose = data
        gp = self.goal_pose.pose.position
        rospy.loginfo(f'new goal pose:({gp.x:.2f},{gp.y:.2f}), pix:{self._model_to_screen(gp.x,gp.y)}')
        
    def __ttbot_pose_cbk(self, data:PoseWithCovarianceStamped):
        """! Callback to catch the position of the vehicle.
        @param  data    PoseWithCovarianceStamped object from amcl.
        @return None.
        """
        self.ttbot_pose.pose = data.pose.pose
        # set confidence true if every cov value is < 0.01
        cov = np.array(data.pose.covariance)
        # self.confident = np.any(cov[abs(cov) > 0.01])
        self.confident = True
        for i in cov:
            if abs(i) > 0.01:
                self.confident = False
                break

        if not self.confident:
            rospy.loginfo(f'Locating...')
        else:
            # rospy.loginfo(f'pose update')

            # Update base_link to odom transform instead of publishing to /initialpose
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
        # linear transformation with y and x switched for orientation
        x_mult, x_const, y_mult, y_const = self.get_lin_params()
        return int(x_mult * y + x_const), int(y_mult * x + y_const)
    def _screen_to_model(self, x_pix, y_pix):
        # linear de-transformation of _model_to_screen
        x_mult, x_const, y_mult, y_const = self.get_lin_params()
        return (y_pix - y_const) / y_mult, (x_pix - x_const) / x_mult
    
    def a_star_path_planner(self, start_pose:PoseStamped, end_pose:PoseStamped):
        """! A Star path planner.
        @param  start_pose    PoseStamped object containing the start of the path to be created.
        @param  end_pose      PoseStamped object containing the end of the path to be created.
        @return path          Path object containing the sequence of waypoints of the created path.
        """
        start_time = time.time_ns()
        path, dist = AStar(self.mp.map, start_pose, end_pose).run()
        self.my_path_pub.publish(path)
        
        rospy.loginfo(f'A* solved:{start_pose.pose.position.x:.3f},{start_pose.pose.position.y:.3f} > end:{end_pose.pose.position.x:.3f},{end_pose.pose.position.y:.3f} in {(time.time_ns() - start_time)/1e6:.2f}ms')
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

        # Calculate distance and heading errors
        cp = current_pose.pose.position
        co = current_pose.pose.orientation
        gp = goal_pose.pose.position
        dist_error = np.sqrt((gp.x - cp.x)**2 + (gp.y - cp.y)**2)
        desired_heading = np.arctan2(gp.y - cp.y, gp.x - cp.x)
        heading_error = desired_heading - euler_from_quaternion([co.x, co.y, co.z, co.w])[2]

        # Check if at goal
        sgp = self.goal_pose.pose.position
        sgo = self.goal_pose.pose.orientation
        dist = np.sqrt((sgp.x - cp.x)**2 + (sgp.y - cp.y)**2)
        if dist < 0.15:
            rospy.loginfo('At goal, waiting')
            # Adjust to goal heading
            dist_error = 0
            heading_error = euler_from_quaternion([sgo.x, sgo.y, sgo.z, sgo.w])[2] - euler_from_quaternion([co.x, co.y, co.z, co.w])[2]
        
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

            # if not sure of localization, spin in a circle
            if not self.confident:
                self.move_ttbot(0, 1.5)
                continue
            
            # if the goal_pose has changed, replan A*
            if self.goal_pose != old_goal_pose:
                self.currIdx = 0
                path = self.a_star_path_planner(self.ttbot_pose, self.goal_pose)
                
            old_goal_pose = self.goal_pose

            # if no path is returned, stop moving
            if path is None:
                self.move_ttbot(0, 0)
                continue

            # find next point to go towards
            self.currIdx = self.get_path_idx(path, self.ttbot_pose, self.currIdx)
            current_goal = path.poses[self.currIdx]
            # rospy.loginfo(f'idx: {self.currIdx+1}/{len(path.poses)}')
            
            self.linear, self.angular = self.pid_controller(self.ttbot_pose, current_goal)
            # rospy.loginfo(f'linear:{self.linear:.3f}\tangular:{self.angular:.3f}')

            # move robot
            self.move_ttbot(self.linear, self.angular)

            # timeout of 10 minutes in case stuck
            if time.time() > start_time + (10 * 60) * 1000:  # timeout after 10 minutes
                rospy.signal_shutdown("Navigation timeout")

        rospy.signal_shutdown("Finished Cleanly")

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
    nav = Navigation(node_name='Navigation')
    try:
        nav.run()
    except rospy.ROSInterruptException:
        print("program interrupted before completion", file=sys.stderr)