#!/usr/bin/env python3

import rospy
from gazebo_msgs.msg import ContactsState
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import PoseStamped
from tf.transformations import euler_from_quaternion
import math


class NavigationMetrics:
    def __init__(self):
        # collision related variables
        self.collision_counts = 0
        self.last_collision_time = None
        self.COLLISION_IGNORE_DURATION = 3.0  # continuous collisions within this seconds will be ignored

        # navigation time related variables
        self.navigation_start_time = None
        self.navigation_duration = 0
        self.NAVIGATION_UNFINISHED_TIME = 1000

        # report as a dictionary of all navigation metrics
        self.report = {'start_poses': [], 'goal_poses': [], 'time_costs': [], 'collision_counts': []}

        # location variables updated in callbacks
        self.robot_name = 'turtlebot3_waffle'
        self.robot_pose = None
        self.goal_pose = None

        # contants to judge robot's proximity to goal
        self.DIST_THR = 0.2  # meter
        self.ANGLE_THR = 0.6  # radian

        # declare all the subscribers and keep the process running
        rospy.init_node('navigation_metrics_node', anonymous=True)
        self.collision_sub = rospy.Subscriber('/robot/bumper_states', ContactsState,
                                              self.collision_callback, queue_size=1)
        self.goal_pose_sub = rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.goal_callback, queue_size=1)
        self.robot_pose_sub = rospy.Subscriber('/gazebo/model_states', ModelStates, self.robot_callback, queue_size=1)
        rospy.loginfo("Navigation metrics are running ...")
        rospy.spin()

    def __call__(self, *args, **kwargs):
        rospy.loginfo(f"Navigation reports for {len(self.report['start_poses'])} run(s):")
        rospy.loginfo(f"time costs: {self.report['time_costs']}")
        rospy.loginfo(f"collision counts: {self.report['collision_counts']}")

    def reset(self, reset_goal: bool = False):
        self.collision_counts = 0
        self.last_collision_time = None
        self.navigation_duration = 0
        if reset_goal:
            self.goal_pose = None

    def collision_callback(self, msg):
        contact_states = msg.states
        if len(contact_states) > 0:
            current_collision_time = msg.header.stamp
            if self.last_collision_time is None:
                self.collision_counts += 1
                self.last_collision_time = current_collision_time
            elif (current_collision_time - self.last_collision_time).to_sec() > self.COLLISION_IGNORE_DURATION:
                self.collision_counts += 1
                self.last_collision_time = current_collision_time
            rospy.loginfo_throttle(1, f"Your current collision counts: {self.collision_counts}.")

    def goal_callback(self, msg):
        # current navigation is preempted, metrics should be differentiable
        if self.goal_pose is not None:
            rospy.logwarn("Current navigation is not finished, but new goal comes, preempt!")
            self.report['time_costs'].append(self.navigation_duration + self.NAVIGATION_UNFINISHED_TIME)
            self.report['collision_counts'].append(self.collision_counts)

        rospy.loginfo("New goal received!")
        self.goal_pose = msg.pose
        if self.robot_pose is None:
            rospy.logwarn("Still waiting for robot pose when received goal pose, abort this goal!")
            return

        self.navigation_start_time = msg.header.stamp
        self.report['start_poses'].append(self.robot_pose)
        self.report['goal_poses'].append(self.goal_pose)
        self.reset()

    @staticmethod
    def dist(x1: float, y1: float, x2: float, y2: float):
        return math.sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2))

    def reached_goal(self):
        if self.goal_pose is None:
            return False

        while self.robot_pose is None:
            rospy.logwarn_throttle(0.5, "Waiting for robot pose to show up ...")

        # judge Euclidean distance
        robot_x, robot_y = self.robot_pose.position.x, self.robot_pose.position.y
        goal_x, goal_y = self.goal_pose.position.x, self.goal_pose.position.y
        if self.dist(robot_x, robot_y, goal_x, goal_y) > self.DIST_THR:
            return False

        # judge heading difference
        robot_quaternion = (self.robot_pose.orientation.x, self.robot_pose.orientation.y,
                            self.robot_pose.orientation.z, self.robot_pose.orientation.w)
        goal_quaternion = (self.goal_pose.orientation.x, self.goal_pose.orientation.y,
                           self.goal_pose.orientation.z, self.goal_pose.orientation.w)
        robot_theta = euler_from_quaternion(robot_quaternion)[2]  # (roll, pitch, yaw)
        goal_theta = euler_from_quaternion(goal_quaternion)[2]
        if abs(robot_theta - goal_theta) > self.ANGLE_THR:
            return False

        return True

    def robot_callback(self, msg):
        # update robot pose
        robot_id = msg.name.index(self.robot_name)
        self.robot_pose = msg.pose[robot_id]

        # update navigation time
        if self.goal_pose is not None:
            assert self.navigation_start_time is not None
            self.navigation_duration = (rospy.Time.now() - self.navigation_start_time).to_sec()

        # if reached goal, fill the report for this session, then reset variables
        if self.reached_goal():
            self.report['time_costs'].append(self.navigation_duration)
            self.report['collision_counts'].append(self.collision_counts)
            self.reset(reset_goal=True)
            rospy.loginfo(f"Navigation finished successfully, waiting for new goal...")
            self.__call__()


if __name__ == '__main__':
    try:
        nav_metrics = NavigationMetrics()
    except rospy.ROSInterruptException:
        pass
    finally:
        nav_metrics()

