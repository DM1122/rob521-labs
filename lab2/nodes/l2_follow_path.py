#!/usr/bin/env python3
from __future__ import division, print_function
import os

import numpy as np
from scipy.linalg import block_diag
from scipy.spatial.distance import cityblock #  calcuate the manhattan distance between two (x, y, z) vectors 
import rospy
import tf2_ros

# msgs
from geometry_msgs.msg import TransformStamped, Twist, PoseStamped
from nav_msgs.msg import Path, Odometry, OccupancyGrid
from visualization_msgs.msg import Marker

# ros and se2 conversion utils
import utils

# ode 
from scipy.integrate import odeint 


TRANS_GOAL_TOL = .1  # m, tolerance to consider a goal complete
ROT_GOAL_TOL = .3  # rad, tolerance to consider a goal complete
TRANS_VEL_OPTS = [0, 0.025, 0.13, 0.26]  # m/s, max of real robot is .26
ROT_VEL_OPTS = np.linspace(-1.82, 1.82, 11)  # rad/s, max of real robot is 1.82
CONTROL_RATE = 5  # Hz, how frequently control signals are sent
# how far into the future the motion planning algorithm looks 
CONTROL_HORIZON = 5  # seconds. if this is set too high and INTEGRATION_DT is too low, code will take a long time to run!
INTEGRATION_DT = .025  # s, delta t to propagate trajectories forward by
# base_link: robots's frame located at the centeral point of a robot 
COLLISION_RADIUS = 0.225  # m, radius from base_link to use for collisions, min of 0.2077 based on dimensions of .281 x .306
ROT_DIST_MULT = .1  # multiplier to change effect of rotational distance in choosing correct control
OBS_DIST_MULT = .1  # multiplier to change the effect of low distance to obstacles on a path
MIN_TRANS_DIST_TO_USE_ROT = TRANS_GOAL_TOL  # m, robot has to be within this distance to use rot distance in cost
PATH_NAME = 'path.npy'  # saved path from l2_planning.py, should be in the same directory as this file

# here are some hardcoded paths to use if you want to develop l2_planning and this file in parallel
# TEMP_HARDCODE_PATH = [[2, 0, 0], [2.75, -1, -np.pi/2], [2.75, -4, -np.pi/2], [2, -4.4, np.pi]]  # almost collision-free
TEMP_HARDCODE_PATH = [[2, -.5, 0], [2.4, -1, -np.pi/2], [2.45, -3.5, -np.pi/2], [1.5, -4.4, np.pi]]  # some possible collisions


class PathFollower():
    def __init__(self):
        # time full path
        self.path_follow_start_time = rospy.Time.now()

        # use tf2 buffer to access transforms between existing frames in tf tree
        self.tf_buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tf_buffer)
        rospy.sleep(1.0)  # time to get buffer running

        # constant transforms
        self.map_odom_tf = self.tf_buffer.lookup_transform('map', 'odom', rospy.Time(0), rospy.Duration(2.0)).transform
        print(self.map_odom_tf)

        # subscribers and publishers
        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.global_path_pub = rospy.Publisher('~global_path', Path, queue_size=1, latch=True)
        self.local_path_pub = rospy.Publisher('~local_path', Path, queue_size=1)
        self.collision_marker_pub = rospy.Publisher('~collision_marker', Marker, queue_size=1)

        # map
        map = rospy.wait_for_message('/map', OccupancyGrid)
        self.map_np = np.array(map.data).reshape(map.info.height, map.info.width)
        self.map_resolution = round(map.info.resolution, 5)
        self.map_origin = -utils.se2_pose_from_pose(map.info.origin)  # negative because of weird way origin is stored
        print(self.map_origin)
        self.map_nonzero_idxes = np.argwhere(self.map_np)
        print(map)


        # collisions
        self.collision_radius_pix = COLLISION_RADIUS / self.map_resolution # map_resolution m/pix 
        self.collision_marker = Marker()
        self.collision_marker.header.frame_id = '/map'
        self.collision_marker.ns = '/collision_radius'
        self.collision_marker.id = 0
        self.collision_marker.type = Marker.CYLINDER
        self.collision_marker.action = Marker.ADD
        self.collision_marker.scale.x = COLLISION_RADIUS * 2
        self.collision_marker.scale.y = COLLISION_RADIUS * 2
        self.collision_marker.scale.z = 1.0
        self.collision_marker.color.g = 1.0
        self.collision_marker.color.a = 0.5

        # transforms
        self.map_baselink_tf = self.tf_buffer.lookup_transform('map', 'base_link', rospy.Time(0), rospy.Duration(2.0))
        self.pose_in_map_np = np.zeros(3) # INIT (x, y, theta) in map configuration
        self.pos_in_map_pix = np.zeros(2) # INIT (x,y) pixel coordinate
        self.update_pose()

        # path variables
        cur_dir = os.path.dirname(os.path.realpath(__file__))

        # to use the temp hardcoded paths above, switch the comment on the following two lines
        # self.path_tuples = np.load(os.path.join(cur_dir, 'path.npy')).T
        self.path_tuples = np.array(TEMP_HARDCODE_PATH)

        self.path = utils.se2_pose_list_to_path(self.path_tuples, 'map')
        self.global_path_pub.publish(self.path)

        # goal
        self.cur_goal = np.array(self.path_tuples[0])
        self.cur_path_index = 0

        # trajectory rollout tools
        # self.all_opts is a Nx2 array with all N possible combinations of the t and v vels, scaled by integration dt
        self.all_opts = np.array(np.meshgrid(TRANS_VEL_OPTS, ROT_VEL_OPTS)).T.reshape(-1, 2) # (44, 2) = (# of possible option * (trans_vel, rot_vel))

        # if there is a [0, 0] option, remove it
        all_zeros_index = (np.abs(self.all_opts) < [0.001, 0.001]).all(axis=1).nonzero()[0]
        if all_zeros_index.size > 0:
            self.all_opts = np.delete(self.all_opts, all_zeros_index, axis=0)
        self.all_opts_scaled = self.all_opts * INTEGRATION_DT

        self.num_opts = self.all_opts_scaled.shape[0]
        self.horizon_timesteps = int(np.ceil(CONTROL_HORIZON / INTEGRATION_DT))

        self.rate = rospy.Rate(CONTROL_RATE)

        rospy.on_shutdown(self.stop_robot_on_shutdown)
        self.follow_path()

    def follow_path(self):
        while not rospy.is_shutdown():
            # timing for debugging...loop time should be less than 1/CONTROL_RATE
            tic = rospy.Time.now()

            self.update_pose()
            self.check_and_update_goal()

            # start trajectory rollout algorithm
            local_paths = np.zeros([self.horizon_timesteps + 1, self.num_opts, 3]) # (# of time steps needed to cover the CONTROLHORIZON, options of (v,w), each local path represented by 3 value )
            local_paths[0] = np.atleast_2d(self.pose_in_map_np).repeat(self.num_opts, axis=0)

            """
            Step 1) Trajectory rollout : Propogate the trajectory forward, storing the resulting points in local_paths
                - perform trajectory rollout to generate potential local paths 
                - reuse the trajectory rollout code from Task 2 (Check)
            """
            def system_dynamics(state, t, v, omega):
                # Define the system of ODEs
                x, y, theta = state
                x_dot = v * np.cos(theta)
                y_dot = v * np.sin(theta)
                theta_dot = omega
                return [x_dot, y_dot, theta_dot]
            
            for t in range(1, self.horizon_timesteps + 1):
                # propogate trajectory forward, assuming perfect control of velocity and no dynamic effects
                for opt in range(self.num_opts):
                    trans_vel, rot_vel = self.all_opts[opt]
                    x, y, theta = local_paths[t - 1, opt]
                    # new_x = x + trans_vel * np.cos(theta) * INTEGRATION_DT # already scaled value 
                    # new_y = y + trans_vel * np.sin(theta) * INTEGRATION_DT
                    # new_theta = theta + rot_vel * INTEGRATION_DT
                    
                    # Solve ODE
                    solution = odeint(
                        func=system_dynamics, y0=[x, y, theta], t=t, args=(trans_vel, rot_vel)
                    )
                    new_x, new_y, new_theta = solution 
                    local_paths[t, opt] = [new_x, new_y, new_theta]

            """
            Step 2) Collision Check 
             - Check the points in local_path_pixels for collisions
             - Check for collsions and remove invalid trajectories 
             - Reuse the collision detection code from Task 1 to perform collision detection on your trajectory 
            """
            # check all trajectory points for collisions
            # first find the closest collision point in the map to each local path point
            """need to check collisions in pixel frame""" 
            local_paths_pixels = (self.map_origin[:2] + local_paths[:, :, :2]) / self.map_resolution
            valid_opts = range(self.num_opts) # INIT 
            local_paths_lowest_collision_dist = np.ones(self.num_opts) * 50 # INIT set to high value 

            # check the points in local_path_"""pixels""" for collisions 
            # local_paths_pixels = (# of time steps needed to cover the CONTROLHORIZON, options of (v,w), each local path represented by 3 value )
            for opt in range(local_paths_pixels.shape[1]):
                for timestep in range(1, self.horizon_timesteps + 1):
                    x, y = local_paths_pixels[timestep, opt, :2]
                    # cityblock = manhanttan distance between two vector 
                    # find the nearest boundary point and then compare it with collision_radius_pix 
                    if np.min(cityblock(self.map_nonzero_idxes, [x, y])) < self.collision_radius_pix:
                        local_paths_lowest_collision_dist[opt] = min(local_paths_lowest_collision_dist[opt], np.min(cityblock(self.map_nonzero_idxes, [x, y])))

            # Remove trajectories that were deemed to have collisions
            valid_opts = [opt for opt in range(self.num_opts) if local_paths_lowest_collision_dist[opt] > self.collision_radius_pix]

            """
            Calculate final cost and choose best option 
                - score each valid trajectory and pick the best one 
            """
            # calculate final cost and choose best option
            final_cost = np.zeros(self.num_opts)

            # for i, opt_idx in enumerate(valid_opts):
            #     trajectory = valid_local_paths[:, opt_idx]
            #     # score based on distance to the current goal 
            #     goal_dist = np.linalg.nort(trajectory[-1, :2] - self.cur_goal[:2])
            #     # score based on distance to nearst obstacle 
            #     obs_dists = local_paths_lowest_collision_dist[opt_idx]
            #     # combine the scores 
            #     final_cost[i] = goal_dist + (OBS_DIST_MULT * obs_dists)

            for opt in valid_opts:
                # Example cost function: distance to goal (more sophisticated cost functions can be used)
                final_cost[opt] = np.linalg.norm(local_paths[-1, opt, :2] - self.cur_goal[:2])

            if final_cost.size == 0:  # hardcoded recovery if all options have collision
                control = [-.1, 0]
            else:
                # best_traj_idx = valid_opts[final_cost.argmin()]
                best_traj_idx = valid_opts[np.argmin(final_cost[valid_opts])]
                best_control = self.all_opts[best_traj_idx] # pick the bast one 
                self.local_path_pub.publish(utils.se2_pose_list_to_path(local_paths[:, best_traj_idx], 'map'))

            # send command to robot
            self.cmd_pub.publish(utils.unicyle_vel_to_twist(best_control))

            # uncomment out for debugging if necessary
            print("Selected control: {control}, Loop time: {time}, Max time: {max_time}".format(
                control=control, time=(rospy.Time.now() - tic).to_sec(), max_time=1/CONTROL_RATE))

            self.rate.sleep()

    def update_pose(self):
        # Update numpy poses with current pose using the tf_buffer
        self.map_baselink_tf = self.tf_buffer.lookup_transform('map', 'base_link', rospy.Time(0)).transform
        self.pose_in_map_np[:] = [self.map_baselink_tf.translation.x, self.map_baselink_tf.translation.y,
                                  utils.euler_from_ros_quat(self.map_baselink_tf.rotation)[2]]
        self.pos_in_map_pix = (self.map_origin[:2] + self.pose_in_map_np[:2]) / self.map_resolution
        self.collision_marker.header.stamp = rospy.Time.now()
        self.collision_marker.pose = utils.pose_from_se2_pose(self.pose_in_map_np)
        self.collision_marker_pub.publish(self.collision_marker)

    def check_and_update_goal(self):
        # iterate the goal if necessary
        dist_from_goal = np.linalg.norm(self.pose_in_map_np[:2] - self.cur_goal[:2])
        abs_angle_diff = np.abs(self.pose_in_map_np[2] - self.cur_goal[2])
        rot_dist_from_goal = min(np.pi * 2 - abs_angle_diff, abs_angle_diff)
        if dist_from_goal < TRANS_GOAL_TOL and rot_dist_from_goal < ROT_GOAL_TOL:
            rospy.loginfo("Goal {goal} at {pose} complete.".format(
                    goal=self.cur_path_index, pose=self.cur_goal))
            if self.cur_path_index == len(self.path_tuples) - 1:
                rospy.loginfo("Full path complete in {time}s! Path Follower node shutting down.".format(
                    time=(rospy.Time.now() - self.path_follow_start_time).to_sec()))
                rospy.signal_shutdown("Full path complete! Path Follower node shutting down.")
            else:
                self.cur_path_index += 1
                self.cur_goal = np.array(self.path_tuples[self.cur_path_index])
        else:
            rospy.logdebug("Goal {goal} at {pose}, trans error: {t_err}, rot error: {r_err}.".format(
                goal=self.cur_path_index, pose=self.cur_goal, t_err=dist_from_goal, r_err=rot_dist_from_goal
            ))

    def stop_robot_on_shutdown(self):
        self.cmd_pub.publish(Twist())
        rospy.loginfo("Published zero vel on shutdown.")


if __name__ == '__main__':
    try:
        rospy.init_node('path_follower', log_level=rospy.DEBUG)
        pf = PathFollower()
    except rospy.ROSInterruptException:
        pass