#!/usr/bin/env python3
# Standard Libraries
from matplotlib.font_manager import json_dump
import numpy as np
import scipy
from typeguard import typechecked
import yaml
import pygame
import time
import matplotlib.image as mpimg
from skimage.draw import disk
from scipy.linalg import block_diag
from pathlib import Path
import sys
import json
from scipy.integrate import odeint

# Conditional import if in a poetry env
print(sys.executable)
if ".venv" in sys.executable:
    from nodes import pygame_utils_custom
else:
    import pygame_utils_custom
from nodes import pygame_utils_custom


def load_map(file_path: Path):
    im = mpimg.imread(file_path)
    if len(im.shape) > 2:
        im = im[:, :, 0]
    im_np = np.array(im)  # Whitespace is true, black is false
    # im_np = np.logical_not(im_np)
    return im_np


def load_map_yaml(file_path: Path):
    with open(file_path, "r") as stream:
        map_settings_dict = yaml.safe_load(stream)
    return map_settings_dict


# Node for building a graph
class Node:
    def __init__(self, point, parent_id, cost):
        self.point = point  # A 3 by 1 vector [x, y, theta]
        self.parent_id = parent_id  # The parent node id that leads to this node (There should only every be one parent in RRT)
        self.cost = cost  # The cost to come to this node
        self.children_ids = []  # The children node ids of this node
        return


# Path Planner
class PathPlanner:
    # A path planner capable of perfomring RRT and RRT*
    def __init__(
        self, map_file_path: Path, map_settings_path: Path, goal_point, stopping_dist
    ):
        # Get map information
        self.occupancy_map = load_map(map_file_path)
        self.map_shape = self.occupancy_map.shape
        self.map_settings_dict = load_map_yaml(map_settings_path)

        # Get the metric bounds of the map
        # origin, the upper right point (real world) because it is multiplied by the resolution
        self.bounds = np.zeros([2, 2])  # m
        self.bounds[0, 0] = self.map_settings_dict["origin"][0]
        self.bounds[1, 0] = self.map_settings_dict["origin"][1]
        self.bounds[0, 1] = (
            self.map_settings_dict["origin"][0]
            + self.map_shape[1] * self.map_settings_dict["resolution"]
        )
        self.bounds[1, 1] = (
            self.map_settings_dict["origin"][1]
            + self.map_shape[0] * self.map_settings_dict["resolution"]
        )
        print(self.bounds)
        # Robot information
        self.robot_radius = 0.22  # m
        self.vel_max = 0.5  # m/s (Feel free to change!)
        self.rot_vel_max = 0.2  # rad/s (Feel free to change!)

        # Goal Parameters
        self.goal_point = goal_point  # m
        self.stopping_dist = stopping_dist  # m # the minimum distance btw the goal point and the final position where the robot should come to stop

        # Trajectory Simulation Parameters
        self.timestep = 1.0  # s
        self.num_substeps = 10

        # Planning storage
        self.nodes = [Node(np.zeros((3, 1)), -1, 0)]

        # RRT* Specific Parameters
        self.lebesgue_free = (
            np.sum(self.occupancy_map) * self.map_settings_dict["resolution"] ** 2
        )
        self.zeta_d = np.pi
        self.gamma_RRT_star = (
            2 * (1 + 1 / 2) ** (1 / 2) * (self.lebesgue_free / self.zeta_d) ** (1 / 2)
        )
        self.gamma_RRT = self.gamma_RRT_star + 0.1
        self.epsilon = 2.5

        # Pygame window for visualization
        self.window = pygame_utils_custom.PygameWindow(
            "Path Planner",
            (1000, 1000),
            self.occupancy_map.shape,
            map_file_path,
            self.map_settings_dict,
            self.goal_point,
            self.stopping_dist,
        )
        return

    # Functions required for RRT
    @typechecked
    def sample_map_space(self) -> np.ndarray:
        """
        select and return the random point lies within the map boundary
        return an [x,y] coordinate to drive the robot towards

        return: (2, 1) shape point
        """
        # self.map_shape, self.boundary
        # (b - a) * random_sample() + a
        random_p = np.random.random_sample(2)
        random_x = (self.bounds[0, 1] - self.bounds[0, 0]) * random_p[0] + self.bounds[
            0, 0
        ]
        random_y = (self.bounds[1, 1] - self.bounds[1, 0]) * random_p[1] + self.bounds[
            1, 0
        ]

        return np.array([random_x, random_y]).reshape(2, 1)

    def check_if_duplicate(self, point) -> bool:
        """
        Check if point is a duplicate of an already existing node in the list
        Class Node object are in self.nodes, and assumes that point is numpy array ((2,1))

        # Assumed that the point is in shape of (2,1)

        Input: A point to check if it already exist in the node list
        Return: Boolean
        """

        # self.nodes = [Node(np.zeros((3, 1)), -1, 0)]
        for cur_node in self.nodes:
            if (
                cur_node.point[0][0] == point[0][0]
                and cur_node.point[1][0] == point[1][0]
            ):
                return True
        return False

    def closest_node(self, point):
        """
        Implement a method to get the closest node to a sapled point.
        Assumed that the point in shape of (2,1)

        Input: A current point needs to find the closest node
        Return: the index of the closest node
        """
        min_distance = 100000000
        min_idx = -1
        if len(self.nodes) >= 1:
            for idx, cur_node in enumerate(self.nodes):
                # print(cur_node.point.shape, point.shape)
                cur_distance = np.sqrt(
                    (cur_node.point[0] - point[0][0]) ** 2
                    + (cur_node.point[1] - point[1][0]) ** 2
                )
                min_distance = min(cur_distance, min_distance)
                if min_distance == cur_distance:
                    min_idx = idx
            return min_idx
        assert len(self.nodes) != 0

    @typechecked
    def simulate_trajectory(
        self, node_i: np.ndarray, point_s: np.ndarray
    ) -> np.ndarray:
        """Simulates the non-holonomic motion of a robot towards a target point.

        This function drives the robot from its current state (node_i) towards a
        specified target point (point_s). It uses a robot controller to calculate
        the necessary linear and angular velocities and then simulates the robot's
        trajectory using these velocities.

        Args:
            node_i (numpy.array): A 3x1 vector representing the current state of the
                                robot. It includes the robot's x and y
                                coordinates and its orientation theta, i.e., [x; y; theta].
            point_s (numpy.array): A 2x1 vector representing the target point in Cartesian
                                coordinates, i.e., [x; y].

        Returns:
            numpy.array: An array representing the simulated trajectory of the robot.
        """
        vel, rot_vel = self.robot_controller(node_i, point_s)

        robot_traj = self.trajectory_rollout(vel, rot_vel)
        return robot_traj

    @typechecked
    def robot_controller(self, node_i, point_s) -> tuple[float, float]:
        """
        This controller determines the velocities that will nominally move the robot from node i to node s.
        Adjust the linear and rotational velocities based on the distance and the angular difference between
        the robot's current position (node_i) and the target point (point_s).
        Max velocities should be enforced.
        """

        x_i, y_i, theta_i = node_i  # robot current position
        x_s, y_s = point_s  # target point

        # Calculate the distance from the robot to the target point
        distance = np.sqrt((x_s - x_i) ** 2 + (y_s - y_i) ** 2)

        # Calculate the angle to the target point considering the robot's current orientation
        angle_to_target = np.arctan2(y_s - y_i, x_s - x_i) - theta_i

        # Normalize the angle_to_target to the range [-pi, pi]
        angle_to_target = (angle_to_target + np.pi) % (2 * np.pi) - np.pi

        # Scale the rotational velocity based on the angular difference
        # Apply maximum angular velocity at ±90 to ±180 degrees
        # Scale linearly to 0 as it approaches ±0 degrees
        if np.abs(angle_to_target) > np.pi / 2:
            rotational_vel = self.rot_vel_max
        else:
            rotational_vel = (2 * self.rot_vel_max / np.pi) * np.abs(angle_to_target)

        # Ensure rotational velocity is in the correct direction
        rotational_vel *= np.sign(angle_to_target)

        # Calculate proportional linear velocity
        linear_vel = 1 * distance

        # Enforce maximum velocity limits
        linear_vel = min(linear_vel, self.vel_max)
        rotational_vel = min(np.abs(rotational_vel), self.rot_vel_max) * np.sign(
            rotational_vel
        )

        return linear_vel, rotational_vel

    @typechecked
    def trajectory_rollout(self, vel: float, rot_vel: float) -> np.ndarray:
        """
        Compute the trajectory of a robot given linear and angular velocities.

        This method calculates the future trajectory of the robot over a fixed time horizon based on the provided linear velocity `vel` and rotational velocity `rot_vel`. The trajectory is determined by solving a system of ordinary differential equations (ODEs) that describe the robot's motion. The initial position of the robot is assumed to be at the origin of its reference frame.
        Assumes the robot starts from the origin (x=0, y=0) with an initial angle of 0 radians.

        Args:
            vel (float): The linear velocity of the robot in meters per second.
            rot_vel (float): The rotational velocity of the robot in radians per second.

        Returns:
            np.ndarray: An N x 3 matrix representing the trajectory of the robot. Each row corresponds to a point in time, with the first column being the x-coordinate, the second column the y-coordinate, and the third column the angle of the robot with respect to its initial orientation.
        """

        def system_dynamics(state, t, v, omega):
            # Define the system of ODEs
            x, y, theta = state
            x_dot = v * np.cos(theta)
            y_dot = v * np.sin(theta)
            theta_dot = omega
            return [x_dot, y_dot, theta_dot]

        # Initial conditions
        x_0 = 0
        y_0 = 0
        theta_0 = 0

        t_proj = 10  # numer of seconds to project into the future
        t = np.linspace(0, t_proj, 100)

        # Solve ODE
        solution = odeint(
            func=system_dynamics, y0=[x_0, y_0, theta_0], t=t, args=(vel, rot_vel)
        )

        return solution  # (N, 3)

    @typechecked
    def point_to_cell(self, point: np.ndarray) -> np.ndarray:
        """Converts a series of [x,y] points in the map to occupancy map cell indices.

        This function computes the cell indices in the occupancy map for each provided [x, y] point.
        The points are assumed to be expressed in the map's bottom left corner reference frame.

        Args:
            point (np.ndarray): An N by 2 matrix of points of interest, where N is the number of points.

        Returns:
            np.ndarray: An array of cell indices in the occupancy map corresponding to each input point.
                        The output is an N by 2 matrix, where the first column contains x indices and the
                        second column contains y indices.
        """
        if point.ndim != 2:
            raise ValueError(
                f"Input array must be 2-dimensional, received {point.ndim}"
            )

        if point.shape[1] != 2:
            raise ValueError(
                f"Input array must have a shape of Nx2, received: {point.shape}"
            )

        return np.floor(point * self.map_settings_dict["resolution"] ** -1).astype(int)

    @typechecked
    def points_to_robot_circle(self, points: np.ndarray) -> list[np.ndarray]:
        """
        Converts a series of [x, y] coordinates to robot map footprints for collision detection.

        This function calculates the pixel locations of a robot's path and converts them into
        footprints based on the robot's radius. These footprints are used for collision detection.

        Args:
            points (np.ndarray): A 2-dimensional numpy array of shape Nx2, where each row
                represents an [x, y] coordinate.

        Returns:
            list[np.ndarray]: A list of numpy arrays. Each array in the list is an Nx2 array
                representing the coordinates of a circle footprint around each point.

        Note:
            The size of the circle footprint is determined by the robot's radius and the map settings.
        """

        if points.ndim != 2:
            raise ValueError(
                f"Input array must be 2-dimensional, received {points.ndim}"
            )

        if points.shape[1] != 2:
            raise ValueError(
                f"Input array must have a shape of Nx2, received: {points.shape}"
            )

        robot_radius_in_cells = (
            self.map_settings_dict["resolution"] ** -1 * self.robot_radius
        )

        circles = []
        for i in range(points.shape[0]):
            point = points[i : i + 1]
            cell_coords = self.point_to_cell(point)[0]
            x_coords, y_coords = disk(
                center=cell_coords,
                radius=robot_radius_in_cells,
                shape=self.map_shape,
            )
            circle_coords = np.vstack(
                (x_coords, y_coords)
            ).T  # Stacking and transposing
            circles.append(circle_coords)

        return circles

    # Note: If you have correctly completed all previous functions, then you should be able to create a working RRT function

    # RRT* specific functions
    def ball_radius(self):
        # Close neighbor distance
        card_V = len(self.nodes)
        return min(
            self.gamma_RRT * (np.log(card_V) / card_V) ** (1.0 / 2.0), self.epsilon
        )

    def connect_node_to_point(self, node_i, point_f):
        """
        Generates a trajectory from the point in node_i to point_f in a 3xN array, where N
        represents the # of substeps. Assumes that we are taking a straight-line path between
        the two points and that the robot's orientation does not change while travelling from
        node_i to point_f.
        Args:
            node_i (node): Origin node
            point_f (point): Destination point

        Returns:
            path (np.array): A Nx3 array representing the path from node_i to point_f
        """
        # Given two nodes find the non-holonomic path that connects them
        # Settings
        # node is a 3 by 1 node
        # point is a 2 by 1 point
        path = np.zeros((3, self.num_substeps))
        path[:, 0] = node_i.point.flatten()

        # Calculate the step size for x and y
        dx = (point_f[0] - node_i.point[0]) / self.num_substeps
        dy = (point_f[1] - node_i.point[1]) / self.num_substeps

        # Fill in the path
        for i in range(1, self.num_substeps):
            path[i, 0] = path[i - 1, 0] + dx
            path[i, 1] = path[i - 1, 1] + dy
            path[i, 2] = node_i.point[2]  # Keep theta the same

        return path

    def cost_to_come(self, trajectory_o):
        """
        Computes the total Euclidean distance travelled between all substeps in a given trajectory.

        Args:
            trajectory_o (np.array): Nx3 array of the path taken betweeen two points.

        Returns:
            cost (float): Cost of traversing the given trajectory.
        """
        # The cost to get to a node from lavalle
        cost = 0.0
        for i in range(1, self.num_substeps):
            dx = trajectory_o[i, 0] - trajectory_o[i - 1, 0]
            dy = trajectory_o[i, 1] - trajectory_o[i - 1, 1]
            cost += np.sqrt(dx**2 + dy**2)
        return cost

    @typechecked
    def update_children(self, node_id: int):
        """
        Recursively updates the cost of child nodes based on the cost of their parent node.

        This method iterates over all children of a specified node, identified by `node_id`, and
        recalculates their cost based on the cost of the parent node and the cost of the trajectory
        between the parent and each child. This updated cost is then assigned to each child node. The
        method is applied recursively to update the cost of all descendant nodes in the tree.

        Args:
            node_id (int): The identifier of the parent node whose children's costs are to be updated.
        """
        node = self.nodes[node_id]
        parent_cost = node.cost

        for child_id in node.children_ids:
            child_node = self.nodes[child_id]

            trajectory = self.connect_node_to_point(node.point, child_node.point)
            trajectory_cost = self.cost_to_come(trajectory)

            child_node.cost = parent_cost + trajectory_cost

            # Recursively update the children of this child node
            self.update_children(child_id)

    @typechecked
    def check_collision(self, trajectory: np.ndarray) -> bool:
        if trajectory.ndim != 2:
            raise ValueError(
                f"Input array must be 2-dimensional, received {trajectory.ndim}"
            )

        if trajectory.shape[1] != 3:
            raise ValueError(
                f"Input array must have a shape of Nx3, received: {trajectory.shape}"
            )

        cell = self.point_to_cell(
            trajectory[:, 0:2]
        )  # convert the all the points to cell coordinates

        for cur_cell in cell:
            # check if the cell is within the bounds of the map
            if not (
                0 <= cur_cell[0]
                and cur_cell[0] < self.map_shape[1]
                and 0 <= cur_cell[1]
                and cur_cell[1] < self.map_shape[0]
            ):
                return True  # out of bound

        # N * (61, 2) = N point, 61 points that make up a circle (61*2)
        footprints = self.points_to_robot_circle(
            trajectory[:, 0:2]
        )  # robot's footprint in map coordinates

        # check each cell in the footprint for collsion
        for footprint in footprints:
            for point in footprint:
                circle_x, circle_y = point
                # Make sure the point is within the map bounds
                if not (
                    0 <= circle_x
                    and circle_x < self.map_shape[1]
                    and 0 <= circle_y
                    and circle_y < self.map_shape[0]
                ):
                    continue  # Skip checking if outside the map
                # Check if the point is an obstacle
                # if 0, the robot cannn't go there there is an obstacle.
                if (
                    self.occupancy_map[circle_y, circle_x] == 0
                ):  # on image, y-th "row", x-th column
                    # Collision detected
                    return True

        return False  # No collsion detected

    # Planner Functions
    def rrt_planning(self):
        """
        RRT alogrithm on the given map and robot
        1) sample one random point
        2) find the closest point in the node list
        3) find trajectory to the closest point
        4) check for the collision (if the trajectory collides with an obstacle)
            - if path to NEW_STATE is collision free
                - Add end point
                - Add path from nearest node to end point
        5) retrun success/failure and current tree
        """

        def is_goal_reached(node_point):
            """
            Check if the goal has been reached within the stopping distance.

            Args:
            node_point (np.ndarray): The current node point as a numpy array [x, y, theta].

            Returns:
            bool: True if the goal is reached, False otherwise.
            """
            # Extract the x, y coordinates of the node point and the goal point
            node_x, node_y = node_point[0], node_point[1]
            goal_x, goal_y = self.goal_point[0], self.goal_point[1]

            # Compute the Euclidean distance between the current node and the goal point
            distance_to_goal = np.sqrt((goal_x - node_x) ** 2 + (goal_y - node_y) ** 2)

            # Check if the distance is within the specified stopping distance
            if distance_to_goal <= self.stopping_dist:
                return True
            else:
                return False

        def rrt_cost_come(trajectory_o):
            cost = 0.0
            for i in range(1, len(trajectory_o)):
                # Calculate the distance between consecutive points
                dist = np.linalg.norm(trajectory_o[i] - trajectory_o[i - 1])

                # Accumulate the cost
                cost += dist

            return cost

        n_iteration = 1000
        for i in range(n_iteration):
            point = self.sample_map_space()

            closest_node_id = self.closest_node(point)
            closest_node = self.nodes[closest_node_id]  # (3, 1)

            # Simulate driving the robot towards the closest point
            trajectory_o = self.simulate_trajectory(
                closest_node.point.reshape(3), point.reshape(2)
            )  # (100,3)

            # Check for collisions and add safe points to list of nodes.
            if not self.check_collision(trajectory_o):
                # If no collision, Add the new node
                new_node_point = trajectory_o[-1]  # The last point of the trajectory
                new_node_cost = (
                    closest_node.cost + rrt_cost_come(trajectory_o)
                )  # update cost-to-come in rrt planning but does not use it to rewire the edge
                new_node = Node(new_node_point, closest_node_id, new_node_cost)
                self.nodes.append(new_node)
                new_node_id = len(self.nodes) - 1
                closest_node.children_ids.append(new_node_id)

                # Step 6: Check if goal is reached
                if is_goal_reached(new_node_point):
                    return self.nodes
        return self.nodes

    def rrt_star_planning(self):
        # This function performs RRT* for the given map and robot
        for i in range(
            1
        ):  # Most likely need more iterations than this to complete the map!
            # Sample
            point = self.sample_map_space()

            # Closest Node
            closest_node_id = self.closest_node(point)

            # Simulate trajectory
            trajectory_o = self.simulate_trajectory(
                self.nodes[closest_node_id].point, point
            )

            # Check for Collision
            print("TO DO: Check for collision.")

            # Last node rewire
            print("TO DO: Last node rewiring")

            # Close node rewire
            print("TO DO: Near point rewiring")

            # Check for early end
            print("TO DO: Check for early end")
        return self.nodes

    def recover_path(self, node_id=-1):
        path = [self.nodes[node_id].point]
        current_node_id = self.nodes[node_id].parent_id
        while current_node_id > -1:
            path.append(self.nodes[current_node_id].point)
            current_node_id = self.nodes[current_node_id].parent_id
        path.reverse()
        return path


def main():
    # Set map information
    map_file_path = Path("../maps/willowgarageworld_05res.png")
    map_settings_path = Path("../maps/willowgarageworld_05res.yaml")

    # robot information
    goal_point = np.array([[10], [10]])  # m
    stopping_dist = 0.5  # m

    # RRT precursor
    path_planner = PathPlanner(
        map_file_path, map_settings_path, goal_point, stopping_dist
    )
    nodes = path_planner.rrt_star_planning()
    node_path_metric = np.hstack(path_planner.recover_path())

    # Leftover test functions
    np.save("shortest_path.npy", node_path_metric)


if __name__ == "__main__":
    main()
