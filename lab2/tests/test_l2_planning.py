from nodes.l2_planning import PathPlanner, RectBounds, Node
import numpy as np
import pytest
from pathlib import Path
import time


def test_willow_test_init():
    """Test constructor"""
    sut = PathPlanner(
        map_file_path=Path("maps/willowgarageworld_05res.png"),
        map_settings_path=Path("maps/willowgarageworld_05res.yaml"),
        goal_point=np.array([10, 10]),
        stopping_dist=0.5,
    )
    time.sleep(10)
    print(sut)


def test_myhal_init():
    """Test constructor"""

    sut = PathPlanner(
        map_file_path=Path("maps/myhal.png"),
        map_settings_path=Path("maps/myhal.yaml"),
        goal_point=np.array([2, 6]),
        stopping_dist=0.1,
    )
    print(sut)


@pytest.mark.parametrize(
    "bounds",
    [
        RectBounds(x=0, y=0, width=10, height=10),
        RectBounds(x=5, y=6, width=15, height=20),
    ],
)
def test_sample_map_space(bounds):
    sut = PathPlanner(
        map_file_path=Path("maps/willowgarageworld_05res.png"),
        map_settings_path=Path("maps/willowgarageworld_05res.yaml"),
        goal_point=np.array([10, 10]),
        stopping_dist=0.5,
    )
    for i in range(10):
        point = sut.sample_map_space(bounds)
        print(point)
    time.sleep(10)
    # Assert that the point lies within the given bounds
    assert bounds.x <= point[0] <= bounds.x + bounds.width
    assert bounds.y <= point[1] <= bounds.y + bounds.height


@pytest.mark.parametrize(
    "test_input, expected_output",
    [
        (np.array([1.0, 2.0]), False),  # Point not in the nodes list
        (np.array([3.0, 4.0]), True),  # Point already exists in the nodes list
        (np.array([5.0, 6.0]), False),  # Another point not in the nodes list
    ],
)
def test_check_if_duplicate(test_input, expected_output):
    sut = PathPlanner(
        map_file_path=Path("maps/willowgarageworld_05res.png"),
        map_settings_path=Path("maps/willowgarageworld_05res.yaml"),
        goal_point=np.array([10, 10]),
        stopping_dist=0.5,
    )

    sut.nodes = [
        Node(
            point=np.array([3.0, 4.0, 0.0]), parent_id=0, cost=0.0
        ),  # Add a node with a point that should be detected as duplicate
    ]

    output = sut.check_if_duplicate(test_input)
    print(output)

    assert output == expected_output


@pytest.mark.parametrize(
    "test_input, expected_output",
    [
        (np.array([1, 1]), 0),  # Closest to the first node
        (np.array([6, 6]), 1),  # Closest to the second node
        (np.array([9, 8]), 2),  # Closest to the third node
    ],
)
def test_closest_node(test_input, expected_output):
    # Setup the PathPlanner
    sut = PathPlanner(
        map_file_path=Path("maps/willowgarageworld_05res.png"),
        map_settings_path=Path("maps/willowgarageworld_05res.yaml"),
        goal_point=np.array([10, 10]),
        stopping_dist=0.5,
    )

    # Add predefined nodes to the PathPlanner's node list
    sut.nodes = [
        Node(point=np.array([0, 0, 0], dtype=float), parent_id=0, cost=0.0),
        Node(point=np.array([5, 5, 0], dtype=float), parent_id=0, cost=0.0),
        Node(point=np.array([10, 10, 0], dtype=float), parent_id=0, cost=0.0),
    ]

    output = sut.closest_node(test_input)

    assert output == expected_output


def test_simulate_trajectory():  # TODO: determine if this function needs to drive to completion
    sut = PathPlanner(
        map_file_path=Path("maps/willowgarageworld_05res.png"),
        map_settings_path=Path("maps/willowgarageworld_05res.yaml"),
        goal_point=np.array([10, 10]),
        stopping_dist=0.5,
    )

    output = sut.simulate_trajectory(
        point_i=np.array([0, 0, 0], dtype=float), point_s=np.array([1, 0], dtype=float)
    )
    print(output)
    if output == None:
        print("output is None")
    else:
        for point in output:
            sut.window.add_se2_pose(point)


def test_robot_controller_max_vel(
    initial_point=np.array([0, 0, 0], dtype=float),
    final_point=np.array(
        [
            10000,
            0,
        ],
        dtype=float,
    ),
):
    """Test to ensure the robot is commanded to its highest velocity and no rotation
    when the target point is far but straight ahead"""
    sut = PathPlanner(
        map_file_path=Path("maps/willowgarageworld_05res.png"),
        map_settings_path=Path("maps/willowgarageworld_05res.yaml"),
        goal_point=np.array([10, 10]),
        stopping_dist=0.5,
    )

    vel, rot_vel = sut.robot_controller(initial_point, final_point)
    print(vel, rot_vel)

    assert vel == sut.vel_max
    assert rot_vel == 0


def test_robot_controller_max_rot_180(
    initial_point=np.array([0, 0, 0], dtype=float),
    final_point=np.array(
        [
            -1,
            0,
        ],
        dtype=float,
    ),
):
    """Test to ensure the robot is commanded to its highest angular velocity when the target
    point is at 180 degrees from it"""
    sut = PathPlanner(
        map_file_path=Path("maps/willowgarageworld_05res.png"),
        map_settings_path=Path("maps/willowgarageworld_05res.yaml"),
        goal_point=np.array([10, 10]),
        stopping_dist=0.5,
    )

    vel, rot_vel = sut.robot_controller(initial_point, final_point)
    print(vel, rot_vel)

    assert rot_vel == -sut.rot_vel_max


def test_robot_controller_max_rot_90(
    initial_point=np.array([0, 0, 0], dtype=float),
    final_point=np.array(
        [
            0,
            1,
        ],
        dtype=float,
    ),
):
    """Test to ensure the robot is commanded to its highest angular velocity when the target
    point is at 90 degrees from it"""
    sut = PathPlanner(
        map_file_path=Path("maps/willowgarageworld_05res.png"),
        map_settings_path=Path("maps/willowgarageworld_05res.yaml"),
        goal_point=np.array([10, 10]),
        stopping_dist=0.5,
    )

    vel, rot_vel = sut.robot_controller(initial_point, final_point)
    print(vel, rot_vel)

    assert np.isclose(rot_vel, sut.rot_vel_max)


def test_robot_controller_max_rot_minus_90(
    initial_point=np.array([0, 0, 0], dtype=float),
    final_point=np.array(
        [
            0,
            -1,
        ],
        dtype=float,
    ),
):
    """Test to ensure the robot is commanded to its highest angular velocity when the target
    point is at -90 degrees from it"""
    sut = PathPlanner(
        map_file_path=Path("maps/willowgarageworld_05res.png"),
        map_settings_path=Path("maps/willowgarageworld_05res.yaml"),
        goal_point=np.array([10, 10]),
        stopping_dist=0.5,
    )

    vel, rot_vel = sut.robot_controller(initial_point, final_point)
    print(vel, rot_vel)

    assert np.isclose(rot_vel, -sut.rot_vel_max)


def test_trajectory_rollout():
    sut = PathPlanner(
        map_file_path=Path("maps/willowgarageworld_05res.png"),
        map_settings_path=Path("maps/willowgarageworld_05res.yaml"),
        goal_point=np.array([10, 10]),
        stopping_dist=0.5,
    )

    # Control inputs
    vel = 5.0  # linear velocity (m/s)
    rot_vel = 1.0  # angular velocity (rad/s)

    output = sut.trajectory_rollout(vel, rot_vel, np.array([1, 1, 0.0], dtype=float))
    print(output)

    for point in output:
        sut.window.add_se2_pose(point)


@pytest.mark.parametrize(
    "test_input, expected_output",
    [
        (
            np.array([[-0.8, 0.85]]),
            np.array([[598, 404]]),
        ),  # should be around bottom left corner of first wall
        (
            np.array([[-0.85, 0.85]]),
            np.array([[598, 403]]),
        ),
    ],
)
def test_point_to_cell(test_input, expected_output):
    sut = PathPlanner(
        map_file_path=Path("maps/willowgarageworld_05res.png"),
        map_settings_path=Path("maps/willowgarageworld_05res.yaml"),
        goal_point=np.array([10, 10]),
        stopping_dist=0.5,
    )

    output = sut.point_to_cell(test_input)

    assert np.array_equal(output, expected_output)


@pytest.mark.parametrize(
    "test_input",
    [
        np.array(
            [
                [-0.8, 0.8],
            ]
        ),
    ],
)
def test_points_to_robot_circle(test_input):
    sut = PathPlanner(
        map_file_path=Path("maps/willowgarageworld_05res.png"),
        map_settings_path=Path("maps/willowgarageworld_05res.yaml"),
        goal_point=np.array([10, 10]),
        stopping_dist=0.5,
    )

    output = sut.points_to_robot_circle(test_input)
    print(output)


def test_ball_radius():
    sut = PathPlanner(
        map_file_path=Path("maps/willowgarageworld_05res.png"),
        map_settings_path=Path("maps/willowgarageworld_05res.yaml"),
        goal_point=np.array([10, 10]),
        stopping_dist=0.5,
    )

    output = sut.ball_radius()
    print(output)


@pytest.mark.parametrize(
    "start_point, end_point",
    [(np.array([0, 0, 0], dtype=float), np.array([10, 2], dtype=float))],
)
def test_connect_node_to_point(start_point, end_point):
    sut = PathPlanner(
        map_file_path=Path("maps/willowgarageworld_05res.png"),
        map_settings_path=Path("maps/willowgarageworld_05res.yaml"),
        goal_point=np.array([10, 0]),
        stopping_dist=0.5,
    )

    output = sut.connect_node_to_point(
        Node(point=start_point, parent_id=-1, cost=0.0),
        end_point,
    )
    print(f"True output: {output}")

    if output is not None:
        assert np.allclose(output[-1][:2], end_point, atol=sut.stopping_dist)

        for point in output:
            sut.window.add_se2_pose(point)
    else:
        print("Collision detected")


@pytest.mark.parametrize(
    "trajectory, expected_cost",
    [
        (
            np.array([[0, 0, 0], [3, 4, 0]], dtype=float),
            5.0,
        ),  # A straight line (3-4-5 triangle), expected cost is 5
        (
            np.array([[0, 0, 0], [0, 0, 0]], dtype=float),
            0.0,
        ),  # No movement, expected cost is 0
        (
            np.array([[0, 0, 0], [3, 4, 0], [6, 8, 0]], dtype=float),
            10.0,
        ),  # A path with 3 points, expected cost is 10 (two straight lines forming a 3-4-5 triangle)
        (
            np.array([[-3, -4, 0], [0, 0, 0]], dtype=float),
            5.0,
        ),  # A line with negative coordinates, expected cost is 5 (mirroring the 3-4-5 triangle)
    ],
)
def test_cost_to_come(trajectory, expected_cost):
    sut = PathPlanner(
        map_file_path=Path("maps/willowgarageworld_05res.png"),
        map_settings_path=Path("maps/willowgarageworld_05res.yaml"),
        goal_point=np.array([10, 10]),
        stopping_dist=0.5,
    )

    output = sut.cost_to_come(trajectory)

    assert np.isclose(output, expected_cost)


def test_update_children():
    sut = PathPlanner(
        map_file_path=Path("maps/willowgarageworld_05res.png"),
        map_settings_path=Path("maps/willowgarageworld_05res.yaml"),
        goal_point=np.array([10, 10]),
        stopping_dist=0.5,
    )

    # Creating sample nodes
    node0 = Node(
        np.array([0, 0, 0], dtype=float), parent_id=-1, cost=0.0, children_ids=[1, 2]
    )
    node1 = Node(
        np.array([1, 1, 0], dtype=float), parent_id=0, cost=10.0, children_ids=[3, 4]
    )
    node2 = Node(
        np.array([1, -1, 0], dtype=float),
        parent_id=0,
        cost=20.0,
    )
    node3 = Node(np.array([3, 2, 0], dtype=float), parent_id=1, cost=30.0)
    node4 = Node(np.array([3, 3, 0], dtype=float), parent_id=1, cost=40.0)
    sut.nodes = [node0, node1, node2, node3, node4]

    # Set a new cost for the parent node (node0)
    new_cost = 5.0
    sut.update_children(0, new_cost)
    for node in sut.nodes:
        print(node.cost)

    # Assert the new costs of child nodes
    assert sut.nodes[0].cost == 5.0  # Updated cost of node0
    assert (
        sut.nodes[1].cost == 15.0
    )  # Updated cost of node1 (original cost 10.0 + delta 5.0)
    assert (
        sut.nodes[2].cost == 25.0
    )  # Updated cost of node2 (original cost 20.0 + delta 5.0)
    assert (
        sut.nodes[3].cost == 35.0
    )  # Updated cost of node3 (original cost 30.0 + delta 5.0)
    assert (
        sut.nodes[4].cost == 45.0
    )  # Updated cost of node4 (original cost 40.0 + delta 5.0)


@pytest.mark.parametrize(
    "test_input, expected_output",  # assumes self.plan_bounds =  RectBounds(x=-5, y=-47, width=55, height=60)
    [
        (np.array([[0, 0], [5, -5]], dtype=float), False),
        (np.array([[10, -12]], dtype=float), False),
        (np.array([[5, -5]], dtype=float), False),
        (np.array([[0, 0], [-30, 0]], dtype=float), True),
        (np.array([[-30, 50]], dtype=float), True),
    ],
)
def test_is_trajectory_out_of_bounds(test_input, expected_output):
    sut = PathPlanner(
        map_file_path=Path("maps/willowgarageworld_05res.png"),
        map_settings_path=Path("maps/willowgarageworld_05res.yaml"),
        goal_point=np.array([15, 0]),
        stopping_dist=0.5,
    )

    print(sut.plan_bounds)

    output = sut.is_trajectory_out_of_bounds(test_input)
    print(output)

    assert output == expected_output


@pytest.mark.parametrize(
    "test_input, expected_output",
    [
        (np.array([[-0.8, 0.85]]), True),  # bottom left corner of first room wall
        (
            np.array([[-0.85, 0.85]]),
            True,
        ),  # one cell to the left of the bottom left corner of first room wall (still true bc robot radius)
        (np.array([[0.0, 0.0]]), False),  # origin
    ],
)
def test_check_collision(test_input, expected_output: bool):
    """Test for trajectory collision checking."""
    sut = PathPlanner(
        map_file_path=Path("maps/willowgarageworld_05res.png"),
        map_settings_path=Path("maps/willowgarageworld_05res.yaml"),
        goal_point=np.array([15, 0]),
        stopping_dist=0.5,
    )

    output = sut.check_collision(test_input)

    assert output == expected_output


@pytest.mark.parametrize(
    "node_point, expected",
    [
        (np.array([3.0, 1.0, 0]), True),  # Node point at the goal
        (np.array([3.1, 1.1, 0]), True),  # Node point within stopping distance
        (np.array([5.0, 1.0, 0]), False),  # Node point outside stopping distance
    ],
)
def test_is_goal_reached(node_point, expected):
    sut = PathPlanner(
        map_file_path=Path("maps/willowgarageworld_05res.png"),
        map_settings_path=Path("maps/willowgarageworld_05res.yaml"),
        goal_point=np.array([3, 1]),
        stopping_dist=0.5,
    )

    output = sut.is_goal_reached(node_point)
    assert output == expected


def test_rrt_willow_planning():
    # the real goal = 43, -43.6
    # intermidiate goal = 3.5, -43.6
    sut = PathPlanner(
        map_file_path=Path("maps/willowgarageworld_05res.png"),
        map_settings_path=Path("maps/willowgarageworld_05res.yaml"),
        goal_point=np.array([43, -43.6]),
        stopping_dist=0.7,
    )
    start_time = time.time()
    nodes = sut.rrt_planning()

    if nodes == None:
        print("None projectory")

    print(f"total time: {time.time() - start_time }")
    time.sleep(30)
    assert isinstance(nodes, list)  # Check if 'nodes' is a list
    assert len(nodes) > 0  # Check if 'nodes' has at least one element


def test_rrt_myhal_planning():
    """Test constructor"""

    sut = PathPlanner(
        map_file_path=Path("maps/myhal.png"),
        map_settings_path=Path("maps/myhal.yaml"),
        goal_point=np.array([2, 6]),
        stopping_dist=0.05,
    )
    start_time = time.time()
    nodes = sut.rrt_planning()

    if nodes == None:
        print("None projectory")
    print(f"total time: {time.time() - start_time }")
    time.sleep(30)
    assert isinstance(nodes, list)  # Check if 'nodes' is a list
    assert len(nodes) > 0  # Check if 'nodes' has at least one element


def test_find_near_nodes():
    sut = PathPlanner(
        map_file_path=Path("maps/willowgarageworld_05res.png"),
        map_settings_path=Path("maps/willowgarageworld_05res.yaml"),
        goal_point=np.array([10, -5]),
        stopping_dist=0.5,
    )
    node0 = Node(np.array([0, 0, 0], dtype=float), -1, 0.0)
    node1 = Node(np.array([1, 0, 0], dtype=float), -1, 0.0)
    node2 = Node(np.array([0, 1, 0], dtype=float), -1, 0.0)
    node3 = Node(np.array([0, 10, 0], dtype=float), -1, 0.0)
    sut.nodes = [node0, node1, node2, node3]
    print("Near nodes at:")
    print(sut.find_near_nodes(node0.point))


def test_is_ancestor():
    sut = PathPlanner(
        map_file_path=Path("maps/willowgarageworld_05res.png"),
        map_settings_path=Path("maps/willowgarageworld_05res.yaml"),
        goal_point=np.array([10, -5]),
        stopping_dist=0.5,
    )

    node0 = Node(np.array([0, 0, 0], dtype=float), -1, 0.0, children_ids=[1, 2])
    node1 = Node(np.array([1, 0, 0], dtype=float), 0, 0.0)
    node2 = Node(np.array([0, 1, 0], dtype=float), 1, 0.0)
    node3 = Node(np.array([0, 10, 0], dtype=float), 0, 0.0)

    sut.nodes = [node0, node1, node2, node3]
    print(node0.point[0], node0.point[1])
    print(sut.goal_point[0], sut.goal_point[1])
    print(
        np.sqrt(
            (node0.point[0] - sut.goal_point[0]) ** 2
            + (node0.point[1] - sut.goal_point[1]) ** 2
        )
    )
    assert sut.is_ancestor(node0, 1) == False
    assert sut.is_ancestor(node1, 0) == True
    assert sut.is_ancestor(node2, 0) == True
    assert sut.is_ancestor(node3, 1) == False


@pytest.mark.skip()
def test_rrt_star_planning():
    sut = PathPlanner(
        map_file_path=Path("maps/willowgarageworld_05res.png"),
        map_settings_path=Path("maps/willowgarageworld_05res.yaml"),
        goal_point=np.array([10, 0]),
        stopping_dist=0.5,
    )
    nodes = sut.rrt_star_planning()
    print(len(nodes))


def test_recover_path():
    sut = PathPlanner(
        map_file_path=Path("maps/willowgarageworld_05res.png"),
        map_settings_path=Path("maps/willowgarageworld_05res.yaml"),
        goal_point=np.array([10, 0]),
        stopping_dist=0.5,
    )

    output = sut.recover_path()

    print(output)


def test_plot_graph():
    sut = PathPlanner(
        map_file_path=Path("maps/willowgarageworld_05res.png"),
        map_settings_path=Path("maps/willowgarageworld_05res.yaml"),
        goal_point=np.array([10, -5]),
        stopping_dist=0.5,
    )

    # Creating sample nodes
    node0 = Node(
        np.array([0, 0, 0], dtype=float), parent_id=-1, cost=0.0, children_ids=[1, 2]
    )
    node1 = Node(
        np.array([1, 1, 0], dtype=float), parent_id=0, cost=10.0, children_ids=[3, 4]
    )
    node2 = Node(
        np.array([1, -1, 0], dtype=float),
        parent_id=0,
        cost=20.0,
    )
    node3 = Node(np.array([3, 2, 0], dtype=float), parent_id=1, cost=30.0)
    node4 = Node(np.array([3, 3, 0], dtype=float), parent_id=1, cost=40.0)
    sut.nodes = [node0, node1, node2, node3, node4]

    output = sut.plot_graph()
    print(output)
