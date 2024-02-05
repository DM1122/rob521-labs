from matplotlib import pyplot as plt
from nodes.l2_planning import PathPlanner
from nodes.l2_planning import Node
import numpy as np
import pytest
from pathlib import Path
import time


@pytest.fixture
def path_planner_instance():
    return PathPlanner(
        map_file_path=Path("maps/willowgarageworld_05res.png"),
        map_settings_path=Path("maps/willowgarageworld_05res.yaml"),
        goal_point=np.array([[10], [0]]),
        stopping_dist=0.5,
    )


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
        goal_point=np.array([[10], [10]]),
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
        goal_point=np.array([[10], [10]]),
        stopping_dist=0.5,
    )

    output = sut.points_to_robot_circle(test_input)
    print(output)


def test_trajectory_rollout():
    sut = PathPlanner(
        map_file_path=Path("maps/willowgarageworld_05res.png"),
        map_settings_path=Path("maps/willowgarageworld_05res.yaml"),
        goal_point=np.array([[10], [10]]),
        stopping_dist=0.5,
    )

    # Control inputs
    vel = 1  # linear velocity (m/s)
    rot_vel = 0.1  # angular velocity (rad/s)

    output = sut.trajectory_rollout(vel, rot_vel)
    print(output.shape)
    print(output)
    # Plot results
    plt.figure()
    plt.plot(output[:, 0], output[:, 1], "b-", label="Trajectory")
    plt.quiver(
        output[:, 0],
        output[:, 1],
        np.cos(output[:, 2]),
        np.sin(output[:, 2]),
        scale=20,
        color="r",
    )
    plt.xlabel("x position")
    plt.ylabel("y position")
    plt.title("Robot Trajectory")
    plt.legend()
    plt.grid()
    plt.show()


def test_robot_controller_max_vel(
    initial_point=np.array([0, 0, 0]),
    final_point=np.array(
        [
            10000,
            0,
        ]
    ),
):
    """Test to ensure the robot is commanded to its highest velocity and no rotation
    when the target point is far but straight ahead"""
    sut = PathPlanner(
        map_file_path=Path("maps/willowgarageworld_05res.png"),
        map_settings_path=Path("maps/willowgarageworld_05res.yaml"),
        goal_point=np.array([[10], [10]]),
        stopping_dist=0.5,
    )

    vel, rot_vel = sut.robot_controller(initial_point, final_point)
    print(vel, rot_vel)
    assert vel == sut.vel_max
    assert rot_vel == 0


def test_robot_controller_max_rot_180(
    initial_point=np.array([0, 0, 0]),
    final_point=np.array(
        [
            -1,
            0,
        ]
    ),
):
    """Test to ensure the robot is commanded to its highest angular velocity when the target
    point is at 180 degrees from it"""
    sut = PathPlanner(
        map_file_path=Path("maps/willowgarageworld_05res.png"),
        map_settings_path=Path("maps/willowgarageworld_05res.yaml"),
        goal_point=np.array([[10], [10]]),
        stopping_dist=0.5,
    )

    vel, rot_vel = sut.robot_controller(initial_point, final_point)
    assert rot_vel == -sut.rot_vel_max


def test_robot_controller_max_rot_90(
    initial_point=np.array([0, 0, 0]),
    final_point=np.array(
        [
            0,
            1,
        ]
    ),
):
    """Test to ensure the robot is commanded to its highest angular velocity when the target
    point is at 90 degrees from it"""
    sut = PathPlanner(
        map_file_path=Path("maps/willowgarageworld_05res.png"),
        map_settings_path=Path("maps/willowgarageworld_05res.yaml"),
        goal_point=np.array([[10], [10]]),
        stopping_dist=0.5,
    )

    vel, rot_vel = sut.robot_controller(initial_point, final_point)

    assert np.isclose(rot_vel, sut.rot_vel_max)


def test_robot_controller_max_rot_minus_90(
    initial_point=np.array([0, 0, 0]),
    final_point=np.array(
        [
            0,
            -1,
        ]
    ),
):
    """Test to ensure the robot is commanded to its highest angular velocity when the target
    point is at -90 degrees from it"""
    sut = PathPlanner(
        map_file_path=Path("maps/willowgarageworld_05res.png"),
        map_settings_path=Path("maps/willowgarageworld_05res.yaml"),
        goal_point=np.array([[10], [10]]),
        stopping_dist=0.5,
    )

    vel, rot_vel = sut.robot_controller(initial_point, final_point)

    assert np.isclose(rot_vel, -sut.rot_vel_max)


@pytest.mark.parametrize("expected_output", np.array([[2, 1]]))
def test_sample_map_space(expected_output):
    sut = PathPlanner(
        map_file_path=Path("maps/willowgarageworld_05res.png"),
        map_settings_path=Path("maps/willowgarageworld_05res.yaml"),
        goal_point=np.array([[10], [10]]),
        stopping_dist=0.5,
    )

    output = sut.sample_map_space()

    assert type(output) == np.ndarray
    assert output.shape == (2, 1)
    assert output[0] <= sut.bounds[0, 1] and output[0] >= sut.bounds[0, 0]
    assert output[1] <= sut.bounds[1, 1] and output[1] >= sut.bounds[1, 0]


@pytest.mark.parametrize(
    "input, expected_output",
    [
        (np.array([[1], [2]]), True),
        (np.array([[3], [26]]), False),
    ],
)
def test_check_if_duplicate(input, expected_output):
    sut = PathPlanner(
        map_file_path=Path("maps/willowgarageworld_05res.png"),
        map_settings_path=Path("maps/willowgarageworld_05res.yaml"),
        goal_point=np.array([[10], [10]]),
        stopping_dist=0.5,
    )

    sut.nodes.append(Node(np.array([[1], [2], [0]]), -1, 0))
    sut.nodes.append(Node(np.array([[3], [25], [0]]), -1, 0))
    sut.nodes.append(Node(np.array([[24], [30], [0]]), -1, 0))
    sut.nodes.append(Node(np.array([[9], [-2], [0]]), -1, 0))

    output = sut.check_if_duplicate(input)
    assert output == expected_output


@pytest.mark.parametrize(
    "input, expected_output",
    [
        (np.array([[1], [2.1]]), 1),
        (np.array([[3], [25.5]]), 2),
        (np.array([[9], [-1.9]]), 4),
    ],
)
def test_closest_node(input, expected_output):
    sut = PathPlanner(
        map_file_path=Path("maps/willowgarageworld_05res.png"),
        map_settings_path=Path("maps/willowgarageworld_05res.yaml"),
        goal_point=np.array([[10], [10]]),
        stopping_dist=0.5,
    )
    # INIT: self.nodes = [Node(np.zeros((3,1)), -1, 0)]
    sut.nodes.append(Node(np.array([[1], [2], [0]]), -1, 0))
    sut.nodes.append(Node(np.array([[3], [25], [0]]), -1, 0))
    sut.nodes.append(Node(np.array([[24], [30], [0]]), -1, 0))
    sut.nodes.append(Node(np.array([[9], [-2], [0]]), -1, 0))

    # print(sut.nodes[0].point[1][0])
    output = sut.closest_node(input)
    assert output == expected_output


def test_rrt_planning(path_planner_instance):
    sut = PathPlanner(
        map_file_path=Path("maps/willowgarageworld_05res.png"),
        map_settings_path=Path("maps/willowgarageworld_05res.yaml"),
        goal_point=np.array([[10], [0]]),
        stopping_dist=0.5,
    )
    # nodes = sut.rrt_planning()

    # print(nodes[0].point.shape, nodes[1].shape)
    # print(nodes[0].point, nodes[1])

    # for state in nodes:
    #     print(state[:2].shape)
    #     print(state[:2].ndim)
    #     break
    # print(nodes.shape)
    # for node in nodes:
    #     print(node.point)

    # print(len(nodes))
    # for node in nodes:
    #     print(node.point)

    # print(len(nodes))
    # assert isinstance(nodes, list)  # Check if 'nodes' is a list
    # assert len(nodes) > 0  # Check if 'nodes' has at least one element


def test_rrt_star_planning(path_planner_instance):
    sut = path_planner_instance
    nodes = sut.rrt_star_planning()
    print(len(nodes))


@pytest.mark.parametrize(
    "test_input, expected_output",
    [
        (np.array([[-0.8, 0.85, 0]]), True),  # bottom left corner of first room wall
        (
            np.array([[-0.85, 0.85, 0]]),
            True,
        ),  # one cell to the left of the bottom left corner of first room wall (still true bc robot radius)
        (np.array([[0, 0, 0]]), False),  # origin
    ],
)
def test_check_collision(test_input, expected_output: bool):
    """Test for trajectory collision checking."""
    sut = PathPlanner(
        map_file_path=Path("maps/willowgarageworld_05res.png"),
        map_settings_path=Path("maps/willowgarageworld_05res.yaml"),
        goal_point=np.array([[15], [0]]),
        stopping_dist=0.5,
    )

    output = sut.check_collision(test_input)

    assert output == expected_output
