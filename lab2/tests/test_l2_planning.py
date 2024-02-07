from matplotlib import pyplot as plt
from nodes.l2_planning import PathPlanner, RectBounds, Node
from nodes.l2_planning import Node
import numpy as np
import pytest
from pathlib import Path


def test_init():
    """Test constructor"""
    sut = PathPlanner(
        map_file_path=Path("maps/willowgarageworld_05res.png"),
        map_settings_path=Path("maps/willowgarageworld_05res.yaml"),
        goal_point=np.array([[10], [10]]),
        stopping_dist=0.5,
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
        goal_point=np.array([[10], [10]]),
        stopping_dist=0.5,
    )

    point = sut.sample_map_space(bounds)
    print(point)

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
        goal_point=np.array([[10], [10]]),
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
        goal_point=np.array([[10], [10]]),
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


@pytest.mark.skip(reason="Not implemented yet")
def test_simulate_trajectory():
    pass


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


def test_ball_radius():
    sut = PathPlanner(
        map_file_path=Path("maps/willowgarageworld_05res.png"),
        map_settings_path=Path("maps/willowgarageworld_05res.yaml"),
        goal_point=np.array([[10], [10]]),
        stopping_dist=0.5,
    )

    output = sut.ball_radius()
    print(output)


@pytest.mark.skip(reason="Not implemented yet")
def test_connect_node_to_point():
    pass


@pytest.mark.skip(reason="Not implemented yet")
def test_cost_to_come():
    pass


@pytest.mark.skip(reason="Not implemented yet")
def test_update_children():
    pass


@pytest.mark.skip(reason="Not implemented yet")
def test_is_trajectory_out_of_bounds():
    pass


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


@pytest.mark.skip(reason="Not implemented yet")
def test_is_goal_reached():
    pass


def test_rrt_planning():
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


def test_rrt_star_planning():
    sut = PathPlanner(
        map_file_path=Path("maps/willowgarageworld_05res.png"),
        map_settings_path=Path("maps/willowgarageworld_05res.yaml"),
        goal_point=np.array([[10], [10]]),
        stopping_dist=0.5,
    )
    nodes = sut.rrt_star_planning()
    print(len(nodes))


@pytest.mark.skip(reason="Not implemented yet")
def test_recover_path():
    pass
