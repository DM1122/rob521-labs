from matplotlib import pyplot as plt
from nodes.l2_planning import PathPlanner
import numpy as np
import pytest
from pathlib import Path


@pytest.mark.parametrize(
    "test_input, expected_output",
    [
        (np.array([[0, 0]]), np.array([[0, 0]])),
        (np.array([[1, 1]]), np.array([[20, 20]])),
        (np.array([[5.44, 3.81]]), np.array([[108, 76]])),
        (
            np.array(
                [
                    [11, 20],
                    [16.8, 9.9],
                    [31.776, 55],
                ]
            ),
            np.array(
                [
                    [220, 400],
                    [336, 198],
                    [635, 1100],
                ]
            ),
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


def test_points_to_robot_circle():
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
    "vector_1, vector_2, expected_output",
    [
        (np.array([1, 1]), np.array([1, 0]), np.pi / 4),  # 45 degrees in radians
        (np.array([1, -1]), np.array([1, 0]), -np.pi / 4),  # 90 degrees in radians
        (np.array([-1, 1]), np.array([1, 1]), np.pi / 2),  # 90 degrees in radians
        (np.array([-2, 0]), np.array([0, 1]), np.pi),  # 180 degrees in radians
    ],
)
def test_calculate_angle_between_vectors(vector_1, vector_2, expected_output):
    output = PathPlanner.calculate_angle_between_vectors(vector_1, vector_2)
    assert np.isclose(output, expected_output)


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

    assert vel == sut.vel_max
    assert rot_vel == 0


def test_robot_controller_max_rot(
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

    assert rot_vel == sut.rot_vel_max
