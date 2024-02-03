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
