from beartype import beartype
from jaxtyping import Float, Int
import numpy as np
import pygame
from pathlib import Path
from pygame.locals import QUIT, KEYUP, K_ESCAPE
import sys 

COLORS = dict(
    w=(255, 255, 255), k=(0, 0, 0), g=(0, 255, 0), r=(255, 0, 0), b=(0, 0, 255)
)


class PygameWindow:
    def __init__(
        self,
        name,
        size,
        real_map_size_pixels,
        map_file_path: Path,
        map_settings_dict,
        goal_point,
        stopping_dist,
    ):
        pygame.init()
        pygame.display.set_caption(name)

        self.size = size
        self.meters_per_pixel = (
            map_settings_dict["resolution"] / self.size[0] * real_map_size_pixels[0]
        )
        self.map_settings_dict = map_settings_dict
        self.origin = np.array(map_settings_dict["origin"])

        self.map_img = pygame.image.load(map_file_path)
        self.map_img = pygame.transform.scale(self.map_img, self.size)

        self.stopping_dist = stopping_dist
        self.goal_point = goal_point

        self.screen = pygame.display.set_mode(self.size)

        pygame.display.flip()

        full_map_height = map_settings_dict["resolution"] * real_map_size_pixels[1]
        # 80 + -49.25 since origin is relative to bottom left corner, but pygame 0, 0 is top left corner
        self.origin_pixels = (
            np.array([-self.origin[0], full_map_height + self.origin[1]])
            / self.meters_per_pixel
        )

        self.refresh_display()

    def refresh_display(self):
        # map
        self.screen.blit(self.map_img, (0, 0))

        # robot pose
        self.add_se2_pose([0, 0, 0], length=5, color=COLORS["r"])

        # goal point
        self.add_point(
            self.goal_point.flatten(),
            radius=self.stopping_dist / self.meters_per_pixel,
            color=COLORS["g"],
        )

    @beartype
    def add_point(
        self,
        map_frame_point: Float[np.ndarray, "2"] | Int[np.ndarray, "2"],
        radius=1,
        width=0,
        color=COLORS["k"],
    ):
        map_frame_point_new = np.copy(map_frame_point)
        map_frame_point_new[1] = -map_frame_point_new[1]  # for top left origin
        point_vec = self.point_to_vec(
            np.array(map_frame_point_new) / self.meters_per_pixel + self.origin_pixels
        )
        pygame.draw.circle(self.screen, color, point_vec, radius, width)
        pygame.display.update()

    def add_se2_pose(self, map_frame_pose, length=1, width=0, color=COLORS["k"]):
        map_frame_pose_new = np.copy(map_frame_pose)
        map_frame_pose_new[1] = -map_frame_pose_new[1]  # for top left origin
        l = length
        p_center = (
            np.array(map_frame_pose_new[:2]) / self.meters_per_pixel
            + self.origin_pixels
        )
        theta = map_frame_pose_new[2]

        # y terms all made opposite of expected here because of top left origin
        p_back = np.array(
            [-l * np.cos(theta) + p_center[0], l * np.sin(theta) + p_center[1]]
        )
        p_1 = np.array(
            [-l / 2 * np.sin(theta) + p_back[0], -l / 2 * np.cos(theta) + p_back[1]]
        )
        p_2 = np.array(
            [l / 2 * np.sin(theta) + p_back[0], l / 2 * np.cos(theta) + p_back[1]]
        )

        c_vec = self.point_to_vec(p_center)
        p1_vec = self.point_to_vec(p_1)
        p2_vec = self.point_to_vec(p_2)

        pygame.draw.polygon(self.screen, color, [c_vec, p1_vec, p2_vec], width=width)
        pygame.display.update()

    def add_line(
        self,
        map_frame_point1: Float[np.ndarray, "2"] | Int[np.ndarray, "2"],
        map_frame_point2: Float[np.ndarray, "2"] | Int[np.ndarray, "2"],
        width=1,
        color=COLORS["k"],
    ):
        map_frame_point1_new = np.copy(map_frame_point1)
        map_frame_point2_new = np.copy(map_frame_point2)
        map_frame_point1_new[1] = -map_frame_point1_new[1]  # for top left origin
        p1 = self.point_to_vec(
            np.array(map_frame_point1_new) / self.meters_per_pixel + self.origin_pixels
        )
        map_frame_point2_new[1] = -map_frame_point2_new[1]  # for top left origin
        p2 = self.point_to_vec(
            np.array(map_frame_point2_new) / self.meters_per_pixel + self.origin_pixels
        )
        pygame.draw.line(self.screen, color, p1, p2, width)
        pygame.display.update()

    # def remove_line(self, p1, p2, width=1, color=COLORS['w']):
    #     pygame.draw.line(self.screen, color, p1, p2, width)

    def point_to_vec(self, point):
        vec = pygame.math.Vector2()
        vec.xy = point
        return vec

    def check_for_close(self):
        for e in pygame.event.get():
            if e.type == QUIT or (e.type == KEYUP and e.key == K_ESCAPE):
                sys.exit("Closing planner.")
