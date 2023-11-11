import random
from typing import *
import gymnasium as gym
import numpy as np
from PIL import Image, ImageDraw  # type: ignore
import pygame
from scipy.ndimage import gaussian_filter  # type: ignore

from paint_rl.experiments.supervised_strokes.gen_supervised import (
    MAX_DIST,
    MIN_DIST,
    gen_curve_points,
    gen_shape,
    rand_point,
)

SCREEN_SCALE = 4


class StrokeEnv(gym.Env):
    """
    An environment where the agent must copy the given stroke.
    """

    def __init__(self, img_size: int, ref_imgs: Optional[List[np.ndarray]] = None, render_mode: Optional[str] = None) -> None:
        super().__init__()
        self.observation_space = gym.spaces.Box(0.0, 1.0, [3, img_size, img_size])
        self.action_space = gym.spaces.Box(
            -1.0,
            1.0,
            [
                4,
            ],
        )
        self.img_size = img_size
        self.ref_imgs = ref_imgs
        self.ref = np.zeros([0, img_size, img_size])
        self.ref_cmp = np.zeros([0, img_size, img_size])
        self.canvas = np.zeros([0, img_size, img_size])
        self.last_pos = (0, 0)
        self.num_strokes = 0
        self.counter = 0
        self.render_mode = render_mode
        self.max_diff = img_size**2
        self.canvas_img = Image.new("1", (self.img_size, self.img_size))
        self.canvas_draw = ImageDraw.Draw(self.canvas_img)
        self.last_diff = 0
        if self.render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode(
                (img_size * SCREEN_SCALE, img_size * SCREEN_SCALE)
            )
            self.clock = pygame.time.Clock()

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = np.clip(action.squeeze() * self.img_size, 0, self.img_size - 1)
        mid_point = (int(action[0]), int(action[1]))
        end_point = (int(action[2]), int(action[3]))
        points = gen_curve_points(self.last_pos, mid_point, end_point)
        self.canvas_draw.line(points, 1, 1)
        new_stroke = np.array(self.canvas_img).swapaxes(0, 1)
        self.canvas = np.minimum(self.canvas + new_stroke, 1.0)

        diff = (np.abs(self.canvas - self.ref_filter)).sum() / self.max_diff
        reward = -(diff - self.last_diff)
        self.last_diff = diff

        self.counter += 1
        trunc = self.counter == self.num_strokes

        pixels_moved = abs(self.last_pos[0] - end_point[0]) + abs(
            self.last_pos[1] - end_point[1]
        )
        if pixels_moved <= 4:
            reward -= 0.1
        self.last_pos = end_point

        pos_channel = self.gen_pos_channel(
            self.last_pos[0], self.last_pos[1], self.img_size
        )
        obs = np.stack([self.canvas, self.ref, pos_channel])

        return obs, reward, False, trunc, {}

    def reset(self, **kwargs) -> tuple[np.ndarray, dict[str, Any]]:
        self.counter = 0
        self.canvas = np.zeros([self.img_size, self.img_size])
        if self.ref_imgs:
            index = random.randrange(0, len(self.ref_imgs))
            img = self.ref_imgs[index]
            self.num_strokes = 10
            self.last_pos = rand_point(
                MIN_DIST, MAX_DIST, prev=(self.img_size // 2, self.img_size // 2)
            )
            self.ref = img.swapaxes(0, 1)
            self.ref_filter = (
                self.ref + gaussian_filter(np.float32(self.ref), sigma=4) * 2
            ).clip(0, 1)
        else:
            img, path, num_strokes = gen_shape(self.img_size)
            self.num_strokes = num_strokes
            self.last_pos = path[0]
            img_data = np.array(img).transpose(2, 1, 0) / 255.0
            self.ref = img_data[1]
            self.canvas = img_data[0]
            self.ref_filter = (
                self.ref * 2 + gaussian_filter(np.float32(self.ref), sigma=4) * 4
            ).clip(0, 1)

        pos_channel = self.gen_pos_channel(
            self.last_pos[0], self.last_pos[1], self.img_size
        )
        obs = np.stack([self.canvas, self.ref, pos_channel])
        self.canvas_draw.rectangle((0, 0, self.img_size, self.img_size), 0)
        self.last_diff = (np.abs(self.canvas - self.ref_filter)).sum() / self.max_diff
        return obs, {}

    def render(self) -> None:
        if self.render_mode == "human":
            pos_channel = np.zeros([self.img_size, self.img_size])
            pos_channel[self.last_pos[0]][self.last_pos[1]] = 1
            img = (
                np.stack([self.canvas, self.ref_filter, pos_channel]).swapaxes(0, 2)
                * 255.0
            )
            img_surf = pygame.surfarray.make_surface(img)
            pygame.transform.scale(
                img_surf,
                (self.img_size * SCREEN_SCALE, self.img_size * SCREEN_SCALE),
                self.screen,
            )
            pygame.display.flip()
            self.clock.tick(20)

    def gen_pos_channel(self, x: int, y: int, img_size: int) -> np.ndarray:
        pos_layer_x = (
            np.abs(np.arange(0 - x, img_size - x))[np.newaxis, ...].repeat(img_size, 0)
            / img_size
        )
        pos_layer_y = (
            np.abs(np.arange(0 - y, img_size - y))[np.newaxis, ...]
            .repeat(img_size, 0)
            .T
            / img_size
        )
        return np.sqrt(pos_layer_x**2 + pos_layer_y**2)