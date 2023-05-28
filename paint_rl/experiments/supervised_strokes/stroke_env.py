import random
from typing import Any, Optional
import gymnasium as gym
import numpy as np
from PIL import Image, ImageDraw #type: ignore
import pygame

from paint_rl.experiments.supervised_strokes.gen_supervised import gen_curve_points, rand_point

SCREEN_SCALE = 4

class StrokeEnv(gym.Env):
    """
    An environment where the agent must copy the given stroke.
    """

    def __init__(self, img_size: int, render_mode: Optional[str] = None) -> None:
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
        self.ref = np.zeros([0, img_size, img_size])
        self.canvas = np.zeros([0, img_size, img_size])
        self.last_pos = (0, 0)
        self.num_strokes = 0
        self.render_mode = render_mode
        if self.render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode((img_size * SCREEN_SCALE, img_size * SCREEN_SCALE))
            self.clock = pygame.time.Clock()


    def step(
        self, action: tuple[float, float, float, float]
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        mid_point = (int((action[0] + 1.0) * self.img_size / 2), int((action[1] + 1.0) * self.img_size / 2))
        end_point = (int((action[2] + 1.0) * self.img_size / 2), int((action[3] + 1.0) * self.img_size / 2))
        img = Image.new("1", (self.img_size, self.img_size))
        draw = ImageDraw.Draw(img)
        points = gen_curve_points(self.last_pos, mid_point, end_point)
        draw.line(points, "white", 4)
        new_stroke = np.array(img)
        self.canvas = np.minimum(self.canvas + new_stroke, 1.0)
        self.last_pos = end_point
        pos_channel = np.zeros([self.img_size, self.img_size])
        pos_channel[self.last_pos[1]][self.last_pos[0]] = 1
        obs = np.stack([self.ref, self.canvas, pos_channel])
        return obs, 0.0, False, False, {}

    def reset(self, **kwargs) -> tuple[np.ndarray, dict[str, Any]]:
        img = Image.new("1", (self.img_size, self.img_size))
        draw = ImageDraw.Draw(img)
        self.num_strokes = random.randrange(3, 8)
        self.last_pos = rand_point(self.img_size)
        curr_pos = self.last_pos
        for _ in range(self.num_strokes):
            mid_point = rand_point(self.img_size)
            end_point = rand_point(self.img_size)
            points = gen_curve_points(curr_pos, mid_point, end_point)
            draw.line(points, "white", 4)
            curr_pos = end_point
        self.ref = np.array(img)
        self.canvas = np.zeros([self.img_size, self.img_size])
        pos_channel = np.zeros([self.img_size, self.img_size])
        pos_channel[self.last_pos[1]][self.last_pos[0]] = 1
        obs = np.stack([self.ref, self.canvas, pos_channel])
        return obs, {}

    def render(self) -> None:
        if self.render_mode == "human":
            pos_channel = np.zeros([self.img_size, self.img_size])
            pos_channel[self.last_pos[1]][self.last_pos[0]] = 1
            img = np.stack([self.ref, self.canvas, pos_channel]).swapaxes(0, 2) * 255.0
            img_surf = pygame.surfarray.make_surface(img)
            pygame.transform.scale(img_surf, (self.img_size * SCREEN_SCALE, self.img_size * SCREEN_SCALE), self.screen)
            pygame.display.flip()
            self.clock.tick(20)