import random
from typing import Any, List, Optional
import gymnasium as gym
import numpy as np
from PIL import Image, ImageDraw  # type: ignore
import pygame
import torch
from torch import nn

from paint_rl.experiments.supervised_strokes.gen_supervised import (
    MAX_DIST,
    MIN_DIST,
    gen_curve_points,
    gen_shape,
    rand_point,
)

SCREEN_SCALE = 4


class RefStrokeEnv(gym.Env):
    """
    An environment where given an image, the agent must draw a stroked version of it.
    """

    def __init__(
        self,
        img_size: int,
        ref_imgs: List[np.ndarray],
        reward_model: Optional[nn.Module],
        render_mode: Optional[str] = None,
    ) -> None:
        """
        img_size: Size of the image.
        ref_imgs: Pool of reference images to choose from. If None, model is not used.
        reward_model: Classifier that acts as a reward signal.
        """
        super().__init__()
        self.observation_space = gym.spaces.Box(0.0, 1.0, [5, img_size, img_size])
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
        self.canvas = np.zeros([0, img_size, img_size])
        self.last_pos = (0, 0)
        self.num_strokes = 0
        self.counter = 0
        self.render_mode = render_mode
        self.reward_model = reward_model
        self.last_score = 0.0
        self.canvas_img = Image.new("1", (self.img_size, self.img_size))
        self.canvas_draw = ImageDraw.Draw(self.canvas_img)
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
        new_stroke = np.array(self.canvas_img)
        self.canvas = np.minimum(self.canvas + new_stroke, 1.0)

        self.counter += 1
        done = self.counter == self.num_strokes

        self.last_pos = end_point

        score = 0.0
        reward = 0.0
        if self.reward_model:
            reward_inpt = (
                torch.from_numpy(
                    np.concatenate(
                        [self.ref, np.array(self.canvas)[np.newaxis, ...]], 0
                    )
                )
                .unsqueeze(0)
                .float()
            )
            score = torch.softmax(self.reward_model(reward_inpt).squeeze(0), 0)[1].item()
            reward = score - self.last_score
        self.last_score = score

        pos_channel = self.gen_pos_channel(
            self.last_pos[0], self.last_pos[1], self.img_size
        )
        obs = np.concatenate([np.stack([self.canvas, pos_channel]), self.ref])

        return obs, reward, done, False, {}

    def reset(self, **kwargs) -> tuple[np.ndarray, dict[str, Any]]:
        self.counter = 0
        self.canvas = np.zeros([self.img_size, self.img_size])
        index = random.randrange(0, len(self.ref_imgs))
        self.ref = self.ref_imgs[index]
        self.num_strokes = 20
        self.last_pos = rand_point(
            MIN_DIST, MAX_DIST, prev=(self.img_size // 2, self.img_size // 2)
        )

        pos_channel = self.gen_pos_channel(
            self.last_pos[0], self.last_pos[1], self.img_size
        )
        obs = np.concatenate([np.stack([self.canvas, pos_channel]), self.ref])
        self.canvas_draw.rectangle((0, 0, self.img_size, self.img_size), 0)
        reward_inpt = (
            torch.from_numpy(
                np.concatenate(
                    [self.ref, np.zeros([1, self.img_size, self.img_size])], 0
                )
            )
            .unsqueeze(0)
            .float()
        )
        self.last_score = 0.0
        if self.reward_model:
            self.last_score = torch.softmax(self.reward_model(reward_inpt).squeeze(0), 0)[1].item()
        return obs, {}

    def render(self) -> None:
        if self.render_mode == "human":
            pos_channel = np.zeros([self.img_size, self.img_size])
            pos_channel[self.last_pos[1]][self.last_pos[0]] = 1
            img = (
                np.stack(
                    [self.canvas, self.ref.mean(0, keepdims=False), pos_channel]
                ).transpose(1, 2, 0)
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
