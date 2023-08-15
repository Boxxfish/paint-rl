import random
from typing import Tuple, List, Optional, Any
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

    ## Actions:

    ### Continuous:
        0. 0.0 - 1.0 midpoint X coord.
        1. 0.0 - 1.0 midpoint Y coord.
        2. 0.0 - 1.0 endpoint X coord.
        3. 0.0 - 1.0 endpoint Y coord.

    ### Discrete:
        0. Pen up.
        1. Pen down.
    """

    def __init__(
        self,
        canvas_size: int,
        img_size: int,
        ref_imgs: List[np.ndarray],
        reward_model: Optional[nn.Module],
        render_mode: Optional[str] = None,
        stroke_width=1,
        max_strokes=50,
    ) -> None:
        """
        canvas_size: Original canvas size. This will be downsampled to `img_size`.
        img_size: Size of the image.
        ref_imgs: Pool of reference images to choose from. If None, model is not used.
        reward_model: Classifier that acts as a reward signal.
        """
        super().__init__()
        self.observation_space = gym.spaces.Box(0.0, 1.0, [7, img_size, img_size])
        self.action_space = gym.spaces.Tuple(
            [
                gym.spaces.Box(
                    -1.0,
                    1.0,
                    [
                        4,
                    ],
                ),
                gym.spaces.Discrete(2),
            ]
        )
        self.max_strokes = max_strokes
        self.canvas_size = canvas_size
        self.stroke_width = stroke_width
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
        self.last_pen_down = True
        self.canvas_img = Image.new("1", (self.canvas_size, self.canvas_size))
        self.canvas_draw = ImageDraw.Draw(self.canvas_img)
        self.prev_frame = np.zeros([2, self.img_size, self.img_size])
        if self.render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode(
                (img_size * SCREEN_SCALE, img_size * SCREEN_SCALE)
            )
            self.clock = pygame.time.Clock()

    def step(
        self,
        action: Tuple[np.ndarray, int],
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        cont_action, disc_action = action
        cont_action = np.clip(
            cont_action.squeeze() * self.canvas_size, 0, self.canvas_size - 1
        )
        mid_point = (int(cont_action[0]), int(cont_action[1]))
        end_point = (int(cont_action[2]), int(cont_action[3]))
        pen_down = disc_action == 1
        if pen_down:
            points = gen_curve_points(self.last_pos, mid_point, end_point)
            self.canvas_draw.line(points, 1, width=self.stroke_width)
            scaled_canvas = self.canvas_img.convert("RGB").resize(
                (self.img_size, self.img_size), resample=Image.Resampling.BILINEAR
            )
            new_stroke = np.array(scaled_canvas).transpose(2, 0, 1)[1] / 255.0
            self.canvas = new_stroke

        self.counter += 1
        trunc = self.counter == self.num_strokes

        self.last_pos = end_point

        score = 0.0
        reward = 0.0

        # Penalize refusing to put down strokes
        if not self.last_pen_down and not pen_down:
            reward = -1.0
        self.last_pen_down = pen_down

        if self.reward_model:
            reward_inpt = (
                torch.from_numpy(
                    add_random(np.concatenate(
                        [self.ref, (np.array(self.canvas))[np.newaxis, ...]], 0
                    ))
                )
                .unsqueeze(0)
                .float()
            )
            with torch.no_grad():
                score = self.reward_model(reward_inpt.cuda()).item()
            reward += score - self.last_score
        self.last_score = score

        # If reward model is very certain, mark as done
        done = False
        if score >= 0.95 and self.counter >= 4:
            print("Threshold hit! Score:", score)
            done = True
        # print(score)
        # input()

        pos_channel = self.gen_pos_channel(
            int(self.last_pos[0] * (self.img_size / self.canvas_size)),
            int(self.last_pos[1] * (self.img_size / self.canvas_size)),
            self.img_size,
        )
        this_frame = np.stack([self.canvas, pos_channel])
        obs = np.concatenate([self.prev_frame, this_frame, self.ref])
        self.prev_frame = this_frame

        return obs, reward, done, trunc, {}

    def reset(self, **kwargs) -> tuple[np.ndarray, dict[str, Any]]:
        self.counter = 0
        self.canvas = np.zeros([self.img_size, self.img_size])
        index = random.randrange(0, len(self.ref_imgs))
        self.ref = self.ref_imgs[index]
        self.num_strokes = self.max_strokes
        self.last_pen_down = True
        self.last_pos = (
            random.randrange(0, self.canvas_size),
            random.randrange(0, self.canvas_size),
        )

        pos_channel = self.gen_pos_channel(
            int(self.last_pos[0] * (self.img_size / self.canvas_size)),
            int(self.last_pos[1] * (self.img_size / self.canvas_size)),
            self.img_size,
        )
        self.prev_frame = np.zeros([2, self.img_size, self.img_size])
        this_frame = np.stack([self.canvas, pos_channel])
        obs = np.concatenate([self.prev_frame, this_frame, self.ref])
        self.prev_frame = this_frame
        self.canvas_draw.rectangle((0, 0, self.canvas_size, self.canvas_size), 0)
        reward_inpt = (
            torch.from_numpy(
                add_random(np.concatenate(
                    [self.ref, np.zeros([1, self.img_size, self.img_size])], 0
                ))
            )
            .unsqueeze(0)
            .float()
        )
        self.last_score = 0.0
        if self.reward_model:
            with torch.no_grad():
                self.last_score = self.reward_model(reward_inpt.cuda()).item()
        return obs, {}

    def render(self) -> None:
        if self.render_mode == "human":
            pos_channel = np.zeros([self.img_size, self.img_size])
            pos_x = int(self.last_pos[0] * (self.img_size / self.canvas_size))
            pos_y = int(self.last_pos[1] * (self.img_size / self.canvas_size))
            pos_channel[pos_y][pos_x] = 1

            img = 0.5 + self.ref / 2.0
            strokes = 1.0 - self.canvas
            img = img * strokes

            img = img.transpose(2, 1, 0) * 255.0
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

def add_random(arr: np.ndarray) -> np.ndarray:
    rand = np.random.normal(0.0, 0.1, arr.shape)
    return np.clip(arr + rand, 0.0, 1.0)