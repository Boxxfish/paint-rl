import random
from typing import Any, List, Optional
import gymnasium as gym
import numpy as np
from PIL import Image, ImageDraw  # type: ignore
import pygame
from scipy.ndimage import gaussian_filter  # type: ignore
from skimage.morphology import binary_dilation, disk  # type: ignore

from paint_rl.experiments.supervised_strokes.gen_supervised import (
    MAX_DIST,
    MIN_DIST,
    gen_curve_points,
    rand_point,
)

SCREEN_SCALE = 8


class OutlineStrokeEnv(gym.Env):
    """
    An environment where the agent must generate a stroke from an image.
    Reward is change in L1 distance between actual stroke and generated strokes.
    """

    def __init__(
        self,
        img_size: int,
        ref_imgs: List[np.ndarray],
        stroke_imgs: List[np.ndarray],
        render_mode: Optional[str] = None,
        stroke_width: int = 1,
        dilation_size: int = 0,
    ) -> None:
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
        self.img_size = img_size
        self.ref_imgs = ref_imgs
        self.stroke_imgs = stroke_imgs
        self.ref = np.zeros([img_size, img_size])
        self.canvas = np.zeros([img_size, img_size])
        self.stroke_width = stroke_width
        self.last_pos = (0, 0)
        self.num_strokes = 0
        self.counter = 0
        self.dilation_size = dilation_size
        self.render_mode = render_mode
        self.max_diff = img_size**2
        self.last_pen_down = True
        self.canvas_img = Image.new("1", (self.img_size, self.img_size))
        self.canvas_draw = ImageDraw.Draw(self.canvas_img)
        self.last_diff = 0
        self.prev_frame = np.zeros([2, self.img_size, self.img_size])
        self.correct_moves: list[tuple[tuple[float, float], tuple[float, float]]] = []
        if self.render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode(
                (img_size * SCREEN_SCALE, img_size * SCREEN_SCALE)
            )
            self.clock = pygame.time.Clock()

    def step(
        self, action: tuple[np.ndarray, int],
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        cont_action, disc_action = action
        cont_action = np.clip(cont_action.squeeze() * self.img_size, 0, self.img_size - 1)
        mid_point = (int(cont_action[0]), int(cont_action[1]))
        end_point = (int(cont_action[2]), int(cont_action[3]))
        pen_down = disc_action == 1
        if pen_down:
            points = gen_curve_points(self.last_pos, mid_point, end_point)
            self.canvas_draw.line(points, 1, width=self.stroke_width)
            scaled_canvas = self.canvas_img.convert("RGB").resize((self.img_size, self.img_size), resample=Image.Resampling.BILINEAR)
            new_stroke = np.array(scaled_canvas).transpose(2, 0, 1)[1] / 255.0
            self.canvas = new_stroke

        diff = (np.abs(self.canvas - self.ref_filter)).sum() / self.max_diff
        reward = -(diff - self.last_diff)
        self.last_diff = diff

         # Penalize refusing to put down strokes
        penalty = -0.001
        if not self.last_pen_down and not pen_down:
            reward += penalty
        self.last_pen_down = pen_down

        done = diff < 0.001
        self.counter += 1
        trunc = self.counter == self.num_strokes

        pixels_moved = abs(self.last_pos[0] - end_point[0]) + abs(
            self.last_pos[1] - end_point[1]
        )
        if pixels_moved <= 4:
            reward += penalty
        self.last_pos = end_point

        pos_channel = self.gen_pos_channel(
            self.last_pos[0], self.last_pos[1], self.img_size
        )
        this_frame = np.stack([self.canvas, pos_channel])
        obs = np.concatenate([self.prev_frame, this_frame, self.ref], 0)
        self.prev_frame = this_frame
        
        return obs, reward, done, trunc, {}

    def reset(self, **kwargs) -> tuple[np.ndarray, dict[str, Any]]:
        self.counter = 0
        index = random.randrange(0, len(self.ref_imgs))
        img = self.ref_imgs[index]
        self.num_strokes = 50
        self.last_pos = rand_point(
            MIN_DIST, MAX_DIST, prev=(self.img_size // 2, self.img_size // 2)
        )
        self.ref = img
        stroke_img = self.stroke_imgs[index]
        
        if self.dilation_size > 0:
            self.ref_filter = binary_dilation(stroke_img, [(disk(1), self.dilation_size)])
        else:
            self.ref_filter = stroke_img

        self.canvas = np.zeros([self.img_size, self.img_size])
        pos_channel = self.gen_pos_channel(
            self.last_pos[0], self.last_pos[1], self.img_size
        )
        self.prev_frame = np.zeros([2, self.img_size, self.img_size])
        this_frame = np.stack([self.canvas, pos_channel])
        obs = np.concatenate([self.prev_frame, this_frame, self.ref])
        self.prev_frame = this_frame
        self.canvas_draw.rectangle((0, 0, self.img_size, self.img_size), 0)
        self.last_diff = (np.abs(self.canvas - self.ref_filter)).sum() / self.max_diff
        return obs, {}

    def render(self) -> None:
        if self.render_mode == "human":
            pos_channel = np.zeros([self.img_size, self.img_size])
            pos_channel[self.last_pos[1]][self.last_pos[0]] = 1
            img_display = self.ref.mean(0)
            img = (
                np.stack([self.canvas, img_display, self.ref_filter]).swapaxes(0, 2)
                * 255.0
            )
            img_surf = pygame.surfarray.make_surface(img)
            pygame.transform.scale(
                img_surf,
                (self.img_size * SCREEN_SCALE, self.img_size * SCREEN_SCALE),
                self.screen,
            )
            pygame.display.flip()
            # self.clock.tick(60)

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

    def set_dilation_size(self, dilation_size: int):
        self.dilation_size = dilation_size