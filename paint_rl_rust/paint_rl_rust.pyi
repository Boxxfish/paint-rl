from typing import Optional
import numpy as np
from torch import Tensor

class TrainingContext:
    def __init__(
        self,
        img_size: int,
        canvas_size: int,
        ref_img_path: str,
        p_net_path: str,
        reward_model_path: str,
        num_envs: int,
        num_workers: int,
        num_steps: int,
        max_strokes: Optional[int] = None,
    ): ...
    def gen_imgs(self, num_imgs: int) -> np.ndarray: ...
    def rollout(
        self,
    ) -> tuple[Tensor, list[Tensor], list[Tensor], Tensor, Tensor, Tensor]: ...
