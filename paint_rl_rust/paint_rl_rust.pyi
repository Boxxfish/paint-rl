import numpy as np


class TrainingContext:
    def __init__(self, img_size: int, canvas_size: int, ref_img_path: str, p_net_path: str, reward_model_path: str, max_strokes: int): ...
    def gen_imgs(self, num_imgs: int, action_scale: float) -> np.ndarray: ...