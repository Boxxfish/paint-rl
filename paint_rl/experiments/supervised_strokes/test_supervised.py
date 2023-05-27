"""
Visually tests that a supervised model has trained correctly.
"""
import numpy as np
import torch
from matplotlib import pyplot as plt  # type: ignore
from PIL import ImageDraw  # type: ignore

from paint_rl.experiments.supervised_strokes.gen_supervised import (
    IMG_SIZE, gen_curve_points, gen_sample)
from paint_rl.experiments.supervised_strokes.train_supervised import StrokeNet


def main():
    img_size = IMG_SIZE
    net = StrokeNet(img_size)
    net.load_state_dict(torch.load("temp/stroke_net.pt"))
    img, path = gen_sample(img_size)
    draw = ImageDraw.Draw(img)
    inpt = torch.from_numpy(np.array(img)).swapaxes(2, 0).unsqueeze(0) / 255.0
    out = (net(inpt).squeeze() + 1.0) * (img_size / 2)
    curve_points = gen_curve_points(
        path[0], [out[0].item(), out[1].item()], [out[2].item(), out[3].item()]
    )
    draw.line(curve_points)
    print(f"Actual: {path[1]}, {path[2]}")
    print(
        f"Predicted: {(out[0].item(), out[1].item())}, {(out[2].item(), out[3].item())}"
    )
    plt.imshow(img)
    plt.show()


if __name__ == "__main__":
    main()
