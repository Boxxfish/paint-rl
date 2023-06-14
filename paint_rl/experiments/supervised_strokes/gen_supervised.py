"""
Generates the dataset for the supervised setting.
"""
import json
import math
import os
import random
import shutil
from pathlib import Path
from typing import List, Optional, Tuple

import bezier  # type: ignore
import numpy as np
from PIL import Image, ImageDraw  # type: ignore
from tqdm import tqdm

IMG_SIZE = 64
MIN_DIST = 8
MAX_DIST = 32
NUM_IMAGES = 1000

CANVAS_COLOR = (255, 0, 0)
TARGET_COLOR = (0, 255, 0)
CANVAS_TARGET_COLOR = (255, 255, 0)
PEN_POS_COLOR = (0, 0, 255)


def rand_point(
    min_: int, max_: int, prev: Optional[Tuple[int, int]]
) -> Tuple[int, int]:
    dist = min_ + random.random() * (max_ - min_)
    angle = random.random() * 2.0 * math.pi
    x = 0
    y = 0
    if prev:
        x = prev[0]
        y = prev[1]
    return (
        clip(x + int(dist * math.cos(angle)), 0, IMG_SIZE),
        clip(0, y + int(dist * math.sin(angle)), IMG_SIZE),
    )


def gen_curve_points(
    p1: Tuple[int, int], p2: Tuple[int, int], p3: Tuple[int, int]
) -> List[Tuple[int, int]]:
    points = []
    nodes = np.swapaxes(np.array([p1, p2, p3]), 0, 1)
    curve = bezier.Curve(nodes, 2)
    num_evals = 10
    for i in range(num_evals):
        pct = i / num_evals
        point = curve.evaluate(pct)
        points.append((point[0], point[1]))
    point = curve.evaluate(1.0)
    points.append((point[0], point[1]))
    return points


def clip(val: int, min_: int, max_: int):
    return min(max(val, min_), max_)


def gen_sample(img_size: int) -> Tuple[Image.Image, List[Tuple[int, int]]]:
    c_img = Image.new("1", (img_size, img_size))
    c_draw = ImageDraw.Draw(c_img)
    r_img = Image.new("1", (img_size, img_size))
    r_draw = ImageDraw.Draw(r_img)
    num_before = random.randrange(0, 2)
    num_after = random.randrange(0, 2)
    lines_before = [rand_point(MIN_DIST, MAX_DIST, prev=(IMG_SIZE // 2, IMG_SIZE // 2))]
    for i in range(1, num_before + 1):
        lines_before.append(rand_point(MIN_DIST, MAX_DIST, prev=lines_before[i - 1]))
    last_point = lines_before[-1]
    mid_point = rand_point(MIN_DIST, MAX_DIST, prev=last_point)
    next_point = rand_point(MIN_DIST, MAX_DIST, prev=last_point)
    path = [last_point, mid_point, next_point]
    points = gen_curve_points(last_point, mid_point, next_point)
    r_draw.line(points, 1, width=1)
    lines_after = [next_point]
    for i in range(num_after):
        lines_after.append(rand_point(MIN_DIST, MAX_DIST, prev=lines_after[i - 1]))
    prev_point_stroke = lines_before[0]
    for i in range(len(lines_before) - 1):
        prev_point = lines_before[i]
        next_point = lines_before[i + 1]
        mid = rand_point(MIN_DIST, MAX_DIST, prev=prev_point)
        points = gen_curve_points(prev_point, mid, next_point)
        r_draw.line(points, 1, width=1)
        next_point_stroke = (
            next_point[0] + random.randrange(-2, 2),
            next_point[1] + random.randrange(-2, 2),
        )
        points = gen_curve_points(
            prev_point_stroke,
            (mid[0] + random.randrange(-2, 2), mid[1] + random.randrange(-2, 2)),
            next_point_stroke,
        )
        c_draw.line(points, 1, width=1)
        prev_point_stroke = next_point_stroke
        last_point = (
            max(min(int(prev_point_stroke[0]), img_size - 1), 0),
            max(min(int(prev_point_stroke[1]), img_size - 1), 0),
        )
    for i in range(len(lines_after) - 1):
        prev_point = lines_after[i]
        next_point = lines_after[i + 1]
        mid = rand_point(MIN_DIST, MAX_DIST, prev=prev_point)
        points = gen_curve_points(prev_point, mid, next_point)
        r_draw.line(points, 1, width=1)
    pos_layer_x = (
        np.abs(np.arange(0 - last_point[0], img_size - last_point[0]))[
            np.newaxis, ...
        ].repeat(img_size, 0)
        / img_size
    )
    pos_layer_y = (
        np.abs(np.arange(0 - last_point[1], img_size - last_point[1]))[np.newaxis, ...]
        .repeat(img_size, 0)
        .T
        / img_size
    )
    pos_layer = np.sqrt(pos_layer_x**2 + pos_layer_y**2).clip(0, 1)
    final = (
        np.stack([np.array(c_img), np.array(r_img), pos_layer])
        .swapaxes(0, 2)
        .swapaxes(0, 1)
    )
    return (Image.fromarray(np.uint8(final * 255)), path)


def gen_shape(img_size: int) -> Tuple[Image.Image, List[Tuple[int, int]]]:
    c_img = Image.new("1", (img_size, img_size))
    c_draw = ImageDraw.Draw(c_img)
    r_img = Image.new("1", (img_size, img_size))
    r_draw = ImageDraw.Draw(r_img)
    num_points = random.randrange(3, 7)
    num_strokes = random.randrange(0, num_points - 1)

    # Generate polygon points
    points = []
    theta = 0.0
    for _ in range(num_points):
        delta_theta = (random.gauss(2.0 * math.pi, 2.0)) / num_points
        theta += delta_theta
        radius = img_size / 4 + (random.gauss(0.0, 1.0)) * img_size / 8
        x = max(min(int(img_size / 2 + math.cos(theta) * radius), img_size), 0)
        y = max(min(int(img_size / 2 + math.sin(theta) * radius), img_size), 0)
        points.append((x, y))
    mid_points = []
    mid_sigma = 2.0
    for i in range(num_points):
        point = points[(i) % num_points]
        next_point = points[(i + 1) % num_points]
        x = int((point[0] + next_point[0]) // 2 + random.gauss(0.0, mid_sigma))
        y = int((point[1] + next_point[1]) // 2 + random.gauss(0.0, mid_sigma))
        mid_points.append((x, y))
        

    # Draw polygon
    start_index = random.randrange(0, num_points)
    last_stroke_point = points[start_index]
    stroke_sigma = 2.0
    for i in range(num_points):
        point = points[(start_index + i) % num_points]
        mid_point = mid_points[(start_index + i) % num_points]
        next_point = points[(start_index + i + 1) % num_points]
        r_draw.line(gen_curve_points(point, mid_point, next_point), 1)
        if i < num_strokes:
            next_stroke_point = (
                int(next_point[0] + random.gauss(0.0, stroke_sigma)),
                int(next_point[1] + random.gauss(0.0, stroke_sigma)),
            )
            c_draw.line([last_stroke_point, mid_point, next_stroke_point], 1)
            last_stroke_point = next_stroke_point
    
    start_point = last_stroke_point
    mid_point = mid_points[(start_index + num_strokes) % num_points]
    end_point = points[(start_index + num_strokes + 1) % num_points]
    path = [start_point, mid_point, end_point]
    
    pos_layer_x = (
        np.abs(np.arange(0 - start_point[0], img_size - start_point[0]))[
            np.newaxis, ...
        ].repeat(img_size, 0)
        / img_size
    )
    pos_layer_y = (
        np.abs(np.arange(0 - start_point[1], img_size - start_point[1]))[
            np.newaxis, ...
        ]
        .repeat(img_size, 0)
        .T
        / img_size
    )
    pos_layer = np.sqrt(pos_layer_x**2 + pos_layer_y**2).clip(0, 1)
    final = (
        np.stack([np.array(c_img), np.array(r_img), pos_layer])
        .swapaxes(0, 2)
        .swapaxes(0, 1)
    )
    return (Image.fromarray(np.uint8(final * 255)), path)


def main():
    folder_name = "temp/supervised"
    if Path(folder_name).exists():
        shutil.rmtree(folder_name)
    os.mkdir(folder_name)
    paths = []
    for i in tqdm(range(NUM_IMAGES)):
        img, path = gen_shape(IMG_SIZE)
        paths.append(path)
        img.save(f"{folder_name}/{i}.png")
    with open("temp/supervised_paths.json", "w") as f:
        json.dump(paths, f)


if __name__ == "__main__":
    main()
