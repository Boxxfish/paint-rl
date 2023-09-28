"""
Trains a model on the supervised stroke task.
This one trains a model to draw a cube, a cylinder, or a sphere.
"""
import json
from argparse import ArgumentParser
import random
import numpy as np
import torch
import wandb
from PIL import Image, ImageDraw  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore
from torch import Tensor, nn
from tqdm import tqdm
from paint_rl import conf
from paint_rl.experiments.stroke_gan.ref_stroke_env import RefStrokeEnv
from matplotlib import pyplot as plt  # type: ignore
from paint_rl.experiments.supervised_strokes.gen_supervised import gen_curve_points


class ResBlock(nn.Module):
    def __init__(self, module: nn.Module):
        nn.Module.__init__(self)
        self.module = module

    def forward(self, x):
        return self.module(x) + x


class SharedNet(nn.Module):
    """
    Shared network that could potentially be used for downstream tasks.
    """

    def __init__(self, size: int):
        nn.Module.__init__(self)
        self.net = nn.Sequential(
            nn.Conv2d(7, 16, 3, stride=2, padding=1),
            ResBlock(
                nn.Sequential(
                    nn.ReLU(),
                    nn.Conv2d(16, 16, 3, padding=1),
                    nn.BatchNorm2d(16),
                    nn.ReLU(),
                    nn.Conv2d(16, 16, 3, padding=1),
                    nn.BatchNorm2d(16),
                    nn.ReLU(),
                )
            ),
            ResBlock(
                nn.Sequential(
                    nn.ReLU(),
                    nn.Conv2d(16, 16, 3, padding=1),
                    nn.BatchNorm2d(16),
                    nn.ReLU(),
                    nn.Conv2d(16, 16, 3, padding=1),
                    nn.BatchNorm2d(16),
                    nn.ReLU(),
                )
            ),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            ResBlock(
                nn.Sequential(
                    nn.ReLU(),
                    nn.Conv2d(32, 32, 5, padding=2),
                    nn.BatchNorm2d(32),
                    nn.ReLU(),
                    nn.Conv2d(32, 32, 5, padding=2),
                    nn.BatchNorm2d(32),
                    nn.ReLU(),
                )
            ),
            nn.Conv2d(32, 64, 5, padding=2),
            ResBlock(
                nn.Sequential(
                    nn.ReLU(),
                    nn.Conv2d(64, 64, 7, padding=3),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.Conv2d(64, 64, 7, padding=3),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                )
            ),
        )
        self.downscale = nn.Conv2d(64, 8, 1)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.net(x)
        x = self.downscale(x)
        x = self.flatten(x)
        return x

    def set_frozen(self, frozen: bool):
        for param in self.parameters():
            param.requires_grad = not frozen


class StrokeNet(nn.Module):
    def __init__(self, size: int, quant_size: int):
        nn.Module.__init__(self)
        self.shared = SharedNet(size)
        self.pen_down = nn.Sequential(
            nn.Linear(quant_size * quant_size * 8, 256),
            nn.ReLU(),
            nn.Linear(256, 2),
            nn.LogSoftmax(1),
        )
        self.stroke_mid = nn.Sequential(
            nn.Linear(quant_size * quant_size * 8, quant_size * quant_size),
            nn.ReLU(),
            nn.Linear(quant_size * quant_size, quant_size * quant_size),
            nn.LogSoftmax(1),
        )
        self.stroke_end = nn.Sequential(
            nn.Linear(quant_size * quant_size * 8, quant_size * quant_size),
            nn.ReLU(),
            nn.Linear(quant_size * quant_size, quant_size * quant_size),
            nn.LogSoftmax(1),
        )

    def forward(self, x):
        x = self.shared(x)
        pen_down = self.pen_down(x)
        stroke_mid = self.stroke_mid(x)
        stroke_end = self.stroke_end(x)
        return stroke_mid, stroke_end, pen_down


def gen_pos_channel(x: int, y: int, img_size: int) -> np.ndarray:
    pos_layer_x = (
        np.abs(np.arange(0 - x, img_size - x))[np.newaxis, ...].repeat(img_size, 0)
        / img_size
    )
    pos_layer_y = (
        np.abs(np.arange(0 - y, img_size - y))[np.newaxis, ...].repeat(img_size, 0).T
        / img_size
    )
    return np.sqrt(pos_layer_x**2 + pos_layer_y**2)


def stroke(
    p1,
    p2,
    p3,
    img_data,
    stroke_img,
    stroke_draw,
    ds_x,
    stroke_mid,
    stroke_end,
    pen_down,
    img_size,
    prev_frame,
    quant_size,
    up=False,
) -> tuple[np.ndarray, np.ndarray]:
    stroke_channel = np.array(stroke_img) / 255.0
    pos_channel = gen_pos_channel(p1[0] * img_size, p1[1] * img_size, img_size)
    this_frame = np.stack(
        [
            stroke_channel + np.random.uniform(0.0, 0.1, [img_size, img_size]),
            pos_channel,
        ]
    )
    ds_x.append(
        np.concatenate(
            [
                prev_frame,
                this_frame,
                img_data + np.random.uniform(0.0, 0.1, [3, img_size, img_size]),
            ]
        )
    )
    stroke_mid.append(quantize_point(p2, quant_size))
    stroke_end.append(quantize_point(p3, quant_size))
    pen_down.append(0 if up else 1)
    last_pos = p3 + np.random.normal(0.0, 0.02, [2])

    if not up:
        stroke_draw.line(
            gen_curve_points(
                p1 * img_size,
                (p2 + np.random.normal(0.0, 0.02, [2])) * img_size,
                (last_pos) * img_size,
            ),
            255,
        )
    return this_frame, last_pos


def quantize_point(point: np.ndarray, quant_size: int) -> int:
    q_point = np.clip(((point * quant_size).round()).astype(int), 0, quant_size - 1)
    return int(q_point[1] * quant_size + q_point[0])


def load_random_ds(
    valid_size: int, size: int, img_size: int, orig_img_size: int, quant_size: int
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    indices = list(range(valid_size, 10_000))
    random.shuffle(indices)
    ds_x: list[np.ndarray] = []
    stroke_mid: list[np.ndarray] = []
    stroke_end: list[np.ndarray] = []
    pen_down: list[np.ndarray] = []
    for i in tqdm(indices):
        if len(ds_x) == size:
            break
        try:
            img = (
                Image.open(f"temp/sketch_outputs/{i}/final.png")
                .convert("RGB")
                .resize((img_size, img_size))
            )
            img_data = np.array(img).transpose(2, 0, 1) / 255.0
            with open(f"temp/sketch_outputs/{i}/strokes.json", "r") as f:
                path = json.load(f)

            all_actions = path["actions"]
            shape_indices = list(range(len(all_actions)))
            random.shuffle(shape_indices)

            stroke_img = Image.new("L", (img_size, img_size))
            stroke_draw = ImageDraw.Draw(stroke_img)

            last_point = np.array([random.random(), random.random()])
            prev_frame = np.zeros([2, img_size, img_size])

            for shape_idx in shape_indices:
                if len(ds_x) == size:
                    break
                cont, disc = all_actions[shape_idx]

                if len(cont) == 0:
                    continue

                for i, (c, d) in enumerate(zip(cont, disc)):
                    if len(ds_x) == size:
                        break

                    mid_point = np.array([c[0], c[1]]).astype(float) / orig_img_size
                    end_point = np.array([c[2], c[3]]).astype(float) / orig_img_size
                    if i == 0:
                        mid_point = (last_point + end_point) / 2.0

                    # 25% chance of actually adding example
                    will_add = random.random() < 0.25
                    prev_frame, last_point = stroke(
                        last_point,
                        mid_point,
                        end_point,
                        img_data,
                        stroke_img,
                        stroke_draw,
                        ds_x if will_add else [],
                        stroke_mid if will_add else [],
                        stroke_end if will_add else [],
                        pen_down if will_add else [],
                        img_size,
                        prev_frame,
                        quant_size,
                        up=d == 0,
                    )
        except KeyboardInterrupt:
            quit()
        except Exception as e:
            print(f"Error processing {i}: {e}")

    # for i in range(len(ds_x)):
    #     plt.imshow(ds_x[i].mean(0))
    #     plt.show()

    stroke_mid_actions = torch.from_numpy(np.stack(stroke_mid))
    del stroke_mid
    stroke_end_actions = torch.from_numpy(np.stack(stroke_end))
    del stroke_end
    pen_down_actions = torch.from_numpy(np.stack(pen_down))
    del pen_down

    ds_x_ = torch.from_numpy(np.stack(ds_x))
    del ds_x
    return (ds_x_.float(), stroke_mid_actions, stroke_end_actions, pen_down_actions)


# Converts discrete stroke probs to a continuous action.
# These can be directly outputted from the model.
def disc_probs_to_cont_actions(
    stroke_mid: np.ndarray, stroke_end: np.ndarray, quant_size: int
) -> np.ndarray:
    mid_index = torch.distributions.Categorical(
        logits=torch.tensor(stroke_mid).exp()
    ).sample([1]).numpy()
    end_index = torch.distributions.Categorical(
        logits=torch.tensor(stroke_end).exp()
    ).sample([1]).numpy()
    # print(stroke_end.exp())
    # plt.imshow(stroke_end.exp().reshape(32, 32))
    # plt.show()
    mid_y = mid_index // quant_size
    mid_x = mid_index - mid_y * quant_size
    end_y = end_index // quant_size
    end_x = end_index - end_y * quant_size
    return np.concatenate([mid_x, mid_y, end_x, end_y], 1) / quant_size


# Converts discrete stroke actions to a continuous one.
def disc_actions_to_cont_actions(
    mid_index: np.ndarray, end_index: np.ndarray, quant_size: int
) -> np.ndarray:
    mid_y = mid_index // quant_size
    mid_x = mid_index - mid_y * quant_size
    end_y = end_index // quant_size
    end_x = end_index - end_y * quant_size
    return np.concatenate([mid_x, mid_y, end_x, end_y], 1) / quant_size


def main():
    orig_img_size = 256
    img_size = 64
    quant_size = 16

    # Argument parsing
    parser = ArgumentParser()
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.eval:
        print("Loading data.")
        imgs = []
        for i in tqdm(range(100)):
            img = (
                Image.open(f"temp/sketch_outputs/{i}/final.png")
                .convert("RGB")
                .resize((img_size, img_size))
            )
            img_data = np.array(img).transpose(2, 0, 1) / 255.0
            imgs.append(img_data)
        with torch.no_grad():
            net = StrokeNet(img_size, quant_size)
            net.load_state_dict(torch.load("temp/stroke_net.pt"))
            env = RefStrokeEnv(
                img_size, img_size, imgs, imgs, None, "human", max_strokes=100
            )
            obs = torch.from_numpy(env.reset()[0]).float().unsqueeze(0)
            while True:
                stroke_mid, stroke_end, pen_down = net(obs)
                cont_action = disc_probs_to_cont_actions(
                    stroke_mid, stroke_end, quant_size
                ).squeeze()
                obs_, _, done, trunc, _ = env.step(
                    (cont_action, pen_down.argmax().item())
                )
                env.render()
                obs = torch.from_numpy(obs_).float().unsqueeze(0)
                if done or trunc:
                    obs = torch.from_numpy(env.reset()[0]).float().unsqueeze(0)

    # Load dataset
    valid_ds_x = []
    valid_stroke_mid_actions = []
    valid_stroke_end_actions = []
    valid_pen_down_actions = []
    valid_size = 8
    train_size = 30_000
    print("Loading data.")
    for i in tqdm(range(1500)):
        if i >= valid_size:
            break
        try:
            img = (
                Image.open(f"temp/sketch_outputs/{i}/final.png")
                .convert("RGB")
                .resize((img_size, img_size))
            )
            img_data = np.array(img).transpose(2, 0, 1) / 255.0
            with open(f"temp/sketch_outputs/{i}/strokes.json", "r") as f:
                path = json.load(f)
            stroke_img = Image.new("L", (img_size, img_size))
            stroke_draw = ImageDraw.Draw(stroke_img)

            cont = sum([a[0] for a in path["actions"]], start=[])
            disc = sum([a[1] for a in path["actions"]], start=[])
            last_point = np.array(path["start"]).astype(float) / orig_img_size
            prev_frame = np.zeros([2, img_size, img_size])
            for c, d in zip(cont, disc):
                prev_frame, last_point = stroke(
                    last_point,
                    np.array([c[0], c[1]]).astype(float) / orig_img_size,
                    np.array([c[2], c[3]]).astype(float) / orig_img_size,
                    img_data,
                    stroke_img,
                    stroke_draw,
                    valid_ds_x,
                    valid_stroke_mid_actions,
                    valid_stroke_end_actions,
                    valid_pen_down_actions,
                    img_size,
                    prev_frame,
                    quant_size,
                    up=d == 0,
                )
            # plt.imshow(ds_x[-1][:3].transpose(1, 2, 0))
            # plt.show()
        except KeyboardInterrupt:
            quit()
        except Exception as e:
            print(f"Error processing {i}: {e}")
    (
        train_x,
        train_stroke_mid,
        train_stroke_end,
        train_pen_down,
    ) = load_random_ds(
        valid_size, train_size, img_size, orig_img_size, quant_size
    )
    valid_x = torch.from_numpy(np.stack(valid_ds_x))
    valid_stroke_mid = torch.from_numpy(np.stack(valid_stroke_mid_actions))
    valid_stroke_end = torch.from_numpy(np.stack(valid_stroke_end_actions))
    valid_pen_down = torch.from_numpy(np.stack(valid_pen_down_actions))
    del (
        valid_ds_x,
        valid_stroke_mid_actions,
        valid_stroke_end_actions,
        valid_pen_down_actions,
    )
    print("Train size:", train_size)
    print("Valid size:", valid_x.shape[0])

    wandb.init(
        project="paint-rl",
        entity=conf.entity,
        config={
            "experiment": "supervised stroke rendering with cubes, spheres, and cylinders",
        },
    )

    # Train model
    net = StrokeNet(img_size, quant_size).cuda().train()
    if args.resume:
        net.load_state_dict(torch.load("temp/stroke_net.pt"))
    opt = torch.optim.Adam(net.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, "min", min_lr=0.00001, factor=1 / 3, patience=30
    )
    batch_size = 512
    train_x = train_x.float()
    valid_batch_x = valid_x.float().cuda()
    valid_batch_stroke_mid = valid_stroke_mid.cuda()
    valid_batch_stroke_end = valid_stroke_end.cuda()
    valid_batch_pen_down = valid_pen_down.cuda()
    disc_crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    for j in tqdm(range(1000000)):
        mean_loss = 0.0
        mean_mid_loss = 0.0
        mean_end_loss = 0.0
        mean_down_loss = 0.0
        num_batches = train_size // batch_size
        indices = torch.from_numpy(
            np.random.choice(train_size, train_size, replace=False)
        )
        for i in range(num_batches):
            batch_indices = indices[i * batch_size : (i + 1) * batch_size]
            batch_x = train_x[batch_indices].cuda()
            batch_mid = train_stroke_mid[batch_indices].cuda()
            batch_end = train_stroke_end[batch_indices].cuda()
            batch_pen_down = train_pen_down[batch_indices].cuda()
            stroke_mid, stroke_end, pen_down = net(batch_x)
            # print(batch_mid.min(), batch_mid.max(), batch_end.min(), batch_end.max())
            stroke_mid_loss = disc_crit(stroke_mid, batch_mid.type(torch.long))
            stroke_end_loss = disc_crit(stroke_end, batch_end.type(torch.long))
            pen_down_loss = disc_crit(pen_down, batch_pen_down.type(torch.long))
            loss = stroke_mid_loss + stroke_end_loss + pen_down_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            mean_loss += loss.item()
            mean_mid_loss += stroke_mid_loss
            mean_end_loss += stroke_end_loss
            mean_down_loss += pen_down_loss
        mean_loss /= num_batches
        mean_mid_loss /= num_batches
        mean_end_loss /= num_batches
        mean_down_loss /= num_batches

        # Evaluate
        with torch.no_grad():
            valid_mid, valid_end, valid_down = net(valid_batch_x)
            valid_loss_mid = disc_crit(
                valid_mid, valid_batch_stroke_mid.type(torch.long)
            )
            valid_loss_end = disc_crit(
                valid_end, valid_batch_stroke_end.type(torch.long)
            )
            valid_loss_down = disc_crit(
                valid_down, valid_batch_pen_down.type(torch.long)
            )
            valid_loss = valid_loss_mid + valid_loss_end + valid_loss_down
            scheduler.step(valid_loss)
            wandb.log(
                {
                    "loss": mean_loss,
                    "valid_loss": valid_loss.item(),
                    "mean_stroke_mid_loss": mean_mid_loss,
                    "mean_stroke_end_loss": mean_end_loss,
                    "mean_pen_down_loss": mean_down_loss,
                    "valid_stroke_mid_loss": valid_loss_mid.item(),
                    "valid_stroke_end_loss": valid_loss_end.item(),
                    "valid_pen_down_loss": valid_loss_down.item(),
                    "lr": opt.param_groups[0]["lr"],
                }
            )

        if (j + 1) % 10 == 0:
            del train_stroke_mid, train_stroke_end, train_pen_down
            (
                train_x,
                train_stroke_mid,
                train_stroke_end,
                train_pen_down,
            ) = load_random_ds(
                valid_size, train_size, img_size, orig_img_size, quant_size
            )

        # Save
        torch.save(net.state_dict(), "temp/stroke_net.pt")


if __name__ == "__main__":
    main()
