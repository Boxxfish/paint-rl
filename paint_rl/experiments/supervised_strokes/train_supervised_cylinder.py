"""
Trains a model on the supervised stroke task.
This one trains a model to draw a cylinder.
"""
import json
from argparse import ArgumentParser
from math import cos, sin
import random
import numpy as np
import torch
import wandb
from PIL import Image, ImageDraw  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore
from torch import nn
from tqdm import tqdm
from paint_rl import conf
from paint_rl.experiments.stroke_gan.ref_stroke_env import RefStrokeEnv
from matplotlib import pyplot as plt  # type: ignore
from paint_rl.experiments.supervised_strokes.gen_supervised import gen_curve_points


class SharedNet(nn.Module):
    """
    Shared network that could potentially be used for downstream tasks.
    Important: Do NOT reinitialize parameters! This erases the positional encoding.
    """

    def __init__(self, size: int):
        nn.Module.__init__(self)
        self.net = nn.Sequential(
            nn.Conv2d(5 + 2, 64, 5, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2),
            nn.ReLU(),
        )
        # Pos encoding
        w = torch.arange(0, size).unsqueeze(0).repeat([size, 1])
        h = torch.arange(0, size).unsqueeze(0).repeat([size, 1]).T
        self.pos = nn.Parameter(
            torch.stack([w.detach(), h.detach()]).unsqueeze(0) / size,
            requires_grad=False,
        )

    def forward(self, x):
        batch_size = x.shape[0]
        pos_enc = self.pos.repeat([batch_size, 1, 1, 1])
        x = torch.cat([x, pos_enc], dim=1)
        x = self.net(x)
        x = torch.max(torch.max(x, 3).values, 2).values
        return x


class StrokeNet(nn.Module):
    def __init__(self, size: int):
        nn.Module.__init__(self)
        self.shared = SharedNet(size)
        self.continuous = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 4),
            nn.Sigmoid(),
        )
        self.discrete = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 2),
            nn.LogSoftmax(1),
        )

    def forward(self, x):
        x = self.shared(x)
        cont = self.continuous(x)
        disc = self.discrete(x)
        return cont, disc


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
    cont_actions,
    disc_actions,
    img_size,
    up=False
):
    stroke_channel = np.array(stroke_img) / 255.0
    pos_channel = gen_pos_channel(p1[0] * img_size, p1[1] * img_size, img_size)
    ds_x.append(np.concatenate([np.stack([stroke_channel, pos_channel]), img_data]))
    cont_actions.append(np.concatenate([p2, p3]))
    disc_actions.append(0 if up else 1)

    if not up:
        stroke_draw.line(gen_curve_points(p1 * img_size, p2 * img_size, p3 * img_size), 255)


def main():
    orig_img_size = 256
    img_size = 64

    # Argument parsing
    parser = ArgumentParser()
    parser.add_argument("--eval", action="store_true")
    args = parser.parse_args()

    if args.eval:
        print("Loading data.")
        imgs = []
        for i in tqdm(range(100)):
            img = (
                Image.open(f"temp/sphere_outputs/{i}/final.png")
                .convert("RGB")
                .resize((img_size, img_size))
            )
            img_data = np.array(img).transpose(2, 0, 1) / 255.0
            imgs.append(img_data)
        with torch.no_grad():
            net = StrokeNet(img_size)
            net.load_state_dict(torch.load("temp/stroke_net.pt"))
            env = RefStrokeEnv(img_size, img_size, imgs, None, "human", max_strokes=20)
            obs = torch.from_numpy(env.reset()[0]).float().unsqueeze(0)
            while True:
                cont_action, disc_action = net(obs)
                obs_, _, done, _, _ = env.step(
                    (cont_action.squeeze().numpy(), disc_action.argmax().item())
                )
                env.render()
                obs = torch.from_numpy(obs_).float().unsqueeze(0)
                if done:
                    obs = torch.from_numpy(env.reset()[0]).float().unsqueeze(0)

    # Load dataset
    ds_x = []
    cont_actions = []
    disc_actions = []
    print("Loading data.")
    for i in tqdm(range(2000)):
        try:
            img = (
                Image.open(f"temp/cylinder_outputs/{i}/final.png")
                .convert("RGB")
                .resize((img_size, img_size))
            )
            img_data = np.array(img).transpose(2, 0, 1) / 255.0
            with open(f"temp/cylinder_outputs/{i}/strokes.json", "r") as f:
                path = json.load(f)
            circles = path["circles"]
            stroke_img = Image.new("L", (img_size, img_size))
            stroke_draw = ImageDraw.Draw(stroke_img)
            for circle in circles:
                center = np.array(circle["center"]) / orig_img_size
                a = int(circle["a"]) / orig_img_size
                b = int(circle["b"]) / orig_img_size
                rot = float(circle["rot"])

                p1 = (center - np.array([cos(rot), sin(rot)]) * b)
                p2 = (center + np.array([-sin(rot), cos(rot)]) * a * 2)
                p3 = (center + np.array([cos(rot), sin(rot)]) * b)
                p4 = (center - np.array([-sin(rot), cos(rot)]) * a * 2)

                # Stroke from anywhere to left point
                rand_x = random.randrange(0, img_size)
                rand_y = random.randrange(0, img_size)
                while np.sqrt(((np.array([rand_x, rand_y]) - p1) ** 2).sum()) < 4:
                    rand_x = random.randrange(0, img_size)
                    rand_y = random.randrange(0, img_size)
                pos_channel = gen_pos_channel(rand_x, rand_y, img_size)
                ds_x.append(
                    np.concatenate(
                        [np.stack([np.zeros([img_size, img_size]), pos_channel]), img_data]
                    )
                )
                cont_actions.append(
                    np.concatenate(
                        [
                            (p1 + np.array([rand_x, rand_y]) / img_size) / 2.0,
                            p1,
                        ]
                    )
                )
                disc_actions.append(0)

                stroke(
                    p1,
                    p2,
                    p3,
                    img_data,
                    stroke_img,
                    stroke_draw,
                    ds_x,
                    cont_actions,
                    disc_actions,
                    img_size,
                )
                stroke(
                    p3,
                    p4,
                    p1,
                    img_data,
                    stroke_img,
                    stroke_draw,
                    ds_x,
                    cont_actions,
                    disc_actions,
                    img_size,
                )

            # Stroke from top circle to bottom circle
            first_c_index = 0 if circles[0]["center"][1] > circles[1]["center"][1] else 1
            
            circle_1 = circles[first_c_index]
            center_1 = np.array(circle_1["center"]) / orig_img_size
            rot_1 = float(circle_1["rot"])
            a_1 = int(circle_1["a"]) / orig_img_size
            b_1 = int(circle_1["b"]) / orig_img_size
            if a_1 * sin(rot_1) > b_1 * -cos(rot_1):
                c1_r = (center_1 - np.array([-sin(rot_1), cos(rot_1)]) * a_1)
                c1_l = (center_1 + np.array([-sin(rot_1), cos(rot_1)]) * a_1)
            else:
                c1_r = (center_1 - np.array([cos(rot_1), sin(rot_1)]) * b_1)
                c1_l = (center_1 + np.array([cos(rot_1), sin(rot_1)]) * b_1)

            circle_2 = circles[first_c_index - 1]
            center_2 = np.array(circle_2["center"]) / orig_img_size
            rot_2 = float(circle_2["rot"])
            a_2 = int(circle_2["a"]) / orig_img_size
            b_2 = int(circle_2["b"]) / orig_img_size
            if a_2 * sin(rot_2) > b_2 * -cos(rot_2):
                c2_r = (center_2 - np.array([-sin(rot_2), cos(rot_2)]) * a_2)
                c2_l = (center_2 + np.array([-sin(rot_2), cos(rot_2)]) * a_2)
            else:
                c2_r = (center_2 - np.array([cos(rot_2), sin(rot_2)]) * b_2)
                c2_l = (center_2 + np.array([cos(rot_2), sin(rot_2)]) * b_2)

            stroke(
                p1,
                (p1 + c1_l) / 2,
                c1_l,
                img_data,
                stroke_img,
                stroke_draw,
                ds_x,
                cont_actions,
                disc_actions,
                img_size,
                up=True,
            )
            
            stroke(
                c1_l,
                (c1_l + c2_l) / 2,
                c2_l,
                img_data,
                stroke_img,
                stroke_draw,
                ds_x,
                cont_actions,
                disc_actions,
                img_size,
            )

            stroke(
                c2_l,
                (c2_l + c1_r) / 2,
                c1_r,
                img_data,
                stroke_img,
                stroke_draw,
                ds_x,
                cont_actions,
                disc_actions,
                img_size,
                up=True,
            )

            stroke(
                c1_r,
                (c1_r + c2_r) / 2,
                c2_r,
                img_data,
                stroke_img,
                stroke_draw,
                ds_x,
                cont_actions,
                disc_actions,
                img_size,
            )
        except KeyboardInterrupt:
            quit()
        except Exception as e:
            print(f"Error processing {i}: {e}")
    ds_x = torch.from_numpy(np.stack(ds_x))
    cont_actions = torch.from_numpy(np.stack(cont_actions))
    disc_actions = torch.from_numpy(np.stack(disc_actions))
    train_x, valid_x, train_cont, valid_cont, train_disc, valid_disc = train_test_split(
        ds_x, cont_actions, disc_actions, train_size=0.8
    )
    del ds_x, cont_actions, disc_actions
    train_size = train_x.shape[0]

    wandb.init(
        project="paint-rl",
        entity=conf.entity,
        config={
            "experiment": "supervised stroke rendering with cylinders",
        },
    )

    # Train model
    net = StrokeNet(img_size).cuda().train()
    opt = torch.optim.Adam(net.parameters(), lr=0.0001)
    batch_size = 512
    train_x = train_x.float()
    train_cont = train_cont.float()
    train_disc = train_disc
    valid_batch_x = valid_x.float().cuda()
    valid_batch_cont = valid_cont.float().cuda()
    valid_batch_disc = valid_disc.cuda()
    disc_crit = nn.NLLLoss()
    noise_distr = torch.distributions.Normal(0.0, 0.04)
    for j in tqdm(range(1000000)):
        mean_loss = 0.0
        num_batches = train_size // batch_size
        indices = torch.from_numpy(
            np.random.choice(train_size, train_size, replace=False)
        )
        for i in range(num_batches):
            batch_indices = indices[i * batch_size : (i + 1) * batch_size]
            noise = noise_distr.sample([batch_size, 3, img_size, img_size])
            noise = torch.concat(
                [torch.zeros([batch_size, 2, img_size, img_size]), noise], 1
            )
            batch_x = (torch.clip(train_x[batch_indices] + noise, 0, 1)).cuda()
            batch_cont = train_cont[batch_indices].cuda()
            batch_disc = train_disc[batch_indices].cuda()
            cont_out, disc_out = net(batch_x)
            cont_loss = ((cont_out - batch_cont) ** 2).mean()
            disc_loss = disc_crit(disc_out, batch_disc.type(torch.long))
            loss = cont_loss + disc_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            mean_loss += loss.item()
        mean_loss /= num_batches

        # Evaluate
        with torch.no_grad():
            valid_cont, valid_disc = net(valid_batch_x)
            valid_loss_cont = ((valid_cont - valid_batch_cont) ** 2).mean()
            valid_loss_disc = disc_crit(valid_disc, valid_batch_disc.type(torch.long))
            valid_loss = valid_loss_cont + valid_loss_disc
            wandb.log({"loss": mean_loss, "valid_loss": valid_loss.item()})

        # Save
        if (j + 1) % 1 == 0:
            torch.save(net.state_dict(), "temp/stroke_net.pt")


if __name__ == "__main__":
    main()
