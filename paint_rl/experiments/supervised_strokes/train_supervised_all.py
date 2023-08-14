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
    Important: Do NOT reinitialize parameters! This erases the positional encoding.
    """

    def __init__(self, size: int):
        nn.Module.__init__(self)
        self.net = nn.Sequential(
            nn.Conv2d(7 + 2, 32, 7, stride=2),
            ResBlock(
                nn.Sequential(
                    nn.ReLU(),
                    nn.Conv2d(32, 32, 5, padding=2),
                    nn.ReLU(),
                    nn.Conv2d(32, 32, 5, padding=2),
                    nn.ReLU(),
                )
            ),
            nn.Conv2d(32, 64, 3, stride=2),
            ResBlock(
                nn.Sequential(
                    nn.ReLU(),
                    nn.Conv2d(64, 64, 3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(64, 64, 3, padding=1),
                    nn.ReLU(),
                )
            ),
            nn.Conv2d(64, 128, 3, stride=2),
            ResBlock(
                nn.Sequential(
                    nn.ReLU(),
                    nn.Conv2d(128, 128, 3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(128, 128, 3, padding=1),
                    nn.ReLU(),
                )
            ),
            nn.Conv2d(128, 256, 3, stride=2),
            ResBlock(
                nn.Sequential(
                    nn.ReLU(),
                    nn.Conv2d(256, 256, 3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(256, 256, 3, padding=1),
                    nn.ReLU(),
                )
            ),
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
    prev_frame,
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
    cont_actions.append(np.concatenate([p2, p3]))
    disc_actions.append(0 if up else 1)
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


def load_random_ds(
    valid_size: int, size: int, img_size: int, orig_img_size: int
) -> tuple[Tensor, Tensor, Tensor]:
    indices = list(range(valid_size, 10_000))
    random.shuffle(indices)
    ds_x: list[np.ndarray] = []
    cont_actions: list[np.ndarray] = []
    disc_actions: list[np.ndarray] = []
    for i in tqdm(indices):
        if len(ds_x) == size:
            break
        try:
            img = (
                Image.open(f"temp/all_outputs/{i}/final.png")
                .convert("RGB")
                .resize((img_size, img_size))
            )
            img_data = np.array(img).transpose(2, 0, 1) / 255.0
            with open(f"temp/all_outputs/{i}/strokes.json", "r") as f:
                path = json.load(f)
            if len(path["cont"]) == 0:
                continue

            stroke_img = Image.new("L", (img_size, img_size))
            stroke_draw = ImageDraw.Draw(stroke_img)

            last_point = np.array(path["start"]).astype(float) / orig_img_size
            prev_frame = np.zeros([2, img_size, img_size])
            for i, (c, d) in enumerate(zip(path["cont"], path["disc"])):
                if len(ds_x) == size:
                    break

                mid_point = np.array([c[0], c[1]]).astype(float) / orig_img_size
                end_point = np.array([c[2], c[3]]).astype(float) / orig_img_size
                if i == 0:
                    last_point = np.array([random.random(), random.random()]).astype(
                        float
                    )
                    mid_point = (last_point + end_point) / 2.0

                # 50% chance of actually adding example
                will_add = random.random() < 0.5
                prev_frame, last_point = stroke(
                    last_point,
                    mid_point,
                    end_point,
                    img_data,
                    stroke_img,
                    stroke_draw,
                    ds_x if will_add else [],
                    cont_actions if will_add else [],
                    disc_actions if will_add else [],
                    img_size,
                    prev_frame,
                    up=d == 0,
                )
        except KeyboardInterrupt:
            quit()
        except Exception as e:
            print(f"Error processing {i}: {e}")

    # for i in range(len(ds_x)):
    #     plt.imshow(ds_x[i].mean(0))
    #     plt.show()

    cont_actions_ = torch.from_numpy(np.stack(cont_actions))
    del cont_actions
    disc_actions_ = torch.from_numpy(np.stack(disc_actions))
    del disc_actions

    ds_x_ = torch.from_numpy(np.stack(ds_x))
    del ds_x
    return (ds_x_.float(), cont_actions_.float(), disc_actions_)


def main():
    orig_img_size = 256
    img_size = 64

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
                Image.open(f"temp/all_outputs/{i}/final.png")
                .convert("RGB")
                .resize((img_size, img_size))
            )
            img_data = np.array(img).transpose(2, 0, 1) / 255.0
            imgs.append(img_data)
        with torch.no_grad():
            net = StrokeNet(img_size)
            net.load_state_dict(torch.load("temp/stroke_net.pt"))
            env = RefStrokeEnv(img_size, img_size, imgs, None, "human", max_strokes=100)
            obs = torch.from_numpy(env.reset()[0]).float().unsqueeze(0)
            while True:
                cont_action, disc_action = net(obs)
                obs_, _, done, trunc, _ = env.step(
                    (cont_action.squeeze().numpy(), disc_action.argmax().item())
                )
                env.render()
                obs = torch.from_numpy(obs_).float().unsqueeze(0)
                if done or trunc:
                    obs = torch.from_numpy(env.reset()[0]).float().unsqueeze(0)

    # Load dataset
    ds_x = []
    cont_actions = []
    disc_actions = []
    valid_ds_x = []
    valid_cont_actions = []
    valid_disc_actions = []
    valid_size = 8
    print("Loading data.")
    for i in tqdm(range(1500)):
        valid = i < valid_size
        try:
            img = (
                Image.open(f"temp/all_outputs/{i}/final.png")
                .convert("RGB")
                .resize((img_size, img_size))
            )
            img_data = np.array(img).transpose(2, 0, 1) / 255.0
            with open(f"temp/all_outputs/{i}/strokes.json", "r") as f:
                path = json.load(f)
            stroke_img = Image.new("L", (img_size, img_size))
            stroke_draw = ImageDraw.Draw(stroke_img)

            last_point = np.array(path["start"]).astype(float) / orig_img_size
            prev_frame = np.zeros([2, img_size, img_size])
            for c, d in zip(path["cont"], path["disc"]):
                prev_frame, last_point = stroke(
                    last_point,
                    np.array([c[0], c[1]]).astype(float) / orig_img_size,
                    np.array([c[2], c[3]]).astype(float) / orig_img_size,
                    img_data,
                    stroke_img,
                    stroke_draw,
                    ds_x if not valid else valid_ds_x,
                    cont_actions if not valid else valid_cont_actions,
                    disc_actions if not valid else valid_disc_actions,
                    img_size,
                    prev_frame,
                    up=d == 0,
                )
            # plt.imshow(ds_x[-1][:3].transpose(1, 2, 0))
            # plt.show()
        except KeyboardInterrupt:
            quit()
        except Exception as e:
            print(f"Error processing {i}: {e}")
    train_x = torch.from_numpy(np.stack(ds_x))
    train_cont = torch.from_numpy(np.stack(cont_actions))
    train_disc = torch.from_numpy(np.stack(disc_actions))
    valid_x = torch.from_numpy(np.stack(valid_ds_x))
    valid_cont = torch.from_numpy(np.stack(valid_cont_actions))
    valid_disc = torch.from_numpy(np.stack(valid_disc_actions))
    del (
        ds_x,
        cont_actions,
        disc_actions,
        valid_ds_x,
        valid_cont_actions,
        valid_disc_actions,
    )
    train_size = train_x.shape[0]
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
    net = StrokeNet(img_size).cuda().train()
    if args.resume:
        net.load_state_dict(torch.load("temp/stroke_net.pt"))
    opt = torch.optim.Adam(net.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, "min", min_lr=0.00001, factor=1 / 3, patience=30
    )
    batch_size = 512
    train_x = train_x.float()
    train_cont = train_cont.float()
    train_disc = train_disc
    valid_batch_x = valid_x.float().cuda()
    valid_batch_cont = valid_cont.float().cuda()
    valid_batch_disc = valid_disc.cuda()
    disc_crit = nn.NLLLoss()
    cont_scale = 40.0
    for j in tqdm(range(1000000)):
        mean_loss = 0.0
        mean_cont_loss = 0.0
        mean_disc_loss = 0.0
        num_batches = train_size // batch_size
        indices = torch.from_numpy(
            np.random.choice(train_size, train_size, replace=False)
        )
        for i in range(num_batches):
            batch_indices = indices[i * batch_size : (i + 1) * batch_size]
            batch_x = train_x[batch_indices].cuda()
            batch_cont = train_cont[batch_indices].cuda()
            batch_disc = train_disc[batch_indices].cuda()
            cont_out, disc_out = net(batch_x)
            cont_loss = ((cont_out - batch_cont) ** 2).mean()
            disc_loss = disc_crit(disc_out, batch_disc.type(torch.long))
            loss = cont_scale * cont_loss + disc_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            mean_loss += loss.item()
            mean_cont_loss += cont_loss
            mean_disc_loss += disc_loss
        mean_loss /= num_batches
        mean_cont_loss /= num_batches
        mean_disc_loss /= num_batches

        # Evaluate
        with torch.no_grad():
            valid_cont, valid_disc = net(valid_batch_x)
            valid_loss_cont = ((valid_cont - valid_batch_cont) ** 2).mean()
            valid_loss_disc = disc_crit(valid_disc, valid_batch_disc.type(torch.long))
            valid_loss = cont_scale * valid_loss_cont + valid_loss_disc
            scheduler.step(valid_loss)
            wandb.log(
                {
                    "loss": mean_loss,
                    "valid_loss": valid_loss.item(),
                    "cont_loss": mean_cont_loss,
                    "disc_loss": mean_disc_loss,
                    "valid_cont_loss": valid_loss_cont.item(),
                    "valid_disc_loss": valid_loss_disc.item(),
                    "lr": opt.param_groups[0]["lr"],
                }
            )

        if (j + 1) % 10 == 0:
            del train_x, train_cont, train_disc
            train_x, train_cont, train_disc = load_random_ds(
                valid_size, train_size, img_size, orig_img_size
            )

        # Save
        torch.save(net.state_dict(), "temp/stroke_net.pt")


if __name__ == "__main__":
    main()
