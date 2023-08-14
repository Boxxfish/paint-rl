"""
Experiment for training a policy net using GAN techiniques.

Training occurs in two phases that repeat over and over again. In the first, the
generator model is trained to output strokes, using the discriminator as a
reward signal. In the second, a dataset of ground truth and generated images are
created from the generator, and the discriminator attempts to classify them as either.
"""
from argparse import ArgumentParser
import copy
from functools import reduce
from itertools import islice
from pathlib import Path
import sys
from typing import Any, Tuple

import gymnasium as gym
from matplotlib import pyplot as plt  # type: ignore
import numpy as np
from paint_rl_rust import TrainingContext
import torch
import torch.nn as nn
import wandb
from gymnasium.wrappers.normalize import NormalizeReward
from torch.distributions import Normal, Categorical
from tqdm import tqdm
from PIL import Image  # type: ignore
from torch.nn.utils.parametrizations import spectral_norm
from paint_rl.algorithms.ppo_multi import train_ppo
from paint_rl.algorithms.rollout_buffer import ActionRolloutBuffer, RolloutBuffer
from paint_rl.conf import entity
from paint_rl.experiments.supervised_strokes.gen_supervised import IMG_SIZE
from paint_rl.experiments.stroke_gan.ref_stroke_env import RefStrokeEnv
from paint_rl.experiments.supervised_strokes.train_supervised_all import (
    StrokeNet as PolicyNet,
)

_: Any

# Hyperparameters
num_envs = 32  # Number of environments to step through at once during sampling.
train_steps = 256  # Number of steps to step through during sampling. Total # of samples is train_steps * num_envs.
iterations = 100  # Number of sample/train iterations.
train_iters = 1  # Number of passes over the samples collected.
train_batch_size = 1024  # Minibatch size while training models.
discount = 0.99  # Discount factor applied to rewards.
lambda_ = 0.99  # Lambda for GAE.
epsilon = 0.2  # Epsilon for importance sample clipping.
max_eval_steps = 300  # Number of eval runs to average over.
eval_steps = 4  # Max number of steps to take during each eval run.
v_lr = 0.001  # Learning rate of the value net.
p_lr = 0.00003  # Learning rate of the policy net.
d_lr = 0.001  # Learning rate of the discriminator.
action_scale = 0.02  # Scale for actions.
gen_steps = 1  # Number of generator steps per iteration.
disc_steps = 2  # Number of discriminator steps per iteration.
disc_ds_size = 1024  # Size of the discriminator dataset. Half will be generated.
disc_batch_size = 64  # Batch size for the discriminator.
stroke_width = 4
canvas_size = 256
device = torch.device("cuda")  # Device to use during training.

# Argument parsing
parser = ArgumentParser()
parser.add_argument("--eval", action="store_true")
parser.add_argument("--test")
parser.add_argument("--resume", action="store_true")
args = parser.parse_args()


class SharedNet(nn.Module):
    """
    A shared architecture for the value and policy networks.
    Enables easy pretraining if need be.
    Takes in an image of 5 channels.
    """

    def __init__(self, size: int):
        nn.Module.__init__(self)
        self.out_size = 512
        self.net = nn.Sequential(
            nn.Conv2d(7 + 2, 128, 3, 2),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, 2),
            nn.ReLU(),
            nn.Conv2d(256, self.out_size, 3, 2),
            nn.ReLU(),
        )
        self.net2 = nn.Sequential(nn.Linear(self.out_size, self.out_size), nn.ReLU())
        # Pos encoding
        w = torch.arange(0, size).unsqueeze(0).repeat([size, 1])
        h = torch.arange(0, size).unsqueeze(0).repeat([size, 1]).T
        self.pos = nn.Parameter(
            torch.stack([w, h]).unsqueeze(0) / size, requires_grad=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        pos_enc = self.pos.repeat([batch_size, 1, 1, 1])
        x = torch.cat([x, pos_enc], dim=1)
        x = self.net(x)
        x = torch.max(torch.max(x, 3).values, 2).values
        x = self.net2(x)
        return x


class ValueNet(nn.Module):
    def __init__(
        self,
        shared_net: SharedNet,
    ):
        nn.Module.__init__(self)
        self.shared_net = shared_net
        self.v_layer2 = nn.Linear(shared_net.out_size, 256)
        self.v_layer3 = nn.Linear(256, 1)
        self.relu = nn.ReLU()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = self.shared_net(input)
        x = self.relu(x)
        x = self.v_layer2(x)
        x = self.relu(x)
        x = self.v_layer3(x)
        return x


# class PolicyNet(nn.Module):
#     def __init__(
#         self,
#         action_count_cont: int,
#         action_count_discrete: int,
#         shared_net: SharedNet,
#     ):
#         nn.Module.__init__(self)
#         self.shared_net = shared_net
#         self.continuous = nn.Sequential(
#             nn.Linear(shared_net.out_size, 64),
#             nn.ReLU(),
#             nn.Linear(64, action_count_cont),
#             nn.Sigmoid(),
#         )
#         self.discrete = nn.Sequential(
#             nn.Linear(shared_net.out_size, 64),
#             nn.ReLU(),
#             nn.Linear(64, action_count_discrete),
#             nn.LogSoftmax(1),
#         )
#         self.relu = nn.ReLU()

#     def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#         x = self.shared_net(input)
#         cont = self.continuous(x)
#         disc = self.discrete(x)
#         return (cont, disc)


class Discriminator(nn.Module):
    """
    Takes the reference image and the drawn image and returns whether it's
    ground truth or generated. 0: Generated, 1: Real.
    Based off the DCGAN discriminator architecture.
    """

    def __init__(
        self,
    ):
        nn.Module.__init__(self)
        self.net = nn.Sequential(
            spectral_norm(nn.Conv2d(4, 32, 3, 2, padding=1)),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Conv2d(32, 64, 3, 2, padding=1)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Conv2d(64, 128, 3, 4, padding=1)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )
        self.net2 = nn.Sequential(
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        x = torch.max(torch.max(x, 3).values, 2).values
        x = self.net2(x)
        return x


def gen_output(
    p_net: PolicyNet,
    ref: np.ndarray,
    canvas_size: int,
    stroke_width: int,
    img_size: int,
    action_scale: float,
    d_net: nn.Module,
) -> np.ndarray:
    """
    Given a reference image, returns the canvas with generated strokes.
    """
    with torch.no_grad():
        temp_env = RefStrokeEnv(
            canvas_size, img_size, [ref], d_net, stroke_width=stroke_width
        )
        obs_, _ = temp_env.reset()
        obs = torch.from_numpy(obs_).float()
        done = False
        trunc = False
        while not (done or trunc):
            action_probs_cont, action_probs_disc = p_net(obs.unsqueeze(0))
            cont_distr = Normal(loc=action_probs_cont.squeeze(), scale=action_scale)
            disc_distr = Categorical(logits=action_probs_disc.squeeze())
            action_cont = cont_distr.sample().numpy()
            action_disc = disc_distr.sample().numpy()
            obs_, _, done, trunc, _ = temp_env.step((action_cont, action_disc))
            obs = torch.from_numpy(obs_).float()
    return np.array(temp_env.canvas)


# Load dataset
img_size = IMG_SIZE
ds_path = Path("temp/all_outputs")
ref_imgs = []
stroke_imgs = []
print("Loading dataset...")

if not args.test:
    amount = 100 if args.eval else 10000
    for dir in tqdm(islice(ds_path.iterdir(), amount)):
        stroke_img = Image.open(dir / "outlines.png")
        stroke_img = stroke_img.resize(
            (img_size, img_size), resample=Image.Resampling.BILINEAR
        )
        stroke_imgs.append(np.array(stroke_img).transpose([2, 0, 1])[1] / 255.0)
        ref_img = Image.open(dir / "final.png")
        ref_img = ref_img.resize((img_size, img_size))
        ref_img = ref_img.convert("RGB")
        ref_imgs.append(np.array(ref_img).transpose([2, 0, 1]) / 255.0)

# Initialize discriminator
d_net = Discriminator()
d_net.eval()
d_opt = torch.optim.Adam(d_net.parameters(), lr=d_lr)

max_strokes = 50
env = gym.vector.SyncVectorEnv(
    [
        lambda: NormalizeReward(
            RefStrokeEnv(
                canvas_size,
                img_size,
                ref_imgs,
                d_net,
                stroke_width=stroke_width,
                max_strokes=max_strokes,
            )
        )
        for _ in range(num_envs)
    ]
)
test_env = RefStrokeEnv(
    canvas_size,
    img_size,
    ref_imgs,
    d_net,
    stroke_width=stroke_width,
    render_mode="human",
    max_strokes=max_strokes,
)

# If evaluating, run the sim
if args.eval:
    d_net.to(device)
    eval_done = False
    d_net.load_state_dict(torch.load("temp/d_net.pt"))
    test_env = RefStrokeEnv(
        canvas_size,
        img_size,
        ref_imgs,
        d_net,
        stroke_width=stroke_width,
        render_mode="human",
    )
    p_net = PolicyNet(IMG_SIZE)
    p_net.load_state_dict(torch.load("temp/p_net.pt"))
    with torch.no_grad():
        reward_total = 0.0
        eval_obs = torch.Tensor(test_env.reset()[0])
        while True:
            steps_taken = 0
            for _ in range(max_eval_steps):
                action_probs_cont, action_probs_disc = p_net(eval_obs.unsqueeze(0))
                actions_cont = (
                    Normal(loc=action_probs_cont.squeeze(), scale=0.001)
                    .sample()
                    .numpy()
                )
                actions_disc = (
                    Categorical(logits=action_probs_disc.squeeze())
                    .sample()
                    .unsqueeze(-1)
                    .numpy()
                )
                obs_, reward, eval_done, eval_trunc, _ = test_env.step(
                    (actions_cont, actions_disc)
                )
                test_env.render()
                eval_obs = torch.Tensor(obs_)
                steps_taken += 1
                if eval_done or eval_trunc:
                    eval_obs = torch.Tensor(test_env.reset()[0])
                    break
                reward_total += reward

# If testing, run the sim
if args.test:
    test_img = Image.open(args.test)
    test_img = test_img.resize((img_size, img_size))
    test_img = test_img.convert("RGB")
    test_img = np.array(test_img).transpose([2, 0, 1]) / 255.0

    eval_done = False
    test_env = RefStrokeEnv(
        canvas_size,
        img_size,
        [test_img],
        None,
        stroke_width=stroke_width,
        render_mode="human",
        max_strokes=100,
    )
    p_net = PolicyNet(IMG_SIZE)
    p_net.load_state_dict(torch.load("temp/p_net.pt"))
    with torch.no_grad():
        reward_total = 0.0
        eval_obs = torch.Tensor(test_env.reset()[0])
        while True:
            steps_taken = 0
            for _ in range(max_eval_steps):
                action_probs_cont, action_probs_disc = p_net(eval_obs.unsqueeze(0))
                actions_cont = (
                    Normal(loc=action_probs_cont.squeeze(), scale=0.001)
                    .sample()
                    .numpy()
                )
                actions_disc = (
                    Categorical(logits=action_probs_disc.squeeze())
                    .sample()
                    .unsqueeze(-1)
                    .numpy()
                )
                obs_, reward, eval_done, eval_trunc, _ = test_env.step(
                    (actions_cont, actions_disc)
                )
                test_env.render()
                eval_obs = torch.Tensor(obs_)
                steps_taken += 1
                if eval_done or eval_trunc:
                    eval_obs = torch.Tensor(test_env.reset()[0])
                    break
                reward_total += reward

wandb.init(
    project="paint-rl",
    entity=entity,
    config={
        "experiment": "stroke GAN (PPO)",
        "num_envs": num_envs,
        "train_steps": train_steps,
        "train_iters": train_iters,
        "train_batch_size": train_batch_size,
        "discount": discount,
        "lambda": lambda_,
        "epsilon": epsilon,
        "max_eval_steps": max_eval_steps,
        "v_lr": v_lr,
        "p_lr": p_lr,
    },
)

# Initialize policy and value networks
obs_space = env.single_observation_space
assert obs_space.shape is not None
obs_shape = torch.Size(obs_space.shape)
act_space = env.single_action_space
assert isinstance(act_space, gym.spaces.Tuple)
assert isinstance(act_space.spaces[0], gym.spaces.Box)
assert isinstance(act_space.spaces[1], gym.spaces.Discrete)
action_count_cont = int(act_space.spaces[0].shape[0])
action_count_discrete = int(act_space.spaces[1].n)
v_net = ValueNet(SharedNet(img_size))
p_net = PolicyNet(img_size)
p_net.load_state_dict(torch.load("temp/stroke_net.pt"))  # For loading from pretraining
v_opt = torch.optim.Adam(v_net.parameters(), lr=v_lr)
p_opt = torch.optim.Adam(p_net.parameters(), lr=p_lr)

# Initialize training context
p_net_path = "temp/training/p_net.ptc"
d_net_path = "temp/training/d_net.ptc"
sample_input_p = torch.from_numpy(obs_space.sample()).unsqueeze(0)
traced = torch.jit.trace(
    p_net,
    (sample_input_p,),
)
traced.save(p_net_path)

sample_input_d = torch.zeros([1, 4, img_size, img_size])
d_net.cpu()
traced = torch.jit.trace(
    d_net,
    (sample_input_d,),
)
traced.save(d_net_path)
training_context = TrainingContext(
    img_size,
    canvas_size,
    "temp/all_outputs",
    p_net_path,
    d_net_path,
    64,
    max_strokes,
    8,
)
d_net.to(device)

# If resuming, load state dicts
if args.resume:
    v_net.load_state_dict(torch.load("temp/v_net.pt"))
    p_net.load_state_dict(torch.load("temp/p_net.pt"))
    d_net.load_state_dict(torch.load("temp/d_net.pt"))

# A rollout buffer stores experience collected during a sampling run
buffer_cont = RolloutBuffer(
    obs_shape,
    torch.Size((action_count_cont,)),
    torch.Size((action_count_cont,)),
    torch.float,
    num_envs,
    train_steps,
)
buffer_disc = ActionRolloutBuffer(
    torch.Size((1,)),
    torch.Size((action_count_discrete,)),
    torch.int,
    num_envs,
    train_steps,
)

obs = torch.Tensor(env.reset()[0])
done = False
orig_action_scale = action_scale
for step in tqdm(range(iterations), position=0):
    # Export models
    traced = torch.jit.trace(
        p_net,
        (sample_input_p,),
    )
    traced.save(p_net_path)

    d_net.cpu()
    traced = torch.jit.trace(
        d_net,
        (sample_input_d,),
    )
    traced.save(d_net_path)
    generated = training_context.gen_imgs(
        disc_ds_size // 2, action_scale
    )  # Shape: (disc_ds_size / 2, 4 img_size, img_size)
    d_net.to(device)

    # Training the discriminator
    ds_indices = np.random.permutation(len(ref_imgs))[: disc_ds_size // 2].tolist()
    ds_refs = np.stack(
        [ref_imgs[i] for i in ds_indices]
    )  # Shape: (disc_ds_size, 3, img_size, img_size)
    ground_truth = np.expand_dims(
        np.stack([stroke_imgs[i] for i in ds_indices]), 1
    )  # Shape: (disc_ds_size / 2, 1 img_size, img_size)
    ds_x_real = (
        torch.from_numpy(np.concatenate([ds_refs, ground_truth], 1)).float()
    )  # Shape: (disc_ds_size / 2, 4, img_size, img_size)
    ds_y_real_batch = torch.from_numpy(np.ones([disc_batch_size])).float().to(device)
    ds_x_generated = (
        torch.from_numpy(generated).float()
    )  # Shape: (disc_ds_size / 2, 4, img_size, img_size)
    ds_y_generated_batch = torch.zeros([disc_batch_size], dtype=torch.float, device=device)
    del ds_refs, ground_truth, generated
    noise = torch.distributions.Normal(0.0, 0.1).sample(ds_x_real.shape)
    ds_x_real = torch.clip(ds_x_real + noise, 0.0, 1.0)
    ds_x_generated = torch.clip(ds_x_generated + noise, 0.0, 1.0)
    d_crit = nn.BCELoss()
    num_batches = disc_ds_size // (2 * disc_batch_size)
    avg_disc_loss_real = 0.0
    avg_disc_loss_generated = 0.0
    d_net.train()

    for _ in tqdm(range(disc_steps), position=1):
        epoch_indices = torch.from_numpy(np.random.permutation(disc_ds_size // 2))

        # Train on generated
        ds_x_generated = ds_x_generated[epoch_indices]
        for i in range(num_batches):
            batch_x = ds_x_generated[i * disc_batch_size : (i + 1) * disc_batch_size].to(device)
            batch_y = ds_y_generated_batch
            d_opt.zero_grad()
            pred_y = d_net(batch_x).squeeze(1)
            loss = d_crit(pred_y, batch_y)
            avg_disc_loss_generated += loss.item()
            loss.backward()
            d_opt.step()

        # Train on real
        ds_x_real = ds_x_real[epoch_indices]
        for i in range(num_batches):
            batch_x = ds_x_real[i * disc_batch_size : (i + 1) * disc_batch_size].to(device)
            batch_y = ds_y_real_batch
            d_opt.zero_grad()
            pred_y = d_net(batch_x).squeeze(1)
            loss = d_crit(pred_y, batch_y)
            avg_disc_loss_real += loss.item()
            loss.backward()
            d_opt.step()

        # with torch.no_grad():
        #     for i in range(5):
        #         pred = d_net(ds_x_real[i].unsqueeze(0)).squeeze(0).item()
        #         print("Prediction: ", pred)
        #         plt.imshow(ds_x_real[i][3].cpu())
        #         plt.show()
        #         plt.imshow(ds_x_real[i][:3].mean(0).cpu())
        #         plt.show()
        #         pred = d_net(ds_x_generated[i].unsqueeze(0)).squeeze(0).item()
        #         print("Prediction: ", pred)
        #         plt.imshow(ds_x_generated[i][3].cpu())
        #         plt.show()
        #         plt.imshow(ds_x_generated[i][:3].mean(0).cpu())
        #         plt.show()
    avg_disc_loss_real /= disc_steps * num_batches
    avg_disc_loss_generated /= disc_steps * num_batches
    d_net.eval()

    # Training the generator
    if step < 4:
        continue
    total_p_loss = 0.0
    total_v_loss = 0.0
    for _ in tqdm(range(gen_steps), position=1):
        action_scale = orig_action_scale * (1.0 - step / iterations)
        # Collect experience for a number of steps and store it in the buffer
        with torch.no_grad():
            for _ in tqdm(range(train_steps), position=2):
                action_probs_cont, action_probs_disc = p_net(obs)
                actions_cont = (
                    Normal(loc=action_probs_cont, scale=action_scale).sample().numpy()
                )
                actions_disc = (
                    Categorical(logits=action_probs_disc).sample().unsqueeze(-1).numpy()
                )
                obs_, rewards, dones, truncs, _ = env.step((actions_cont, actions_disc))
                buffer_cont.insert_step(
                    obs,
                    torch.from_numpy(actions_cont),
                    action_probs_cont,
                    list(rewards),
                    list(dones),
                    list(truncs),
                )
                buffer_disc.insert_step(
                    torch.from_numpy(actions_disc),
                    action_probs_disc,
                )
                obs = torch.from_numpy(obs_)
                if done:
                    obs = torch.Tensor(env.reset()[0])
                    done = False
            buffer_cont.insert_final_step(obs)

        # Train
        step_p_loss, step_v_loss = train_ppo(
            p_net,
            v_net,
            p_opt,
            v_opt,
            (buffer_cont, buffer_disc),
            device,
            train_iters,
            train_batch_size,
            discount,
            lambda_,
            epsilon,
            action_scale,
            gradient_steps=2
        )
        total_p_loss += step_p_loss
        total_v_loss += step_v_loss
        buffer_cont.clear()
        buffer_disc.clear()

    # Evaluate the network's performance after this training iteration.
    with torch.no_grad():
        # Visualize
        reward_total = 0.0
        eval_obs = torch.Tensor(test_env.reset()[0])
        for _ in range(eval_steps):
            steps_taken = 0
            for _ in range(max_eval_steps):
                action_probs_cont, action_probs_disc = p_net(eval_obs.unsqueeze(0))
                actions_cont = (
                    Normal(loc=action_probs_cont.squeeze(), scale=action_scale)
                    .sample()
                    .numpy()
                )
                actions_disc = (
                    Categorical(logits=action_probs_disc.squeeze())
                    .sample()
                    .unsqueeze(-1)
                    .numpy()
                )
                obs_, reward, eval_done, eval_trunc, _ = test_env.step(
                    (actions_cont, actions_disc)
                )
                test_env.render()
                eval_obs = torch.Tensor(obs_)
                steps_taken += 1
                reward_total += reward
                if eval_done or eval_trunc:
                    eval_obs = torch.Tensor(test_env.reset()[0])
                    break

    wandb.log(
        {
            "avg_eval_episode_reward": reward_total / eval_steps,
            "avg_v_loss": total_v_loss / (train_iters * gen_steps),
            "avg_p_loss": total_p_loss / (train_iters * gen_steps),
            "action_scale": action_scale,
            "avg_disc_loss_real": avg_disc_loss_real,
            "avg_disc_loss_generated": avg_disc_loss_generated,
        }
    )

    torch.save(p_net.state_dict(), "temp/p_net.pt")
    torch.save(v_net.state_dict(), "temp/v_net.pt")
    torch.save(d_net.state_dict(), "temp/d_net.pt")
