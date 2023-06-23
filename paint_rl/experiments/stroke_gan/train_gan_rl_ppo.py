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
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import wandb
from gymnasium.wrappers.time_limit import TimeLimit
from gymnasium.wrappers.normalize import NormalizeReward
from torch.distributions import Normal
from tqdm import tqdm
from PIL import Image  # type: ignore

from paint_rl.algorithms.ppo_cont import train_ppo
from paint_rl.algorithms.rollout_buffer import RolloutBuffer
from paint_rl.conf import entity
from paint_rl.experiments.supervised_strokes.gen_supervised import IMG_SIZE
from paint_rl.experiments.stroke_gan.ref_stroke_env import RefStrokeEnv
from paint_rl.utils import init_orthogonal

_: Any

# Hyperparameters
num_envs = 128  # Number of environments to step through at once during sampling.
train_steps = 64  # Number of steps to step through during sampling. Total # of samples is train_steps * num_envs/
iterations = 500  # Number of sample/train iterations.
train_iters = 2  # Number of passes over the samples collected.
train_batch_size = 2048  # Minibatch size while training models.
discount = 0.95  # Discount factor applied to rewards.
lambda_ = 0.95  # Lambda for GAE.
epsilon = 0.2  # Epsilon for importance sample clipping.
max_eval_steps = 10  # Number of eval runs to average over.
eval_steps = 1  # Max number of steps to take during each eval run.
v_lr = 0.001  # Learning rate of the value net.
p_lr = 0.0001  # Learning rate of the policy net.
d_lr = 0.001  # Learning rate of the discriminator.
action_scale = 0.2  # Scale for actions.
gen_steps = 1  # Number of generator steps per iteration.
disc_steps = 1  # Number of discriminator steps per iteration.
disc_ds_size = 1000  # Size of the discriminator dataset. Half will be generated.
disc_batch_size = 64  # Batch size for the discriminator.
device = torch.device("cuda")  # Device to use during training.


class SharedNet(nn.Module):
    """
    A shared architecture for the value and policy networks.
    Enables easy pretraining if need be.
    Takes in an image of 6 channels.
    """

    def __init__(self, size: int):
        nn.Module.__init__(self)
        self.out_size = 64
        self.net = nn.Sequential(
            nn.Conv2d(5 + 2, 12, 3, 2),
            nn.ReLU(),
            nn.Conv2d(12, 32, 3, 2),
            nn.ReLU(),
            nn.Conv2d(32, self.out_size, 3, 2),
            nn.ReLU(),
        )
        self.net2 = nn.Sequential(nn.Linear(self.out_size, self.out_size), nn.ReLU())
        # Pos encoding
        w = torch.arange(0, size).unsqueeze(0).repeat([size, 1])
        h = torch.arange(0, size).unsqueeze(0).repeat([size, 1]).T
        self.pos = nn.Parameter(
            torch.stack([w, h]).unsqueeze(0) / size, requires_grad=False
        )
        init_orthogonal(self)

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
        self.v_layer2 = nn.Linear(shared_net.out_size, 64)
        self.v_layer3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        init_orthogonal(self)
        self.shared_net = shared_net

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = self.shared_net(input)
        x = self.relu(x)
        x = self.v_layer2(x)
        x = self.relu(x)
        x = self.v_layer3(x)
        return x


class PolicyNet(nn.Module):
    def __init__(
        self,
        action_count: int,
        shared_net: SharedNet,
    ):
        nn.Module.__init__(self)
        self.shared_net = shared_net
        self.a_layer2 = nn.Linear(shared_net.out_size, 64)
        self.a_layer3 = nn.Linear(64, action_count)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        init_orthogonal(self)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = self.shared_net(input)
        x = self.relu(x)
        x = self.a_layer2(x)
        x = self.relu(x)
        x = self.a_layer3(x)
        x = self.sigmoid(x)
        return x


class Discriminator(nn.Module):
    """
    Takes the reference image and the drawn image and returns whether it's
    ground truth or generated. 0: Generated, 1: Real.
    Outputs UNNORMALIZED logits.
    """

    def __init__(
        self,
    ):
        nn.Module.__init__(self)
        self.net = nn.Sequential(
            nn.Conv2d(4, 12, 3, 2),
            nn.ReLU(),
            nn.Conv2d(12, 32, 3, 2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2),
            nn.ReLU(),
        )
        self.net2 = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )
        init_orthogonal(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        x = torch.max(torch.max(x, 3).values, 2).values
        x = self.net2(x)
        return x


def gen_output(p_net: PolicyNet, ref: np.ndarray, img_size: int) -> np.ndarray:
    """
    Given a reference image, returns the canvas with generated strokes.
    """
    temp_env = RefStrokeEnv(img_size, [ref], None)
    obs_, _ = temp_env.reset()
    obs = torch.from_numpy(obs_).float()
    done = False
    while not done:
        distr = Normal(loc=p_net(obs.unsqueeze(0)).squeeze(), scale=0.0001)
        action = distr.sample().numpy()
        obs_, _, done, _, _ = temp_env.step(action)
        obs = torch.from_numpy(obs_).float()
    return np.array(temp_env.canvas)


# Load dataset
img_size = IMG_SIZE
ds_path = Path("temp/form_outputs")
ref_imgs = []
stroke_imgs = []
print("Loading dataset...")

for dir in tqdm(islice(ds_path.iterdir(), 2000)):
    stroke_img = Image.open(dir / "outlines.png")
    stroke_img = stroke_img.resize((img_size, img_size))
    stroke_imgs.append(np.array(stroke_img).transpose([2, 0, 1])[1] / 255.0)
    ref_img = Image.open(dir / "final.png")
    ref_img = ref_img.resize((img_size, img_size))
    ref_img = ref_img.convert("RGB")
    ref_imgs.append(np.array(ref_img).transpose([2, 0, 1]) / 255.0)

# Initialize discriminator
d_net = Discriminator()
d_opt = torch.optim.Adam(d_net.parameters(), lr=d_lr)

env = gym.vector.SyncVectorEnv(
    [
        lambda: NormalizeReward(TimeLimit(RefStrokeEnv(img_size, ref_imgs, d_net), 10))
        for _ in range(num_envs)
    ]
)
test_env = RefStrokeEnv(img_size, ref_imgs, d_net, render_mode="human")

# Argument parsing
parser = ArgumentParser()
parser.add_argument("--eval", action="store_true")
args = parser.parse_args()

# If evaluating, run the sim
if args.eval:
    eval_done = False
    test_env = TimeLimit(StrokeEnv(img_size, render_mode="human"), 10)  # type: ignore
    p_net = PolicyNet(action_count=4, shared_net=SharedNet(img_size))
    p_net.load_state_dict(torch.load("temp/p_net.pt"))
    with torch.no_grad():
        reward_total = 0.0
        eval_obs = torch.Tensor(test_env.reset()[0])
        while True:
            steps_taken = 0
            for _ in range(max_eval_steps):
                distr = Normal(loc=p_net(eval_obs.unsqueeze(0)).squeeze(), scale=0.0001)
                action = distr.sample().numpy()
                obs_, reward, eval_done, eval_trunc, _ = test_env.step(action)
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
assert isinstance(act_space, gym.spaces.Box)
v_net = ValueNet(SharedNet(img_size))
p_net = PolicyNet(int(act_space.shape[0]), SharedNet(img_size))
v_opt = torch.optim.Adam(v_net.parameters(), lr=v_lr)
p_opt = torch.optim.Adam(p_net.parameters(), lr=p_lr)

# A rollout buffer stores experience collected during a sampling run
buffer = RolloutBuffer(
    obs_shape,
    torch.Size((int(act_space.shape[0]),)),
    torch.Size((int(act_space.shape[0]),)),
    torch.float,
    num_envs,
    train_steps,
)

obs = torch.Tensor(env.reset()[0])
done = False
orig_action_scale = action_scale
for step in tqdm(range(iterations), position=0):
    # Training the generator
    total_p_loss = 0.0
    total_v_loss = 0.0
    for _ in range(gen_steps):
        action_scale = orig_action_scale * (1.0 - step / iterations)
        # Collect experience for a number of steps and store it in the buffer
        with torch.no_grad():
            for _ in tqdm(range(train_steps), position=1):
                action_probs = p_net(obs)
                actions = Normal(loc=action_probs, scale=action_scale).sample().numpy()
                obs_, rewards, dones, truncs, _ = env.step(actions)
                buffer.insert_step(
                    obs,
                    torch.from_numpy(actions),
                    action_probs,
                    list(rewards),
                    list(dones),
                    list(truncs),
                )
                obs = torch.from_numpy(obs_)
                if done:
                    obs = torch.Tensor(env.reset()[0])
                    done = False
            buffer.insert_final_step(obs)

        # Train
        step_p_loss, step_v_loss = train_ppo(
            p_net,
            v_net,
            p_opt,
            v_opt,
            buffer,
            device,
            train_iters,
            train_batch_size,
            discount,
            lambda_,
            epsilon,
            action_scale,
        )
        total_p_loss += step_p_loss
        total_v_loss += step_v_loss
        buffer.clear()

    # Training the discriminator
    ds_indices = np.random.permutation(len(ref_imgs))[:disc_ds_size].tolist()
    ds_refs = np.stack([ref_imgs[i] for i in ds_indices]) # Shape: (disc_ds_size, 3, img_size, img_size)
    ground_truth = [stroke_imgs[i] for i in ds_indices[: disc_ds_size // 2]]
    generated = [
        gen_output(p_net, ref_imgs[i], img_size)
        for i in ds_indices[disc_ds_size // 2 :]
    ]
    ds_canvases = np.expand_dims(np.stack(ground_truth + generated), 1) # Shape: (disc_ds_size, 1, img_size, img_size)
    ds_x = np.concatenate([ds_refs, ds_canvases], 1) # Shape: (disc_ds_size, 4, img_size, img_size)
    del ds_refs, ds_canvases
    ds_y = np.concatenate(
        [
            np.ones([disc_ds_size // 2], dtype=np.int64),
            np.zeros([disc_ds_size // 2], dtype=np.int64),
        ],
        0,
    )
    d_crit = nn.CrossEntropyLoss()
    num_batches = disc_ds_size // disc_batch_size
    avg_disc_loss = 0.0
    for _ in range(disc_steps):
        epoch_indices = np.random.permutation(disc_ds_size)
        epoch_x = torch.from_numpy(ds_x[epoch_indices]).float()
        epoch_y = torch.from_numpy(ds_y[epoch_indices])
        for i in range(num_batches):
            batch_x = epoch_x[i * disc_batch_size : (i + 1) * disc_batch_size]
            batch_y = epoch_y[i * disc_batch_size : (i + 1) * disc_batch_size]
            d_opt.zero_grad()
            pred_y = d_net(batch_x)
            loss = d_crit(pred_y, batch_y)
            avg_disc_loss += loss.item()
            loss.backward()
            d_opt.step()
    avg_disc_loss /= disc_steps * num_batches

    # Evaluate the network's performance after this training iteration.
    with torch.no_grad():
        # Visualize
        reward_total = 0.0
        eval_obs = torch.Tensor(test_env.reset()[0])
        for _ in range(eval_steps):
            steps_taken = 0
            for _ in range(max_eval_steps):
                # Action scale is very small since it should learn a deterministic policy
                distr = Normal(loc=p_net(eval_obs.unsqueeze(0)).squeeze(), scale=0.0001)
                action = distr.sample().numpy()
                obs_, reward, eval_done, eval_trunc, _ = test_env.step(action)
                test_env.render()
                eval_obs = torch.Tensor(obs_)
                steps_taken += 1
                if eval_done or eval_trunc:
                    eval_obs = torch.Tensor(test_env.reset()[0])
                    break
                reward_total += reward

    wandb.log(
        {
            "avg_eval_episode_reward": reward_total / eval_steps,
            "avg_v_loss": total_v_loss / (train_iters * gen_steps),
            "avg_p_loss": total_p_loss / (train_iters * gen_steps),
            "action_scale": action_scale,
            "avg_disc_loss": avg_disc_loss,
        }
    )

    if step % 5 == 0:
        torch.save(p_net.state_dict(), "temp/p_net.pt")
