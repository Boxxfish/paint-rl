"""
Experiment for using RL to train a pretrained stroke model with PPO.
The strokes to copy are from a preexisting dataset.
This generates strokes from an actual image.
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
from gymnasium.wrappers.normalize import NormalizeReward
from torch.distributions import Normal, Categorical
from tqdm import tqdm
from PIL import Image, ImageOps  # type: ignore

from paint_rl.algorithms.ppo_multi import train_ppo
from paint_rl.algorithms.rollout_buffer import RolloutBuffer, ActionRolloutBuffer
from paint_rl.conf import entity
from paint_rl.experiments.supervised_strokes.gen_supervised import IMG_SIZE
from paint_rl.experiments.supervised_strokes.outline_stroke_env import OutlineStrokeEnv
from paint_rl.experiments.supervised_strokes.train_supervised_all import (
    StrokeNet as PolicyNet,
    SharedNet,
)

_: Any

# Hyperparameters
num_envs = 256  # Number of environments to step through at once during sampling.
train_steps = 64  # Number of steps to step through during sampling. Total # of samples is train_steps * num_envs/
iterations = 1000  # Number of sample/train iterations.
train_iters = 2  # Number of passes over the samples collected.
train_batch_size = 4096  # Minibatch size while training models.
discount = 0.9  # Discount factor applied to rewards.
lambda_ = 0.95  # Lambda for GAE.
epsilon = 0.2  # Epsilon for importance sample clipping.
max_eval_steps = 20  # Number of eval runs to average over.
eval_steps = 8  # Max number of steps to take during each eval run.
v_lr = 0.001  # Learning rate of the value net.
p_lr = 0.0001  # Learning rate of the policy net.
action_scale = 0.1  # Scale for actions.
device = torch.device("cuda")  # Device to use during training.


class ValueNet(nn.Module):
    def __init__(self, obs_shape: torch.Size):
        nn.Module.__init__(self)
        self.v_layer2 = nn.Linear(256, 256)
        self.v_layer3 = nn.Linear(256, 1)
        self.relu = nn.ReLU()
        self.shared_net = nn.Sequential(
            nn.Conv2d(5, 64, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2),
            nn.ReLU(),
        )

    def forward(self, input: torch.Tensor):
        x = self.shared_net(input)
        x = self.relu(x)
        x = torch.max(torch.max(x, 3).values, 2).values
        x = self.v_layer2(x)
        x = self.relu(x)
        x = self.v_layer3(x)
        return x


img_size = IMG_SIZE

# Argument parsing
parser = ArgumentParser()
parser.add_argument("--eval", action="store_true")
args = parser.parse_args()

# Load dataset
ds_path = Path("temp/all_outputs")
ref_imgs = []
stroke_imgs = []
print("Loading dataset...")

for dir in tqdm(ds_path.iterdir()):
    stroke_img = Image.open(dir / "outlines.png")
    stroke_img = stroke_img.resize((img_size, img_size))
    stroke_imgs.append(np.array(stroke_img).transpose([2, 0, 1])[1] > 20)
    ref_img = Image.open(dir / "final.png")
    ref_img = ref_img.resize((img_size, img_size))
    ref_img = ref_img.convert("RGB")
    ref_imgs.append(np.array(ref_img).transpose([2, 0, 1]) / 255.0)

env = gym.vector.SyncVectorEnv(
    [
        lambda: NormalizeReward(
            OutlineStrokeEnv(img_size, ref_imgs=ref_imgs, stroke_imgs=stroke_imgs)
        )
        for _ in range(num_envs)
    ]
)
test_env = OutlineStrokeEnv(
    img_size, ref_imgs=ref_imgs[:20], stroke_imgs=stroke_imgs[:20], render_mode="human"
)

# If evaluating, run the sim
if args.eval:
    eval_done = False
    test_env = OutlineStrokeEnv(
        img_size, ref_imgs=ref_imgs, stroke_imgs=stroke_imgs, render_mode="human"
    )
    obs_space = test_env.observation_space
    assert obs_space.shape is not None
    obs_shape = torch.Size(obs_space.shape)
    act_space = env.single_action_space
    assert isinstance(act_space, gym.spaces.Tuple)
    assert isinstance(act_space.spaces[0], gym.spaces.Box)
    assert isinstance(act_space.spaces[1], gym.spaces.Discrete)
    p_net = PolicyNet(img_size)
    p_net.load_state_dict(torch.load("temp/p_net.pt"))
    with torch.no_grad():
        reward_total = 0.0
        eval_obs = torch.Tensor(test_env.reset()[0])
        while True:
            steps_taken = 0
            for _ in range(max_eval_steps):
                action_probs_cont, action_probs_disc = p_net(eval_obs.unsqueeze(0))
                actions_cont = (
                    Normal(loc=action_probs_cont.squeeze(), scale=0.00001)
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
                    print("Total reward:", reward_total)
                    reward_total = 0
                    eval_obs = torch.Tensor(test_env.reset()[0])
                    break
                reward_total += reward
del ref_imgs
wandb.init(
    project="paint-rl",
    entity=entity,
    config={
        "experiment": "supervised stroke with RL (PPO) for spheres",
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
p_net = PolicyNet(img_size)
p_net.load_state_dict(torch.load("temp/stroke_net.pt"))
v_net = ValueNet(obs_shape)
v_opt = torch.optim.Adam(v_net.parameters(), lr=v_lr)
p_opt = torch.optim.Adam(p_net.parameters(), lr=p_lr)

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
    step_amount = (1.0 - step / iterations)
    action_scale = orig_action_scale * step_amount
    dilation_size = int(2 * step_amount)
    for i in range(num_envs):
        env.envs[i].unwrapped.set_dilation_size(dilation_size) # type: ignore
    test_env.set_dilation_size(dilation_size)

    # Collect experience for a number of steps and store it in the buffer
    with torch.no_grad():
        for _ in tqdm(range(train_steps), position=1):
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
    total_p_loss, total_v_loss = train_ppo(
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
    )
    buffer_cont.clear()
    buffer_disc.clear()

    # Evaluate the network's performance after this training iteration.
    with torch.no_grad():
        # Visualize
        reward_total = 0.0
        avg_eval_entropy_disc = 0.0
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
                actions_disc_distr = Categorical(logits=action_probs_disc.squeeze())
                actions_disc = actions_disc_distr.sample().unsqueeze(-1).numpy()
                avg_eval_entropy_disc += actions_disc_distr.entropy()
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

    wandb.log(
        {
            "avg_eval_episode_reward": reward_total / eval_steps,
            "avg_v_loss": total_v_loss / train_iters,
            "avg_p_loss": total_p_loss / train_iters,
            "avg_eval_entropy_disc": avg_eval_entropy_disc / eval_steps,
            "action_scale": action_scale,
        }
    )

    if step % 5 == 0:
        torch.save(p_net.state_dict(), "temp/p_net.pt")
        torch.save(v_net.state_dict(), "temp/v_net.pt")
