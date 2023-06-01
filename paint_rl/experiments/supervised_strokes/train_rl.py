"""
Experiment for using RL to train a pretrained stroke model.
"""
import copy
import random
from functools import reduce
from typing import Any

import gymnasium as gym
from gymnasium.wrappers.time_limit import TimeLimit
import numpy as np  # type: ignore
import torch
import torch.nn as nn
import wandb
from tqdm import tqdm
from paint_rl.algorithms.ddpg import train_ddpg

import paint_rl.conf
from paint_rl.algorithms.td3 import train_td3
from paint_rl.algorithms.replay_buffer import ReplayBuffer
from paint_rl.experiments.supervised_strokes.gen_supervised import IMG_SIZE, gen_sample
from paint_rl.experiments.supervised_strokes.stroke_env import StrokeEnv
from paint_rl.experiments.supervised_strokes.train_supervised import (
    SharedNet,
    StrokeNet,
)
from paint_rl.utils import init_orthogonal
from matplotlib import pyplot as plt  # type: ignore

_: Any

# Hyperparameters
num_envs = 2  # Number of environments to step through at once during sampling.
train_steps = 64  # Number of steps to step through during sampling. Total # of samples is train_steps * num_envs.
iterations = 10000  # Number of sample/train iterations.
train_iters = 20  # Number of passes over the samples collected.
train_batch_size = 128  # Minibatch size while training models.
discount = 0.99  # Discount factor applied to rewards.
q_epsilon = 0.4  # Amount of noise added to actions during training.
eval_steps = 1  # Number of eval runs to average over.
max_eval_steps = 20  # Max number of steps to take during each eval run.
q_lr = 0.0003  # Learning rate of the q net.
p_lr = 0.0003  # Learning rate of the p net.
polyak = 0.005  # Polyak averaging value.
noise_scale = 0.02  # Target smoothing noise scale.
noise_clip = 0.1  # Target smoothing clip value.
warmup_steps = 0  # For the first n number of steps, we will only sample randomly.
buffer_size = 5000  # Number of elements that can be stored in the buffer.
device = torch.device("cuda")

wandb.init(
    project="paint-rl",
    entity=paint_rl.conf.entity,
    config={
        "experiment": "supervised stroke with RL",
        "num_envs": num_envs,
        "train_steps": train_steps,
        "train_iters": train_iters,
        "train_batch_size": train_batch_size,
        "discount": discount,
        "q_epsilon": q_epsilon,
        "max_eval_steps": max_eval_steps,
        "q_lr": q_lr,
        "p_lr": p_lr,
        "polyak": polyak,
    },
)


# The Q network takes in an observation and a action and returns the predicted return.
class QNet(nn.Module):
    def __init__(
        self,
        obs_shape: torch.Size,
        act_count: int,
        shared_net: SharedNet,
    ):
        nn.Module.__init__(self)
        self.action_head = nn.Sequential(
            nn.Linear(act_count, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )
        self.net = nn.Sequential(
            nn.Linear(32 + 32, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        init_orthogonal(self)
        self.states_head = copy.deepcopy(shared_net)

    def forward(self, states: torch.Tensor, actions: torch.Tensor):
        states = self.states_head(states)
        actions = self.action_head(actions)
        x = self.net(torch.cat([states, actions], dim=1))
        return x

class PNet(nn.Module):
    def __init__(
        self,
        obs_shape: torch.Size,
        act_count: int,
        trained_net: StrokeNet,
    ):
        nn.Module.__init__(self)
        self.trained_net = copy.deepcopy(trained_net)

    def forward(self, x: torch.Tensor):
        x = self.trained_net(x)
        return x

img_size = IMG_SIZE
env = gym.vector.SyncVectorEnv(
    [lambda: TimeLimit(StrokeEnv(img_size), 10) for _ in range(num_envs)]
)
test_env = StrokeEnv(img_size, render_mode="human")
perfect_env = StrokeEnv(img_size)

# Initialize networks
obs_space = env.single_observation_space
assert obs_space.shape is not None
obs_shape = torch.Size(obs_space.shape)
act_space = env.single_action_space
assert isinstance(act_space, gym.spaces.Box)
trained_net = StrokeNet(img_size)
trained_net.load_state_dict(torch.load("temp/stroke_net.pt"))
shared_net = trained_net.shared
q_net_1 = QNet(obs_shape, int(act_space.shape[0]), shared_net)
q_net_1_target = copy.deepcopy(q_net_1)
q_net_1_target.to(device)
q_1_opt = torch.optim.Adam(q_net_1.parameters(), lr=q_lr)
q_net_2 = QNet(obs_shape, int(act_space.shape[0]), shared_net)
q_net_2_target = copy.deepcopy(q_net_2)
q_net_2_target.to(device)
q_2_opt = torch.optim.Adam(q_net_2.parameters(), lr=q_lr)
p_net = PNet(obs_shape, int(act_space.shape[0]), trained_net)
p_net_target = copy.deepcopy(p_net)
p_net_target.to(device)
p_opt = torch.optim.Adam(p_net.parameters(), lr=p_lr)

del shared_net

act_low = torch.from_numpy(act_space.low)
act_high = torch.from_numpy(act_space.high)

# A replay buffer stores experience collected over all sampling runs
buffer = ReplayBuffer(
    obs_shape,
    torch.Size((int(act_space.shape[0]),)),
    torch.float,
    buffer_size,
)

obs = torch.Tensor(env.reset()[0])
done = False
for step in tqdm(range(iterations), position=0):
    # Collect experience
    with torch.no_grad():
        for _ in range(train_steps):
            if step < warmup_steps:
                actions_ = (
                    torch.distributions.Uniform(act_low, act_high)
                    .sample(torch.Size([num_envs]))
                    .numpy()
                )
            else:
                eps = torch.distributions.Normal(0, q_epsilon).sample(
                    torch.Size([num_envs, act_low.shape[0]])
                )
                actions_ = torch.clamp(
                    p_net(obs) + eps,
                    act_low.repeat([num_envs, 1]),
                    act_high.repeat([num_envs, 1]),
                ).numpy()
            obs_, rewards, dones, truncs, _ = env.step(actions_)
            next_obs = torch.from_numpy(obs_)
            buffer.insert_step(
                obs,
                next_obs,
                torch.from_numpy(actions_).squeeze(0),
                list(rewards),
                list(dones | truncs),
                None,
                None,
            )
            obs = next_obs

    # Collect perfect experience
    for _ in range(16):
        perf_obs = torch.from_numpy(perfect_env.reset()[0])
        for i in range(perfect_env.num_strokes):
            mid_point, end_point = perfect_env.correct_moves[i]
            eps = torch.distributions.Normal(0, 0.04).sample(
                torch.Size([act_low.shape[0]])
            ).numpy()
            action_ = np.clip((np.array([mid_point[0], mid_point[1], end_point[0], end_point[1]]) / img_size) + eps, -1, 1)
            obs_, reward_, done_, trunc_, _ = perfect_env.step(action_)
            next_obs = torch.from_numpy(obs_)
            buffer.insert_step(
                perf_obs.unsqueeze(0).float(),
                next_obs.unsqueeze(0).float(),
                torch.from_numpy(action_).unsqueeze(0).float(),
                [reward_],
                [done_ or trunc_],
                None,
                None,
            )
            perfect_env.render()
            perf_obs = next_obs

    # Train
    if buffer.filled:
        total_q_loss, total_p_loss = train_td3(
            q_net_1,
            q_net_1_target,
            q_1_opt,
            q_net_2,
            q_net_2_target,
            q_2_opt,
            p_net,
            p_net_target,
            p_opt,
            buffer,
            device,
            train_iters,
            train_batch_size,
            discount,
            polyak,
            noise_clip,
            noise_scale,
            act_low,
            act_high,
            update_policy_every=10 if step > 300 else 999,
        )

        # Evaluate the network's performance after this training iteration.
        eval_done = False
        with torch.no_grad():
            reward_total = 0.0
            pred_reward_total = 0.0
            obs_, info = test_env.reset()
            eval_obs = torch.from_numpy(np.array(obs_)).float()
            for _ in range(eval_steps):
                steps_taken = 0
                for _ in range(max_eval_steps):
                    action = torch.clamp(
                        p_net(eval_obs.unsqueeze(0)).squeeze(1), act_low, act_high
                    ).squeeze(0)
                    pred_reward_total += q_net_1(
                        eval_obs.unsqueeze(0), action.unsqueeze(0)
                    ).item()
                    obs_, reward, eval_done, eval_trunc, _ = test_env.step(
                        action.numpy()
                    )
                    test_env.render()
                    eval_obs = torch.from_numpy(np.array(obs_)).float()
                    steps_taken += 1
                    reward_total += reward
                    if eval_done or eval_trunc:
                        obs_, info = test_env.reset()
                        eval_obs = torch.from_numpy(np.array(obs_)).float()
                        break

        wandb.log(
            {
                "avg_eval_episode_reward": reward_total / eval_steps,
                "avg_eval_episode_predicted_reward": pred_reward_total / eval_steps,
                "avg_q_loss": total_q_loss / train_iters,
                "avg_p_loss": total_p_loss / train_iters,
            }
        )

        if step % 5 == 0:
            torch.save(q_net_1.state_dict(), "temp/q_net_1.pt")
            torch.save(q_net_2.state_dict(), "temp/q_net_2.pt")
            torch.save(p_net.state_dict(), "temp/p_net.pt")
