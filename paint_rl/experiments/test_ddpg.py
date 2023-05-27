"""
Experiment for checking that DDPG works.

The Deep Deterministic Policy Gradient (DDPG) algorithm is a popular offline
deep reinforcement learning algorithm for continuous spaces. It's intuitive to
understand, and it gets reliable results, though it can take longer to run.
"""
import copy
import random
from functools import reduce
from typing import Any

import envpool  # type: ignore
import numpy as np  # type: ignore
import torch
import torch.nn as nn
import wandb
from gymnasium.envs.classic_control.pendulum import PendulumEnv
from tqdm import tqdm

import paint_rl.conf
from paint_rl.algorithms.ddpg import train_ddpg
from paint_rl.algorithms.replay_buffer import ReplayBuffer
from paint_rl.utils import init_orthogonal, polyak_avg

_: Any

# Hyperparameters
num_envs = 2  # Number of environments to step through at once during sampling.
train_steps = 32  # Number of steps to step through during sampling. Total # of samples is train_steps * num_envs.
iterations = 10000  # Number of sample/train iterations.
train_iters = 1  # Number of passes over the samples collected.
train_batch_size = 128  # Minibatch size while training models.
discount = 0.99  # Discount factor applied to rewards.
q_epsilon = 0.1  # Amount of noise added to actions during training.
eval_steps = 8  # Number of eval runs to average over.
max_eval_steps = 300  # Max number of steps to take during each eval run.
q_lr = 0.001  # Learning rate of the q net.
p_lr = 0.001  # Learning rate of the p net.
polyak = 0.005  # Polyak averaging value.
warmup_steps = 100  # For the first n number of steps, we will only sample randomly.
buffer_size = 10000  # Number of elements that can be stored in the buffer.
device = torch.device("cuda")

wandb.init(
    project="tests",
    entity=paint_rl.conf.entity,
    config={
        "experiment": "ddpg",
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
    ):
        nn.Module.__init__(self)
        flat_obs_dim = reduce(lambda e1, e2: e1 * e2, obs_shape, 1)
        self.states_head = nn.Sequential(
            nn.Linear(flat_obs_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )
        self.action_head = nn.Sequential(
            nn.Linear(act_count, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )
        self.net = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        init_orthogonal(self)

    def forward(self, states: torch.Tensor, actions: torch.Tensor):
        states = self.states_head(states)
        actions = self.action_head(actions)
        x = self.net(torch.cat([states, actions], dim=1))
        return x


# The P network takes in an observation and returns the best action.
class PNet(nn.Module):
    def __init__(
        self,
        obs_shape: torch.Size,
        act_count: int,
    ):
        nn.Module.__init__(self)
        flat_obs_dim = reduce(lambda e1, e2: e1 * e2, obs_shape, 1)
        self.net = nn.Sequential(
            nn.Linear(flat_obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, act_count),
        )
        init_orthogonal(self)

    def forward(self, input: torch.Tensor):
        x = self.net(input)
        return x


env = envpool.make("Pendulum-v1", "gym", num_envs=num_envs)
test_env = PendulumEnv()

# Initialize networks
obs_space = env.observation_space
act_space = env.action_space
q_net = QNet(obs_space.shape, int(act_space.shape[0]))
q_net_target = copy.deepcopy(q_net)
q_net_target.to(device)
q_opt = torch.optim.Adam(q_net.parameters(), lr=q_lr)
p_net = PNet(obs_space.shape, int(act_space.shape[0]))
p_net_target = copy.deepcopy(p_net)
p_net_target.to(device)
p_opt = torch.optim.Adam(p_net.parameters(), lr=p_lr)

act_low = torch.from_numpy(act_space.low)
act_high = torch.from_numpy(act_space.high)

# A replay buffer stores experience collected over all sampling runs
buffer = ReplayBuffer(
    torch.Size(obs_space.shape),
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
                eps = torch.distributions.Uniform(-q_epsilon, q_epsilon).sample(
                    torch.Size([num_envs])
                )
                actions_ = (
                    torch.clamp(
                        p_net(obs).squeeze(1) + eps,
                        act_low.repeat([num_envs]),
                        act_high.repeat([num_envs]),
                    )
                    .unsqueeze(1)
                    .numpy()
                )
            obs_, rewards, dones, truncs, _ = env.step(actions_)
            next_obs = torch.from_numpy(obs_)
            buffer.insert_step(
                obs,
                next_obs,
                torch.from_numpy(actions_).squeeze(0),
                rewards,
                dones,
                None,
                None,
            )
            obs = next_obs

    # Train
    if buffer.filled:
        total_q_loss, total_p_loss = train_ddpg(
            q_net,
            q_net_target,
            q_opt,
            p_net,
            p_net_target,
            p_opt,
            buffer,
            device,
            train_iters,
            train_batch_size,
            discount,
            polyak,
        )

        # Evaluate the network's performance after this training iteration.
        eval_done = False
        with torch.no_grad():
            reward_total = 0
            pred_reward_total = 0
            obs_, info = test_env.reset()
            eval_obs = torch.from_numpy(np.array(obs_)).float()
            for _ in range(eval_steps):
                steps_taken = 0
                score = 0
                for _ in range(max_eval_steps):
                    action = p_net(eval_obs.unsqueeze(0)).squeeze(1)
                    pred_reward_total += q_net(
                        eval_obs.unsqueeze(0), action.unsqueeze(0)
                    ).item()
                    action = action.numpy()
                    obs_, reward, eval_done, eval_trunc, _ = test_env.step(action)
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
            }
        )
