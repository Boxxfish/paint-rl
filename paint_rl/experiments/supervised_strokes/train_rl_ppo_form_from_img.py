# """
# Experiment for using RL to train a pretrained stroke model with PPO.
# The strokes to copy are from a preexisting dataset.
# This generates strokes from an actual image.
# """
# import copy
# from functools import reduce
# from pathlib import Path
# import sys
# from typing import Any

# import gymnasium as gym
# import numpy as np
# import torch
# import torch.nn as nn
# import wandb
# from gymnasium.wrappers.time_limit import TimeLimit
# from gymnasium.wrappers.normalize import NormalizeReward
# from torch.distributions import Normal
# from tqdm import tqdm
# from PIL import Image, ImageOps # type: ignore

# from paint_rl.algorithms.ppo_cont import train_ppo
# from paint_rl.algorithms.rollout_buffer import RolloutBuffer
# from paint_rl.conf import entity
# from paint_rl.experiments.supervised_strokes.gen_supervised import IMG_SIZE
# from paint_rl.experiments.supervised_strokes.stroke_env import StrokeEnv
# from paint_rl.experiments.supervised_strokes.train_supervised import (
#     SharedNet,
#     StrokeNet,
# )
# from paint_rl.utils import init_orthogonal

# _: Any

# # Hyperparameters
# num_envs = 128  # Number of environments to step through at once during sampling.
# train_steps = 64  # Number of steps to step through during sampling. Total # of samples is train_steps * num_envs/
# iterations = 1000  # Number of sample/train iterations.
# train_iters = 2  # Number of passes over the samples collected.
# train_batch_size = 2048  # Minibatch size while training models.
# discount = 0.99  # Discount factor applied to rewards.
# lambda_ = 0.95  # Lambda for GAE.
# epsilon = 0.2  # Epsilon for importance sample clipping.
# max_eval_steps = 10  # Number of eval runs to average over.
# eval_steps = 1  # Max number of steps to take during each eval run.
# v_lr = 0.001  # Learning rate of the value net.
# p_lr = 0.0001  # Learning rate of the policy net.
# action_scale = 0.3  # Scale for actions.
# device = torch.device("cuda")  # Device to use during training.


# class ValueNet(nn.Module):
#     def __init__(
#         self,
#         obs_shape: torch.Size,
#     ):
#         nn.Module.__init__(self)
#         self.v_layer2 = nn.Linear(64, 64)
#         self.v_layer3 = nn.Linear(64, 1)
#         self.relu = nn.ReLU()
#         self.shared_net = nn.Sequential(
#             nn.Conv2d(5, 16, 5, stride=2, padding=2),
#             nn.ReLU(),
#             nn.Conv2d(16, 32, 5, padding=2),
#             nn.ReLU(),
#             nn.Conv2d(32, 64, 5, padding=2),
#             nn.ReLU(),
#             nn.Conv2d(64, 64, 1),
#             nn.ReLU(),
#             nn.Conv2d(64, 64, 1),
#             nn.ReLU(),
#         )
#         init_orthogonal(self)

#     def forward(self, input: torch.Tensor):
#         x = self.shared_net(input)
#         x = torch.max(torch.max(x, 3).values, 2).values
#         x = self.relu(x)
#         x = self.v_layer2(x)
#         x = self.relu(x)
#         x = self.v_layer3(x)
#         return x


# class PolicyNet(nn.Module):
#     def __init__(
#         self,
#         obs_shape: torch.Size,
#         action_count: int,
#     ):
#         nn.Module.__init__(self)
#         self.shared_net = nn.Sequential(
#             nn.Conv2d(5, 16, 5, stride=2, padding=2),
#             nn.ReLU(),
#             nn.Conv2d(16, 32, 5, padding=2),
#             nn.ReLU(),
#             nn.Conv2d(32, 64, 5, padding=2),
#             nn.ReLU(),
#             nn.Conv2d(64, 64, 1),
#             nn.ReLU(),
#             nn.Conv2d(64, 64, 1),
#             nn.ReLU(),
#         )
#         self.a_layer2 = nn.Linear(64, 64)
#         self.a_layer3 = nn.Linear(64, action_count)
#         self.relu = nn.ReLU()
#         self.sigmoid = nn.Sigmoid()
#         init_orthogonal(self)

#     def forward(self, input: torch.Tensor):
#         x = self.shared_net(input)
#         x = torch.max(torch.max(x, 3).values, 2).values
#         x = self.relu(x)
#         x = self.a_layer2(x)
#         x = self.relu(x)
#         x = self.a_layer3(x)
#         x = self.sigmoid(x)
#         return x

# img_size = IMG_SIZE

# # Testing on real image
# if len(sys.argv) > 1 and sys.argv[1] == "test":
#     img_path = sys.argv[2]
#     img = Image.open(img_path)
#     img = img.resize((img_size, img_size), resample=Image.Resampling.BILINEAR)
#     img = img.convert("RGB")
#     img_ref = np.array(img).transpose(2, 0, 1) / 255.0
#     img_ref_filter = np.array(ImageOps.grayscale(img)) / 255.0
#     eval_done = False
#     new_test_env = StrokeEnv(img_size, ref_imgs=[img_ref], stroke_imgs=[img_ref[0]], render_mode="human")
#     obs_space = new_test_env.observation_space
#     assert obs_space.shape is not None
#     obs_shape = torch.Size(obs_space.shape)
#     act_space = new_test_env.action_space
#     assert isinstance(act_space, gym.spaces.Box)
#     p_net = PolicyNet(obs_shape, int(act_space.shape[0]))
#     p_net.load_state_dict(torch.load("temp/p_net.pt"))
#     with torch.no_grad():
#         reward_total = 0.0
#         eval_obs = torch.Tensor(new_test_env.reset()[0])
#         new_test_env.ref = img_ref
#         new_test_env.ref_filter = img_ref_filter
#         new_test_env.num_strokes = 20
#         steps_taken = 0
#         while True:
#             distr = Normal(loc=p_net(eval_obs.unsqueeze(0)).squeeze(), scale=0.0001)
#             action = distr.sample().numpy()
#             obs_, reward, eval_done, eval_trunc, _ = new_test_env.step(action)
#             new_test_env.render()
#             eval_obs = torch.Tensor(obs_)
#             steps_taken += 1
#             if eval_done or eval_trunc:
#                 eval_obs = torch.Tensor(new_test_env.reset()[0])
#                 new_test_env.ref = img_ref
#                 new_test_env.ref_filter = img_ref_filter
#                 new_test_env.num_strokes = 20
#             reward_total += reward
#     input("Press any button to quit...")
#     quit()

# # Load dataset
# ds_path = Path("temp/form_dataset")
# ref_imgs = []
# stroke_imgs = []
# print("Loading dataset...")

# for dir in tqdm(ds_path.iterdir()):
#     stroke_img = Image.open(dir / "outlines.png")
#     stroke_img = stroke_img.resize((img_size, img_size))
#     stroke_imgs.append(np.array(stroke_img).transpose([2, 0, 1])[1] / 255.0)
#     ref_img = Image.open(dir / "final.png")
#     ref_img = ref_img.resize((img_size, img_size))
#     ref_img = ref_img.convert("RGB")
#     ref_imgs.append(np.array(ref_img).transpose([2, 0, 1]) / 255.0)

# env = gym.vector.SyncVectorEnv(
#     [
#         lambda: NormalizeReward(TimeLimit(StrokeEnv(img_size, ref_imgs=ref_imgs, stroke_imgs=stroke_imgs), 10))
#         for _ in range(num_envs)
#     ]
# )
# test_env = StrokeEnv(img_size, ref_imgs=ref_imgs[:20], stroke_imgs=stroke_imgs[:20], render_mode="human")
# del ref_imgs

# # If evaluating, run the sim
# if len(sys.argv) > 1 and sys.argv[1] == "eval":
#     eval_done = False
#     test_env = TimeLimit(StrokeEnv(img_size, render_mode="human"), 10)  # type: ignore
#     obs_space = test_env.observation_space
#     assert obs_space.shape is not None
#     obs_shape = torch.Size(obs_space.shape)
#     act_space = env.action_space
#     assert isinstance(act_space, gym.spaces.Box)
#     p_net = PolicyNet(obs_shape, int(act_space.shape[0]))
#     p_net.load_state_dict(torch.load("temp/p_net.pt"))
#     with torch.no_grad():
#         reward_total = 0.0
#         eval_obs = torch.Tensor(test_env.reset()[0])
#         while True:
#             steps_taken = 0
#             for _ in range(max_eval_steps):
#                 distr = Normal(loc=p_net(eval_obs.unsqueeze(0)).squeeze(), scale=0.0001)
#                 action = distr.sample().numpy()
#                 obs_, reward, eval_done, eval_trunc, _ = test_env.step(action)
#                 test_env.render()
#                 eval_obs = torch.Tensor(obs_)
#                 steps_taken += 1
#                 if eval_done or eval_trunc:
#                     eval_obs = torch.Tensor(test_env.reset()[0])
#                     break
#                 reward_total += reward

# wandb.init(
#     project="paint-rl",
#     entity=entity,
#     config={
#         "experiment": "supervised stroke with RL (PPO)",
#         "num_envs": num_envs,
#         "train_steps": train_steps,
#         "train_iters": train_iters,
#         "train_batch_size": train_batch_size,
#         "discount": discount,
#         "lambda": lambda_,
#         "epsilon": epsilon,
#         "max_eval_steps": max_eval_steps,
#         "v_lr": v_lr,
#         "p_lr": p_lr,
#     },
# )


# # Initialize policy and value networks
# obs_space = env.single_observation_space
# assert obs_space.shape is not None
# obs_shape = torch.Size(obs_space.shape)
# act_space = env.single_action_space
# assert isinstance(act_space, gym.spaces.Box)
# v_net = ValueNet(obs_shape)
# p_net = PolicyNet(obs_shape, int(act_space.shape[0]))
# v_opt = torch.optim.Adam(v_net.parameters(), lr=v_lr)
# p_opt = torch.optim.Adam(p_net.parameters(), lr=p_lr)

# # A rollout buffer stores experience collected during a sampling run
# buffer = RolloutBuffer(
#     obs_shape,
#     torch.Size((int(act_space.shape[0]),)),
#     torch.Size((int(act_space.shape[0]),)),
#     torch.float,
#     num_envs,
#     train_steps,
# )

# obs = torch.Tensor(env.reset()[0])
# done = False
# orig_action_scale = action_scale
# for step in tqdm(range(iterations), position=0):
#     action_scale = orig_action_scale * (1.0 - step / iterations)
#     # Collect experience for a number of steps and store it in the buffer
#     with torch.no_grad():
#         for _ in tqdm(range(train_steps), position=1):
#             action_probs = p_net(obs)
#             actions = Normal(loc=action_probs, scale=action_scale).sample().numpy()
#             obs_, rewards, dones, truncs, _ = env.step(actions)
#             buffer.insert_step(
#                 obs,
#                 torch.from_numpy(actions),
#                 action_probs,
#                 list(rewards),
#                 list(dones),
#                 list(truncs),
#             )
#             obs = torch.from_numpy(obs_)
#             if done:
#                 obs = torch.Tensor(env.reset()[0])
#                 done = False
#         buffer.insert_final_step(obs)

#     # Train
#     total_p_loss, total_v_loss = train_ppo(
#         p_net,
#         v_net,
#         p_opt,
#         v_opt,
#         buffer,
#         device,
#         train_iters,
#         train_batch_size,
#         discount,
#         lambda_,
#         epsilon,
#         action_scale,
#     )
#     buffer.clear()

#     # Evaluate the network's performance after this training iteration.
#     with torch.no_grad():
#         # Visualize
#         reward_total = 0.0
#         eval_obs = torch.Tensor(test_env.reset()[0])
#         for _ in range(eval_steps):
#             steps_taken = 0
#             for _ in range(max_eval_steps):
#                 # Action scale is very small since it should learn a deterministic policy
#                 distr = Normal(loc=p_net(eval_obs.unsqueeze(0)).squeeze(), scale=0.0001)
#                 action = distr.sample().numpy()
#                 obs_, reward, eval_done, eval_trunc, _ = test_env.step(action)
#                 test_env.render()
#                 eval_obs = torch.Tensor(obs_)
#                 steps_taken += 1
#                 if eval_done or eval_trunc:
#                     eval_obs = torch.Tensor(test_env.reset()[0])
#                     break
#                 reward_total += reward

#     wandb.log(
#         {
#             "avg_eval_episode_reward": reward_total / eval_steps,
#             "avg_v_loss": total_v_loss / train_iters,
#             "avg_p_loss": total_p_loss / train_iters,
#             "action_scale": action_scale,
#         }
#     )

#     if step % 5 == 0:
#         torch.save(p_net.state_dict(), "temp/p_net.pt")
