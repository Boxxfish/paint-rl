"""
A rollout buffer for use with on-policy algorithms. Unlike a replay buffer,
rollouts only store experience collected under a single policy.
"""
from typing import List, Optional, Tuple
from typing_extensions import Protocol

import torch
from torch import nn


class MergeableBuffer(Protocol):
    def get_actions(self) -> torch.Tensor:
        ...

    def get_action_probs(self) -> torch.Tensor:
        ...

    def get_masks(self) -> torch.Tensor:
        ...


class ActionRolloutBuffer(MergeableBuffer):
    """
    Stores just the actions and action probabilties.
    Requires a `RolloutBuffer` to merge with.
    """

    def __init__(
        self,
        action_shape: torch.Size,
        action_probs_shape: torch.Size,
        action_dtype: torch.dtype,
        num_envs: int,
        num_steps: int,
    ):
        k = torch.float
        action_shape = torch.Size([num_steps, num_envs] + list(action_shape))
        action_probs_shape = torch.Size(
            [num_steps, num_envs] + list(action_probs_shape)
        )
        self.num_envs = num_envs
        self.num_steps = num_steps
        self.next = 0
        d = torch.device("cpu")
        self.actions = torch.zeros(
            action_shape, dtype=action_dtype, device=d, requires_grad=False
        )
        self.action_probs = torch.zeros(
            action_probs_shape, dtype=k, device=d, requires_grad=False
        )
        self.masks = torch.zeros(
            action_probs_shape, dtype=torch.int, device=d, requires_grad=False
        )

    def insert_step(
        self,
        actions: torch.Tensor,
        action_probs: torch.Tensor,
        masks: Optional[List[int]] = None,
    ):
        """
        Inserts a transition from each environment into the buffer. Make sure
        more data than steps aren't inserted. Insert the state that was observed
        PRIOR to performing the action.
        """
        d = torch.device("cpu")
        with torch.no_grad():
            self.actions[self.next].copy_(actions)
            self.action_probs[self.next].copy_(action_probs)
            if masks:
                self.masks[self.next].copy_(
                    torch.tensor(masks, dtype=torch.int, device=d)
                )

        self.next += 1

    def get_actions(self) -> torch.Tensor:
        return self.actions

    def get_action_probs(self) -> torch.Tensor:
        return self.action_probs

    def get_masks(self) -> torch.Tensor:
        return self.masks
    
    def clear(self):
        """
        Clears the buffer.
        """
        self.next = 0


class RolloutBuffer(MergeableBuffer):
    """
    Stores transitions and generates mini batches from the latest policy. Also
    computes advantage estimates.
    """

    def __init__(
        self,
        state_shape: torch.Size,
        action_shape: torch.Size,
        action_probs_shape: torch.Size,
        action_dtype: torch.dtype,
        num_envs: int,
        num_steps: int,
    ):
        k = torch.float
        state_shape = torch.Size([num_steps + 1, num_envs] + list(state_shape))
        action_shape = torch.Size([num_steps, num_envs] + list(action_shape))
        action_probs_shape = torch.Size(
            [num_steps, num_envs] + list(action_probs_shape)
        )
        self.num_envs = num_envs
        self.num_steps = num_steps
        self.next = 0
        d = torch.device("cpu")
        self.states = torch.zeros(state_shape, dtype=k, device=d, requires_grad=False)
        self.actions = torch.zeros(
            action_shape, dtype=action_dtype, device=d, requires_grad=False
        )
        self.action_probs = torch.zeros(
            action_probs_shape, dtype=k, device=d, requires_grad=False
        )
        self.rewards = torch.zeros(
            [num_steps, num_envs], dtype=k, device=d, requires_grad=False
        )
        # Technically this is the "terminated" flag
        self.dones = torch.zeros(
            [num_steps, num_envs], dtype=k, device=d, requires_grad=False
        )
        self.truncs = torch.zeros(
            [num_steps, num_envs], dtype=k, device=d, requires_grad=False
        )
        self.masks = torch.zeros(
            action_probs_shape, dtype=torch.int, device=d, requires_grad=False
        )

    def insert_step(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        action_probs: torch.Tensor,
        rewards: List[float],
        dones: List[bool],
        truncs: List[bool],
        masks: Optional[List[int]] = None,
    ):
        """
        Inserts a transition from each environment into the buffer. Make sure
        more data than steps aren't inserted. Insert the state that was observed
        PRIOR to performing the action. The final state returned will be
        inserted using `insert_final_step`.
        """
        d = torch.device("cpu")
        with torch.no_grad():
            self.states[self.next].copy_(states)
            self.actions[self.next].copy_(actions)
            self.action_probs[self.next].copy_(action_probs)
            self.rewards[self.next].copy_(
                torch.tensor(rewards, dtype=torch.float, device=d)
            )
            self.dones[self.next].copy_(
                torch.tensor(dones, dtype=torch.float, device=d)
            )
            self.truncs[self.next].copy_(
                torch.tensor(truncs, dtype=torch.float, device=d)
            )
            if masks:
                self.masks[self.next].copy_(
                    torch.tensor(masks, dtype=torch.int, device=d)
                )

        self.next += 1

    def insert_final_step(self, states: torch.Tensor):
        """
        Inserts the final observation observed.
        """
        with torch.no_grad():
            self.states[self.next].copy_(states)

    def samples(
        self, batch_size: int, discount: float, lambda_: float, v_net: nn.Module
    ) -> list[
        Tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ]
    ]:
        """
        Generates minibatches of experience, incorporating advantage estimates.
        Returns states, actions, action probabilities, returns, advantages, and
        dones.
        """
        with torch.no_grad():
            d = torch.device("cpu")
            returns = torch.zeros(
                [self.num_steps, self.num_envs], dtype=torch.float, device=d
            )
            advantages = torch.zeros(
                [self.num_steps, self.num_envs], dtype=torch.float, device=d
            )
            step_returns: torch.Tensor = v_net(self.states[self.next]).squeeze()

            # Calculate advantage estimates and rewards to go
            state_values = step_returns.clone()
            step_advantages = torch.zeros([self.num_envs], dtype=torch.float, device=d)
            for i in reversed(range(self.num_steps)):
                prev_states = self.states[i]
                rewards = self.rewards[i]
                inv_dones = 1.0 - self.dones[i]
                inv_truncs = 1.0 - self.truncs[i]
                prev_state_values: torch.Tensor = v_net(prev_states).squeeze()
                # Delta is the difference between the 1 step bootstrap (reward +
                # value prediction of next state) and the value prediction of
                # the current state
                delta = (
                    rewards + discount * inv_dones * state_values - prev_state_values
                )
                state_values = prev_state_values
                step_advantages = (
                    delta
                    + discount * lambda_ * inv_dones * step_advantages * inv_truncs
                )
                step_returns = state_values + step_advantages
                advantages[i] = step_advantages
                returns[i] = step_returns

            # Permute transitions to decorrelate them
            exp_count = self.num_envs * self.num_steps
            indices = torch.randperm(exp_count, dtype=torch.int, device=d)
            rand_prev_states = self.states.flatten(0, 1).index_select(0, indices)
            rand_actions = self.actions.flatten(0, 1).index_select(0, indices)
            rand_action_probs = self.action_probs.flatten(0, 1).index_select(0, indices)
            rand_masks = self.masks.flatten(0, 1).index_select(0, indices)
            rand_returns = returns.flatten(0, 1).index_select(0, indices)
            rand_advantages = advantages.flatten(0, 1).index_select(0, indices)
            batch_count = exp_count // batch_size
            batches = []
            for i in range(batch_count):
                start = i * batch_size
                end = (i + 1) * batch_size
                batches.append(
                    (
                        rand_prev_states[start:end].reshape(
                            [batch_size] + list(self.states.shape)[2:]
                        ),
                        rand_actions[start:end].reshape(
                            [batch_size] + list(self.actions.shape)[2:]
                        ),
                        rand_action_probs[start:end].reshape(
                            [batch_size] + list(self.action_probs.shape)[2:]
                        ),
                        rand_returns[start:end].reshape([batch_size, 1]),
                        rand_advantages[start:end].reshape([batch_size, 1]),
                        rand_masks[start:end].reshape(
                            [batch_size] + list(self.action_probs.shape)[2:]
                        ),
                    )
                )
            return batches

    def samples_merged(
        self,
        other: MergeableBuffer,
        batch_size: int,
        discount: float,
        lambda_: float,
        v_net: nn.Module,
    ) -> List[
        Tuple[
            torch.Tensor,
            List[torch.Tensor],
            List[torch.Tensor],
            torch.Tensor,
            torch.Tensor,
            List[torch.Tensor],
        ]
    ]:
        """
        Similar to the `samples` method, except the actions and probabilities from another rollout buffer will be returned as well.
        This is useful for environments with multiple action types.
        """
        with torch.no_grad():
            d = torch.device("cpu")
            returns = torch.zeros(
                [self.num_steps, self.num_envs], dtype=torch.float, device=d
            )
            advantages = torch.zeros(
                [self.num_steps, self.num_envs], dtype=torch.float, device=d
            )
            step_returns: torch.Tensor = v_net(self.states[self.next]).squeeze()

            # Calculate advantage estimates and rewards to go
            state_values = step_returns.clone()
            step_advantages = torch.zeros([self.num_envs], dtype=torch.float, device=d)
            for i in reversed(range(self.num_steps)):
                prev_states = self.states[i]
                rewards = self.rewards[i]
                inv_dones = 1.0 - self.dones[i]
                inv_truncs = 1.0 - self.truncs[i]
                prev_state_values: torch.Tensor = v_net(prev_states).squeeze()
                # Delta is the difference between the 1 step bootstrap (reward +
                # value prediction of next state) and the value prediction of
                # the current state
                delta = (
                    rewards + discount * inv_dones * state_values - prev_state_values
                )
                state_values = prev_state_values
                step_advantages = (
                    delta
                    + discount * lambda_ * inv_dones * step_advantages * inv_truncs
                )
                step_returns = state_values + step_advantages
                advantages[i] = step_advantages
                returns[i] = step_returns

            # Permute transitions to decorrelate them
            exp_count = self.num_envs * self.num_steps
            indices = torch.randperm(exp_count, dtype=torch.int, device=d)
            rand_prev_states = self.states.flatten(0, 1).index_select(0, indices)
            rand_returns = returns.flatten(0, 1).index_select(0, indices)
            rand_advantages = advantages.flatten(0, 1).index_select(0, indices)

            # Merge actions
            sources = (self, other)
            rand_actions = [
                source.get_actions().flatten(0, 1).index_select(0, indices)
                for source in sources
            ]
            rand_action_probs = [
                source.get_action_probs().flatten(0, 1).index_select(0, indices)
                for source in sources
            ]
            rand_masks = [
                source.get_masks().flatten(0, 1).index_select(0, indices)
                for source in sources
            ]

            batch_count = exp_count // batch_size
            batches = []
            for i in range(batch_count):
                start = i * batch_size
                end = (i + 1) * batch_size
                batches.append(
                    (
                        rand_prev_states[start:end].reshape(
                            [batch_size] + list(self.states.shape)[2:]
                        ),
                        [
                            actions[start:end].reshape(
                                [batch_size] + list(source.get_actions().shape)[2:]
                            )
                            for actions, source in zip(rand_actions, sources)
                        ],
                        [
                            action_probs[start:end].reshape(
                                [batch_size] + list(source.get_action_probs().shape)[2:]
                            )
                            for action_probs, source in zip(rand_action_probs, sources)
                        ],
                        rand_returns[start:end].reshape([batch_size, 1]),
                        rand_advantages[start:end].reshape([batch_size, 1]),
                        [
                            masks[start:end].reshape(
                                [batch_size] + list(source.get_action_probs().shape)[2:]
                            )
                            for masks, source in zip(rand_masks, sources)
                        ],
                    )
                )
            return batches

    def clear(self):
        """
        Clears the buffer.
        """
        self.next = 0

    def get_actions(self) -> torch.Tensor:
        return self.actions

    def get_action_probs(self) -> torch.Tensor:
        return self.action_probs

    def get_masks(self) -> torch.Tensor:
        return self.masks
