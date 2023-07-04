import copy
from typing import Tuple

import torch
from torch import nn
from torch.distributions import Normal, Categorical
from tqdm import tqdm

from .rollout_buffer import ActionRolloutBuffer, RolloutBuffer


def train_ppo(
    p_net: nn.Module,
    v_net: nn.Module,
    p_opt: torch.optim.Optimizer,
    v_opt: torch.optim.Optimizer,
    buffer: Tuple[RolloutBuffer, ActionRolloutBuffer],
    device: torch.device,
    train_iters: int,
    train_batch_size: int,
    discount: float,
    lambda_: float,
    epsilon: float,
    act_scale: float,
    gradient_steps: int = 1,
) -> Tuple[float, float]:
    """
    Performs the PPO training loop. Returns a tuple of total policy loss and
    total value loss.
    This accepts a continuous rollout buffer and a discrete one, in that order.

    Args:
        gradient_steps: Number of batches to step through before before
        adjusting weights.
        use_masks: If True, masks are passed to the model.
    """
    p_net.train()
    v_net_frozen = copy.deepcopy(v_net)
    v_net.train()
    if device.type != "cpu":
        p_net.to(device)
        v_net.to(device)

    total_v_loss = 0.0
    total_p_loss = 0.0

    p_opt.zero_grad()
    v_opt.zero_grad()

    for _ in tqdm(range(train_iters), position=1):
        batches = buffer[0].samples_merged(buffer[1], train_batch_size, discount, lambda_, v_net_frozen)
        for (
            i,
            (prev_states, actions, action_probs, returns, advantages, _),
        ) in enumerate(batches):
            # Move batch to device if applicable
            prev_states = prev_states.to(device=device)
            returns = returns.to(device=device)
            advantages = advantages.to(device=device)

            actions_cont, actions_disc = [action.to(device=device) for action in actions]
            action_probs_cont, action_probs_disc = [action_prob.to(device=device) for action_prob in action_probs]

            # Train policy network
            with torch.no_grad():
                old_act_probs_cont = torch.sum(Normal(loc=action_probs_cont, scale=act_scale).log_prob(
                    actions_cont.squeeze()
                ), 1, keepdim=False)
                old_act_probs_disc = Categorical(logits=action_probs_disc).log_prob(
                    actions_disc.squeeze()
                )
                old_act_probs = old_act_probs_cont + old_act_probs_disc
            new_log_probs_cont, new_log_probs_disc = p_net(prev_states)
            new_act_probs_cont = torch.sum(Normal(loc=new_log_probs_cont, scale=act_scale).log_prob(
                actions_cont.squeeze()
            ), 1, keepdim=False)
            new_act_probs_disc = Categorical(logits=new_log_probs_disc).log_prob(
                actions_disc.squeeze()
            )
            new_act_probs = new_act_probs_cont + new_act_probs_disc
            term1 = (new_act_probs - old_act_probs).exp() * advantages
            term2 = (1.0 + epsilon * advantages.sign()) * advantages
            p_loss = -term1.min(term2).mean() / gradient_steps
            p_loss.backward()
            total_p_loss += p_loss.item()

            # Train value network
            diff = v_net(prev_states) - returns
            v_loss = (diff * diff).mean() / gradient_steps
            v_loss.backward()
            total_v_loss += v_loss.item()

        if (i + 1) % gradient_steps == 0:
            p_opt.step()
            v_opt.step()
            p_opt.zero_grad()
            v_opt.zero_grad()

    if device.type != "cpu":
        p_net.cpu()
        v_net.cpu()
    p_net.eval()
    v_net.eval()

    return (total_p_loss, total_v_loss)
