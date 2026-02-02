# /// script
# dependencies = [
#   "fire",
#   "gymnasium",
#   "gymnasium[other]",
#   "metacontroller-pytorch",
#   "minigrid",
#   "tqdm",
#   "wandb"
# ]
# ///

from fire import Fire
from pathlib import Path
from tqdm import tqdm

import numpy as np
import torch
from torch import cat, tensor, stack, zeros
from torch.optim import Adam
from torch.nn.utils.rnn import pad_sequence

from einops import rearrange

from accelerate import Accelerator

from babyai_env import create_env
from metacontroller.metacontroller import Transformer, MetaController, z_score, extract_grpo_data
from metacontroller.transformer_with_resnet import TransformerWithResnet

# research entry point

def reward_shaping_fn(
    cumulative_rewards: torch.Tensor,
    all_rewards: torch.Tensor,
    episode_lens: torch.Tensor
) -> torch.Tensor | None:
    """
    researchers can modify this function to engineer rewards
    or return None to reject the entire batch
    
    cumulative_rewards: (num_episodes,)
    all_rewards: (num_episodes, max_timesteps)
    episode_lens: (num_episodes,)
    """
    return cumulative_rewards

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# main

def main(
    npy_skipfile = None,
    env_name = 'BabyAI-BossLevel-v0',
    num_episodes = int(10e6),
    max_timesteps = 500,
    render_every_eps = 1_000,
    video_folder = None,
    seed: int | None = None,
    transformer_weights_path: str | None = None,
    meta_controller_weights_path: str | None = None,
    output_meta_controller_path = 'metacontroller_rl_trained.pt',
    use_resnet = False,
    lr = 3e-4,
    save_steps = 100,
    num_groups = 16,
    max_grad_norm = 1.0,
    use_wandb = False,
    wandb_project = 'metacontroller-babyai-rl'
):

    def store_checkpoint(step: int):
        if accelerator.is_main_process:
            meta_controller_checkpoint_path_with_step = output_meta_controller_path.replace('.pt', f'_step_{step}.pt')
            unwrapped_meta_controller = accelerator.unwrap_model(meta_controller)
            unwrapped_meta_controller.save(meta_controller_checkpoint_path_with_step)
            accelerator.print(f"MetaController to {meta_controller_checkpoint_path_with_step}")

    # seeds to skip

    skip_seeds = set(np.load(npy_skipfile)) if exists(npy_skipfile) else set()

    def random_seed_not_in_skip():
        while True:
            s = torch.randint(0, 1000000, (1,)).item()
            if s not in skip_seeds:
                return s

    # accelerator

    accelerator = Accelerator(log_with = 'wandb' if use_wandb else None)

    if use_wandb:
        accelerator.init_trackers(wandb_project)

    # environment

    env = create_env(
        env_name,
        render_mode = 'rgb_array',
        video_folder = video_folder,
        render_every_eps = render_every_eps
    )

    # load models

    model = None
    if exists(transformer_weights_path):
        weights_path = Path(transformer_weights_path)
        assert weights_path.exists(), f"transformer weights not found at {weights_path}"
        
        transformer_klass = TransformerWithResnet if use_resnet else Transformer
        model = transformer_klass.init_and_load(str(weights_path), strict = False)
        model.eval()

    meta_controller = None
    if exists(meta_controller_weights_path):
        weights_path = Path(meta_controller_weights_path)
        assert weights_path.exists(), f"meta controller weights not found at {weights_path}"
        meta_controller = MetaController.init_and_load(str(weights_path), strict = False)
        meta_controller.eval()

    meta_controller = default(meta_controller, getattr(model, 'meta_controller', None))
    assert exists(meta_controller), "MetaController must be present for reinforcement learning"

    # optimizer

    optim = Adam(meta_controller.internal_rl_parameters(), lr = lr)

    # prepare

    model, meta_controller, optim = accelerator.prepare(model, meta_controller, optim)

    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_meta_controller = accelerator.unwrap_model(meta_controller)

    # rollouts

    num_batch_updates = num_episodes // num_groups

    pbar = tqdm(range(num_batch_updates), desc = 'training')

    for gradient_step in pbar:

        all_states = []
        all_log_probs = []
        all_switch_betas = []
        all_latent_actions = []
        all_cumulative_rewards = []
        all_step_rewards = []
        all_episode_lens = []

        # every group has a shared seed (for GRPO relative comparison)

        group_seed = default(seed, random_seed_not_in_skip())

        for _ in range(num_groups):

            state, *_ = env.reset(seed = group_seed)

            cache = None
            past_action_id = None

            states = []
            log_probs = []
            switch_betas = []
            latent_actions = []

            total_reward = 0.
            step_rewards = []
            episode_len = max_timesteps

            # one episode rollout

            for step in range(max_timesteps):

                image = state['image']
                image_tensor = torch.from_numpy(image).float().to(accelerator.device)

                if use_resnet:
                    image_tensor = rearrange(image_tensor, 'h w c -> 1 1 h w c')
                    image_tensor = unwrapped_model.visual_encode(image_tensor)
                else:
                    image_tensor = rearrange(image_tensor, 'h w c -> 1 1 (h w c)')

                if torch.is_tensor(past_action_id):
                    past_action_id = past_action_id.long()

                with torch.no_grad():
                    logits, cache = unwrapped_model(
                        image_tensor,
                        past_action_id,
                        meta_controller = unwrapped_meta_controller,
                        return_cache = True,
                        return_raw_action_dist = True,
                        cache = cache
                    )

                action = unwrapped_model.action_readout.sample(logits)
                past_action_id = action
                action = action.squeeze()

                # GRPO collection

                grpo_data = extract_grpo_data(unwrapped_meta_controller, cache)

                states.append(grpo_data.state)
                log_probs.append(grpo_data.log_prob)
                switch_betas.append(grpo_data.switch_beta)
                latent_actions.append(grpo_data.action)

                next_state, reward, terminated, truncated, *_ = env.step(action.cpu().numpy())

                total_reward += reward
                step_rewards.append(reward)
                done = terminated or truncated

                if done:
                    episode_len = step + 1
                    break

                state = next_state

            # store episode data (concatenate timesteps)
            # each has shape (1, timesteps, ...)

            all_states.append(cat(states, dim=1).squeeze(0))           # (timesteps, state_dim)
            all_log_probs.append(cat(log_probs, dim=1).squeeze(0))     # (timesteps,)
            all_switch_betas.append(cat(switch_betas, dim=1).squeeze(0))  # (timesteps,)
            all_latent_actions.append(cat(latent_actions, dim=1).squeeze(0))  # (timesteps,)

            all_cumulative_rewards.append(tensor(total_reward))
            all_step_rewards.append(tensor(step_rewards))
            all_episode_lens.append(episode_len)

        # compute advantages via z-score (GRPO style)

        cumulative_rewards = stack(all_cumulative_rewards)
        episode_lens = tensor(all_episode_lens, device=accelerator.device)

        # pad step rewards for reward shaping hook

        max_len = max(all_episode_lens)
        padded_step_rewards = zeros(num_groups, max_len)

        for i, (rewards, length) in enumerate(zip(all_step_rewards, all_episode_lens)):
            padded_step_rewards[i, :length] = rewards

        # reward shaping hook

        shaped_rewards = reward_shaping_fn(cumulative_rewards, padded_step_rewards, episode_lens)

        if not exists(shaped_rewards):
            continue

        # skip if no variance in rewards (z-score would be all zeros)
        if shaped_rewards.std() < 1e-8:
            pbar.set_postfix(
                loss = 'skip (no variance)',
                reward = f'{cumulative_rewards.mean().item():.4f}'
            )
            continue

        group_advantages = z_score(shaped_rewards).to(accelerator.device)  # (num_groups,)

        # pad episodes to same length for batching

        padded_states = pad_sequence(all_states, batch_first=True)           # (num_groups, max_len, state_dim)
        padded_log_probs = pad_sequence(all_log_probs, batch_first=True)     # (num_groups, max_len)
        padded_switch_betas = pad_sequence(all_switch_betas, batch_first=True)  # (num_groups, max_len)
        padded_latent_actions = pad_sequence(all_latent_actions, batch_first=True)  # (num_groups, max_len)

        # learn on this group directly (on-policy GRPO)

        meta_controller.train()

        loss = meta_controller.policy_loss(
            padded_states,
            padded_log_probs,
            padded_latent_actions,
            group_advantages,
            padded_switch_betas == 1.,
            episode_lens = episode_lens
        )

        accelerator.backward(loss)

        grad_norm = accelerator.clip_grad_norm_(meta_controller.parameters(), max_grad_norm)

        optim.step()
        optim.zero_grad()

        meta_controller.eval()

        pbar.set_postfix(
            loss = f'{loss.item():.4f}',
            grad_norm = f'{grad_norm.item():.4f}',
            reward = f'{cumulative_rewards.mean().item():.4f}'
        )

        accelerator.log({
            'loss': loss.item(),
            'grad_norm': grad_norm.item(),
            'reward': cumulative_rewards.mean().item(),
            'reward_std': cumulative_rewards.std().item(),
        })

        accelerator.print(f'loss: {loss.item():.4f}, grad_norm: {grad_norm.item():.4f}, reward: {cumulative_rewards.mean().item():.4f}')

        if gradient_step % save_steps == 0:
            store_checkpoint(gradient_step)

    env.close()

    # save

    if exists(output_meta_controller_path):
        unwrapped_meta_controller.save(output_meta_controller_path)
        accelerator.print(f'MetaController weights saved to {output_meta_controller_path}')

if __name__ == '__main__':
    Fire(main)
