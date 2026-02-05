# /// script
# dependencies = [
#   "fire",
#   "gymnasium",
#   "gymnasium[other]",
#   "metacontroller-pytorch>=0.0.57",
#   "torch-einops-utils @ git+https://github.com/lucidrains/torch-einops-utils.git",
#   "minigrid",
#   "tqdm",
#   "wandb",
#   "sentence-transformers"
# ]
# ///

from fire import Fire
from pathlib import Path
from tqdm import tqdm

import numpy as np
import torch
from torch import cat, tensor, stack, Tensor
from torch.optim import Adam

from einops import rearrange

from torch_einops_utils import pad_sequence, pad_sequence_and_cat, lens_to_mask

from accelerate import Accelerator

from babyai_env import (
    create_env,
    get_missions_embeddings,
    transform_to_symbolic
)
import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv
from metacontroller.metacontroller import Transformer, MetaController, z_score, extract_grpo_data
from metacontroller.transformer_with_resnet import TransformerWithResnet
from functools import partial

# Patch x_transformers so checkpoint config can unpickle (Identity was removed/moved in some versions)
try:
    import x_transformers.x_transformers as _xt_module
    if not hasattr(_xt_module, "Identity"):
        _xt_module.Identity = torch.nn.Identity
except Exception:
    pass

# research entry point

def reward_shaping_fn(
    cumulative_rewards: Tensor, # float(num_episodes,)
    all_rewards: Tensor,        # float(num_episodes, max_timesteps)
    episode_lens: Tensor,       # int(num_episodes,)
    reject_threshold_cumulative_reward_variance: float = 0.
) -> Tensor | None:
    """
    researchers can modify this function to engineer rewards
    or return None to reject the entire batch
    """

    if cumulative_rewards.var() < reject_threshold_cumulative_reward_variance:
        return None

    return cumulative_rewards

def should_reject_group_based_on_switch_betas(
    switch_betas: Tensor,
    episode_lens: Tensor
):
    return switch_betas.sum().item() == 0.

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def divisible_by(num, den):
    return (num % den) == 0


# main

def main(
    seed: int | None = None,
    npy_seedfile = None,
    env_name = 'BabyAI-BossLevel-v0',
    num_episodes = int(10e6),
    max_timesteps = 500,
    render_every_eps = 1_000,
    video_folder = './recordings',
    transformer_weights_path: str | None = None,
    meta_controller_weights_path: str | None = None,
    output_meta_controller_path = 'metacontroller_rl_trained.pt',
    use_resnet = False,
    lr = 3e-5,
    save_steps = 100,
    batch_size = 16,
    max_grad_norm = 1.0,
    use_wandb = False,
    wandb_project = 'metacontroller-babyai-rl',
    reject_threshold_cumulative_reward_variance = 0.1,
    condition_on_mission_embed = False,
    env_shared_memory = True,
    env_context = 'fork'
):

    def store_checkpoint(step: int):
        if accelerator.is_main_process:
            meta_controller_checkpoint_path_with_step = output_meta_controller_path.replace('.pt', f'_step_{step}.pt')
            unwrapped_meta_controller = accelerator.unwrap_model(meta_controller)
            unwrapped_meta_controller.save(meta_controller_checkpoint_path_with_step)
            accelerator.print(f"MetaController to {meta_controller_checkpoint_path_with_step}")

    # seed selection priority: 1) fixed seed arg, 2) numpy seedfile, 3) random

    group_seeds = None
    if not exists(seed) and exists(npy_seedfile):
        cap_seeds = 10
        group_seeds = np.load(npy_seedfile, allow_pickle=True).astype(int).tolist()
        group_seeds = group_seeds[:cap_seeds]
        
    seed_index = 0

    def get_next_seed():
        nonlocal seed_index
        if exists(seed):
            return seed
        if group_seeds is not None:
            s = group_seeds[seed_index % len(group_seeds)]
            seed_index += 1
            return s
        return torch.randint(0, 1000000, (1,)).item()

    # batch_size = number of simultaneous trajectories (the group) = number of parallel envs
    assert batch_size >= 2, "batch_size must be at least 2 for relative comparison (GRPO)"

    # accelerator

    accelerator = Accelerator(log_with = 'wandb' if use_wandb else None)

    if use_wandb:
        accelerator.init_trackers(wandb_project)

    # environment

    env_make_fn = partial(
        create_env,
        env_name,
        render_mode = 'rgb_array',
        video_folder = video_folder,
        render_every_eps = render_every_eps,
        use_symbolic = False
    )

    env = AsyncVectorEnv([env_make_fn] * batch_size, shared_memory = env_shared_memory, context = env_context)


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

    num_batch_updates = num_episodes // batch_size

    pbar = tqdm(range(num_batch_updates), desc = 'training')
    
    print("starting training")

    for gradient_step in pbar:

        all_states = []
        all_log_probs = []
        all_switch_betas = []
        all_latent_actions = []
        all_step_rewards = []
        all_episode_lens = []

        # one group of batch_size trajectories per gradient step (for GRPO relative comparison)
        group_seed = get_next_seed()
        env_seeds = [group_seed] * batch_size

        state, _ = env.reset(seed = env_seeds)

        if condition_on_mission_embed:
            missions = env.get_attr('mission')
            mission_embed = get_missions_embeddings(missions)
            mission_embed = mission_embed.to(accelerator.device)

        cache = None
        past_action_id = None

        iteration_states = []
        iteration_log_probs = []
        iteration_switch_betas = []
        iteration_latent_actions = []
        iteration_rewards = []

        dones = torch.zeros(batch_size, dtype = torch.bool)

        all_steps_dones = []

        # rollout: batch of batch_size parallel trajectories (one env per trajectory)
        for step in range(max_timesteps):
            # state['image']: (batch_size, H, W, C)
            image = state['image']
            image_tensor = torch.from_numpy(image).float().to(accelerator.device)

            if use_resnet:
                image_tensor = rearrange(image_tensor, 'b h w c -> b 1 h w c')
                image_tensor = unwrapped_model.visual_encode(image_tensor)
            else:
                image_tensor = rearrange(image_tensor, 'b h w c -> b 1 (h w c)')

            if torch.is_tensor(past_action_id):
                past_action_id = past_action_id.long()

            with torch.no_grad():
                logits, cache = unwrapped_model(
                    image_tensor,
                    past_action_id,
                    meta_controller = unwrapped_meta_controller,
                    return_cache = True,
                    return_raw_action_dist = True,
                    cache = cache,
                    condition = mission_embed if condition_on_mission_embed else None
                )

            action = unwrapped_model.action_readout.sample(logits)
            past_action_id = action

            grpo_data = extract_grpo_data(unwrapped_meta_controller, cache)

            iteration_states.append(grpo_data.state)
            iteration_log_probs.append(grpo_data.log_prob)
            iteration_switch_betas.append(grpo_data.switch_beta)
            iteration_latent_actions.append(grpo_data.action)

            next_state, reward, terminated, truncated, _ = env.step(action.cpu().numpy())
            next_state['image'] = transform_to_symbolic(next_state['image'])

            iteration_rewards.append(reward)

            all_steps_dones.append(dones.clone())
            dones |= torch.from_numpy(terminated | truncated)

            if dones.all():
                break

            state = next_state

        episode_lens = (~torch.stack(all_steps_dones)).sum(dim = 0)

        cur_states = torch.cat(iteration_states, dim = 1)
        cur_log_probs = torch.cat(iteration_log_probs, dim = 1)
        cur_switch_betas = torch.cat(iteration_switch_betas, dim = 1)
        cur_latent_actions = torch.cat(iteration_latent_actions, dim = 1)
        cur_rewards = tensor(np.stack(iteration_rewards))  # (timesteps, batch_size)

        all_states.append(cur_states)
        all_log_probs.append(cur_log_probs)
        all_switch_betas.append(cur_switch_betas)
        all_latent_actions.append(cur_latent_actions)
        all_step_rewards.append(cur_rewards)
        all_episode_lens.append(episode_lens)

        # pack group data (single rollout → one group of batch_size trajectories)

        group_states = pad_sequence_and_cat(all_states, dim = 1, dim_cat = 0)
        group_log_probs = pad_sequence_and_cat(all_log_probs, dim = 1, dim_cat = 0)
        group_switch_betas = pad_sequence_and_cat(all_switch_betas, dim = 1, dim_cat = 0)
        group_latent_actions = pad_sequence_and_cat(all_latent_actions, dim = 1, dim_cat = 0)
        
        group_step_rewards = pad_sequence_and_cat(all_step_rewards, dim = 0, dim_cat = 1)
        group_step_rewards = rearrange(group_step_rewards, 't g -> g t')

        episode_lens = cat(all_episode_lens, dim = 0)

        # mask rewards after done

        mask = lens_to_mask(episode_lens, group_step_rewards.shape[-1])
        group_step_rewards = group_step_rewards * mask

        # compute advantages via z-score (GRPO style)

        cumulative_rewards = group_step_rewards.sum(dim = -1)

        # reward shaping hook

        shaped_rewards = reward_shaping_fn(
            cumulative_rewards,
            group_step_rewards,
            episode_lens,
            reject_threshold_cumulative_reward_variance = reject_threshold_cumulative_reward_variance
        )

        if not exists(shaped_rewards):
            accelerator.print(f'group rejected - variance of {cumulative_rewards.var().item():.4f} is lower than threshold of {reject_threshold_cumulative_reward_variance}')
            continue

        grouped_shaped_rewards = rearrange(shaped_rewards, '(g n) -> g n', n = batch_size)
        grouped_advantages = z_score(grouped_shaped_rewards, dim = 1)
        group_advantages = rearrange(grouped_advantages, 'g n -> (g n)').float()
        

        if torch.any(torch.isnan(group_advantages)):
            accelerator.print(f'group rejected - advantages contained NaNs')
            continue

        # whether to reject group based on switch betas (as it determines the mask for learning)

        if should_reject_group_based_on_switch_betas(group_switch_betas, episode_lens):
            accelerator.print(f'group rejected - switch betas for the entire group does not meet criteria for learning')
            continue

        # learn on this group directly (on-policy GRPO)

        meta_controller.train()

        loss = meta_controller.policy_loss(
            group_states,
            group_log_probs,
            group_latent_actions,
            group_advantages,
            group_switch_betas == 1.,
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
            reward = f'{cumulative_rewards.mean().item():.4f}',
            seeds = f'{group_seeds}'
        )

        accelerator.log({
            'loss': loss.item(),
            'grad_norm': grad_norm.item(),
            'reward': cumulative_rewards.mean().item(),
            'reward_std': cumulative_rewards.std().item(),
        })

        accelerator.print(f'loss: {loss.item():.4f}, grad_norm: {grad_norm.item():.4f}, reward: {cumulative_rewards.mean().item():.4f}, seeds: {group_seeds}')

        if gradient_step % save_steps == 0:
            store_checkpoint(gradient_step)

    env.close()

    # save

    if exists(output_meta_controller_path):
        unwrapped_meta_controller.save(output_meta_controller_path)
        accelerator.print(f'MetaController weights saved to {output_meta_controller_path}')

if __name__ == '__main__':
    Fire(main)
