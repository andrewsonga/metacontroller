# /// script
# dependencies = [
#   "fire",
#   "gymnasium",
#   "gymnasium[other]",
#   "memmap-replay-buffer>=0.0.12",
#   "metacontroller-pytorch",
#   "minigrid",
#   "tqdm"
# ]
# ///

from fire import Fire
from tqdm import tqdm
from shutil import rmtree
from pathlib import Path

import torch
from einops import rearrange

import gymnasium as gym
import minigrid
from minigrid.wrappers import FullyObsWrapper, SymbolicObsWrapper

from memmap_replay_buffer import ReplayBuffer
from metacontroller.metacontroller import Transformer

# functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def divisible_by(num, den):
    return (num % den) == 0

# main

def main(
    env_name = 'BabyAI-BossLevel-v0',
    num_episodes = int(10e6),
    max_timesteps = 500,
    buffer_size = 5_000,
    render_every_eps = 1_000,
    video_folder = './recordings',
    seed = None,
    weights_path = None
):

    # environment

    env = gym.make(env_name, render_mode = 'rgb_array')
    env = FullyObsWrapper(env.unwrapped)
    env = SymbolicObsWrapper(env.unwrapped)

    rmtree(video_folder, ignore_errors = True)

    env = gym.wrappers.RecordVideo(
        env = env,
        video_folder = video_folder,
        name_prefix = 'babyai',
        episode_trigger = lambda eps_num: divisible_by(eps_num, render_every_eps),
        disable_logger = True
    )

    # maybe load model

    model = None
    if exists(weights_path):
        weights_path = Path(weights_path)
        assert weights_path.exists(), f"weights not found at {weights_path}"
        model = Transformer.init_and_load(str(weights_path), strict = False)
        model.eval()

    # replay

    replay_buffer = ReplayBuffer(
        './replay-data',
        max_episodes = buffer_size,
        max_timesteps = max_timesteps + 1,
        fields = dict(
            action = 'int',
            state_image = ('float', (7, 7, 3)),
            state_direction = 'int'
        ),
        overwrite = True,
        circular = True
    )

    # rollouts

    for _ in tqdm(range(num_episodes)):

        state, *_ = env.reset(seed = seed)

        cache = None
        past_action_id = None

        for _ in range(max_timesteps):

            if exists(model):
                # preprocess state
                # assume state is a dict with 'image'
                image = state['image']
                image_tensor = torch.from_numpy(image).float()
                image_tensor = rearrange(image_tensor, 'h w c -> 1 1 (h w c)')

                if exists(past_action_id) and torch.is_tensor(past_action_id):
                    past_action_id = past_action_id.long()

                with torch.no_grad():
                    logits, cache = model(
                        image_tensor,
                        past_action_id,
                        return_cache = True,
                        return_raw_action_dist = True,
                        cache = cache
                    )

                action = model.action_readout.sample(logits)
                past_action_id = action
                action = action.squeeze()
            else:
                action = torch.randint(0, 7, ())

            next_state, reward, terminated, truncated, *_ = env.step(action.cpu().numpy())

            done = terminated or truncated

            if done:
                break

            state = next_state

    env.close()

if __name__ == '__main__':
    Fire(main)
