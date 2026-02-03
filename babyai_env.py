import torch
from functools import lru_cache
from pathlib import Path
from shutil import rmtree

import gymnasium as gym
import minigrid
from minigrid.wrappers import FullyObsWrapper, SymbolicObsWrapper

from sentence_transformers import SentenceTransformer

# sbert

_sbert = None

def get_sbert(local_files_only: bool = False, device: str = "cpu"):
    global _sbert
    if _sbert is None:
        _sbert = SentenceTransformer('all-MiniLM-L6-v2', local_files_only=local_files_only, device=device)
    return _sbert

@lru_cache(maxsize = 1000)
def get_mission_embedding(mission: str, sbert: object = None):
    if not sbert: sbert = get_sbert()
    embedding = sbert.encode(mission, convert_to_tensor = True)
    return embedding

_mission_cache = {}

def get_missions_embeddings(missions: list[str], sbert: object = None, batch_size: int = 64):
    if not sbert: sbert = get_sbert()
    
    # identify uncached missions
    
    unique_missions = list(set(missions))
    uncached_missions = [m for m in unique_missions if m not in _mission_cache]
    
    if len(uncached_missions) > 0:
        embeddings = sbert.encode(uncached_missions, convert_to_tensor=True, batch_size=batch_size, show_progress_bar=True)
        
        for mission, embedding in zip(uncached_missions, embeddings):
            _mission_cache[mission] = embedding

    # return stacked embeddings

    return torch.stack([_mission_cache[m] for m in missions])

# functions

def divisible_by(num, den):
    return (num % den) == 0

# env creation

def create_env(
    env_id,
    render_mode = 'rgb_array',
    video_folder = None,
    render_every_eps = 1000
):
    # register minigrid environments if needed
    minigrid.register_minigrid_envs()

    # environment
    env = gym.make(env_id, render_mode = render_mode)
    env = FullyObsWrapper(env)
    env = SymbolicObsWrapper(env)

    if video_folder is not None:
        video_folder = Path(video_folder)
        rmtree(video_folder, ignore_errors = True)

        env = gym.wrappers.RecordVideo(
            env = env,
            video_folder = str(video_folder),
            name_prefix = 'babyai',
            episode_trigger = lambda eps_num: divisible_by(eps_num, render_every_eps),
            disable_logger = True
        )

    return env
