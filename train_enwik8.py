# /// script
# dependencies = [
#   "accelerate",
#   "fire",
#   "torch",
#   "einops",
#   "tqdm",
#   "numpy"
# ]
# ///

import os
import math
import gzip
import random
import tqdm
import numpy as np
from pathlib import Path

import torch
from torch import Tensor
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from accelerate import Accelerator
from einops import rearrange

from metacontroller import MetaController, Transformer

# constants

NUM_BATCHES = int(1e5)
BATCH_SIZE = 4
GRAD_ACCUM_EVERY = 4
LEARNING_RATE = 1e-4
VALIDATE_EVERY = 100
PRIME_LENGTH = 128
GENERATE_EVERY = 250
GENERATE_LENGTH = 512
SEQ_LEN = 128

# helpers

def exists(v):
    return v is not None

def cycle(loader):
    while True:
        for data in loader:
            yield data

def divisible_by(num, den):
    return (num % den) == 0

def decode_token(token):
    return str(chr(max(32, token)))

def decode_tokens(tokens):
    return "".join(list(map(decode_token, tokens)))

# sampling

@torch.no_grad()
def sample(
    model,
    prompt: Tensor,
    seq_len: int,
    temperature = 1.,
):
    model.eval()

    prompt_seq_len = prompt.shape[-1]
    sample_num_times = max(0, seq_len - prompt_seq_len)

    state = prompt
    action = prompt[:, 1:]

    (action_dist, _), cache = model(
        state = state,
        actions = action,
        return_cache = True,
        return_state_action_cache = True,
        force_behavior_cloning = True
    )

    next_action = next_state = model.action_readout.sample(action_dist[:, -1:], temperature = temperature)

    state = torch.cat((state, next_state), dim = -1)

    for _ in range(sample_num_times):

        (action_dist, _), next_cache = model(
            state = state[:, -1:],
            actions = state[:, -1:],
            cache = cache,
            return_state_action_cache = True,
            force_behavior_cloning = True
        )

        next_state = model.action_readout.sample(action_dist[:, -1:], temperature = temperature)

        state = torch.cat((state, next_state), dim = -1)

        cache = next_cache

    return state[:, prompt_seq_len:]

# accelerator

accelerator = Accelerator()

# dataset

data_path = Path("./data/enwik8.gz")
if not data_path.exists():
    data_path = Path("/Users/philwang/dl/metacontroller/data/enwik8.gz")

with gzip.open(str(data_path)) as file:
    data = np.frombuffer(file.read(int(95e6)), dtype=np.uint8).copy()
    np_train, np_valid = np.split(data, [int(90e6)])
    data_train, data_val = torch.from_numpy(np_train), torch.from_numpy(np_valid)

class TextSamplerDataset(Dataset):
    def __init__(self, data, seq_len):
        super().__init__()
        self.data = data
        self.seq_len = seq_len

    def __len__(self):
        return self.data.size(0) // self.seq_len

    def __getitem__(self, index):
        rand_start = torch.randint(0, self.data.size(0) - self.seq_len - 1, (1,))
        full_seq = self.data[rand_start : rand_start + self.seq_len + 1].long()
        return full_seq

train_dataset = TextSamplerDataset(data_train, SEQ_LEN)
val_dataset = TextSamplerDataset(data_val, SEQ_LEN)
train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True)
val_loader = DataLoader(val_dataset, batch_size = BATCH_SIZE, shuffle = False)

# model

meta_controller = MetaController(
    dim_model = 512,
    dim_meta_controller = 256,
    dim_latent = 128
)

model = Transformer(
    dim = 512,
    state_embed_readout = dict(num_discrete = 256),
    action_embed_readout = dict(num_discrete = 256),
    lower_body = dict(depth = 3, heads = 8, attn_dim_head = 48),
    upper_body = dict(depth = 3, heads = 8, attn_dim_head = 48),
    meta_controller = meta_controller
)

# optimizer

optim = AdamW(model.parameters(), lr = LEARNING_RATE)

model, optim, train_loader, val_loader = accelerator.prepare(
    model, optim, train_loader, val_loader
)

train_loader = cycle(train_loader)
val_loader = cycle(val_loader)

# training

pbar = tqdm.tqdm(range(NUM_BATCHES), mininterval = 10.0, desc = "training")

for i in pbar:
    model.train()
    
    is_discovering = i > 5000 

    if is_discovering:
        model.train_discovery()
    else:
        model.train()

    total_loss = 0.

    for _ in range(GRAD_ACCUM_EVERY):
        data = next(train_loader)
        
        tokens = data
        state = tokens[:, :-1]
        actions = tokens[:, 1:]

        outputs = model(
            state = state,
            actions = actions,
            discovery_phase = is_discovering,
            force_behavior_cloning = not is_discovering
        )

        if is_discovering:
            obs_loss, action_recon_loss, kl_loss, ratio_loss = outputs
            loss = (action_recon_loss + 0.5) + (obs_loss + 0.5) + kl_loss * 0.01 + ratio_loss
            
            curr_state_loss = obs_loss.item()
            curr_action_loss = action_recon_loss.item()
        else:
            state_loss, action_loss = outputs
            loss = (action_loss + 0.5) + (state_loss + 0.5)
            
            curr_state_loss = state_loss.item()
            curr_action_loss = action_loss.item()

        accelerator.backward(loss / GRAD_ACCUM_EVERY)
        total_loss += loss.item()

    if divisible_by(i, 10):
        phase = 'discovering' if is_discovering else 'cloning'
        action_loss_key = 'action_recon_loss' if is_discovering else 'action_loss'

        pbar_description = f"{i}: loss: {total_loss / GRAD_ACCUM_EVERY:.3f} ({phase}) state_loss: {curr_state_loss:.3f} {action_loss_key}: {curr_action_loss:.3f}"

        tqdm.tqdm.write(pbar_description)
        pbar.set_postfix(state_loss=f"{curr_state_loss:.3f}", action_loss=f"{curr_action_loss:.3f}")

    accelerator.clip_grad_norm_(model.parameters(), 0.5)
    optim.step()
    optim.zero_grad()

    if divisible_by(i, VALIDATE_EVERY):
        model.eval()
        with torch.no_grad():
            valid_data = next(val_loader)

            tokens = valid_data
            state = tokens[:, :-1]
            actions = tokens[:, 1:]
            
            outputs = model(
                state = state,
                actions = actions,
                discovery_phase = is_discovering,
                force_behavior_cloning = not is_discovering
            )
            
            if is_discovering:
                loss = outputs.action_recon + outputs.state_pred
            else:
                loss = outputs.action + outputs.state

            accelerator.print(f"{i}: validation loss: {loss.item():.3f}")

    if not is_discovering and divisible_by(i, GENERATE_EVERY):
        model.eval()
        inp = random.choice(val_dataset)[:PRIME_LENGTH]
        inp = inp.to(accelerator.device)

        prime = decode_tokens(inp)
        accelerator.print(f"\n\nINPUT: {prime}")

        prompt = inp[None, ...]
        sampled = sample(model, prompt, GENERATE_LENGTH)
        
        output = decode_tokens(sampled[0])
        accelerator.print(f"\n\nOUTPUT: {output}\n")
