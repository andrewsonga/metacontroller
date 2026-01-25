#!/bin/bash
set -e

# 1. Gather trajectories
echo "Gathering trajectories..."
uv run gather_babyai_trajs.py --num_seeds 10 --num_episodes_per_seed 10 --output_dir end_to_end_trajectories --env_id BabyAI-MiniBossLevel-v0

# 2. Behavioral cloning
echo "Training behavioral cloning model..."
uv run train_behavior_clone_babyai.py --epochs 1 --batch_size 16 --input_dir end_to_end_trajectories --env_id BabyAI-MiniBossLevel-v0 --checkpoint_path end_to_end_model.pt

# 3. Inference rollouts
echo "Running inference rollouts..."
uv run train_babyai.py --weights_path end_to_end_model.pt --env_name BabyAI-MiniBossLevel-v0 --num_episodes 5 --buffer_size 100 --max_timesteps 100
