"""
Evaluate a trained Continuous PPO checkpoint on Pendulum-v1.

Usage:
    python eval_pendulum.py --checkpoint checkpoints/pendulum_ppo
"""

import argparse
import gymnasium as gym
import numpy as np
from ezppo import ContinuousPPOAgent


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained PPO checkpoint on Pendulum-v1")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/pendulum_ppo",
                        help="Path to the saved model checkpoint (without extension)")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Number of evaluation episodes")
    args = parser.parse_args()

    # Environment setup
    env_id = "Pendulum-v1"
    env = gym.make(env_id, render_mode="human")

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    print(f"Evaluating on {env_id}")
    print(f"State dim: {state_dim}, Action dim: {action_dim}")
    print(f"Action range: [{env.action_space.low[0]}, {env.action_space.high[0]}]")

    # Create agent — matches the hyperparams used in training
    agent = ContinuousPPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        architecture={
            "actor": [state_dim, 256, 256, action_dim],
            "critic": [state_dim, 256, 256, 1]
        },
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        epsilon=0.2,
        critic_coef=0.5,
        entropy_coef=0.01,
        num_envs=1,
        trajectory_size=512,
        device="cpu",
        normalize_observations=False,
        default_num_epochs=10,
        default_batch_size=64
    )

    # Load checkpoint — normalizer path only needed if normalization was enabled during training
    agent.load(args.checkpoint)
    print(f"Loaded checkpoint from: {args.checkpoint}\n")

    # Evaluation
    rewards = []
    print(f"--- Evaluation ({args.episodes} episodes) ---")
    for ep in range(args.episodes):
        state, info = env.reset()
        total_reward = 0
        done = False
        steps = 0
        while not done:
            action = agent.act_deterministic(state)[0]  # Deterministic action
            action_scaled = action * env.action_space.high[0]
            state, reward, terminated, truncated, info = env.step(action_scaled)
            done = terminated or truncated
            total_reward += reward
            steps += 1
        rewards.append(total_reward)
        print(f"  Episode {ep+1}/{args.episodes}: reward={total_reward:.1f}, steps={steps}")

    print(f"\nAverage reward: {np.mean(rewards):.1f} +/- {np.std(rewards):.1f}")
    print(f"Best reward: {np.max(rewards):.1f}")
    print(f"Worst reward: {np.min(rewards):.1f}")


if __name__ == "__main__":
    main()
