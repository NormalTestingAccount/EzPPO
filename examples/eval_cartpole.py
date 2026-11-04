"""
Evaluate a trained Discrete PPO checkpoint on CartPole-v1.

Usage:
    python eval_cartpole.py --checkpoint checkpoints/cartpole_ppo
"""

import argparse
import gymnasium as gym
import numpy as np
from ezppo import DiscretePPOAgent


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained PPO checkpoint on CartPole-v1")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/cartpole_ppo",
                        help="Path to the saved model checkpoint (without extension)")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Number of evaluation episodes")
    args = parser.parse_args()

    # Environment setup
    env_id = "CartPole-v1"
    env = gym.make(env_id, render_mode="human")

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    print(f"Evaluating on {env_id}")
    print(f"State dim: {state_dim}, Action dim: {action_dim}")

    # Create agent — matches the hyperparams used in training
    agent = DiscretePPOAgent(
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
        entropy_coef=0.001,
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
    lengths = []
    print(f"--- Evaluation ({args.episodes} episodes) ---")
    for ep in range(args.episodes):
        state, info = env.reset()
        total_reward = 0
        done = False
        steps = 0
        while not done:
            print(agent.act(state))
            action = agent.act_deterministic(state)
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1
        rewards.append(total_reward)
        lengths.append(steps)
        print(f"  Episode {ep+1}/{args.episodes}: reward={total_reward:.1f}, steps={steps}")

    print(f"\nAverage reward: {np.mean(rewards):.1f} +/- {np.std(rewards):.1f}")
    print(f"Average length: {np.mean(lengths):.1f} +/- {np.std(lengths):.1f}")
    print(f"Best reward: {np.max(rewards):.1f}")
    print(f"Worst reward: {np.min(rewards):.1f}")
    success_rate = np.mean([r >= 500 for r in rewards]) * 100
    print(f"Success rate (reward >= 500): {success_rate:.0f}%")


if __name__ == "__main__":
    main()
