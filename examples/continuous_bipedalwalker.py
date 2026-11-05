"""
Continuous PPO example: BipedalWalker-v3

Trains a PPO agent to walk a bipedal robot across terrain.

The environment has 24 observation dimensions and 4 continuous actions
(each in [-1, 1]): left_engine, right_engine, left_leg_up, right_leg_up.

Requires Box2D: pip install swig && pip install "gymnasium[box2d]"
"""

import gymnasium as gym
import numpy as np
from ezppo import ContinuousPPOAgent


def main():
    # Environment setup
    env_id = "BipedalWalker-v3"
    env = gym.make(env_id, render_mode="rgb_array")

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    print(f"Training on {env_id}")
    print(f"State dim: {state_dim}, Action dim: {action_dim}")
    print(f"Action range: [{env.action_space.low[0]}, {env.action_space.high[0]}]")

    # Create agent — tuned for complex continuous control
    agent = ContinuousPPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        architecture={
            "actor": [state_dim, 256, 256, action_dim],
            "critic": [state_dim, 256, 256, 1]
        },
        lr=3e-4,
        gamma=0.995,
        gae_lambda=0.98,
        epsilon=0.1,
        critic_coef=0.5,
        entropy_coef=0.01,
        num_envs=1,
        trajectory_size=2048,
        device="cpu",
        normalize_observations=False,
        default_num_epochs=4,
        default_batch_size=128
    )

    # Training loop
    max_episodes = 5000
    best_reward = -float("inf")

    for episode in range(max_episodes):
        state, info = env.reset()
        total_reward = 0
        done = False

        while not done:
            action, log_prob, raw_action = agent.act(state)
            # Scale action to env range: tanh gives [-1,1], scale to action space
            action_scaled = action * env.action_space.high[0]
            next_state, reward, terminated, truncated, info = env.step(action_scaled)
            done = terminated or truncated
            total_reward += reward

            agent.store((state, raw_action, reward, log_prob, next_state, terminated))
            state = next_state

        # Train when trajectory is full
        if len(agent.trajectory) >= agent._trajectory_size:
            agent.train(verbose=False)
            agent.clear()

        if total_reward > best_reward:
            best_reward = total_reward

        if (episode + 1) % 100 == 0:
            print(f"Episode {episode+1}/{max_episodes} | Reward: {total_reward:.1f} | Best: {best_reward:.1f}")

    # Save checkpoint
    agent.save("checkpoints/bipedalwalker_ppo", None)
    print("Checkpoint saved!")

    # Evaluation
    print("\n--- Evaluation ---")
    eval_env = gym.make(env_id, render_mode="human")
    for eval_ep in range(5):
        state, info = eval_env.reset()
        total_reward = 0
        done = False
        while not done:
            action = agent.act_deterministic(state)[0]  # Get deterministic action
            action_scaled = action * eval_env.action_space.high[0]
            state, reward, terminated, truncated, info = eval_env.step(action_scaled)
            done = terminated or truncated
            total_reward += reward
        print(f"Eval episode: {total_reward}")


if __name__ == "__main__":
    main()
