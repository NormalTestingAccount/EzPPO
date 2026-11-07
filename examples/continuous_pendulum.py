"""
Continuous PPO example: Pendulum-v1

Trains a PPO agent to swing up and balance the pendulum.
"""

import gymnasium as gym
import numpy as np
from ezppo import ContinuousPPOAgent

from gymnasium.vector import AsyncVectorEnv


# Define a helper function (thunk) to create each sub-environment
def make_env(env_id, seed):
  def thunk():
    e = gym.make(env_id, render_mode="rgb_array")
    e.action_space.seed(seed)
    return e

  return thunk


def main():
    # Environment setup
    env_id = "Pendulum-v1"
    env = AsyncVectorEnv([make_env(env_id, seed=i) for i in range(16)])

    state_dim = env.single_observation_space.shape[0]
    action_dim = env.single_action_space.shape[0]

    print(f"Training on {env_id}")
    print(f"State dim: {state_dim}, Action dim: {action_dim}")
    print(f"Action range: [{env.single_action_space.low[0]}, {env.single_action_space.high[0]}]")

    # Create agent — performant hyperparams, normalization off for stability
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
        num_envs=16,
        trajectory_size=2048,
        device="cpu",
        normalize_observations=False,
        default_num_epochs=3,
        default_batch_size=64
    )

    # Training loop
    max_episodes = 5000
    best_reward = -float("inf")

    for episode in range(max_episodes):
        state, info = env.reset()
        total_reward = 0
        done = False

        while len(agent.trajectory) < agent._trajectory_size:
            action, log_prob, raw_action = agent.act(state)
            # Scale action to env range: tanh gives [-1,1], scale to action space
            action_scaled = action * 2 #env.action_space.high[0]
            next_state, reward, terminated, truncated, info = env.step(action_scaled)
            #done = terminated or truncated
            total_reward += np.average(reward)

            agent.store((state, raw_action, reward, log_prob, next_state, terminated, truncated))
            state = next_state

        # Train when trajectory is full
        if len(agent.trajectory) >= agent._trajectory_size:
            agent.train(verbose=False)
            agent.clear()

        if total_reward > best_reward:
            agent.save("checkpoints/pendulum_ppo", None)
            print("New best saved!")
            best_reward = total_reward

        #if (episode + 1) % 100 == 0:
        print(f"Episode {episode+1}/{max_episodes} | Reward: {total_reward:.1f} | Best: {best_reward:.1f}")

    # Save checkpoint
    #agent.save("checkpoints/pendulum_ppo", None)
    #print("Checkpoint saved!")

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
            done = np.logical_or(terminated, truncated)
            total_reward += reward
        print(f"Eval episode: {total_reward}")


if __name__ == "__main__":
    main()
