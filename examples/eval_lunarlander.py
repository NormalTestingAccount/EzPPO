"""
Discrete PPO example: LunarLander-v2

Trains a PPO agent to land the lunar module safely.
"""

import gymnasium as gym
from ezppo import DiscretePPOAgent


def main():
    env_id = "LunarLander-v3"
    env = gym.make(env_id, render_mode="rgb_array")

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    print(f"Training on {env_id}")
    print(f"State dim: {state_dim}, Action dim: {action_dim}")

    # Create agent with performant hyperparams
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

    # Save checkpoint
    agent.load("checkpoints/lunarlander_ppo")
    print("Checkpoint loaded!")

    # Evaluation
    print("\n--- Evaluation ---")
    eval_env = gym.make(env_id, render_mode="human")
    for eval_ep in range(5):
        state, info = eval_env.reset()
        total_reward = 0
        done = False
        while not done:
            action, _ = agent.act(state)
            state, reward, terminated, truncated, info = eval_env.step(action.tolist())
            done = terminated or truncated
            total_reward += reward
        print(f"Eval episode: {total_reward}")


if __name__ == "__main__":
    main()
