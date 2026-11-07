"""
Discrete PPO example: CliffWalking-v1

Trains a PPO agent to find the shortest path from start to goal
while avoiding the cliff edge.

Note: CliffWalking uses integer state IDs (0-47). We pass state_dim=48
to the agent so it knows the one-hot encoding size, and manually
one-hot encode the state before calling act(). The agent's train()
also handles one-hot encoding automatically.
"""

import gymnasium as gym
import torch
from ezppo import DiscretePPOAgent


def one_hot(state: int, num_classes: int) -> torch.Tensor:
    """Convert integer state to one-hot vector."""
    return torch.nn.functional.one_hot(torch.tensor(state), num_classes).float()


def main():
    env_id = "CliffWalking-v1"
    env = gym.make(env_id, render_mode="human")

    state_dim = env.observation_space.n          # Discrete states (grid cells)
    action_dim = env.action_space.n               # 4 actions (up, down, left, right)

    print(f"Training on {env_id}")
    print(f"State space: {state_dim} discrete states")
    print(f"Action space: {action_dim} discrete actions")

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

    # Training loop
    max_episodes = 1000
    best_reward = -float("inf")

    for episode in range(max_episodes):
        state, info = env.reset()
        total_reward = 0
        done = False

        while not done:
            # One-hot encode discrete state ID
            state_vec = one_hot(state, state_dim)
            action, log_prob = agent.act(state_vec)
            next_state, reward, terminated, truncated, info = env.step(action.item())
            done = terminated or truncated
            total_reward += reward

            agent.store((state_vec.numpy(), action.item(), reward, log_prob.item(),
                         one_hot(next_state, state_dim).numpy(), terminated, truncated))
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
    agent.save("checkpoints/cliffwalking_ppo", None)
    print("Checkpoint saved!")

    # Evaluation
    print("\n--- Evaluation ---")
    eval_env = gym.make(env_id, render_mode="human")
    for eval_ep in range(5):
        state, info = eval_env.reset()
        total_reward = 0
        done = False
        while not done:
            state_vec = one_hot(state, state_dim)
            action = agent.act_deterministic(state_vec)
            state, reward, terminated, truncated, info = eval_env.step(action)
            done = terminated or truncated
            total_reward += reward
        print(f"Eval episode: {total_reward}")


if __name__ == "__main__":
    main()
