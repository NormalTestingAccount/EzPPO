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

    # Create agent with LunarLander-optimized hyperparams
    # Key changes from CartPole defaults:
    #   - higher entropy (more exploration)
    #   - higher gamma (longer horizon)
    #   - higher gae_lambda (lower variance)
    #   - lower epsilon (more stable updates)
    #   - larger trajectory (more diverse data)
    #   - fewer epochs (less overfitting)
    agent = DiscretePPOAgent(
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
    max_episodes = 3000
    best_reward = -float("inf")

    for episode in range(max_episodes):
        state, info = env.reset()
        total_reward = 0
        done = False

        while not done:
            action, log_prob = agent.act(state)
            next_state, reward, terminated, truncated, info = env.step(action.item())
            done = terminated or truncated
            total_reward += reward

            agent.store((state, action.item(), reward, log_prob.item(), next_state, terminated))
            state = next_state

        # Train when trajectory is full
        if len(agent.trajectory) >= agent._trajectory_size:
            agent.train(verbose=False)
            agent.clear()

        if total_reward > best_reward:
            best_reward = total_reward

            agent.save("checkpoints/lunarlander_ppo", None)
            print("Checkpoint saved!")

        if (episode + 1) % 100 == 0:
            print(f"Episode {episode+1}/{max_episodes} | Reward: {total_reward:.1f} | Best: {best_reward:.1f}")

    if total_reward > best_reward:
        # Save checkpoint
        agent.save("checkpoints/lunarlander_ppo", None)
        print("Checkpoint saved!")

    # Evaluation
    print("\n--- Evaluation ---")
    eval_env = gym.make(env_id, render_mode="human")
    for eval_ep in range(5):
        state, info = eval_env.reset()
        total_reward = 0
        done = False
        while not done:
            action = agent.act_deterministic(state)
            state, reward, terminated, truncated, info = eval_env.step(action)
            done = terminated or truncated
            total_reward += reward
        print(f"Eval episode: {total_reward}")


if __name__ == "__main__":
    main()
