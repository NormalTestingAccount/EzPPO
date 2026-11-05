"""
Discrete PPO example: CartPole-v1

Trains a PPO agent to balance the pole on a cart.
"""

import gymnasium as gym
from ezppo import DiscretePPOAgent




def main():
    # Environment setup
    env_id = "CartPole-v1"
    env = gym.make(env_id, render_mode="rgb_array")

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    print(f"Training on {env_id}")
    print(f"State dim: {state_dim}, Action dim: {action_dim}")

    # Create agent — performant hyperparams based on old reference impl
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
    max_episodes = 2000
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

        # Track best
        if total_reward > best_reward:
            best_reward = total_reward

        if (episode + 1) % 100 == 0:
            print(f"Episode {episode+1}/{max_episodes} | Reward: {total_reward:.1f} | Best: {best_reward:.1f}")

    # Save checkpoint
    agent.save("checkpoints/cartpole_ppo", "checkpoints/cartpole_ppo_norm")
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
