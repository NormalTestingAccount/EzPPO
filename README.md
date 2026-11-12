# EzPPO

A simple, unified PPO (Proximal Policy Optimization) library for PyTorch that supports **discrete** and **continuous** action spaces.

## Features

- PPO agents for discrete and continuous actions
- Optional online observation normalization
- Generalized Advantage Estimation (GAE)
- Clipped surrogate objective, mini-batches, and gradient clipping
- Checkpoint save/load and vector-environment support

## Installation

```bash
pip install torch gymnasium numpy
```

Some environments require extra dependencies. For BipedalWalker, install
`gymnasium[box2d]` and `swig`.

## Training with one environment

### Discrete actions

```python
import gymnasium as gym
from ezppo import DiscretePPOAgent

env = gym.make("CartPole-v1")
agent = DiscretePPOAgent(
    state_dim=env.observation_space.shape[0],
    action_dim=env.action_space.n,
)

for episode in range(1000):
    state, _ = env.reset()
    done = False

    while not done:
        action, log_prob = agent.act(state)
        next_state, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated

        agent.store((
            state, action.item(), reward, log_prob.item(), next_state,
            terminated, truncated,
        ))
        state = next_state

    agent.train()
    agent.clear()

agent.save("checkpoints/cartpole_ppo", "checkpoints/cartpole_ppo_norm.npy")
```

### Continuous actions

```python
import gymnasium as gym
from ezppo import ContinuousPPOAgent

env = gym.make("Pendulum-v1")
agent = ContinuousPPOAgent(
    state_dim=env.observation_space.shape[0],
    action_dim=env.action_space.shape[0],
)

for episode in range(1000):
    state, _ = env.reset()
    done = False

    while not done:
        action, log_prob, raw_action = agent.act(state)
        next_state, reward, terminated, truncated, _ = env.step(
            action.numpy() * env.action_space.high
        )
        done = terminated or truncated

        agent.store((
            state, raw_action.numpy(), reward, log_prob.item(), next_state,
            terminated, truncated,
        ))
        state = next_state

    agent.train()
    agent.clear()

agent.save("checkpoints/pendulum_ppo", "checkpoints/pendulum_ppo_norm.npy")
```

Store both `terminated` and `truncated` in every transition. PPO bootstraps
through truncations using `next_state`, but stops GAE recursion at either
episode boundary. For a regular, non-vector Gymnasium environment,
`next_state` is the transition's final observation.

## Training with multiple environments

### Discrete actions

```python
import gymnasium as gym
import numpy as np
from gymnasium.vector import AsyncVectorEnv, AutoresetMode
from ezppo import DiscretePPOAgent

def make_env():
    return gym.make("CartPole-v1")

num_envs = 4
envs = AsyncVectorEnv(
    [make_env for _ in range(num_envs)],
    autoreset_mode=AutoresetMode.SAME_STEP,
)
agent = DiscretePPOAgent(
    state_dim=envs.single_observation_space.shape[0],
    action_dim=envs.single_action_space.n,
    num_envs=num_envs,
)

observations, _ = envs.reset()
for _ in range(128):
    actions, log_probs = agent.act(observations)
    next_observations, rewards, terminated, truncated, infos = envs.step(
        actions.numpy()
    )

    transition_next_observations = next_observations.copy()
    if np.any(truncated):
        transition_next_observations[truncated] = infos["final_obs"][truncated]

    agent.store((
        observations, actions.numpy(), rewards, log_probs.numpy(),
        transition_next_observations, terminated, truncated,
    ))
    observations = next_observations

agent.train()
agent.clear()
```

### Continuous actions

```python
import gymnasium as gym
import numpy as np
from gymnasium.vector import AsyncVectorEnv, AutoresetMode
from ezppo import ContinuousPPOAgent

def make_env():
    return gym.make("Pendulum-v1")

num_envs = 4
envs = AsyncVectorEnv(
    [make_env for _ in range(num_envs)],
    autoreset_mode=AutoresetMode.SAME_STEP,
)
agent = ContinuousPPOAgent(
    state_dim=envs.single_observation_space.shape[0],
    action_dim=envs.single_action_space.shape[0],
    num_envs=num_envs,
)

observations, _ = envs.reset()
for _ in range(128):
    actions, log_probs, raw_actions = agent.act(observations)
    next_observations, rewards, terminated, truncated, infos = envs.step(
        actions.numpy() * 2
    )

    transition_next_observations = next_observations.copy()
    if np.any(truncated):
        transition_next_observations[truncated] = infos["final_obs"][truncated]

    agent.store((
        observations, raw_actions.numpy(), rewards, log_probs.numpy(),
        transition_next_observations, terminated, truncated,
    ))
    observations = next_observations

agent.train()
agent.clear()
```

Each vector-environment step stores one transition per environment, so these
examples collect `128 * num_envs` transitions. `SAME_STEP` autoreset returns
the reset observation in `next_observations`; for truncated episodes, the
examples replace it with `infos["final_obs"]` before storing so the value
function bootstraps from the final observation.

## Evaluation

Use `act_deterministic()` for evaluation.

### Discrete actions

```python
import gymnasium as gym
from ezppo import DiscretePPOAgent

env = gym.make("CartPole-v1")
agent = DiscretePPOAgent(
    state_dim=env.observation_space.shape[0],
    action_dim=env.action_space.n,
)
agent.load("checkpoints/cartpole_ppo", "checkpoints/cartpole_ppo_norm.npy")

for episode in range(5):
    state, _ = env.reset()
    total_reward = 0
    done = False

    while not done:
        action = agent.act_deterministic(state)
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward

    print(f"Episode {episode + 1}: reward={total_reward:.1f}")
```

### Continuous actions

```python
import gymnasium as gym
from ezppo import ContinuousPPOAgent

env = gym.make("Pendulum-v1")
agent = ContinuousPPOAgent(
    state_dim=env.observation_space.shape[0],
    action_dim=env.action_space.shape[0],
)
agent.load("checkpoints/pendulum_ppo", "checkpoints/pendulum_ppo_norm.npy")

for episode in range(5):
    state, _ = env.reset()
    total_reward = 0
    done = False

    while not done:
        action = agent.act_deterministic(state)[0]
        state, reward, terminated, truncated, _ = env.step(
            action.numpy() * env.action_space.high
        )
        done = terminated or truncated
        total_reward += reward

    print(f"Episode {episode + 1}: reward={total_reward:.1f}")
```

## API summary

- Set `device="mixed"` to run `act()` and `act_deterministic()` on CPU and
  training on CUDA; this mode requires CUDA to be available.
- `DiscretePPOAgent.act(state)` returns `(action, log_prob)`.
- `ContinuousPPOAgent.act(state)` returns `(action, log_prob, raw_action)`.
- For continuous PPO, pass the squashed `action` to the environment and store
  `raw_action` in the transition.
- Store transitions as
  `(state, action, reward, log_prob, next_state, terminated, truncated)`.
- `agent.train()` updates from stored transitions. Call `agent.clear()` before
  collecting the next rollout.
- `agent.save(model_path)` and `agent.load(model_path)` save and load weights.
  Supply a normalizer path to both when observation normalization is enabled.

## License

MIT
