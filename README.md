# EzPPO

A simple, unified PPO (Proximal Policy Optimization) library for PyTorch. Supports both **discrete** and **continuous** action spaces.

## Features

- **Unified API**: Same training loop for discrete and continuous
- **Observation normalization**: Online running statistics
- **GAE (Generalized Advantage Estimation)**: For stable advantage estimation
- **Clipped surrogate objective**: Standard PPO clipping
- **Mini-batch training**: With random shuffling
- **Gradient clipping**: Prevents exploding gradients
- **Checkpoint save/load**: Resume training easily
- **Multi-environment support**: Parallel environments built-in
- **GPU support**: Automatic device management

## Quick Start

### Discrete Actions (CartPole)

```python
from ezppo import DiscretePPOAgent
import gymnasium as gym

agent = DiscretePPOAgent(state_dim=4, action_dim=2)
env = gym.make("CartPole-v1")

state, _ = env.reset()
action, log_prob = agent.act(state)  # Stochastic action
# action = agent.act_deterministic(state)  # Greedy action (for eval)
```

### Continuous Actions (Pendulum)

```python
from ezppo import ContinuousPPOAgent
import gymnasium as gym

agent = ContinuousPPOAgent(state_dim=3, action_dim=1)
env = gym.make("Pendulum-v1")

state, _ = env.reset()
action, log_prob, raw_action = agent.act(state)
# action = agent.act_deterministic(state)[0]  # Mean action (for eval)
```

## Installation

```bash
pip install torch gymnasium numpy
```

## Architecture

```
ezppo/
├── __init__.py          # Public API
├── networks.py          # DiscretePPO, ContinuousPPO, BasePPO
├── normalizer.py        # RunningNormalizer
└── agents.py            # DiscretePPOAgent, ContinuousPPOAgent

examples/
├── discrete_cartpole.py
└── continuous_pendulum.py
```

## Training Loop Template

Store both `terminated` and `truncated` in each transition. The agent bootstraps
from `next_state` after truncation but stops GAE recursion at either episode
boundary. For auto-reset vector environments, pass the final observation as
`next_state` for a truncated transition rather than the reset observation.

```python
trajectory = []
max_trajectory = 2048

for episode in range(num_episodes):
    state, _ = env.reset()
    done = False

    while not done:
        action, log_prob = agent.act(state)
        next_state, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated

        trajectory.append((state, action.item(), reward, log_prob.item(),
                           next_state, terminated, truncated))
        state = next_state

    if len(trajectory) >= max_trajectory:
        agent.train(num_epochs=4, batch_size=64)
        trajectory.clear()
```

## API Reference

### Networks

| Class | Purpose |
|-------|---------|
| `BasePPO` | Base class with configurable architecture |
| `DiscretePPO` | Discrete actions → logits output |
| `ContinuousPPO` | Continuous actions → mean + learnable log_std |

### Agents

| Class | Purpose |
|-------|---------|
| `DiscretePPOAgent` | Discrete action space |
| `ContinuousPPOAgent` | Continuous action space |

### Key Methods

**Agent.act(state)** → `(action, log_prob)` or `(action, log_prob, raw_action)`
**Agent.act_deterministic(state)** → Deterministic action (for evaluation)
**Agent.train(num_epochs=4, batch_size=64)** → Train on trajectory
**Agent.store(packet)** → Add transition to trajectory
**Agent.clear()** → Clear trajectory buffer
**Agent.save(path)** → Save checkpoint
**Agent.load(path)** → Load checkpoint

## Configuration

```python
agent = DiscretePPOAgent(
    state_dim=4, action_dim=2,
    architecture={                    # Custom network architecture
        "actor": [state_dim, 64, 64, action_dim],
        "critic": [state_dim, 64, 64, 1]
    },
    lr=3e-4, gamma=0.99, 
    gae_lambda=0.95, epsilon=0.2,
    critic_coef=0.5, entropy_coef=0.01,
    num_envs=1, trajectory_size=2048,
    device="cpu"
)
```

## License

MIT
