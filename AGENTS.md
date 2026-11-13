# EzPPO - Agent Guide

This file documents the EzPPO library structure, conventions, and implementation details for AI assistants working on this codebase.

## Library Purpose

EzPPO is a **simple, unified PPO (Proximal Policy Optimization) library** for PyTorch supporting both **discrete** and **continuous** action spaces. It's designed to be easy to use for learning RL while maintaining correctness.

## Project Structure

```
ezppo/                    # Core library (installable package)
├── __init__.py           # Public API exports
├── networks.py           # Neural network modules
│   ├── BasePPO           # Shared base with configurable architecture
│   ├── DiscretePPO       # Actor outputs logits, critic outputs values
│   └── ContinuousPPO     # Actor outputs (mean, log_std), critic outputs values
├── normalizer.py         # RunningNormalizer (Welford's online stats)
└── agents.py             # Agent classes
    ├── BasePPOAgent      # Shared trajectory/GAE/training logic
    ├── DiscretePPOAgent  # Discrete action space agent
    └── ContinuousPPOAgent # Continuous action space agent (tanh squashing)

examples/
├── discrete_cartpole.py          # Training example (CartPole-v1)
├── discrete_lunarlander.py       # Training example (LunarLander-v3)
├── discrete_mountaincar.py       # Training example (MountainCar-v0)
├── discrete_cliffwalking.py      # Training example (CliffWalking-v0)
├── continuous_pendulum.py       # Training example (Pendulum-v1)
├── continuous_bipedalwalker.py  # Training example (BipedalWalker-v3)
├── eval_cartpole.py              # Checkpoint evaluation (CartPole)
├── eval_pendulum.py              # Checkpoint evaluation (Pendulum)
├── eval_lunarlander.py           # Checkpoint evaluation (LunarLander)
└── eval_bipedalwalker.py        # Checkpoint evaluation (BipedalWalker)
```

## Key Implementation Details

### Network Architecture (`networks.py`)

- Architecture dict: `{"actor": [input, h1, h2, ..., output], "critic": [input, h1, h2, ..., 1]}`
- Optional `"shared"` backbone for weight sharing between actor/critic
- **Output layers NEVER get activation** — this is critical. The loop in `_init_layer` must skip activation for the output layer (when `i+1 == len(arch)-1`)
- Orthogonal init: `sqrt(2)` for hidden layers, `0.01` for output layers
- `ContinuousPPO` has learnable `log_std` parameter per action dimension

### Agent Training (`agents.py`)

- **Observation normalization is OPTIONAL** — agent constructors default to enabled; examples may disable it
  - When `normalize_observations=False`, `obs_normalizer` is `None`
  - All `act()`, `act_deterministic()`, `save()`, `load()`, `train()` methods must guard against `None`
- **`load(checkpoint_path, load_optimizer=True)`** — load `model` and `optim` from a checkpoint directory, plus `norm.npy` when observation normalization is enabled
- GAE computation in `_compute_gae()`: supports both single-env `[T]` and multi-env `[T, B]` shapes
- Every stored transition has the shape `(state, action, reward, log_prob, next_state, terminated, truncated)`. Both end flags are required.
- GAE uses separate masks: value bootstrapping stops only on termination, while GAE recursion stops on termination or truncation.
- For auto-reset vector environments, the rollout collector must store the final observation as `next_state` on truncation (and should use final observations for all ended environments). The library does not extract final observations from vector `info`.
- Advantage normalization: `(adv - mean) / (std + 1e-8)` before training
- Gradient clipping: `max_norm=0.5` on every step

### RunningNormalizer (`normalizer.py`)

- Uses Welford's parallel algorithm for numerically stable online stats
- Shape can be `int`, `list[int]`, or `tuple[int]` (all flattened to 1D)
- Initial state: `mean=0`, `var=1`, `count=1e-8`
- Clipping at ±10 for outlier protection
- `load()` preserves device placement via `.to(self.mean.device)`

### Continuous Action Handling

- Uses **tanh squashing** with Jacobian correction:
  ```python
  raw_action = dist.rsample()
  action = torch.tanh(raw_action)
  log_prob = dist.log_prob(raw_action) - log(1 - tanh^2(raw_action) + eps)
  ```
- `log_std` clamped to `[-5, 2]` during sampling
- Agent's `act()` returns `(action, log_prob, raw_action)`
- `act_deterministic()` returns `tanh(mean)` (no sampling)

## Default Hyperparameters (Performant)

These are the agent constructor defaults; examples may override them:

| Param | Discrete | Continuous |
|-------|----------|------------|
| Hidden units | 64, 64 | 64, 64 |
| Training epochs | 4 | 4 |
| Trajectory size | 2048 | 2048 |
| Normalize observations | **True** | **True** |
| Batch size | 64 | 64 |
| Entropy coefficient | 0.001 | 0.01 |
| Critic coefficient | 0.5 | 0.5 |
| Learning rate | 3e-4 | 3e-4 |
| Gamma / GAE lambda | 0.99 / 0.95 | 0.99 / 0.95 |
| PPO clip epsilon | 0.2 | 0.2 |

## Common Patterns

### Training Loop and Transition Semantics

Pass both Gymnasium end flags to `store()`. `next_state` must be the observation immediately after the action; for vector environments using auto-reset, the client must take the final observation from `info` when appropriate.

```python
for episode in range(max_episodes):
    state, _ = env.reset()
    done = False
    while not done:
        if isinstance(agent, ContinuousPPOAgent):
            action, log_prob, raw_action = agent.act(state)
            scaled_action = action.cpu().numpy()
            env_action = env.action_space.low + (
                (scaled_action + 1.0) * 0.5
                * (env.action_space.high - env.action_space.low)
            )
            stored_action = raw_action.cpu().numpy()
        else:
            action, log_prob = agent.act(state)
            env_action = action.item()
            stored_action = action.item()
        next_state, reward, terminated, truncated, info = env.step(env_action)
        done = terminated or truncated
        agent.store((
            state, stored_action, reward, log_prob.item(), next_state,
            terminated, truncated,
        ))
        state = next_state
    if len(agent.trajectory) >= trajectory_size:
        agent.train()
        agent.clear()
```

- `terminated=True`: do not bootstrap the value target; stop GAE recursion.
- `truncated=True`: bootstrap from the supplied final `next_state`; stop GAE recursion.
- If both are true, termination takes precedence for bootstrapping.
- For discrete PPO, store the sampled action. For continuous PPO, store `raw_action` returned by `act()` (not the squashed environment action).

### Checkpoint Save/Load
```python
# Save model and optimizer, plus the normalizer when enabled
agent.save("checkpoints/model_name")

# Load model and optimizer, plus the normalizer when enabled
agent.load("checkpoints/model_name")
```

## What to Watch Out For

1. **Output layer activation bug**: Any new network-building code must ensure the output layer does NOT get an activation function appended
2. **Null safety**: When `normalize_observations=False`, all `obs_normalizer` accesses must be guarded with `if self.obs_normalizer is not None`
3. **Shape handling**: `RunningNormalizer` accepts `int`, `list`, or `tuple` for shape — the `isinstance` check must use `(list, tuple)`
4. **Device consistency**: After `load()`, tensors stay on the device where the normalizer was; don't assume CPU
5. **Architecture validation**: When using shared backbone, actor/critic head first dimensions must match backbone output dimension

## Testing

Run tests: `python test_ezppo.py`

Tests cover:
- Network forward passes (discrete + continuous)
- Agent act/act_deterministic
- Training with small trajectory
- Save/load with and without normalization

## Package Info

- Name: `ezppo`
- Python: `>=3.8`
- Dependencies: `torch`, `gymnasium`, `numpy`
- Build: setuptools (pyproject.toml)
