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
├── continuous_pendulum.py        # Training example (Pendulum-v1)
├── eval_cartpole.py              # Checkpoint evaluation (CartPole)
└── eval_pendulum.py              # Checkpoint evaluation (Pendulum)
```

## Key Implementation Details

### Network Architecture (`networks.py`)

- Architecture dict: `{"actor": [input, h1, h2, ..., output], "critic": [input, h1, h2, ..., 1]}`
- Optional `"shared"` backbone for weight sharing between actor/critic
- **Output layers NEVER get activation** — this is critical. The loop in `_init_layer` must skip activation for the output layer (when `i+1 == len(arch)-1`)
- Orthogonal init: `sqrt(2)` for hidden layers, `0.01` for output layers
- `ContinuousPPO` has learnable `log_std` parameter per action dimension

### Agent Training (`agents.py`)

- **Observation normalization is OPTIONAL** — disabled by default for stability
  - When `normalize_observations=False`, `obs_normalizer` is `None`
  - All `act()`, `act_deterministic()`, `save()`, `load()`, `train()` methods must guard against `None`
- **`load(model_path)`** — normalizer_path defaults to `None`; only loads normalizer if `obs_normalizer is not None`
- GAE computation in `_compute_gae()`: supports both single-env `[T]` and multi-env `[T, B]` shapes
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

These are the tuned defaults based on reference implementation comparison:

| Param | Discrete | Continuous |
|-------|----------|------------|
| Hidden units | 256 | 256 |
| Training epochs | 10 | 10 |
| Trajectory size | 512 | 512 |
| Normalize obs | **False** | **False** |
| Batch size | 64 | 64 |
| Entropy coef | 0.001 | 0.01 |
| Critic coef | 0.5 | 0.5 |

## Common Patterns

### Training Loop
```python
for episode in range(max_episodes):
    state, _ = env.reset()
    while not done:
        action, log_prob = agent.act(state)
        next_state, reward, terminated, truncated, _ = env.step(...)
        agent.store((state, action, reward, log_prob, next_state, terminated))
        state = next_state
    if len(agent.trajectory) >= agent._trajectory_size:
        agent.train()  # Uses defaults if not specified
        agent.clear()
```

### Checkpoint Save/Load
```python
# Save (normalizer_path only needed if normalize_observations=True)
agent.save("checkpoints/model_name")

# Load (no normalizer_path needed when disabled)
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
