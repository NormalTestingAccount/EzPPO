"""
PyTorch neural network modules for Discrete and Continuous PPO.

Both networks share a common base class (BasePPO) that provides:
- Configurable architecture via dict (actor, critic, optional shared backbone)
- Orthogonal weight initialization (standard for RL)
- Small output layer initialization (std=0.01) for stability
"""

import torch
import torch.nn as nn
import numpy as np


class BasePPO(nn.Module):
    """
    Base PPO network with configurable architecture.
    
    Architecture dict format:
        {
            "actor": [input_dim, hidden1, hidden2, ..., output_dim],
            "critic": [input_dim, hidden1, hidden2, ..., 1],
            "shared": [input_dim, hidden1, ...]  # optional: shared backbone
        }
    
    If "shared" is provided, both actor and critic share an initial backbone.
    Otherwise, actor and critic are independent networks.
    """

    def __init__(self, architecture: dict, hidden_activation=nn.Tanh):
        super().__init__()

        self.actor_head = self._init_layer(architecture["actor"], hidden_activation)
        self.critic_head = self._init_layer(architecture["critic"], hidden_activation)

        self.backbone = self._init_layer(
            architecture["shared"], hidden_activation
        ) if "shared" in architecture else None

        self.input_dim = architecture["shared"][0] if self.backbone else architecture["actor"][0]
        self.output_dim = architecture["actor"][-1]

        # Choose forward methods based on whether we have a shared backbone
        if self.backbone:
            self._act_forward = self._act_forward_with_backbone
            self._crit_forward = self._crit_forward_with_backbone
        else:
            self._act_forward = self._act_forward_independent
            self._crit_forward = self._crit_forward_independent

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Actor forward pass (returns logits for discrete, mean for continuous)."""
        return self._act_forward(x)

    def predict_value(self, x: torch.Tensor) -> torch.Tensor:
        """Critic forward pass (returns state value estimate)."""
        return self._crit_forward(x)

    def _init_layer(self, arch: list[int], activation):
        """Build a sequential network from an architecture list."""
        seq = nn.Sequential()
        for i, v in enumerate(arch):
            if i == len(arch) - 1:
                break  # Don't add output layer activation
            input_n = v
            output_n = arch[i + 1]

            # Initialize weights with orthogonal init
            if i + 1 == len(arch) - 1:
                # Output layer: use small initialization for stability
                linear = self._ortho_init(nn.Linear(input_n, output_n), std=0.01)
            else:
                # Hidden layers: use sqrt(2) orthogonal init (standard for RL)
                linear = self._ortho_init(nn.Linear(input_n, output_n))

            # Only add activation for hidden layers, not the output layer
            if i + 1 != len(arch) - 1:
                seq.extend([linear, activation()])
            else:
                seq.append(linear)
        return seq

    def _ortho_init(self, layer, std=np.sqrt(2), bias_const=0.0):
        """Initialize a linear layer with orthogonal weights."""
        nn.init.orthogonal_(layer.weight, std)
        nn.init.constant_(layer.bias, bias_const)
        return layer

    # --- Forward method variants ---

    def _act_forward_independent(self, x):
        """Actor forward without shared backbone."""
        return self.actor_head(x)

    def _crit_forward_independent(self, x):
        """Critic forward without shared backbone."""
        return self.critic_head(x)

    def _act_forward_with_backbone(self, x):
        """Actor forward with shared backbone."""
        return self.actor_head(self.backbone(x))

    def _crit_forward_with_backbone(self, x):
        """Critic forward with shared backbone."""
        return self.critic_head(self.backbone(x))


class DiscretePPO(BasePPO):
    """
    Discrete action PPO network.
    
    Output:
        forward(x) -> action logits [batch, n_actions]
        predict_value(x) -> value estimates [batch, 1]
    """

    def __init__(self, architecture: dict, hidden_activation=nn.Tanh):
        super().__init__(architecture, hidden_activation)


class ContinuousPPO(BasePPO):
    """
    Continuous action PPO network with learnable log_std.
    
    Output:
        forward(x) -> (mean [batch, n_actions], log_std [n_actions])
        predict_value(x) -> value estimates [batch, 1]
    
    The log_std parameter is shared across all batches and is clamped
    during action sampling to prevent explosion.
    """

    def __init__(self, architecture: dict, hidden_activation=nn.Tanh, default_log=-1.0):
        super().__init__(architecture, hidden_activation)

        # Learnable log_std parameter (shared across all layers/batches)
        self.log_std = nn.Parameter(
            torch.ones(self.output_dim) * default_log,
            requires_grad=True
        )

    def forward(self, x: torch.Tensor):
        """Returns (mean, log_std). Mean has shape [batch, n_actions]."""
        mean = super().forward(x)
        return mean, self.log_std
