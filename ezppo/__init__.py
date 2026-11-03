"""
EzPPO - A simple, unified PPO library for Discrete and Continuous environments.

Usage:
    from ezppo import DiscretePPOAgent, ContinuousPPOAgent

# Discrete example (CartPole):
    agent = DiscretePPOAgent(state_dim=4, action_dim=2)
    action, log_prob = agent.act(state)

# Continuous example (Pendulum):
    agent = ContinuousPPOAgent(state_dim=3, action_dim=1)
    action, log_prob, raw_action = agent.act(state)
"""

from .networks import DiscretePPO, ContinuousPPO, BasePPO
from .normalizer import RunningNormalizer
from .agents import BasePPOAgent, DiscretePPOAgent, ContinuousPPOAgent

__version__ = "0.1.0"
__all__ = [
    "DiscretePPO",
    "ContinuousPPO",
    "BasePPO",
    "RunningNormalizer",
    "BasePPOAgent",
    "DiscretePPOAgent",
    "ContinuousPPOAgent",
]
