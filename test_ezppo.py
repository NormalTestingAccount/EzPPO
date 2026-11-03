"""
Quick test script to verify EzPPO works correctly.
"""

import torch
import numpy as np
from ezppo import DiscretePPO, ContinuousPPO, DiscretePPOAgent, ContinuousPPOAgent


def test_discrete_network():
    print("=== Testing DiscretePPO Network ===")
    arch = {"actor": [4, 64, 64, 2], "critic": [4, 64, 64, 1]}
    net = DiscretePPO(arch)
    
    # Test forward
    x = torch.randn(3, 4)
    logits = net(x)
    assert logits.shape == (3, 2), f"Expected (3, 2), got {logits.shape}"
    print(f"  Forward: {logits.shape} OK")
    
    # Test predict_value
    values = net.predict_value(x)
    assert values.shape == (3, 1), f"Expected (3, 1), got {values.shape}"
    print(f"  Predict value: {values.shape} OK")
    
    print("  DiscretePPO Network: OK\n")


def test_continuous_network():
    print("=== Testing ContinuousPPO Network ===")
    arch = {"actor": [3, 64, 64, 1], "critic": [3, 64, 64, 1]}
    net = ContinuousPPO(arch)
    
    # Test forward
    x = torch.randn(3, 3)
    mean, log_std = net(x)
    assert mean.shape == (3, 1), f"Expected (3, 1), got {mean.shape}"
    assert log_std.shape == (1,), f"Expected (1,), got {log_std.shape}"
    print(f"  Forward: mean={mean.shape}, log_std={log_std.shape} OK")
    
    # Test predict_value
    values = net.predict_value(x)
    assert values.shape == (3, 1), f"Expected (3, 1), got {values.shape}"
    print(f"  Predict value: {values.shape} OK")
    
    print("  ContinuousPPO Network: OK\n")


def test_discrete_agent():
    print("=== Testing DiscretePPOAgent ===")
    agent = DiscretePPOAgent(state_dim=4, action_dim=2, device="cpu")
    
    # Test act
    state = np.random.randn(4).astype(np.float32)
    action, log_prob = agent.act(state)
    print(f"  Act: action={action}, log_prob={float(log_prob):.3f} OK")
    
    # Test deterministic act
    action_det = agent.act_deterministic(state)
    print(f"  Act deterministic: action={action_det} OK")
    
    # Test storing and training
    for _ in range(256):
        action, log_prob = agent.act(state)
        agent.store((state, action.item(), 0.0, log_prob.item(), state, 0))
        state = np.random.randn(4).astype(np.float32)
    
    print(f"  Trajectory size: {len(agent.trajectory)} OK")
    agent.train(num_epochs=2, batch_size=64)
    print(f"  Training: OK")
    agent.clear()
    print("  DiscretePPOAgent: OK\n")


def test_continuous_agent():
    print("=== Testing ContinuousPPOAgent ===")
    agent = ContinuousPPOAgent(state_dim=3, action_dim=1, device="cpu")
    
    # Test act
    state = np.random.randn(3).astype(np.float32)
    action, log_prob, raw_action = agent.act(state)
    print(f"  Act: action={float(action):.3f}, log_prob={float(log_prob):.3f} OK")
    
    # Test deterministic act
    action_det = agent.act_deterministic(state)
    print(f"  Act deterministic: action={float(action_det[0]):.3f} OK")
    
    # Test storing and training
    for _ in range(256):
        action, log_prob, raw_action = agent.act(state)
        agent.store((state, raw_action, 0.0, log_prob, state, 0))
        state = np.random.randn(3).astype(np.float32)
    
    print(f"  Trajectory size: {len(agent.trajectory)} OK")
    agent.train(num_epochs=2, batch_size=64)
    print(f"  Training: OK")
    agent.clear()
    print("  ContinuousPPOAgent: OK\n")


if __name__ == "__main__":
    print("EzPPO Test Suite\n")
    test_discrete_network()
    test_continuous_network()
    test_discrete_agent()
    test_continuous_agent()
    print("All tests passed!")
