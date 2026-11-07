"""
Quick test script to verify EzPPO works correctly.
"""

import torch
import numpy as np
from unittest.mock import patch
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
        agent.store((state, action.item(), 0.0, log_prob.item(), state, 0, 0))
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
        agent.store((state, raw_action, 0.0, log_prob, state, 0, 0))
        state = np.random.randn(3).astype(np.float32)
    
    print(f"  Trajectory size: {len(agent.trajectory)} OK")
    agent.train(num_epochs=2, batch_size=64)
    print(f"  Training: OK")
    agent.clear()
    print("  ContinuousPPOAgent: OK\n")


def test_discrete_agent_multi_env():
    print("=== Testing DiscretePPOAgent (multi-env) ===")
    num_envs = 4
    agent = DiscretePPOAgent(state_dim=4, action_dim=2, num_envs=num_envs, device="cpu")
    
    # Test batched act
    states = np.random.randn(num_envs, 4).astype(np.float32)
    actions, log_probs = agent.act(states)
    assert actions.shape == (num_envs,), f"Expected {(num_envs,)}, got {actions.shape}"
    assert log_probs.shape == (num_envs,), f"Expected {(num_envs,)}, got {log_probs.shape}"
    print(f"  Batched act: actions={actions.shape}, log_probs={log_probs.shape} OK")
    
    # Test single state expansion
    single_state = np.random.randn(4).astype(np.float32)
    actions_exp, log_probs_exp = agent.act(single_state)
    assert actions_exp.shape == (num_envs,), f"Expected {(num_envs,)}, got {actions_exp.shape}"
    print(f"  Single state expansion: {actions_exp.shape} OK")
    
    # Test storing and training with batched data
    for _ in range(64):
        states = np.random.randn(num_envs, 4).astype(np.float32)
        actions, log_probs = agent.act(states)
        next_states = np.random.randn(num_envs, 4).astype(np.float32)
        rewards = np.random.randn(num_envs).astype(np.float32) * 0.1
        terminateds = np.zeros(num_envs, dtype=np.int32)
        truncateds = np.zeros(num_envs, dtype=np.int32)
        agent.store((states, actions.numpy(), rewards, log_probs.numpy(), next_states, terminateds, truncateds))
    
    print(f"  Trajectory size: {len(agent.trajectory)} OK")
    agent.train(num_epochs=2, batch_size=32)
    print(f"  Training: OK")
    agent.clear()
    print("  DiscretePPOAgent (multi-env): OK\n")


def test_continuous_agent_multi_env():
    print("=== Testing ContinuousPPOAgent (multi-env) ===")
    num_envs = 4
    agent = ContinuousPPOAgent(state_dim=3, action_dim=1, num_envs=num_envs, device="cpu")
    
    # Test batched act
    states = np.random.randn(num_envs, 3).astype(np.float32)
    actions, log_probs, raw_actions = agent.act(states)
    assert actions.shape == (num_envs, 1), f"Expected {(num_envs, 1)}, got {actions.shape}"
    assert log_probs.shape == (num_envs,), f"Expected {(num_envs,)}, got {log_probs.shape}"
    print(f"  Batched act: actions={actions.shape}, log_probs={log_probs.shape} OK")
    
    # Test single state expansion
    single_state = np.random.randn(3).astype(np.float32)
    actions_exp, log_probs_exp, raw_exp = agent.act(single_state)
    assert actions_exp.shape == (num_envs, 1), f"Expected {(num_envs, 1)}, got {actions_exp.shape}"
    print(f"  Single state expansion: {actions_exp.shape} OK")
    
    # Test storing and training with batched data
    for _ in range(64):
        states = np.random.randn(num_envs, 3).astype(np.float32)
        actions, log_probs, raw_actions = agent.act(states)
        next_states = np.random.randn(num_envs, 3).astype(np.float32)
        rewards = np.random.randn(num_envs).astype(np.float32) * 0.1
        terminateds = np.zeros(num_envs, dtype=np.int32)
        truncateds = np.zeros(num_envs, dtype=np.int32)
        agent.store((states, raw_actions.numpy(), rewards, log_probs.numpy(), next_states, terminateds, truncateds))
    
    print(f"  Trajectory size: {len(agent.trajectory)} OK")
    agent.train(num_epochs=2, batch_size=32)
    print(f"  Training: OK")
    agent.clear()
    print("  ContinuousPPOAgent (multi-env): OK\n")


def test_continuous_training_and_inference_shapes():
    """Check continuous tensor shapes through inference and PPO updates."""
    state_dim = 3
    action_dim = 2
    rollout_steps = 7
    batch_size = 4

    for num_envs in (1, 3):
        agent = ContinuousPPOAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            num_envs=num_envs,
            normalize_observations=False,
            default_num_epochs=1,
            default_batch_size=batch_size,
        )

        actor_shapes = []
        value_shapes = []
        gae_shapes = []
        gae_masks = []
        minibatch_shapes = []
        exp_input_shapes = []

        def record_actor_shapes(_module, inputs, output):
            mean, log_std = output
            actor_shapes.append(
                (tuple(inputs[0].shape), tuple(mean.shape), tuple(log_std.shape))
            )

        agent.network.register_forward_hook(record_actor_shapes)

        predict_value = agent.network.predict_value

        def record_value_shapes(states):
            values = predict_value(states)
            value_shapes.append((tuple(states.shape), tuple(values.shape)))
            return values

        agent.network.predict_value = record_value_shapes

        compute_gae = agent._compute_gae

        def record_gae_shapes(rewards, values, next_values, bootstrap_mask, gae_mask):
            advantages, targets = compute_gae(
                rewards, values, next_values, bootstrap_mask, gae_mask
            )
            gae_shapes.append(
                (
                    tuple(rewards.shape),
                    tuple(values.shape),
                    tuple(next_values.shape),
                    tuple(bootstrap_mask.shape),
                    tuple(gae_mask.shape),
                    tuple(advantages.shape),
                    tuple(targets.shape),
                )
            )
            gae_masks.append(
                (bootstrap_mask.detach().cpu().clone(), gae_mask.detach().cpu().clone())
            )
            return advantages, targets

        agent._compute_gae = record_gae_shapes

        get_log_prob_and_entropy = agent._get_log_prob_and_entropy

        def record_minibatch_shapes(states, raw_actions):
            log_probs, entropy = get_log_prob_and_entropy(states, raw_actions)
            minibatch_shapes.append(
                (
                    tuple(states.shape),
                    tuple(raw_actions.shape),
                    tuple(log_probs.shape),
                    tuple(entropy.shape),
                )
            )
            return log_probs, entropy

        agent._get_log_prob_and_entropy = record_minibatch_shapes

        inference_states = np.random.randn(num_envs, state_dim).astype(np.float32)
        if num_envs == 1:
            inference_states = inference_states[0]
        actions, log_probs, raw_actions = agent.act(inference_states)
        expected_env_shape = () if num_envs == 1 else (num_envs,)
        expected_action_shape = (action_dim,) if num_envs == 1 else (num_envs, action_dim)
        assert tuple(actions.shape) == expected_action_shape
        assert tuple(raw_actions.shape) == expected_action_shape
        assert tuple(log_probs.shape) == expected_env_shape
        deterministic_actions = agent.act_deterministic(inference_states)[0]
        assert tuple(deterministic_actions.shape) == expected_action_shape
        assert actor_shapes == [
            (
                (num_envs, state_dim),
                (num_envs, action_dim),
                (action_dim,),
            ),
            (
                (num_envs, state_dim),
                (num_envs, action_dim),
                (action_dim,),
            ),
        ]

        for step in range(rollout_steps):
            states = np.random.randn(num_envs, state_dim).astype(np.float32)
            if num_envs == 1:
                states = states[0]
            actions, log_probs, raw_actions = agent.act(states)
            next_states = np.random.randn(num_envs, state_dim).astype(np.float32)
            rewards = np.random.randn(num_envs).astype(np.float32)
            terminateds = np.zeros(num_envs, dtype=np.bool_)
            truncateds = np.zeros(num_envs, dtype=np.bool_)
            if step == 2:
                truncateds[0] = True
            if step == 4:
                terminateds[-1] = True
            if num_envs == 1:
                next_states = next_states[0]
                rewards = rewards[0]
                terminateds = terminateds[0]
                truncateds = truncateds[0]
            agent.store(
                (states, raw_actions, rewards, log_probs, next_states, terminateds, truncateds)
            )

        actor_shapes.clear()
        original_exp = torch.exp

        def record_exp_input(values):
            exp_input_shapes.append(tuple(values.shape))
            return original_exp(values)

        with patch("torch.exp", side_effect=record_exp_input):
            agent.train(num_epochs=1, batch_size=batch_size, verbose=False)

        total_samples = rollout_steps * num_envs
        expected_batch_sizes = [
            min(batch_size, total_samples - start)
            for start in range(0, total_samples, batch_size)
        ]
        assert [entry[0] for entry in minibatch_shapes] == [
            (size, state_dim) for size in expected_batch_sizes
        ]
        assert [entry[1:] for entry in minibatch_shapes] == [
            ((size, action_dim), (size,), (size,))
            for size in expected_batch_sizes
        ]
        assert exp_input_shapes == [(size,) for size in expected_batch_sizes]
        assert value_shapes[:2] == [
            ((rollout_steps, num_envs, state_dim), (rollout_steps, num_envs, 1)),
            ((rollout_steps, num_envs, state_dim), (rollout_steps, num_envs, 1)),
        ]
        assert value_shapes[2:] == [
            ((size, state_dim), (size, 1)) for size in expected_batch_sizes
        ]
        assert gae_shapes == [
            (
                (rollout_steps, num_envs),
                (rollout_steps, num_envs),
                (rollout_steps, num_envs),
                (rollout_steps, num_envs),
                (rollout_steps, num_envs),
                (rollout_steps, num_envs),
                (rollout_steps, num_envs),
            )
        ]
        expected_bootstrap_mask = torch.ones(rollout_steps, num_envs)
        expected_bootstrap_mask[4, -1] = 0
        expected_gae_mask = torch.ones(rollout_steps, num_envs)
        expected_gae_mask[2, 0] = 0
        expected_gae_mask[4, -1] = 0
        torch.testing.assert_close(gae_masks[0][0], expected_bootstrap_mask)
        torch.testing.assert_close(gae_masks[0][1], expected_gae_mask)
        assert actor_shapes == [
            ((size, state_dim), (size, action_dim), (action_dim,))
            for size in expected_batch_sizes
        ]


def test_gae_bootstrap_and_episode_boundary_masks():
    """Truncations bootstrap from next-state values but stop GAE recursion."""
    agent = ContinuousPPOAgent(
        state_dim=1,
        action_dim=1,
        gamma=0.5,
        gae_lambda=0.8,
        normalize_observations=False,
    )
    rewards = torch.tensor([[1.0], [2.0], [3.0]])
    values = torch.tensor([[10.0], [20.0], [30.0]])
    next_values = torch.tensor([[30.0], [40.0], [50.0]])
    terminateds = torch.tensor([[False], [False], [True]])
    truncateds = torch.tensor([[True], [False], [False]])

    bootstrap_mask = (~terminateds).float()
    gae_mask = (~(terminateds | truncateds)).float()
    advantages, targets = agent._compute_gae(
        rewards, values, next_values, bootstrap_mask, gae_mask
    )

    expected_advantages = torch.tensor([[6.0], [-8.8], [-27.0]])
    torch.testing.assert_close(advantages, expected_advantages)
    torch.testing.assert_close(targets, expected_advantages + values)


if __name__ == "__main__":
    print("EzPPO Test Suite\n")
    test_discrete_network()
    test_continuous_network()
    test_discrete_agent()
    test_continuous_agent()
    test_discrete_agent_multi_env()
    test_continuous_agent_multi_env()
    test_continuous_training_and_inference_shapes()
    test_gae_bootstrap_and_episode_boundary_masks()
    print("All tests passed!")
