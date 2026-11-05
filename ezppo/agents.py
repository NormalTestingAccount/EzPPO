"""
Agent classes for Discrete and Continuous PPO.

Both agents share a common base class with:
- Trajectory collection (store/clear)
- GAE computation
- Observation normalization
- Mini-batch training loop
- Device management
- Save/load checkpointing
"""

import torch
import numpy as np
import torch.optim as optim
from torch.distributions import Categorical, Normal
from torch.nn import MSELoss

from .networks import DiscretePPO, ContinuousPPO
from .normalizer import RunningNormalizer


class BasePPOAgent:
    """Shared base for both discrete and continuous agents."""

    def __init__(self, network, lr=3e-4, gamma=0.99, gae_lambda=0.95,
                 epsilon=0.2, critic_coef=0.5, entropy_coef=0.01,
                 num_envs=1, trajectory_size=2048, device="cpu",
                 normalize_observations=True, default_num_epochs=4,
                 default_batch_size=64):
        self.network = network
        self.optimizer = optim.Adam(network.parameters(), lr=lr, eps=1e-5)

        self.trajectory = []
        self._num_envs = num_envs
        self._trajectory_size = trajectory_size

        # PPO hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.epsilon = epsilon
        self.critic_coef = critic_coef
        self.entropy_coef = entropy_coef
        self.lr = lr
        self.device = torch.device(device)

        self.normalize_observations = normalize_observations
        self.default_num_epochs = default_num_epochs
        self.default_batch_size = default_batch_size

        # Set up observation normalizer (optional)
        if self.normalize_observations:
            self.obs_normalizer = RunningNormalizer(network.input_dim)
            self.obs_normalizer.to(self.device)
        else:
            self.obs_normalizer = None

        self.network.to(self.device)

    def store(self, packet: tuple) -> None:
        """Store a transition in the trajectory."""
        self.trajectory.append(packet)

    def clear(self) -> None:
        """Clear the trajectory buffer."""
        self.trajectory = []

    def _compute_gae(self, states, rewards, values, next_values, terminal_mask):
        """
        Compute Generalized Advantage Estimation (GAE).
        
        Args:
            states: [T, B, obs_dim] or [T, obs_dim]
            rewards: [T, B] or [T]
            values: [T, B] or [T] - value of current states
            next_values: [T, B] or [T] - value of next states
            terminal_mask: [T, B] or [T] - 0 if terminal, 1 if not
            
        Returns:
            advantages: [T, B] or [T]
            critic_targets: [T, B] or [T] - advantages + values (for critic loss)
        """
        if rewards.ndim == 1:
            # Single environment: [T]
            T = rewards.shape[0]
            advantages = torch.zeros(T, device=self.device)
            last_advantage = 0.0
            for t in reversed(range(T)):
                td_error = rewards[t] + self.gamma * next_values[t] * terminal_mask[t] - values[t]
                last_advantage = td_error + self.gamma * self.gae_lambda * last_advantage * terminal_mask[t]
                advantages[t] = last_advantage
        else:
            # Multi-environment: [T, B]
            T, B = rewards.shape
            advantages = torch.zeros(T, B, device=self.device)
            last_advantage = torch.zeros(B, device=self.device)
            for t in reversed(range(T)):
                td_error = rewards[t] + self.gamma * next_values[t] * terminal_mask[t] - values[t]
                last_advantage = td_error + self.gamma * self.gae_lambda * last_advantage * terminal_mask[t]
                advantages[t] = last_advantage

        critic_targets = advantages + values
        return advantages, critic_targets

    def _flatten_batch(self, *tensors):
        """Flatten [T, B, ...] to [T*B, ...] for batched training."""
        return [t.flatten(end_dim=1) for t in tensors]

    def train(self, num_epochs=None, batch_size=None):
        """
        Train PPO on the collected trajectory.
        
        Args:
            num_epochs: Number of PPO update epochs (uses default if None).
            batch_size: Mini-batch size for training (uses default if None).
        """
        if not self.trajectory:
            return

        num_epochs = num_epochs if num_epochs is not None else self.default_num_epochs
        batch_size = batch_size if batch_size is not None else self.default_batch_size

        # Unpack trajectory
        states, actions, rewards, old_log_probs, next_states, terminateds = zip(*self.trajectory)

        # Convert to tensors
        states_np = np.stack(states)
        actions_np = np.stack(actions)
        rewards_np = np.stack(rewards)
        old_log_probs_np = np.stack(old_log_probs)
        next_states_np = np.stack(next_states)
        terminateds_np = np.stack(terminateds)

        # Handle discrete state spaces (integer IDs -> one-hot)
        # For discrete envs like CliffWalking, states are integer IDs (0-d or scalar states)
        if states_np.ndim == 1 and states_np.dtype in (np.int64, np.int32, np.int16, np.int8):
            states = torch.nn.functional.one_hot(torch.from_numpy(states_np).long(), self.network.input_dim).float().to(self.device)
            next_states = torch.nn.functional.one_hot(torch.from_numpy(next_states_np).long(), self.network.input_dim).float().to(self.device)
        else:
            states = torch.from_numpy(states_np).float().to(self.device)
            next_states = torch.from_numpy(next_states_np).float().to(self.device)
        actions = torch.from_numpy(actions_np).float().to(self.device)
        rewards = torch.from_numpy(rewards_np).float().to(self.device)
        old_log_probs = torch.from_numpy(old_log_probs_np).float().to(self.device)
        terminateds = torch.from_numpy(terminateds_np).float().to(self.device)
        # Ensure consistent [T, B] shapes
        # For single env, stacking gives [T], [T, obs_dim] etc.
        # For multi env, stacking gives [T, num_envs], [T, num_envs, obs_dim] etc.
        if rewards.ndim == 1:
            # Single environment case
            rewards = rewards.unsqueeze(1)      # [T, 1]
            old_log_probs = old_log_probs.unsqueeze(1)  # [T, 1]
            terminateds = terminateds.unsqueeze(1)  # [T, 1]
            actions = actions.unsqueeze(1)      # [T, 1]
            if states.ndim == 2:
                states = states.unsqueeze(1)    # [T, 1, obs_dim]
            if next_states.ndim == 2:
                next_states = next_states.unsqueeze(1)  # [T, 1, obs_dim]

        terminal_mask = (1 - terminateds).float()

        # Compute values
        values = self.network.predict_value(states).squeeze(-1)
        next_values = self.network.predict_value(next_states).squeeze(-1)

        # Compute GAE
        advantages, critic_targets = self._compute_gae(
            states, rewards, values, next_values, terminal_mask
        )

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Detach targets (they shouldn't be used for backprop)
        old_log_probs = old_log_probs.detach()
        advantages = advantages.detach()
        critic_targets = critic_targets.detach()

        # Flatten for batched training
        states, actions, old_log_probs, critic_targets, advantages = \
            self._flatten_batch(states, actions, old_log_probs, critic_targets, advantages)

        total_samples = states.shape[0]
        avg_policy_loss = 0
        avg_critic_loss = 0
        avg_entropy = 0
        num_updates = 0

        # Training loop
        for _ in range(num_epochs):
            indices = np.arange(total_samples)
            np.random.shuffle(indices)

            for start in range(0, total_samples, batch_size):
                end = min(start + batch_size, total_samples)
                batch_idx = indices[start:end]

                batch_states = states[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_critic_targets = critic_targets[batch_idx]
                batch_advantages = advantages[batch_idx]

                # Get new log probs and entropy (implemented by subclass)
                new_log_probs, entropy = self._get_log_prob_and_entropy(batch_states, batch_actions)

                # Compute ratios
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                clipped_ratio = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon)

                # Policy loss (clipped surrogate)
                policy_loss = -torch.min(
                    ratio * batch_advantages,
                    clipped_ratio * batch_advantages
                ).mean()

                # Critic loss (MSE)
                new_values = self.network.predict_value(batch_states).squeeze(-1)
                critic_loss = MSELoss()(new_values, batch_critic_targets)

                # Entropy bonus
                entropy_loss = -entropy.mean()

                # Total loss
                total_loss = policy_loss + self.critic_coef * critic_loss + self.entropy_coef * entropy_loss

                # Optimizer step
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=0.5)
                self.optimizer.step()

                avg_policy_loss += policy_loss.item()
                avg_critic_loss += critic_loss.item()
                avg_entropy += entropy_loss.item()
                num_updates += 1

        # Update observation normalizer (if enabled)
        if self.obs_normalizer is not None:
            self.obs_normalizer.update(states.cpu().detach())

        # Move back to the agent's device
        self.network.to(self.device)
        if self.obs_normalizer is not None:
            self.obs_normalizer.to(self.device)

        # Log
        if num_updates > 0:
            print(f"[{type(self).__name__}] Policy: {avg_policy_loss/num_updates:.4f} | "
                  f"Critic: {avg_critic_loss/num_updates:.4f} | "
                  f"Entropy: {avg_entropy/num_updates:.4f}")

    def save(self, model_path: str, normalizer_path: str = None) -> None:
        """Save checkpoint."""
        torch.save(self.network.state_dict(), model_path)
        if self.obs_normalizer is not None:
            self.obs_normalizer.save(normalizer_path)

    def load(self, model_path: str, normalizer_path: str | None = None) -> None:
        """Load checkpoint.
        
        Args:
            model_path: Path to the network state dict file.
            normalizer_path: Path to the normalizer file (only used if
                             normalize_observations=True; defaults to None).
        """
        self.network.load_state_dict(torch.load(model_path, weights_only=True))
        if self.obs_normalizer is not None:
            self.obs_normalizer.load(normalizer_path)
        self.network.to(self.device)
        if self.obs_normalizer is not None:
            self.obs_normalizer.to(self.device)


class DiscretePPOAgent(BasePPOAgent):
    """
    PPO agent for discrete action spaces.
    
    Usage:
        agent = DiscretePPOAgent(state_dim=4, action_dim=2)
        action, log_prob = agent.act(state)
        agent.store((state, action, reward, log_prob, next_state, terminated))
        agent.train()
    """

    def __init__(self, state_dim: int, action_dim: int,
                 architecture: dict = None,
                 lr: float = 3e-4, gamma: float = 0.99,
                 gae_lambda: float = 0.95, epsilon: float = 0.2,
                 critic_coef: float = 0.5, entropy_coef: float = 0.001,
                 num_envs: int = 1, trajectory_size: int = 2048,
                 device: str = "cpu", checkpoint: str = None,
                 normalize_observations: bool = True,
                 default_num_epochs: int = 4,
                 default_batch_size: int = 64):
        
        # Default architecture
        if architecture is None:
            architecture = {
                "actor": [state_dim, 64, 64, action_dim],
                "critic": [state_dim, 64, 64, 1]
            }

        network = DiscretePPO(architecture)
        super().__init__(network, lr, gamma, gae_lambda, epsilon,
                         critic_coef, entropy_coef, num_envs,
                         trajectory_size, device,
                         normalize_observations, default_num_epochs,
                         default_batch_size)

        if checkpoint:
            if self.obs_normalizer is not None:
                self.load(checkpoint + ".pth", checkpoint + "_norm.npy")
            else:
                self.load(checkpoint + ".pth", None)

    def act(self, state) -> tuple:
        """
        Select an action (stochastic).
        
        Args:
            state: Observation. Single env: [obs_dim] or [1, obs_dim].
                   Multi env: [num_envs, obs_dim].
            
        Returns:
            (action, log_prob) - both shaped [num_envs] if batched, else scalar
        """
        state = torch.as_tensor(state, dtype=torch.float32)
        # Handle 0-d (scalar) or 1-d states: add batch dim
        if state.dim() == 0:
            state = state.unsqueeze(0)        # [1]
        if state.dim() == 1:
            state = state.unsqueeze(0)        # [1, obs_dim]
        # Expand single state to match num_envs, or validate batch size
        if state.shape[0] == 1 and self._num_envs > 1:
            state = state.expand(self._num_envs, -1)  # [num_envs, obs_dim]
        elif state.shape[0] != self._num_envs:
            raise ValueError(
                f"Expected {self._num_envs} states, got {state.shape[0]}"
            )
        if self.normalize_observations:
            state = self.obs_normalizer.normalize(state).to(self.device)
        else:
            state = state.to(self.device)

        logits = self.network(state)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        # Remove batch dim if single env
        if self._num_envs == 1:
            return action.squeeze(0).cpu().detach(), log_prob.squeeze(0).cpu().detach()
        return action.cpu().detach(), log_prob.cpu().detach()

    def act_deterministic(self, state):
        """Select action greedily (no sampling)."""
        state = torch.as_tensor(state, dtype=torch.float32)
        # Handle 0-d (scalar) or 1-d states: add batch dim
        if state.dim() == 0:
            state = state.unsqueeze(0)        # [1]
        if state.dim() == 1:
            state = state.unsqueeze(0)        # [1, obs_dim]
        # Expand single state to match num_envs, or validate batch size
        if state.shape[0] == 1 and self._num_envs > 1:
            state = state.expand(self._num_envs, -1)  # [num_envs, obs_dim]
        elif state.shape[0] != self._num_envs:
            raise ValueError(
                f"Expected {self._num_envs} states, got {state.shape[0]}"
            )
        if self.normalize_observations:
            state = self.obs_normalizer.normalize(state).to(self.device)
        else:
            state = state.to(self.device)

        logits = self.network(state)
        action = logits.argmax(dim=-1)

        # Return scalar for single env, array for multi env
        if self._num_envs == 1:
            return action.squeeze(0).cpu().detach().item()
        return action.cpu().detach().tolist()

    def _get_log_prob_and_entropy(self, states, actions):
        """Compute log probability and entropy for a batch."""
        logits = self.network(states)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        # Categorical.entropy() returns [batch], NOT [batch, 1]
        entropies = dist.entropy()
        return log_probs, entropies


class ContinuousPPOAgent(BasePPOAgent):
    """
    PPO agent for continuous action spaces with tanh squashing.
    
    Uses reparameterization trick with tanh:
    - Sample raw action from Gaussian
    - Squash with tanh: action = tanh(raw)
    - Correct log_prob with Jacobian: log_prob = raw_log_prob - log(1 - tanh^2 + eps)
    
    Usage:
        agent = ContinuousPPOAgent(state_dim=3, action_dim=1)
        action, log_prob, raw_action = agent.act(state)
        agent.store((state, raw_action, reward, log_prob, next_state, terminated))
        agent.train()
    """

    def __init__(self, state_dim: int, action_dim: int,
                 architecture: dict = None,
                 lr: float = 3e-4, gamma: float = 0.99,
                 gae_lambda: float = 0.95, epsilon: float = 0.2,
                 critic_coef: float = 0.5, entropy_coef: float = 0.01,
                 num_envs: int = 1, trajectory_size: int = 2048,
                 device: str = "cpu", checkpoint: str = None,
                 normalize_observations: bool = True,
                 default_num_epochs: int = 4,
                 default_batch_size: int = 64):
        
        # Default architecture
        if architecture is None:
            architecture = {
                "actor": [state_dim, 64, 64, action_dim],
                "critic": [state_dim, 64, 64, 1]
            }

        network = ContinuousPPO(architecture)
        super().__init__(network, lr, gamma, gae_lambda, epsilon,
                         critic_coef, entropy_coef, num_envs,
                         trajectory_size, device,
                         normalize_observations, default_num_epochs,
                         default_batch_size)

        if checkpoint:
            if self.obs_normalizer is not None:
                self.load(checkpoint + ".pth", checkpoint + "_norm.npy")
            else:
                self.load(checkpoint + ".pth", None)

    def act(self, state) -> tuple:
        """
        Select an action (stochastic, with tanh squashing).
        
        Args:
            state: Observation. Single env: [obs_dim] or [1, obs_dim].
                   Multi env: [num_envs, obs_dim].
            
        Returns:
            (action, log_prob, raw_action) - shaped [num_envs, action_dim] etc.
            Returns scalars/1-dim if _num_envs=1, otherwise arrays.
        """
        state = torch.as_tensor(state, dtype=torch.float32)
        # Handle 0-d (scalar) or 1-d states: add batch dim
        if state.dim() == 0:
            state = state.unsqueeze(0)        # [1]
        if state.dim() == 1:
            state = state.unsqueeze(0)        # [1, obs_dim]
        # Expand single state to match num_envs, or validate batch size
        if state.shape[0] == 1 and self._num_envs > 1:
            state = state.expand(self._num_envs, -1)  # [num_envs, obs_dim]
        elif state.shape[0] != self._num_envs:
            raise ValueError(
                f"Expected {self._num_envs} states, got {state.shape[0]}"
            )
        if self.normalize_observations:
            state = self.obs_normalizer.normalize(state).to(self.device)
        else:
            state = state.to(self.device)

        mean, log_std = self.network(state)
        log_std = torch.clamp(log_std, -5, 2)
        std = log_std.exp()

        dist = Normal(mean, std)
        raw_action = dist.rsample()
        action = torch.tanh(raw_action)

        # Jacobian correction for tanh squashing
        log_prob = (dist.log_prob(raw_action) - torch.log(
            1 - action ** 2 + 1e-6
        )).sum(-1)

        if self._num_envs == 1:
            return action.squeeze(0).cpu().detach(), \
                   log_prob.cpu().detach(), \
                   raw_action.squeeze(0).cpu().detach()
        return action.cpu().detach(), log_prob.cpu().detach(), raw_action.cpu().detach()

    def act_deterministic(self, state) -> tuple:
        """
        Select action deterministically (mean of distribution).
        Useful for evaluation.
        """
        state = torch.as_tensor(state, dtype=torch.float32)
        # Handle 0-d (scalar) or 1-d states: add batch dim
        if state.dim() == 0:
            state = state.unsqueeze(0)        # [1]
        if state.dim() == 1:
            state = state.unsqueeze(0)        # [1, obs_dim]
        # Expand single state to match num_envs, or validate batch size
        if state.shape[0] == 1 and self._num_envs > 1:
            state = state.expand(self._num_envs, -1)  # [num_envs, obs_dim]
        elif state.shape[0] != self._num_envs:
            raise ValueError(
                f"Expected {self._num_envs} states, got {state.shape[0]}"
            )
        if self.normalize_observations:
            state = self.obs_normalizer.normalize(state).to(self.device)
        else:
            state = state.to(self.device)

        mean, log_std = self.network(state)
        action = torch.tanh(mean)

        log_prob = torch.zeros(mean.shape[0], device=self.device)
        raw_action = torch.zeros(mean.shape[0], device=self.device)

        if self._num_envs == 1:
            return action.squeeze(0).cpu().detach(), \
                   log_prob.cpu().detach(), \
                   raw_action.cpu().detach()
        return action.cpu().detach(), log_prob.cpu().detach(), raw_action.cpu().detach()

    def _get_log_prob_and_entropy(self, states, raw_actions):
        """Compute log probability and entropy for a batch of raw actions."""
        mean, log_std = self.network(states)
        log_std = torch.clamp(log_std, -5, 2)
        std = log_std.exp()

        dist = Normal(mean, std)
        actions = torch.tanh(raw_actions)

        # Jacobian correction
        log_probs = (dist.log_prob(raw_actions) - torch.log(
            1 - actions ** 2 + 1e-6
        )).sum(-1)

        entropy = dist.entropy().sum(-1)
        return log_probs, entropy
