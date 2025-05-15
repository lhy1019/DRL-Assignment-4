# sac_dmc_humanoid/networks.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

# Define log_std bounds for numerical stability
LOG_STD_MAX = 2
LOG_STD_MIN = -20 # Original SAC paper uses -20, some impl use -5 or -10
EPSILON = 1e-6 # For numerical stability in log_prob calculation

class Actor(nn.Module):
    """
    Actor Network for SAC.
    Outputs parameters for a squashed Gaussian policy.
    """
    def __init__(self, obs_dim, action_dim, hidden_dim=256, action_bound=1.0):
        super(Actor, self).__init__()
        self.action_dim = action_dim
        self.action_bound = torch.tensor(action_bound, dtype=torch.float32) # Store as tensor

        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean_fc = nn.Linear(hidden_dim, action_dim)
        self.log_std_fc = nn.Linear(hidden_dim, action_dim)

        # Ensure action_bound is on the same device as parameters later
        self.register_buffer('action_bound_const', self.action_bound)


    def forward(self, state):
        """
        Given a state, outputs mean and log_std for the action distribution.
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.mean_fc(x)
        log_std = self.log_std_fc(x)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, state, reparameterize=True):
        """
        Samples an action from the policy, applies Tanh squashing,
        and computes the log probability of the squashed action.
        The reparameterize flag is for the actor loss calculation.
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal_dist = Normal(mean, std)

        if reparameterize:
            # Reparameterization trick (mean + std * N(0,1))
            # This allows gradients to flow back through the sampling process
            # u is the sample from the Normal distribution (before squashing)
            u = normal_dist.rsample()
        else:
            u = normal_dist.sample()

        # Squash the action using Tanh
        # action_tanh is in [-1, 1]
        action_tanh = torch.tanh(u)
        
        # Scale action to [-action_bound, action_bound]
        # This is the action taken in the environment
        action = action_tanh * self.action_bound_const.to(u.device) # Ensure device consistency

        # Calculate log probability
        # Log prob of u from the Normal distribution
        log_prob_u = normal_dist.log_prob(u)

        # Correct log prob for Tanh squashing and action scaling
        # The Tanh squashing correction is: sum_dims(log(1 - tanh(u)^2 + epsilon))
        # The action scaling correction (if action_bound is not 1 for all dims): sum_dims(log(action_bound_const))
        # So, log_prob(action) = log_prob(u) - sum_dims(log(1 - tanh(u)^2 + epsilon)) - sum_dims(log(action_bound_const))
        # This can be written as: log_prob(u) - sum_dims(log( (1 - tanh(u)^2) * action_bound_const + epsilon))
        # However, common implementations (e.g. SpinningUp) use:
        # log_prob = normal_dist.log_prob(u).sum(axis=-1)
        # log_prob -= (2 * (np.log(2) - u - F.softplus(-2 * u))).sum(axis=-1)
        # This is a more numerically stable way to compute log(1 - tanh(u)^2)
        # log(1 - tanh(x)^2) = log(sech(x)^2) = 2 * log(sech(x))
        # sech(x) = 2 / (e^x + e^-x) = 1 / cosh(x)
        # log(1 - tanh(x)^2) = -2 * (x + softplus(-2x) - log(2)) for PyTorch Tanh definition
        # For our case, action_tanh = tanh(u)
        # Correction term: sum_i (log(1 - action_tanh_i^2 + EPSILON))
        
        # log_prob_correction = torch.log(1.0 - action_tanh.pow(2) + EPSILON)
        # If action_bound is a scalar, log(action_bound) is added action_dim times.
        # If action_bound is a vector, sum(log(action_bound_i))
        
        # Let's use the direct derivation: jacobian of a = B * tanh(u) is B * (1 - tanh(u)^2)
        # log_prob(action) = log_prob(u) - sum_i log(B_i * (1 - tanh(u_i)^2))
        
        log_prob = log_prob_u - torch.log(self.action_bound_const.to(u.device) * (1.0 - action_tanh.pow(2)) + EPSILON)
        log_prob = log_prob.sum(dim=-1, keepdim=True) # Sum over action dimensions

        return action, log_prob, mean, log_std # Return mean and log_std as they might be useful for KL divergence if needed elsewhere

class Critic(nn.Module):
    """
    Critic Network (Q-function) for SAC.
    Outputs a Q-value given a state-action pair.
    """
    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()

        self.fc1 = nn.Linear(obs_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        """
        Given a state and action, outputs the Q-value.
        """
        # Concatenate state and action
        # Ensure action is on the same device as state if it comes from actor.sample()
        # and action_bound_const might have been moved.
        if state.device != action.device:
            action = action.to(state.device)
            
        sa = torch.cat([state, action], 1)

        q = F.relu(self.fc1(sa))
        q = F.relu(self.fc2(q))
        q = self.fc3(q)
        return q

# --- Example Usage (for testing the networks) ---
if __name__ == '__main__':
    # Environment parameters (as per your description)
    obs_dim_env = 67
    action_dim_env = 21
    # For Humanoid, action_bound is scalar 1.0
    # If it were per-dimension, action_bound_env would be a list/array
    action_bound_env_scalar = 1.0 

    # Test with scalar action_bound
    print("--- Actor Test (Scalar Action Bound) ---")
    actor_net_scalar = Actor(obs_dim_env, action_dim_env, action_bound=action_bound_env_scalar)
    
    # Move model to a device if you want to test GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    actor_net_scalar.to(device)

    batch_size = 4
    dummy_state = torch.randn(batch_size, obs_dim_env).to(device)
    
    mean, log_std = actor_net_scalar(dummy_state)
    sampled_action, log_prob, _, _ = actor_net_scalar.sample(dummy_state)

    print("State shape:", dummy_state.shape)
    print("Mean shape:", mean.shape)
    print("Log_std shape:", log_std.shape)
    print("Sampled Action shape:", sampled_action.shape)
    print("Log_prob shape:", log_prob.shape)
    assert sampled_action.shape == (batch_size, action_dim_env)
    assert log_prob.shape == (batch_size, 1)
    # Check bounds with a small tolerance for floating point
    assert (sampled_action.detach().cpu().numpy() >= -action_bound_env_scalar - EPSILON).all() and \
           (sampled_action.detach().cpu().numpy() <= action_bound_env_scalar + EPSILON).all(), "Action out of bounds"
    print("Actor output (scalar bound) seems correct.")

    print("\n--- Critic Test ---")
    critic_net = Critic(obs_dim_env, action_dim_env)
    critic_net.to(device)
    q_value = critic_net(dummy_state, sampled_action) 

    print("State shape:", dummy_state.shape)
    print("Action shape:", sampled_action.shape)
    print("Q-value shape:", q_value.shape)
    assert q_value.shape == (batch_size, 1)
    print("Critic output seems correct.")

    # Test with a different scalar action_bound
    print("\n--- Actor Test with different scalar action_bound (e.g., 2.0) ---")
    action_bound_test_scalar = 2.0
    actor_net_ab_scalar = Actor(obs_dim_env, action_dim_env, action_bound=action_bound_test_scalar)
    actor_net_ab_scalar.to(device)
    sampled_action_ab_scalar, log_prob_ab_scalar, _, _ = actor_net_ab_scalar.sample(dummy_state)
    print("Sampled Action AB scalar shape:", sampled_action_ab_scalar.shape)
    print("Log_prob AB scalar shape:", log_prob_ab_scalar.shape)
    assert (sampled_action_ab_scalar.detach().cpu().numpy() >= -action_bound_test_scalar - EPSILON).all() and \
           (sampled_action_ab_scalar.detach().cpu().numpy() <= action_bound_test_scalar + EPSILON).all(), "Action out of bounds with custom scalar bound"
    print("Actor output with custom scalar action_bound seems correct.")

    # Test with vector action_bound (if applicable, though DMC humanoid is scalar)
    # For Humanoid, action_bound is a scalar 1.0.
    # If it were a vector, e.g., action_bound_env_vector = np.array([1.0, 2.0] * (action_dim_env // 2) + [1.0]*(action_dim_env % 2))
    # actor_net_vector = Actor(obs_dim_env, action_dim_env, action_bound=action_bound_env_vector)
    # ... and then test
    # This part is mostly for future-proofing if environments have per-dimension bounds.
    # For this humanoid case, scalar action_bound=1.0 is correct.