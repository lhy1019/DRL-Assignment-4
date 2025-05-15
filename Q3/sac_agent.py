# sac_dmc_humanoid/sac_agent.py

import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import copy # For deep copying target networks

from networks import Actor, Critic # Assuming networks.py is in the same directory orPYTHONPATH
# If networks.py is in a subdirectory, e.g., sac_dmc_humanoid.networks
# from .networks import Actor, Critic 


class SACAgent:
    def __init__(self,
                 obs_dim,
                 action_dim,
                 action_bound, # Should be a scalar for DMC, e.g., 1.0
                 hidden_dim=256,
                 actor_lr=3e-4,
                 critic_lr=3e-4,
                 alpha_lr=3e-4,
                 gamma=0.99,
                 tau=0.005,
                 initial_log_alpha=0.0, # Initial value for log_alpha
                 target_entropy=None, # If None, use heuristic: -action_dim
                 device='cpu'):

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.action_bound = action_bound # scalar for scaling action output
        self.gamma = gamma
        self.tau = tau
        self.device = torch.device(device)

        # Actor Network
        self.actor = Actor(obs_dim, action_dim, hidden_dim, action_bound).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)

        # Critic Networks (Twin Q-networks)
        self.critic1 = Critic(obs_dim, action_dim, hidden_dim).to(self.device)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=critic_lr)
        self.critic_target1 = Critic(obs_dim, action_dim, hidden_dim).to(self.device)
        self.critic_target1.load_state_dict(self.critic1.state_dict()) # Initialize target same as critic
        for p in self.critic_target1.parameters(): # Target networks are not trained directly
            p.requires_grad = False

        self.critic2 = Critic(obs_dim, action_dim, hidden_dim).to(self.device)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=critic_lr)
        self.critic_target2 = Critic(obs_dim, action_dim, hidden_dim).to(self.device)
        self.critic_target2.load_state_dict(self.critic2.state_dict())
        for p in self.critic_target2.parameters():
            p.requires_grad = False

        # Alpha (Temperature for entropy regularization)
        self.log_alpha = torch.tensor(initial_log_alpha, dtype=torch.float32, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)
        self.alpha = self.log_alpha.exp().detach() # Keep alpha detached for use in losses, grads flow via log_alpha

        if target_entropy is None:
            # Heuristic for target entropy: -|A|
            self.target_entropy = -float(action_dim)
        else:
            self.target_entropy = float(target_entropy)
        
        print(f"SAC Agent Initialized on {self.device}")
        print(f"  obs_dim: {obs_dim}, action_dim: {action_dim}, action_bound: {action_bound}")
        print(f"  hidden_dim: {hidden_dim}")
        print(f"  actor_lr: {actor_lr}, critic_lr: {critic_lr}, alpha_lr: {alpha_lr}")
        print(f"  gamma: {gamma}, tau: {tau}")
        print(f"  initial_log_alpha: {initial_log_alpha}, current_alpha: {self.alpha.item()}")
        print(f"  target_entropy: {self.target_entropy}")


    def select_action(self, state, deterministic=False, add_batch_dim=True):
        """
        Selects an action given the current state.
        Args:
            state (np.ndarray): The current state.
            deterministic (bool): If True, select action deterministically (using mean).
                                  Otherwise, sample from the policy distribution.
            add_batch_dim (bool): If True, assumes state is a single sample and adds batch dim.
        Returns:
            np.ndarray: The action to take.
        """
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).to(self.device)
        
        if add_batch_dim and state.ndim == 1: # If state is (obs_dim,)
            state = state.unsqueeze(0) # Add batch dimension: (1, obs_dim)
        
        self.actor.eval() # Set actor to evaluation mode
        with torch.no_grad():
            if deterministic:
                mean, _ = self.actor(state)
                # Action is tanh(mean) scaled by action_bound
                action = torch.tanh(mean) * self.action_bound
            else:
                # sample() returns action, log_prob, mean, log_std
                # We only need the action here, reparameterize=False as we're not training here
                action, _, _, _ = self.actor.sample(state, reparameterize=False)
        self.actor.train() # Set actor back to training mode

        if add_batch_dim:
            action = action.squeeze(0) # Remove batch dimension for single action: (action_dim,)
            
        return action.cpu().numpy()


    def _soft_update_target_network(self, source_net, target_net):
        for target_param, source_param in zip(target_net.parameters(), source_net.parameters()):
            target_param.data.copy_(self.tau * source_param.data + (1.0 - self.tau) * target_param.data)

    def update_parameters(self, batch_data):
        """
        Performs a single update step for actor, critics, and alpha.
        Args:
            batch_data (dict): A dictionary of tensors from the replay buffer.
                               Keys: 'obs', 'actions', 'rewards', 'next_obs', 'dones'.
        Returns:
            dict: A dictionary containing loss values for logging.
        """
        obs_batch = batch_data['obs']
        actions_batch = batch_data['actions']
        rewards_batch = batch_data['rewards']
        next_obs_batch = batch_data['next_obs']
        dones_batch = batch_data['dones']

        # --- Update Critic Networks ---
        with torch.no_grad(): # Target Q computations should not affect critic gradients
            # Get next actions and their log probabilities from the current policy
            next_actions, next_log_pi, _, _ = self.actor.sample(next_obs_batch, reparameterize=True) # reparam for consistency, though not strictly needed for target Q value calc

            # Compute target Q-values from target critics
            q1_target_next = self.critic_target1(next_obs_batch, next_actions)
            q2_target_next = self.critic_target2(next_obs_batch, next_actions)
            q_target_next_min = torch.min(q1_target_next, q2_target_next)

            # Add entropy term: Q_target = r + gamma * (1-d) * (min_Q_next - alpha * log_pi_next)
            # self.alpha is detached, so its gradient won't flow here
            q_target = rewards_batch + self.gamma * (1.0 - dones_batch) * (q_target_next_min - self.alpha * next_log_pi)

        # Current Q estimates
        q1_current = self.critic1(obs_batch, actions_batch)
        q2_current = self.critic2(obs_batch, actions_batch)

        # Critic loss (MSE)
        critic1_loss = F.mse_loss(q1_current, q_target)
        critic2_loss = F.mse_loss(q2_current, q_target)
        
        # Optimize critic 1
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        # Optimize critic 2
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        # --- Update Actor Network (and Alpha) ---
        # Actor update is typically done after critic updates (or at a different frequency)
        # For simplicity, we update it here in the same call.
        
        # Freeze Q-networks while computing policy gradient (prevents gradients from Qs flowing to actor through Q values)
        # This is implicitly handled as we only call backward on actor_loss w.r.t actor params.
        # for p in self.critic1.parameters(): p.requires_grad = False
        # for p in self.critic2.parameters(): p.requires_grad = False

        # Sample actions and log_probs from current policy (with reparameterization for backprop)
        pi_actions, log_pi, _, _ = self.actor.sample(obs_batch, reparameterize=True)

        # Q-values for these policy actions from current critics
        q1_pi = self.critic1(obs_batch, pi_actions)
        q2_pi = self.critic2(obs_batch, pi_actions)
        q_pi_min = torch.min(q1_pi, q2_pi)

        # Actor loss: E[alpha * log_pi - min_Q]
        # self.alpha is detached here. The gradient for alpha comes from alpha_loss.
        actor_loss = (self.alpha * log_pi - q_pi_min).mean()

        # Optimize actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Unfreeze Q-networks (if they were frozen, not strictly necessary with separate optimizers)
        # for p in self.critic1.parameters(): p.requires_grad = True
        # for p in self.critic2.parameters(): p.requires_grad = True

        # --- Update Alpha (Temperature) ---
        # Alpha loss: E[-log_alpha * (log_pi + target_entropy)]
        # We want log_pi to be around -target_entropy.
        # log_pi is detached here as we are optimizing alpha, not the policy via this loss.
        alpha_loss = -(self.log_alpha * (log_pi.detach() + self.target_entropy)).mean()
        
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        self.alpha = self.log_alpha.exp().detach() # Update alpha value, keep it detached

        # --- Soft update target networks ---
        self._soft_update_target_network(self.critic1, self.critic_target1)
        self._soft_update_target_network(self.critic2, self.critic_target2)

        return {
            'critic1_loss': critic1_loss.item(),
            'critic2_loss': critic2_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha_loss': alpha_loss.item(),
            'alpha': self.alpha.item(),
            'log_pi_mean': log_pi.mean().item() # For monitoring
        }

    def save_models(self, path_prefix):
        """Saves actor, critics, and log_alpha."""
        torch.save(self.actor.state_dict(), f"{path_prefix}_actor.pth")
        torch.save(self.critic1.state_dict(), f"{path_prefix}_critic1.pth")
        torch.save(self.critic2.state_dict(), f"{path_prefix}_critic2.pth")
        torch.save(self.critic_target1.state_dict(), f"{path_prefix}_critic_target1.pth")
        torch.save(self.critic_target2.state_dict(), f"{path_prefix}_critic_target2.pth")
        # Save log_alpha tensor directly, not just its value
        torch.save({'log_alpha': self.log_alpha}, f"{path_prefix}_log_alpha.pth") 
        print(f"Models saved to {path_prefix}_*.pth")

    def load_models(self, path_prefix, evaluate=False, load_optimizer_states=False): # Added evaluate and load_optimizer_states
        print(f"Loading models from prefix: {path_prefix} with evaluate={evaluate}")
        self.actor.load_state_dict(torch.load(f"{path_prefix}_actor.pth", map_location=self.device))
        
        if not evaluate: # Only load critics and targets if continuing training or need them for some reason
            self.critic1.load_state_dict(torch.load(f"{path_prefix}_critic1.pth", map_location=self.device))
            self.critic2.load_state_dict(torch.load(f"{path_prefix}_critic2.pth", map_location=self.device))
            self.critic_target1.load_state_dict(torch.load(f"{path_prefix}_critic_target1.pth", map_location=self.device))
            self.critic_target2.load_state_dict(torch.load(f"{path_prefix}_critic_target2.pth", map_location=self.device))

            try:
                log_alpha_state = torch.load(f"{path_prefix}_log_alpha.pth", map_location=self.device)
                # Ensure log_alpha tensor exists and is properly configured for grad if continuing training
                if not hasattr(self, 'log_alpha') or self.log_alpha is None: # If log_alpha wasn't initialized (e.g. agent init for eval)
                     # This case shouldn't happen if evaluate=False, as __init__ would set up log_alpha
                    self.log_alpha = torch.tensor(0.0, dtype=torch.float32, requires_grad=True, device=self.device)
                
                self.log_alpha.data.copy_(log_alpha_state['log_alpha'].data) # Copy data
                
                # Ensure alpha_optimizer targets the current self.log_alpha
                if not hasattr(self, 'alpha_optimizer') or self.alpha_optimizer is None:
                    # Get alpha_lr from defaults if not available - this is a bit hacky
                    # Better if SACAgent stores its alpha_lr
                    alpha_lr_val = 3e-4 # Fallback, ideally get from self.config or __init__ params
                    if hasattr(self, 'alpha_optimizer_lr_default'): # If you stored it
                        alpha_lr_val = self.alpha_optimizer_lr_default
                    self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr_val)
                else:
                     # Re-assign params to be sure if log_alpha was recreated (it shouldn't be if done carefully)
                    self.alpha_optimizer.param_groups[0]['params'] = [self.log_alpha]

                self.alpha = self.log_alpha.exp().detach()
                print(f"Loaded log_alpha for continued training: {self.log_alpha.item():.4f} (current alpha: {self.alpha.item():.4f})")

                if load_optimizer_states:
                    try:
                        self.actor_optimizer.load_state_dict(torch.load(f"{path_prefix}_actor_optimizer.pth", map_location=self.device))
                        self.critic1_optimizer.load_state_dict(torch.load(f"{path_prefix}_critic1_optimizer.pth", map_location=self.device))
                        self.critic2_optimizer.load_state_dict(torch.load(f"{path_prefix}_critic2_optimizer.pth", map_location=self.device))
                        print("Loaded actor and critic optimizer states.")
                    except FileNotFoundError:
                        print("Optimizer state files not found for actor/critics. Optimizers will start fresh.")
            except FileNotFoundError:
                print(f"Warning: {path_prefix}_log_alpha.pth not found. Alpha auto-tuning might not resume correctly if continuing training.")
                # If log_alpha is essential for continued training and not found, could raise error or use initial_log_alpha
                # For now, it means self.alpha will be based on whatever self.log_alpha was initialized with in __init__
        
        # Always set actor to eval mode if evaluate=True. If evaluate=False, it's assumed training continues, so mode is handled by select_action.
        self.actor.eval() # Set actor to eval mode for deterministic actions
        if not evaluate:
            # If continuing training, other networks might be in train or eval mode depending on usage
            self.critic1.train() 
            self.critic2.train()
            # Target networks are always in eval mode as they are not directly trained
            self.critic_target1.eval()
            self.critic_target2.eval()
        else: # if evaluate=True, set all loaded online networks to eval
            if hasattr(self, 'critic1'): self.critic1.eval()
            if hasattr(self, 'critic2'): self.critic2.eval()
            if hasattr(self, 'critic_target1'): self.critic_target1.eval() # Though not strictly needed for pure eval
            if hasattr(self, 'critic_target2'): self.critic_target2.eval() # Though not strictly needed for pure eval


        if evaluate:
            print(f"Models (Actor only, or all if full load) loaded from {path_prefix}_*.pth and set to eval mode.")
        else:
            print(f"Models loaded from {path_prefix}_*.pth for continued training.")


# --- Example Usage (Conceptual - would be in main.py) ---
if __name__ == '__main__':
    # Dummy parameters for testing SACAgent initialization
    obs_dim_test = 67
    action_dim_test = 21
    action_bound_test = 1.0 # DMC humanoid actions are [-1, 1]
    device_test = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"--- SACAgent Test ---")
    agent = SACAgent(obs_dim_test, action_dim_test, action_bound_test, device=device_test, hidden_dim=64, actor_lr=1e-4, critic_lr=1e-4, alpha_lr=1e-4)

    # Test select_action
    print("\n--- Testing select_action ---")
    dummy_state_np = np.random.randn(obs_dim_test).astype(np.float32)
    action_stochastic = agent.select_action(dummy_state_np, deterministic=False)
    action_deterministic = agent.select_action(dummy_state_np, deterministic=True)
    print(f"State (numpy shape): {dummy_state_np.shape}")
    print(f"Stochastic action (numpy shape): {action_stochastic.shape}, Value: {action_stochastic[:3]}...")
    print(f"Deterministic action (numpy shape): {action_deterministic.shape}, Value: {action_deterministic[:3]}...")
    assert action_stochastic.shape == (action_dim_test,)
    assert action_deterministic.shape == (action_dim_test,)
    assert (action_stochastic >= -action_bound_test).all() and (action_stochastic <= action_bound_test).all()
    assert (action_deterministic >= -action_bound_test).all() and (action_deterministic <= action_bound_test).all()
    
    # Test select_action with batch of states (add_batch_dim=False)
    dummy_batch_state_np = np.random.randn(4, obs_dim_test).astype(np.float32) # Batch of 4 states
    action_batch_stochastic = agent.select_action(dummy_batch_state_np, deterministic=False, add_batch_dim=False)
    print(f"Batch State (numpy shape): {dummy_batch_state_np.shape}")
    print(f"Batch Stochastic action (numpy shape): {action_batch_stochastic.shape}")
    assert action_batch_stochastic.shape == (4, action_dim_test)


    # Test update_parameters (requires a ReplayBuffer and some data)
    print("\n--- Testing update_parameters (conceptual) ---")
    # We need ReplayBuffer from replay_buffer.py for a full test
    # For now, let's create a dummy batch directly
    batch_size_test = 32
    dummy_obs_batch = torch.randn(batch_size_test, obs_dim_test).to(device_test)
    dummy_actions_batch = torch.rand(batch_size_test, action_dim_test).to(device_test) * 2 - 1 # Actions in [-1, 1]
    dummy_rewards_batch = torch.randn(batch_size_test, 1).to(device_test)
    dummy_next_obs_batch = torch.randn(batch_size_test, obs_dim_test).to(device_test)
    dummy_dones_batch = torch.randint(0, 2, (batch_size_test, 1)).float().to(device_test)

    dummy_batch_data = {
        'obs': dummy_obs_batch,
        'actions': dummy_actions_batch,
        'rewards': dummy_rewards_batch,
        'next_obs': dummy_next_obs_batch,
        'dones': dummy_dones_batch
    }
    
    # Perform one update
    agent.actor.train() # Ensure actor is in train mode for updates
    losses = agent.update_parameters(dummy_batch_data)
    print("Losses after one update:")
    for k, v in losses.items():
        print(f"  {k}: {v:.4f}")
    assert 'critic1_loss' in losses and isinstance(losses['critic1_loss'], float)
    assert 'actor_loss' in losses and isinstance(losses['actor_loss'], float)
    assert 'alpha_loss' in losses and isinstance(losses['alpha_loss'], float)
    assert 'alpha' in losses and losses['alpha'] > 0

    print("\n--- Testing save/load models ---")
    import os
    model_path_prefix = "test_sac_model"
    agent.save_models(model_path_prefix)
    
    # Create a new agent and load
    agent_loaded = SACAgent(obs_dim_test, action_dim_test, action_bound_test, device=device_test, hidden_dim=64)
    agent_loaded.load_models(model_path_prefix)
    
    # Check if a parameter is the same (e.g., actor's first layer bias)
    # Need to be careful with floating point comparisons.
    # A simpler check is that alpha is loaded correctly.
    print(f"Original agent alpha: {agent.alpha.item()}, Loaded agent alpha: {agent_loaded.alpha.item()}")
    assert abs(agent.alpha.item() - agent_loaded.alpha.item()) < 1e-6, "Alpha not loaded correctly"

    # Clean up dummy model files
    files_to_remove = [
        f"{model_path_prefix}_actor.pth", f"{model_path_prefix}_critic1.pth",
        f"{model_path_prefix}_critic2.pth", f"{model_path_prefix}_critic_target1.pth",
        f"{model_path_prefix}_critic_target2.pth", f"{model_path_prefix}_log_alpha.pth"
    ]
    for f_name in files_to_remove:
        if os.path.exists(f_name):
            os.remove(f_name)
    print("Dummy model files cleaned up.")
    
    print("\nSACAgent basic tests passed.")