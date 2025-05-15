# student_agent.py
import gymnasium as gym
import numpy as np
import torch
import os

# --- Ensure these imports work ---
try:
    from sac_agent import SACAgent
    from networks import Actor, Critic # Actor/Critic might be needed if SACAgent re-initializes them internally
except ImportError:
    import sys
    sys.path.append(os.path.dirname(__file__))
    from sac_agent import SACAgent
    from networks import Actor, Critic


class Agent(object):
    def __init__(self):
        # --- Agent Configuration ---
        # REPLACE with your actual obs_dim from training Humanoid-Walk
        self.trained_obs_dim = 67  # EXAMPLE: From your main_train.py output for humanoid-walk
                                   # e.g., Box(-inf, inf, (67,), float32) -> 67
        self.action_dim = 21
        self.action_bound = 1.0 
        
        # Match hidden_dim used during training for the loaded actor
        hidden_dim = 512 # EXAMPLE: From your config['hidden_dim']

        self.device = torch.device("cpu")

        # --- Initialize SACAgent ---
        # For evaluation, many SACAgent __init__ params are not strictly needed if we only load the actor.
        # However, to use the existing SACAgent class, we might need to provide them.
        # The `evaluate=True` flag in `load_models` will then ensure only necessary parts are used.
        self.sac_agent_instance = SACAgent(
            obs_dim=self.trained_obs_dim,
            action_dim=self.action_dim,
            action_bound=self.action_bound,
            hidden_dim=hidden_dim,
            actor_lr=0, critic_lr=0, alpha_lr=0, # Dummy values, not used for inference
            gamma=0.99, tau=0.005,                # Dummy values
            initial_log_alpha=0.0,                # Dummy, will be overwritten if model has it, or ignored for eval
            target_entropy=-self.action_dim,      # Dummy
            device=self.device
        )

        # --- Load Trained Model Weights ---
        current_script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Path to the directory containing the models
        models_dir_name = "models_sac_humanoid_gym" # As per your description
        models_directory_path = os.path.join(current_script_dir, models_dir_name)

        # Model prefix (e.g., the part before _actor.pth, _critic1.pth)
        # This should match how you saved them.
        # If you saved as "sac_humanoid_walk_best", then use that.
        model_file_prefix = "sac_humanoid_walk_best" # EXAMPLE - REPLACE if different
        
        full_model_path_prefix = os.path.join(models_directory_path, model_file_prefix)

        print(f"Attempting to load model from prefix: {full_model_path_prefix}")
        try:
            # Call load_models with evaluate=True
            # This tells load_models to only load what's needed for acting (mainly actor)
            # and set it to eval mode.
            self.sac_agent_instance.load_models(full_model_path_prefix, evaluate=True)
            print(f"Successfully loaded trained model: {full_model_path_prefix}")
            self.model_loaded_successfully = True
        except FileNotFoundError:
            print(f"ERROR: Model files not found at prefix: {full_model_path_prefix}. Check path and filenames.")
            self.model_loaded_successfully = False
        except Exception as e:
            print(f"ERROR: An unexpected error occurred while loading the model: {e}")
            import traceback
            traceback.print_exc()
            self.model_loaded_successfully = False
        
        # Fallback action space if model loading fails
        self.action_space_fallback = gym.spaces.Box(-1.0, 1.0, (self.action_dim,), np.float64)


    def act(self, observation):
        if not self.model_loaded_successfully:
            # print("Act: Model not loaded, returning random action.") # Avoid excessive printing in eval
            return self.action_space_fallback.sample()

        if not isinstance(observation, np.ndarray):
            observation = np.array(observation, dtype=np.float32)

        if observation.shape[0] != self.trained_obs_dim:
            # print(f"Warning: Obs dim mismatch! Expected {self.trained_obs_dim}, got {observation.shape[0]}. Random action.") # Avoid print
            return self.action_space_fallback.sample()

        # self.sac_agent_instance.select_action should handle device placement and numpy conversion
        action = self.sac_agent_instance.select_action(observation, deterministic=True, add_batch_dim=True)
        
        return action.astype(np.float64) # Ensure correct dtype for submission