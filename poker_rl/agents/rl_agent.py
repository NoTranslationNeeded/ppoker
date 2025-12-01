
import ray
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.policy.policy import Policy
import numpy as np
import torch

class RLAgent:
    def __init__(self, checkpoint_path, policy_id="main_policy"):
        self.checkpoint_path = checkpoint_path
        self.policy_id = policy_id
        
        # Restore algorithm
        # Note: We might need to register the custom model before restoring if not already done in the script
        from poker_rl.models.masked_lstm import MaskedLSTM
        from ray.rllib.models import ModelCatalog
        ModelCatalog.register_custom_model("masked_lstm", MaskedLSTM)
        
        self.algo = Algorithm.from_checkpoint(checkpoint_path)
        self.policy = self.algo.get_policy(policy_id)
        
        # Initialize LSTM state
        self.state = self.get_initial_state()
        
    def get_initial_state(self):
        # Get initial state from the model
        # The model is accessible via policy.model
        if hasattr(self.policy, "model") and hasattr(self.policy.model, "get_initial_state"):
            # get_initial_state returns a list of tensors/arrays
            # We need to ensure they are in the format expected by compute_single_action
            return self.policy.model.get_initial_state()
        return []

    def reset_state(self):
        self.state = self.get_initial_state()

    def compute_action(self, observation, action_mask):
        """
        Compute action using the RL policy and maintain LSTM state.
        
        Args:
            observation: The observation vector (numpy array)
            action_mask: The action mask (numpy array)
        """
        
        # Construct input dict for compute_single_action
        # RLlib expects observation to be the full observation space structure
        # Our env uses Dict space: {"observations": ..., "action_mask": ...}
        
        full_obs = {
            "observations": observation,
            "action_mask": action_mask
        }
        
        # compute_single_action signature:
        # compute_single_action(observation, state=None, prev_action=None, prev_reward=None, info=None, ...)
        # It returns (action, state_out, info)
        
        action, state_out, info = self.policy.compute_single_action(
            obs=full_obs,
            state=self.state,
            explore=False # Deterministic for evaluation/play
        )
        
        # Update state
        self.state = state_out
        
        return action
