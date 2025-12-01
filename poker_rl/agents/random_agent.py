import numpy as np

class RandomAgent:
    """
    An agent that selects a random legal action based on the action mask.
    Compatible with POKERENGINE environment.
    """
    def __init__(self):
        pass

    def compute_action(self, observation, action_mask):
        """
        Selects a random legal action.
        
        Args:
            observation: Unused.
            action_mask: Binary array of shape (8,) where 1 indicates legal.
            
        Returns:
            int: Selected action index (0-7).
        """
        mask = np.array(action_mask)
        legal_indices = np.where(mask == 1)[0]
        
        if len(legal_indices) == 0:
            # Fallback to Fold (0) if no actions legal (should not happen)
            return 0
            
        return np.random.choice(legal_indices)
