import numpy as np

class CallStationAgent:
    """
    An agent that always Checks or Calls if possible.
    Otherwise Folds.
    Compatible with POKERENGINE environment.
    """
    def __init__(self):
        pass

    def compute_action(self, observation, action_mask):
        """
        Selects Check/Call (1) if legal. Otherwise Fold (0).
        
        Args:
            observation: Unused.
            action_mask: Binary array of shape (8,) where 1 indicates legal.
            
        Returns:
            int: Selected action index.
        """
        mask = np.array(action_mask)
        
        # Index 1 is Check/Call
        if len(mask) > 1 and mask[1] == 1:
            return 1
            
        # Fallback to Fold (0)
        return 0
