
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.torch_utils import FLOAT_MIN
import torch
import torch.nn as nn
from gymnasium import spaces

class MaskedMLP(TorchModelV2, nn.Module):
    """
    Custom Model that handles Action Masking
    """
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        
        # Obs space is Dict:
        # "observations": Box(310)
        # "action_mask": Box(7)
        
        # We need the shape of the "observations" part
        # obs_space.original_space['observations'].shape[0]
        # But RLlib might pass flattened obs_space if not handled carefully.
        # However, for Dict space, RLlib usually keeps structure if we access input_dict['obs'].
        
        # Get input size dynamically from obs_space
        # obs_space is a Dict space, so we access the 'observations' component
        if hasattr(obs_space, "original_space"):
            input_size = obs_space.original_space["observations"].shape[0]
        else:
            # Fallback if not wrapped or different structure
            input_size = obs_space["observations"].shape[0]
        
        # FC Layers
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 256)
        
        # Policy Head
        self.logits = nn.Linear(256, num_outputs)
        
        # Value Head
        self.value = nn.Linear(256, 1)
        
        self._value = None
        
    def forward(self, input_dict, state, seq_lens):
        # input_dict["obs"] is the observation
        # Since we use Dict space, it should be a dictionary of tensors
        obs = input_dict["obs"]["observations"]
        action_mask = input_dict["obs"]["action_mask"]
        
        # Forward pass
        x = torch.relu(self.fc1(obs))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        logits = self.logits(x)
        
        # Apply Action Masking
        # Mask is 1 for legal, 0 for illegal
        # We want to set illegal logits to -inf
        inf_mask = torch.clamp(torch.log(action_mask), min=FLOAT_MIN)
        masked_logits = logits + inf_mask
        
        self._value = self.value(x).squeeze(1)
        
        return masked_logits, state
    
    def value_function(self):
        return self._value
