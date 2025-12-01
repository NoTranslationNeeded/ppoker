
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.torch_utils import FLOAT_MIN
import torch
import torch.nn as nn

class MaskedLSTM(TorchModelV2, nn.Module):
    """
    Custom Model that handles Action Masking and LSTM memory
    """
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        
        # Get input size dynamically from obs_space
        # obs_space is a Dict space, so we access the 'observations' component
        if hasattr(obs_space, "original_space"):
            input_size = obs_space.original_space["observations"].shape[0]
        else:
            # Fallback if not wrapped or different structure
            input_size = obs_space["observations"].shape[0]
            
        self.lstm_state_size = model_config.get("lstm_cell_size", 256)
        
        # FC Layers for feature extraction
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 512)
        
        # LSTM Layer
        self.lstm = nn.LSTM(512, self.lstm_state_size, batch_first=True)
        
        # Policy Head
        self.logits = nn.Linear(self.lstm_state_size, num_outputs)
        
        # Value Head
        self.value = nn.Linear(self.lstm_state_size, 1)
        
        self._value = None
        
        # Initialization
        self._initialize_weights()

    def _initialize_weights(self):
        # FC Layers: Orthogonal or Xavier
        for name, param in self.named_parameters():
            if "fc" in name:
                if "weight" in name:
                    nn.init.orthogonal_(param)
                elif "bias" in name:
                    nn.init.constant_(param, 0.0)
            elif "lstm" in name:
                if "weight_ih" in name:
                    nn.init.orthogonal_(param)
                elif "weight_hh" in name:
                    nn.init.orthogonal_(param)
                elif "bias" in name:
                    # Initialize forget gate bias to 1.0
                    # PyTorch LSTM bias is [b_ii, b_if, b_ig, b_io]
                    # We want to set b_if to 1.0
                    n = param.shape[0]
                    start, end = n // 4, n // 2
                    nn.init.constant_(param, 0.0)
                    param.data[start:end].fill_(1.0)
            elif "logits" in name or "value" in name:
                if "weight" in name:
                    nn.init.orthogonal_(param, gain=0.01)
                elif "bias" in name:
                    nn.init.constant_(param, 0.0)
        
    @override(ModelV2)
    def get_initial_state(self):
        # Return initial hidden states for LSTM (h_0, c_0)
        # RLlib expects a list of numpy arrays or tensors? 
        # Actually RLlib expects list of values that match the shapes.
        # But usually we return list of zeros.
        return [
            torch.zeros(self.lstm_state_size),
            torch.zeros(self.lstm_state_size)
        ]

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        # input_dict["obs"] is the observation
        # Since we use Dict space, it should be a dictionary of tensors
        obs = input_dict["obs"]["observations"]
        action_mask = input_dict["obs"]["action_mask"]
        
        # Forward pass through FC layers
        x = torch.relu(self.fc1(obs))
        x = torch.relu(self.fc2(x))
        
        # LSTM requires [Batch, Seq, Feature]
        # RLlib passes flattened inputs [Batch * Seq, Feature] if seq_lens is None?
        # But when using LSTM, RLlib usually handles the time dimension.
        # We need to unsqueeze if necessary or rely on RLlib's view_requirements.
        # However, standard RLlib LSTM wrapper logic:
        
        # If we are in a trajectory, inputs are [Batch, Seq, Feature] ?
        # Actually, RLlib's RecurrentNetwork handles this. 
        # But here we are inheriting from TorchModelV2 directly.
        # We need to reshape x to [Batch, Seq, Feature]
        
        output, new_state = self.forward_rnn(x, state, seq_lens)
        
        logits = self.logits(output)
        
        # Apply Action Masking
        # Mask is 1 for legal, 0 for illegal
        # We want to set illegal logits to -inf
        inf_mask = torch.clamp(torch.log(action_mask), min=FLOAT_MIN)
        masked_logits = logits + inf_mask
        
        # Value function
        # We use the output of LSTM for value function as well
        # But we need to flatten it back if it was reshaped?
        # forward_rnn returns [Batch * Seq, Feature] usually?
        
        # Let's look at how forward_rnn is usually implemented or if we need to implement it manually.
        # Since we didn't inherit from RecurrentNetwork, we do it manually.
        
        self._value = self.value(output).squeeze(1)
        
        return masked_logits, new_state

    def forward_rnn(self, inputs, state, seq_lens):
        """
        Manually handle LSTM forward pass
        """
        # inputs: [Batch * Time, Features]
        # state: List of [Batch, Hidden]
        # seq_lens: [Batch]
        
        if seq_lens is not None:
            # We have a batch of sequences
            max_seq_len = inputs.shape[0] // seq_lens.shape[0]
            # Reshape to [Batch, Time, Features]
            # Note: This assumes inputs are padded/structured correctly by RLlib
            # But RLlib passes a flat batch. We need to use add_time_dimension
            from ray.rllib.policy.rnn_sequencing import add_time_dimension
            
            inputs_time_major = add_time_dimension(
                inputs,
                seq_lens=seq_lens,
                framework="torch",
                time_major=False,
            )
            
            # Unpack state
            h_0, c_0 = state
            # Add batch dimension to state if needed? 
            # LSTM expects (num_layers, batch, hidden_size)
            # state comes as [batch, hidden_size]
            h_0 = h_0.unsqueeze(0)
            c_0 = c_0.unsqueeze(0)
            
            output, (h_n, c_n) = self.lstm(inputs_time_major, (h_0, c_0))
            
            # Flatten output back to [Batch * Time, Features]
            output = torch.reshape(output, [-1, self.lstm_state_size])
            
            # Return new state (squeeze back to [Batch, Hidden])
            return output, [h_n.squeeze(0), c_n.squeeze(0)]
        else:
            # Single step (e.g. inference without time dimension explicitly passed as batch)
            # Or simple batch without seq_lens?
            # Usually seq_lens is provided if state is provided.
            
            # If seq_lens is None, it might be a simple forward pass (batch=1, time=1) or just batch
            # But for LSTM we need state.
            
            # Defensive coding: Check dimensions
            if inputs.dim() == 2:
                # [Batch, Features] -> [Batch, 1, Features]
                inputs = inputs.unsqueeze(1)
            
            # Ensure state is correct shape
            h_0, c_0 = state
            if h_0.dim() == 2:
                # [Batch, Hidden] -> [1, Batch, Hidden]
                h_0 = h_0.unsqueeze(0)
                c_0 = c_0.unsqueeze(0)
            
            output, (h_n, c_n) = self.lstm(inputs, (h_0, c_0))
            output = output.squeeze(1)
            
            return output, [h_n.squeeze(0), c_n.squeeze(0)]

    @override(ModelV2)
    def value_function(self):
        return self._value
