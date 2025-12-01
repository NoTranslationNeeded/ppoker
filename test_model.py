
import torch
from gymnasium import spaces
from poker_rl.models.masked_lstm import MaskedLSTM

def test_lstm():
    print("Testing MaskedLSTM...")
    
    # Mock spaces
    obs_space = spaces.Dict({
        "observations": spaces.Box(low=0, high=1, shape=(150,)),
        "action_mask": spaces.Box(low=0, high=1, shape=(7,))
    })
    action_space = spaces.Discrete(7)
    
    model_config = {
        "lstm_cell_size": 256
    }
    
    model = MaskedLSTM(obs_space, action_space, 7, model_config, "test_model")
    print("Model initialized successfully.")
    
    # Test forward pass
    # RLlib passes flattened inputs [Batch, Features]
    # But for LSTM we need to handle it.
    
    # Case 1: Single step (inference)
    obs = torch.randn(1, 150)
    mask = torch.ones(1, 7)
    input_dict = {
        "obs": {
            "observations": obs,
            "action_mask": mask
        }
    }
    state = model.get_initial_state()
    # Expand state to batch size 1
    state = [s.unsqueeze(0) for s in state]
    seq_lens = torch.tensor([1])
    
    print("Running forward pass (single step)...")
    output, new_state = model.forward(input_dict, state, seq_lens)
    print("Output shape:", output.shape)
    print("New state shapes:", [s.shape for s in new_state])
    
    # Case 2: Batch with time dimension (training)
    # Batch=2, Time=5 -> Total 10
    print("Running forward pass (batch sequence)...")
    obs = torch.randn(10, 150)
    mask = torch.ones(10, 7)
    input_dict = {
        "obs": {
            "observations": obs,
            "action_mask": mask
        }
    }
    state = model.get_initial_state()
    # Expand state to batch size 2
    state = [s.repeat(2, 1) for s in state]
    seq_lens = torch.tensor([5, 5])
    
    output, new_state = model.forward(input_dict, state, seq_lens)
    print("Output shape:", output.shape)
    print("New state shapes:", [s.shape for s in new_state])

if __name__ == "__main__":
    test_lstm()
