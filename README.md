# Texas Hold'em AI Training Guide

## 🚀 Quick Start

### 1. Start Training
Run the batch script to start training with the correct Python environment:
```cmd
run_training.bat
```
This script uses the specific Python 3.10 environment where all dependencies (`ray`, `ompeval`, `rlcard`) are installed.

### 2. Monitor Progress
Open TensorBoard to view training metrics:
```cmd
tensorboard --logdir=./ray_results --port 6006
```
Then open your browser to [http://localhost:6006](http://localhost:6006).

### 3. Watch Games
To watch the AI play after training (or during training using a checkpoint):
```cmd
python watch_tournament.py --checkpoint ray_results/deepstack_7actions_dense_v3_ompeval/.../checkpoint_000010
```

## 🛠️ Troubleshooting

### "Python not found" or "App Execution Alias"
If you see errors about Python not being found or opening the Microsoft Store, it's because the default `python` command is broken.
**Solution:** Always use `run_training.bat` or explicitly use the full path:
`C:\Users\99san\AppData\Local\Programs\Python\Python310\python.exe`

### "AttributeError: action_space_size"
Ensure `tournament_poker_env.py` has `self.action_space_size = 7` in `__init__`.

### "ValueError: X is not a valid Action"
This means the Agent Action (0-6) was passed directly to RLCard. Ensure `tournament_poker_env.py`'s `step()` method maps the action back to RLCard format.
