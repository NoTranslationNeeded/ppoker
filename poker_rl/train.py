
import os
import sys

# Add project root to path so we can import poker_rl when running as script
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

import ray
import numpy as np
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env

from poker_rl.env import PokerMultiAgentEnv
from poker_rl.models.masked_mlp import MaskedMLP
from poker_rl.models.masked_lstm import MaskedLSTM

def env_creator(config):
    # Suppress numpy overflow warnings in worker processes (RLlib internal issue)
    import numpy as np
    import warnings
    np.seterr(all='ignore')
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    return PokerMultiAgentEnv(config)

def train():
    # Initialize Ray with runtime_env to suppress warnings in all actors
    ray.init(runtime_env={"env_vars": {"PYTHONWARNINGS": "ignore"}})
    
    # Suppress warnings in driver
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    
    # Register Environment
    register_env("poker_env", env_creator)
    
    # Register Model
    ModelCatalog.register_custom_model("masked_mlp", MaskedMLP)
    ModelCatalog.register_custom_model("masked_lstm", MaskedLSTM)
    
    # Configuration
    config = (
        PPOConfig()
        .environment("poker_env")
        .framework("torch")
        .training(
            model={
                "custom_model": "masked_lstm",
                "custom_model_config": {
                    "lstm_cell_size": 256,
                },
                "max_seq_len": 20, # Length of history to train on
            },
            train_batch_size=32768, # Increased to >30000 as requested
            gamma=0.99,
            lr=3e-4,
            # PPO specific
            clip_param=0.2,
            lambda_=0.95,
            entropy_coeff=0.01,
            num_epochs=10,
        )
        .multi_agent(
            policies={
                "main_policy": (None, PokerMultiAgentEnv().observation_space, PokerMultiAgentEnv().action_space, {}),
            },
            policy_mapping_fn=lambda agent_id, episode, worker, **kwargs: "main_policy",
        )
        .resources(num_gpus=1) # Set to 1 if GPU available
        .env_runners(num_env_runners=4)
        .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
    )
    
    # Run training
    print("Starting training...")
    tuner = tune.Tuner(
        "PPO",
        param_space=config.to_dict(),
        run_config=tune.RunConfig(
            stop={"training_iteration": 1000}, # Run for 1000 iterations
            storage_path=os.path.abspath("experiments/logs"),
            name="poker_ppo_v2_fixed",
        ),
    )
    
    results = tuner.fit()
    print("Training completed.")
    
    # Print results
    best_result = results.get_best_result(metric="episode_reward_mean", mode="max")
    print("Available metrics:", best_result.metrics.keys())
    if "episode_reward_mean" in best_result.metrics:
        print(f"Best result: {best_result.metrics['episode_reward_mean']}")
    elif "env_runners/episode_reward_mean" in best_result.metrics:
        print(f"Best result: {best_result.metrics['env_runners/episode_reward_mean']}")
    else:
        print("Could not find episode_reward_mean in metrics.")

if __name__ == "__main__":
    train()
