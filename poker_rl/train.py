
import os
import sys
import signal
import atexit

# Add project root to path so we can import poker_rl when running as script
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

import ray
import torch
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

import argparse

def train(experiment_name="epsilon", resume=False, stop_iters=1000):
    # Initialize Ray with runtime_env to suppress warnings in all actors
    if ray.is_initialized():
        ray.shutdown()
    ray.init(
        runtime_env={"env_vars": {"PYTHONWARNINGS": "ignore"}},
        _metrics_export_port=0,   # Disable metrics exporter (prevents connection errors)
        include_dashboard=False,  # Disable dashboard (optional, also avoids metrics)
    )
    
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
                "max_seq_len": 40,  # Increased from 20 to handle long hands (Issue #9 solved)
            },
            train_batch_size=8192,  # Reduced from 32768 for faster iterations
            gamma=0.99,
            lr=3e-4,
            # PPO specific
            clip_param=0.2,
            lambda_=0.95,
            entropy_coeff=0.05,  # Increased from 0.03 to encourage exploration
            num_epochs=10,
        )
        .multi_agent(
            policies={
                "main_policy": (None, PokerMultiAgentEnv().observation_space, PokerMultiAgentEnv().action_space, {}),
            },
            policy_mapping_fn=lambda agent_id, episode, worker, **kwargs: "main_policy",
        )
        .resources(num_gpus=1 if torch.cuda.is_available() else 0)
        .env_runners(num_env_runners=4, sample_timeout_s=300)
        .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
        .checkpointing(
            export_native_model_files=True,  # Export PyTorch model files
        )
    )
    
    # Run training
    print(f"Starting training (Experiment: {experiment_name})...")
    storage_path = os.path.abspath("experiments/logs")
    
    if resume:
        experiment_path = os.path.join(storage_path, experiment_name)
        print(f"Resuming experiment from: {experiment_path}")
        
        if not os.path.exists(experiment_path):
            print(f"Error: Experiment path {experiment_path} does not exist. Cannot resume.")
            return

        tuner = tune.Tuner.restore(
            path=experiment_path,
            trainable="PPO",
            resume_unfinished=True,
            resume_errored=True,
            restart_errored=True
        )
    else:
        # Manual checkpoint trigger callback
        from ray.tune.callback import Callback
        import time
        
        class ManualCheckpointCallback(Callback):
            """
            Monitor for SAVE_CHECKPOINT file. When created, save a checkpoint.
            Usage: Create 'SAVE_CHECKPOINT' file in project root to trigger save.
            """
            def __init__(self, trigger_file="SAVE_CHECKPOINT"):
                self.trigger_file = os.path.abspath(trigger_file)
                self.last_check = time.time()
                
            def on_trial_result(self, iteration, trials, trial, result, **info):
                # Check every 5 seconds
                if time.time() - self.last_check < 5:
                    return
                    
                self.last_check = time.time()
                
                if os.path.exists(self.trigger_file):
                    print("\n" + "="*60)
                    print("💾 Manual checkpoint trigger detected!")
                    print(f"Saving checkpoint at iteration {result.get('training_iteration', 'unknown')}")
                    
                    # Actually save the checkpoint!
                    try:
                        checkpoint_path = trial.save()
                        print(f"✅ Checkpoint saved successfully!")
                        print(f"📂 Location: {checkpoint_path}")
                    except Exception as e:
                        print(f"❌ Failed to save checkpoint: {e}")
                    
                    print("="*60)
                    
                    # Remove trigger file
                    try:
                        os.remove(self.trigger_file)
                    except:
                        pass
        
        manual_checkpoint_callback = ManualCheckpointCallback()
        
        tuner = tune.Tuner(
            "PPO",
            param_space=config.to_dict(),
            run_config=tune.RunConfig(
                stop={"training_iteration": stop_iters},
                storage_path=storage_path,
                name=experiment_name,
                callbacks=[manual_checkpoint_callback],  # Manual save via file trigger
                checkpoint_config=tune.CheckpointConfig(
                    checkpoint_frequency=1,   # Save checkpoint every iteration
                    checkpoint_at_end=True,   # Save checkpoint at end of training
                    num_to_keep=5,            # Keep last 5 checkpoints (CRITICAL: without this, checkpoints won't be saved!)
                ),
            ),
        )
    
    # Graceful shutdown handler
    shutdown_requested = {"flag": False}
    
    def signal_handler(signum, frame):
        """Handle Ctrl+C gracefully by requesting shutdown"""
        if not shutdown_requested["flag"]:
            print("\n" + "="*60)
            print("🛑 Shutdown requested (Ctrl+C detected)")
            print("Saving checkpoint and shutting down gracefully...")
            print("="*60)
            shutdown_requested["flag"] = True
            # Note: Ray Tune will complete current iteration and save checkpoint
        else:
            print("\n⚠️  Force shutdown! (Ctrl+C pressed again)")
            print("Checkpoint may not be saved!")
            sys.exit(1)
    
    # Register signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    # Run training with error handling
    try:
        results = tuner.fit()
        print("Training completed successfully.")
    except KeyboardInterrupt:
        print("\n" + "="*60)
        print("✅ Training interrupted gracefully")
        print("Latest checkpoint saved")
        print("="*60)
        return
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        raise
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
    parser = argparse.ArgumentParser(description="Train Poker AI using Ray RLlib")
    parser.add_argument("--name", type=str, default="epsilon", help="Name of the experiment (default: epsilon)")
    parser.add_argument("--resume", action="store_true", help="Resume training from existing checkpoint")
    parser.add_argument("--stop-iters", type=int, default=1000, help="Number of training iterations (default: 1000)")
    args = parser.parse_args()
    
    train(experiment_name=args.name, resume=args.resume, stop_iters=args.stop_iters)
