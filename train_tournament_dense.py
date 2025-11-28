"""
Tournament Poker Training with ICM + Survival Reward
Uses advanced reward functions for better tournament strategy learning
"""
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
from ray.air.integrations.mlflow import MLflowLoggerCallback
import os
import mlflow
from gymnasium import spaces
import numpy as np

# Import our custom environment and model
from tournament_pettingzoo import TournamentPokerParallelEnv
from transformer_model import TransformerPokerModel
from tournament_logger import TournamentLoggingCallback

# Register custom Transformer model
ModelCatalog.register_custom_model("transformer_poker", TransformerPokerModel)

def env_creator(config):
    """
    Create tournament poker environment with Pure Dense rewards
    """
    env = TournamentPokerParallelEnv(
        starting_chips=100,
        randomize_stacks=True
    )
    return env

# Register environment
register_env("tournament_poker_dense", lambda config: ParallelPettingZooEnv(env_creator(config)))


def train_tournament_dense():
    # Initialize Ray
    ray.init(ignore_reinit_error=True)
    
    # Define spaces - 60 dimensions with blind level
    obs_space = spaces.Box(low=0, high=1, shape=(60,), dtype=np.float32)
    act_space = spaces.Discrete(7)  # 7 actions: Fold, Check/Call, 33%, 75%, 100%, 150%, All-in
    
    # MLflow setup
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("deepstack-7actions-dense-v2-fair")

    # Configure PPO with Transformer
    config = (
        PPOConfig()
        .environment(
            "tournament_poker_dense",
            env_config={}
        )
        .framework("torch")
        .env_runners(num_env_runners=4)  # 4 runners for faster data collection
        .training(
            model={
                "custom_model": "transformer_poker",
                "custom_model_config": {
                    "d_model": 128,
                    "nhead": 8,
                    "num_layers": 4,
                    "dim_feedforward": 512,
                    "dropout": 0.1,
                    "max_seq_len": 20,
                },
                "max_seq_len": 20,
            },
            gamma=0.99,
            lambda_=0.95,              # GAE lambda for advantage estimation
            clip_param=0.2,            # PPO clip range (conservative for Dense Reward)
            lr=0.0003,
            train_batch_size=4000,     # Optimized for Dense Reward (was 8000)
            # sgd_minibatch_size=256,  # Removed due to TypeError in Ray 2.52
            num_sgd_iter=10,
            entropy_coeff=0.02,        # Increased exploration for 7-action space (was 0.01)
        )
        .multi_agent(
            policies={
                "player_0": (None, obs_space, act_space, {"lr": 0.0003}),
                "player_1": (None, obs_space, act_space, {"lr": 0.0003}),
            },
            policy_mapping_fn=lambda agent_id, *args, **kwargs: agent_id,
            policies_to_train=["player_0", "player_1"],
        )
        .callbacks(TournamentLoggingCallback)
        .experimental(_validate_config=False)
    )
    
    # Training configuration
    stop = {
        "timesteps_total": 1_000_000,  # 1 Million timesteps (~10-15 min)
    }
    
    print("=" * 80)
    print(" Training Tournament Poker AI with PURE DENSE REWARD")
    print("=" * 80)
    print("")
    print("Architecture:")
    print("   Model: Transformer Encoder")
    print("     - d_model: 128")
    print("     - Attention Heads: 8")
    print("     - Encoder Layers: 4")
    print("")
    print("Reward Function:")
    print("   Pure Dense Reward: (chip_payoff / BB) / 250")
    print("   Applied at: Every hand end (including tournament end)")
    print("   Range: -1.0 to +1.0 (normalized)")
    print("")
    print("Why Pure Dense Reward:")
    print("   * Immediate feedback for every decision")
    print("   * Chip EV maximization = optimal poker")
    print("   * No artificial incentives")
    print("   * Consistent with pro poker strategy")
    print("   * Simple, clear learning signal")
    print("")
    print("Environment:")
    print("   * ompeval C++ Monte Carlo Equity (Ultra-Fast)")
    print("   * Bitmask Optimization")
    print("=" * 80)
    
    # Run training
    results = tune.run(
        "PPO",
        name="deepstack_7actions_dense_v2_fair",
        config=config.to_dict(),
        stop=stop,
        checkpoint_freq=10,
        storage_path=os.path.abspath("./ray_results"),
        verbose=1,
        callbacks=[
            MLflowLoggerCallback(
                tracking_uri="file:./mlruns",
                experiment_name="deepstack-7actions-dense-v2-fair",
                save_artifact=True,
            ),
        ],
    )
    
    print("")
    print("=" * 80)
    print(" Training complete!")
    print("")
    print("View results:")
    print("  TensorBoard: tensorboard --logdir=./ray_results")
    print("  MLflow UI:   mlflow ui")
    print("")
    print("Watch games:")
    print("  python watch_tournament.py --checkpoint <path_to_checkpoint>")
    print("")
    print("=" * 80)

    ray.shutdown()

if __name__ == "__main__":
    train_tournament_icm()
