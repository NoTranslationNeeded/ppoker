from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.env.single_agent_episode import SingleAgentEpisode
from ray.rllib.policy import Policy
from typing import Dict, Optional, Union
import numpy as np
import logging
from logging.handlers import RotatingFileHandler
import os

# Configure dedicated game logger
def setup_game_logger():
    """Setup a separate logger for poker game logs only"""
    logger = logging.getLogger('poker_games')
    logger.setLevel(logging.INFO)
    logger.propagate = False  # Don't propagate to Ray's root logger
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Create rotating file handler (10MB max, keep 3 backup files)
    handler = RotatingFileHandler(
        'poker_games.log',
        maxBytes=10*1024*1024,  # 10MB
        backupCount=3,
        encoding='utf-8'
    )
    
    # Simple formatter - we'll handle formatting in the code
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    
    logger.addHandler(handler)
    return logger

# Initialize game logger
game_logger = setup_game_logger()


class TournamentLoggingCallback(DefaultCallbacks):
    """
    Custom callback that logs detailed tournament information
    for the first episode of each training iteration
    """
    
    def __init__(self):
        super().__init__()
        self.episode_count = 0
        self.current_iteration = 0
        self.iteration_first_episode = None  # Track first episode of current iteration
        self.pending_log = None  # Store episode data for logging
        
    def on_episode_start(
        self,
        *,
        episode: Union[EpisodeV2, SingleAgentEpisode],
        **kwargs
    ):
        """Called at the start of each episode (tournament)"""
        # Initialize tournament log
        # Use custom_data for SingleAgentEpisode compatibility
        data = episode.custom_data if hasattr(episode, "custom_data") else episode.user_data
        
        data["tournament_log"] = []
        data["hand_count"] = 0
        data["starting_chips"] = None
        data["blind_level_changes"] = []
        
    def on_episode_step(
        self,
        *,
        episode: Union[EpisodeV2, SingleAgentEpisode],
        **kwargs
    ):
        """Called at each step"""
        # Get info from last step
        # SingleAgentEpisode uses get_infos() or similar, but let's try last_info_for first
        # If it's SingleAgentEpisode, it might not have last_info_for
        if hasattr(episode, "last_info_for"):
            info = episode.last_info_for()
        elif hasattr(episode, "get_infos"):
            # For SingleAgentEpisode, get the last info
            try:
                info = episode.get_infos(-1)
            except (IndexError, KeyError):
                info = {}
        else:
            info = {}
        
        if info:
            data = episode.custom_data if hasattr(episode, "custom_data") else episode.user_data
            
            # Record starting chips on first step
            if data["starting_chips"] is None and "starting_chips" in info:
                data["starting_chips"] = info["starting_chips"]
            
            # Record hand endings
            if "hand_ended" in info:
                hand_log = {
                    "hand_num": data["hand_count"] + 1,
                    "payoffs": info.get("hand_payoffs", [0, 0]),
                    "chips": info.get("current_chips", [0, 0]),
                    "blind_level": info.get("blind_level", 0),
                    "cards": info.get("cards", {})  # Store card information
                }
                data["tournament_log"].append(hand_log)
                data["hand_count"] += 1
    
    def on_episode_end(
        self,
        *,
        episode: Union[EpisodeV2, SingleAgentEpisode],
        metrics_logger: Optional["MetricsLogger"] = None,
        **kwargs
    ):
        """Called at the end of each episode (tournament)"""
        self.episode_count += 1
        
        # Get final tournament info
        if hasattr(episode, "last_info_for"):
            info = episode.last_info_for()
        elif hasattr(episode, "get_infos"):
            try:
                info = episode.get_infos(-1)
            except (IndexError, KeyError):
                info = {}
        else:
            info = {}
        
        # Log custom metrics
        if info:
            metrics = {
                "hands_played": info.get("hands_played", 0),
                "tournament_winner": info.get("tournament_winner", 0),
            }
            if "final_chips" in info:
                metrics["final_chips_p0"] = info["final_chips"][0]
                metrics["final_chips_p1"] = info["final_chips"][1]
            
            # Use MetricsLogger if available (New API)
            if metrics_logger:
                for key, value in metrics.items():
                    metrics_logger.log_value(key, value)
            # Fallback to custom_metrics (Old API)
            elif hasattr(episode, "custom_metrics"):
                for key, value in metrics.items():
                    episode.custom_metrics[key] = value
        
        # Store first episode of iteration for detailed logging
        if self.iteration_first_episode is None:
            self.iteration_first_episode = self.episode_count
            self.pending_log = {
                'episode': episode,
                'info': info,
                'episode_number': self.episode_count
            }
    
    def _print_detailed_tournament_log(self, episode: Union[EpisodeV2, SingleAgentEpisode], final_info: Dict):
        """Print detailed tournament progression to poker_games.log"""
        
        # Get tournament data
        data = episode.custom_data if hasattr(episode, "custom_data") else episode.user_data
        starting_chips = data.get("starting_chips", [100, 100])
        tournament_log = data.get("tournament_log", [])
        
        if not tournament_log:
            return  # No hands to log
        
        # Header
        episode_num = self.pending_log['episode_number'] if self.pending_log else "?"
        iteration_num = self.current_iteration
        
        game_logger.info("=" * 80)
        game_logger.info(f"ITERATION {iteration_num} | Episode {episode_num} | Tournament Start")
        game_logger.info("=" * 80)
        game_logger.info(f"Starting Chips: P0={starting_chips[0]}, P1={starting_chips[1]}")
        
        # Get initial blind level
        if tournament_log:
            first_blind = tournament_log[0].get("blind_level", 0)
            blind_levels_list = [(1,2), (2,4), (3,6), (5,10), (10,20), (15,30), (25,50), (50,100)]
            if first_blind < len(blind_levels_list):
                sb, bb = blind_levels_list[first_blind]
                game_logger.info(f"Initial Blinds: {sb}/{bb}")
        game_logger.info("")
        
        # Log each hand
        prev_blind_level = -1
        for hand in tournament_log:
            hand_num = hand["hand_num"]
            payoffs = hand["payoffs"]
            chips = hand["chips"]
            blind_level = hand["blind_level"]
            cards = hand.get("cards", {})
            
            # Calculate blind values
            blind_levels_list = [(1,2), (2,4), (3,6), (5,10), (10,20), (15,30), (25,50), (50,100)]
            if blind_level < len(blind_levels_list):
                sb, bb = blind_levels_list[blind_level]
                blind_str = f"{sb}/{bb}"
            else:
                blind_str = "N/A"
            
            # Check for blind level increase
            if blind_level != prev_blind_level and prev_blind_level != -1:
                game_logger.info("─" * 80)
                game_logger.info(f"[!] BLIND LEVEL INCREASED TO {blind_str}")
                game_logger.info("─" * 80)
            prev_blind_level = blind_level
            
            # Hand header
            game_logger.info("─" * 80)
            game_logger.info(f"Hand #{hand_num} | Blinds: {blind_str}")
            game_logger.info("─" * 80)
            
            # Card information
            p0_hand = cards.get('player_0_hand', [])
            p1_hand = cards.get('player_1_hand', [])
            community = cards.get('community', [])
            
            if p0_hand:
                game_logger.info(f"  P0 Cards: [{' '.join(p0_hand)}]")
            if p1_hand:
                game_logger.info(f"  P1 Cards: [{' '.join(p1_hand)}]")
            if community:
                game_logger.info(f"  Community: [{' '.join(community)}]")
            
            # Result
            game_logger.info("")
            
            # Determine winner
            if payoffs[0] > 0:
                winner_str = "P0 wins"
            elif payoffs[1] > 0:
                winner_str = "P1 wins"
            else:
                winner_str = "Split pot"
            
            pot_size = abs(payoffs[0]) + abs(payoffs[1])
            game_logger.info(f"  Result: {winner_str} (pot: {pot_size:.1f})")
            game_logger.info(f"  Payoffs: P0=({payoffs[0]:+.1f}), P1=({payoffs[1]:+.1f})")
            game_logger.info(f"  Chips After: P0={chips[0]:.1f}, P1={chips[1]:.1f}")
            game_logger.info("")
        
        # Final results
        if final_info:
            game_logger.info("=" * 80)
            game_logger.info("Tournament Result")
            game_logger.info("=" * 80)
            
            winner = final_info.get("tournament_winner", 0)
            hands_played = final_info.get("hands_played", 0)
            final_chips = final_info.get("final_chips", [0, 0])
            
            game_logger.info(f"  Duration: {hands_played} hands")
            game_logger.info(f"  Winner: Player {winner}")
            game_logger.info(f"  Final Chips: P0={final_chips[0]:.1f}, P1={final_chips[1]:.1f}")
            game_logger.info(f"  Chip Differential: P0=({final_chips[0] - starting_chips[0]:+.1f}), "
                           f"P1=({final_chips[1] - starting_chips[1]:+.1f})")
            game_logger.info("=" * 80)
            game_logger.info("")  # Empty line for separation
        
        # Also print brief console summary
        print(f"\n{'='*80}")
        print(f">>> ITERATION {iteration_num} - Tournament Logged (Episode {episode_num})")
        print(f"{'='*80}")
        if final_info:
            winner = final_info.get("tournament_winner", 0)
            hands_played = final_info.get("hands_played", 0)
            final_chips = final_info.get("final_chips", [0, 0])
            print(f"  * {hands_played} hands played")
            print(f"  * Winner: Player {winner}")
            print(f"  * Final chips: P0={final_chips[0]:.1f}, P1={final_chips[1]:.1f}")
            print(f"  * Detailed log written to: poker_games.log")
        print(f"{'='*80}\n")
    
    def on_train_result(
        self,
        *,
        algorithm,
        result: dict,
        **kwargs
    ):
        """Called after each training iteration"""
        iteration = result.get("training_iteration", 0)
        self.current_iteration = iteration
        
        # Print detailed log for first tournament of this iteration
        if self.pending_log is not None:
            print(f"\n{'='*100}")
            print(f">>> ITERATION {iteration} - First Tournament (Episode {self.pending_log['episode_number']})")
            print(f"{'='*100}")
            self._print_detailed_tournament_log(
                self.pending_log['episode'],
                self.pending_log['info']
            )
            # Reset for next iteration
            self.pending_log = None
            self.iteration_first_episode = None
        
        # Print iteration summary
        print(f"\n{'='*100}")
        print(f"--- Training Iteration {iteration} Summary")
        print(f"{'='*100}")
        print(f"Episodes this iteration: {result.get('episodes_this_iter', 0)}")
        print(f"Mean reward: {result.get('episode_reward_mean', 0):.2f}")
        print(f"Mean episode length: {result.get('episode_len_mean', 0):.1f} steps")
        
        # Try to get policy loss
        learner_info = result.get('info', {}).get('learner', {})
        if learner_info:
            for policy_id, policy_info in learner_info.items():
                if 'learner_stats' in policy_info:
                    stats = policy_info['learner_stats']
                    if 'policy_loss' in stats:
                        print(f"Policy loss: {stats['policy_loss']:.4f}")
                        break
        
        # Custom metrics if available
        custom = result.get('custom_metrics', {})
        if custom:
            print(f"\nCustom Metrics:")
            if 'hands_played_mean' in custom:
                print(f"  Avg hands per tournament: {custom['hands_played_mean']:.1f}")
            if 'tournament_winner_mean' in custom:
                win_rate = custom['tournament_winner_mean']
                print(f"  Player 0 win rate: {(1-win_rate)*100:.1f}%")
                print(f"  Player 1 win rate: {win_rate*100:.1f}%")
        
        print(f"{'='*100}\n")


if __name__ == "__main__":
    print("Tournament Logging Callback")
    print("This module provides detailed logging for first tournament of each iteration")
    print("\nUsage:")
    print("  callbacks = TournamentLoggingCallback()")
    print("  config.callbacks(TournamentLoggingCallback)")
