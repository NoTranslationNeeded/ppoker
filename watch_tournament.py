"""
Watch Tournament Poker Games
Visualize AI playing complete tournaments with blind escalation

Usage:
    python watch_tournament.py                                    # Random play
    python watch_tournament.py --checkpoint <path>               # Trained AI
    python watch_tournament.py --checkpoint <path> --num-games 5 # 5 tournaments
"""
import ray
from ray.rllib.algorithms.ppo import PPO
from tournament_poker_env import TournamentPokerEnv
import time
import argparse

def watch_tournament(checkpoint_path=None, num_games=3):
    """
    Watch AI play tournament poker games with detailed logging
    
    Args:
        checkpoint_path: Path to trained model (None = random actions)
        num_games: Number of tournaments to watch
    """
    
    # Initialize environment
    env = TournamentPokerEnv(randomize_stacks=True)
    
    # Load trained model if checkpoint provided
    # Load trained model if checkpoint provided
    policy = None
    if checkpoint_path:
        try:
            ray.init(ignore_reinit_error=True)
            # Try loading using Policy API first (more robust)
            from ray.rllib.policy import Policy
            try:
                policy = Policy.from_checkpoint(checkpoint_path)
                print(f"  Loaded policy from: {checkpoint_path}\n")
            except Exception as e1:
                print(f"  Policy.from_checkpoint failed: {e1}")
                # Fallback to Algorithm API
                algo = PPO.from_checkpoint(checkpoint_path)
                policy = algo.get_policy("player_0")
                print(f"  Loaded algorithm from: {checkpoint_path}\n")
        except Exception as e:
            print(f"[ERROR] Error loading checkpoint: {e}")
            print("  Make sure Ray versions match and path is correct.")
            return
    else:
        print("  Playing with random actions (no checkpoint loaded)\n")
    
    # Play tournaments
    for tournament_num in range(1, num_games + 1):
        print("=" * 80)
        print(f"  TOURNAMENT {tournament_num}")
        print("=" * 80)
        
        obs = env.reset()
        print(f"\n  Starting Conditions:")
        print(f"  Player 0: {env.chips[0]:.1f} chips ({env.chips[0]/2:.1f} BB)")
        print(f"  Player 1: {env.chips[1]:.1f} chips ({env.chips[1]/2:.1f} BB)")
        print(f"  Blinds: {env.small_blind}/{env.big_blind}")
        
        done = False
        hand_num = 0
        prev_blind_level = 0
        
        while not done:
            current_player = env.current_player
            legal_actions = env.get_legal_actions()
            
            # Display hand start
            if env.hand_count != hand_num:
                hand_num = env.hand_count
                
                # Check blind level change
                if env.current_blind_level != prev_blind_level:
                    print(f"\n{'^ BLIND LEVEL UP! ':=^80}")
                    print(f"  New blinds: {env.small_blind}/{env.big_blind}")
                    print(f"  Player 0: {env.chips[0]:.1f} chips ({env.chips[0]/env.big_blind:.1f} BB)")
                    print(f"  Player 1: {env.chips[1]:.1f} chips ({env.chips[1]/env.big_blind:.1f} BB)")
                    print("=" * 80)
                    prev_blind_level = env.current_blind_level
                
                print(f"\n--- Hand {hand_num} (Blind Level {env.current_blind_level + 1}: {env.small_blind}/{env.big_blind}) ---")
            
            # Choose action
            if policy and current_player == 0:
                # Use trained AI for Player 0
                action, _, _ = policy.compute_single_action(obs, explore=False)
            else:
                # Random action for Player 1 (or if no policy)
                import numpy as np
                action = np.random.choice(legal_actions)
            
            # Step environment
            obs, reward, done, info = env.step(action)
            
            # Check if hand ended
            if 'hand_ended' in info:
                payoffs = info['hand_payoffs']
                chips = info['current_chips']
                print(f"    Hand result: P0={payoffs[0]:+.1f}, P1={payoffs[1]:+.1f}")
                print(f"    Chip stacks: P0={chips[0]:.1f} ({chips[0]/env.big_blind:.1f}BB), P1={chips[1]:.1f} ({chips[1]/env.big_blind:.1f}BB)")
        
        # Tournament ended
        print("\n" + "=" * 80)
        print("  TOURNAMENT OVER")
        print(f"  Total hands played: {info['hands_played']}")
        print(f"  Final blind level: {env.current_blind_level + 1} ({env.small_blind}/{env.big_blind})")
        print(f"  Winner: Player {info['tournament_winner']}")
        print(f"\n  Final Results:")
        print(f"    Player 0: Started with {info['starting_chips'][0]:.1f}, ended with {info['final_chips'][0]:.1f} ({info['final_chips'][0]-info['starting_chips'][0]:+.1f})")
        print(f"    Player 1: Started with {info['starting_chips'][1]:.1f}, ended with {info['final_chips'][1]:.1f} ({info['final_chips'][1]-info['starting_chips'][1]:+.1f})")
        print("=" * 80 + "\n")
        time.sleep(2)
    
    if checkpoint_path:
        ray.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Watch AI poker tournaments")
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to checkpoint (optional, uses random if not provided)"
    )
    parser.add_argument(
        "--num-games",
        type=int,
        default=3,
        help="Number of tournaments to watch (default: 3)"
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print("  TOURNAMENT POKER WATCHER")
    print("=" * 80)
    
    watch_tournament(args.checkpoint, args.num_games)
