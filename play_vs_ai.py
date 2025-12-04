
import os
import sys
import argparse
import numpy as np
import ray
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.policy import Policy

# Add POKERENGINE to path
current_dir = os.path.dirname(os.path.abspath(__file__))
poker_engine_path = os.path.join(current_dir, 'POKERENGINE')
sys.path.append(poker_engine_path)

try:
    from poker_engine import PokerGame, Action, ActionType, Card
except ImportError:
    print("Error: Could not import poker_engine. Make sure POKERENGINE folder exists.")
    sys.exit(1)

# Import optimized modules
from poker_rl.utils.obs_builder import ObservationBuilder
from poker_rl.models.masked_lstm import MaskedLSTM
from poker_rl.models.masked_mlp import MaskedMLP

# Register custom models
ModelCatalog.register_custom_model("masked_lstm", MaskedLSTM)
ModelCatalog.register_custom_model("masked_mlp", MaskedMLP)

def get_action_name(action_idx, amount=0):
    if action_idx == 0: return "Fold"
    if action_idx == 1: return "Check/Call"
    if action_idx == 2: return "Min Raise"
    if action_idx == 13: return "All-in"
    
    pcts = [10, 25, 33, 50, 75, 100, 125, 150, 200, 300]
    if 3 <= action_idx <= 12:
        return f"Bet {pcts[action_idx-3]}%"
    return "Unknown"

def print_game_state(game, player_id, hole_cards):
    print("\n" + "="*50)
    print(f"Community Cards: {[str(c) for c in game.community_cards]}")
    print(f"Pot: {game.get_pot_size():.1f} (Current Bet: {game.current_bet:.1f})")
    print("-" * 50)
    
    p0 = game.players[0]
    p1 = game.players[1]
    
    # Show opponent info (hidden cards)
    opp_id = 1 - player_id
    opp = game.players[opp_id]
    print(f"Opponent (P{opp_id}): Chips {opp.chips:.1f} | Bet {opp.bet_this_round:.1f}")
    
    # Show my info
    me = game.players[player_id]
    print(f"YOU (P{player_id}):      Chips {me.chips:.1f} | Bet {me.bet_this_round:.1f}")
    print(f"Hole Cards: {[str(c) for c in hole_cards]}")
    print("="*50 + "\n")

def get_human_action(game, player_id):
    legal_mask = ObservationBuilder._get_legal_actions_mask(game, player_id)
    legal_indices = np.where(legal_mask == 1)[0]
    
    print("Available Actions:")
    for idx in legal_indices:
        print(f"[{idx}] {get_action_name(idx)}")
    
    while True:
        try:
            choice = input("Enter action index: ")
            action_idx = int(choice)
            if action_idx in legal_indices:
                return action_idx
            else:
                print("Invalid action index. Try again.")
        except ValueError:
            print("Please enter a number.")

def map_action(game, action_idx, player_id):
    # Logic copied from env_fast.py
    player = game.players[player_id]
    pot = game.get_pot_size()
    to_call = game.current_bet - player.bet_this_round
    
    if action_idx == 0:
        return Action.fold()
    elif action_idx == 1:
        return Action.check() if to_call == 0 else Action.call(to_call)
    elif action_idx == 13:
        return Action.all_in(player.chips)
    elif action_idx == 2:
        if game.current_bet > 0:
            target = game.current_bet + game.min_raise + 1e-5
            max_bet = player.chips + player.bet_this_round
            if target >= max_bet: return Action.all_in(player.chips)
            return Action.raise_to(target)
        else:
            return Action.check()
    else:
        pcts = [0.10, 0.25, 0.33, 0.50, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]
        pct = pcts[action_idx - 3]
        bet_amount = pot * pct
        
        if game.current_bet > 0:
            target = game.current_bet + bet_amount
            min_raise_target = game.current_bet + game.min_raise + 1e-5
            target = max(target, min_raise_target)
            max_bet = player.chips + player.bet_this_round
            if target >= max_bet: return Action.all_in(player.chips)
            return Action.raise_to(target)
        else:
            bet_amount = max(bet_amount, game.big_blind)
            if bet_amount >= player.chips: return Action.all_in(player.chips)
            return Action.bet(bet_amount)

def play(checkpoint_path, sb=50.0, bb=100.0, chips=10000.0):
    # Initialize Ray
    ray.init(ignore_reinit_error=True, log_to_driver=False)
    
    print(f"Loading checkpoint from {checkpoint_path}...")
    # Load the trained algorithm
    # Note: We use Algorithm.from_checkpoint which restores the Policy
    algo = Algorithm.from_checkpoint(checkpoint_path)
    policy = algo.get_policy("main_policy")
    print("Model loaded successfully!")
    
    # Game Setup
    stack = chips # Fixed stack for play mode
    
    while True:
        print("\n" + "*"*30)
        print("STARTING NEW HAND")
        print("*"*30)
        
        # Random button
        button = np.random.randint(0, 2)
        human_id = 0 # Human is always P0 for simplicity in display, but we can swap roles
        # Actually let's fix Human as P0 and AI as P1, but swap button
        
        game = PokerGame(small_blind=sb, big_blind=bb)
        game.start_hand(players_info=[(0, stack), (1, stack)], button=button)
        
        action_history = {'preflop': [], 'flop': [], 'turn': [], 'river': []}
        start_stacks = [stack, stack]
        
        # LSTM State (if needed)
        # We need to maintain internal state for the AI if it uses LSTM
        # Initial state is usually zeros
        ai_state = policy.get_initial_state()
        
        while not game.is_hand_over:
            current_player = game.get_current_player()
            
            # Print state for Human
            if current_player == human_id:
                print_game_state(game, human_id, game.players[human_id].hand)
                action_idx = get_human_action(game, human_id)
            else:
                # AI Turn
                print(f"\nAI (P{current_player}) is thinking...")
                
                # Build observation
                obs_dict = ObservationBuilder.get_observation(
                    game, current_player, action_history, start_stacks
                )
                obs = obs_dict["observations"]
                mask = obs_dict["action_mask"]
                
                # Compute Action
                # compute_single_action returns (action, state_out, info)
                # We need to pass state if LSTM
                action_idx, ai_state, _ = policy.compute_single_action(
                    obs, 
                    state=ai_state, 
                    explore=False # Deterministic play (or True for some randomness)
                )
                
                # Check legality (just in case)
                if mask[action_idx] == 0:
                    print(f"AI attempted illegal action {action_idx}. Fallback to Check/Fold.")
                    action_idx = 1 if mask[1] == 1 else 0
                
                print(f"AI chose: {get_action_name(action_idx)}")
            
            # Execute Action
            engine_action = map_action(game, action_idx, current_player)
            
            # Record history
            pot_before = game.get_pot_size()
            street = game.street.value
            bet_amount = engine_action.amount if engine_action.amount > 0 else 0
            if pot_before > 0: ratio = bet_amount / pot_before
            else: ratio = 0.0
            
            if street in action_history:
                action_history[street].append((action_idx, current_player, ratio, bet_amount))
            
            game.process_action(current_player, engine_action)
            
        # Hand Over
        print("\n=== HAND OVER ===")
        print(f"Community: {[str(c) for c in game.community_cards]}")
        print(f"My Hand: {[str(c) for c in game.players[human_id].hand]}")
        print(f"AI Hand: {[str(c) for c in game.players[1].hand]}")
        print(f"Pot: {game.get_pot_size()}")
        print(f"Winner: {game.winners}")
        
        # Update stacks (simplified, just reset for now or track?)
        # Let's track
        p0_chips = game.players[0].chips
        p1_chips = game.players[1].chips
        print(f"Resulting Stacks: Me {p0_chips}, AI {p1_chips}")
        
        if input("\nPlay again? (y/n): ").lower() != 'y':
            break

def find_latest_checkpoint(experiment_name=None):
    base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiments", "logs")
    
    if not os.path.exists(base_path):
        print(f"Error: Log directory not found at {base_path}")
        return None

    # Determine search paths
    if experiment_name:
        search_paths = [os.path.join(base_path, experiment_name)]
    else:
        # Search all experiment folders
        search_paths = [os.path.join(base_path, d) for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]

    candidates = []
    
    for exp_dir in search_paths:
        if not os.path.exists(exp_dir): continue
        
        # Find PPO run directories
        ppo_dirs = [os.path.join(exp_dir, d) for d in os.listdir(exp_dir) if d.startswith("PPO_poker_env")]
        
        for ppo_dir in ppo_dirs:
            if not os.path.isdir(ppo_dir): continue
            
            # Find checkpoints
            checkpoints = [os.path.join(ppo_dir, d) for d in os.listdir(ppo_dir) if d.startswith("checkpoint_")]
            
            for ckpt in checkpoints:
                if os.path.isdir(ckpt):
                    # Get modification time
                    mtime = os.path.getmtime(ckpt)
                    candidates.append((mtime, ckpt))
    
    if not candidates:
        return None
    
    # Sort by time descending
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]

def list_experiments(base_path):
    if not os.path.exists(base_path):
        return []
    return [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]

def select_experiment_interactively():
    base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiments", "logs")
    experiments = list_experiments(base_path)
    
    if not experiments:
        print("No experiments found in experiments/logs.")
        sys.exit(1)
        
    print("\nAvailable Experiments:")
    for i, exp in enumerate(experiments):
        print(f"[{i}] {exp}")
        
    while True:
        try:
            choice = input("\nSelect experiment index: ")
            idx = int(choice)
            if 0 <= idx < len(experiments):
                return experiments[idx]
            print("Invalid index.")
        except ValueError:
            print("Please enter a number.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play Poker against AI")
    parser.add_argument("--model", type=str, default=None, help="Experiment name (e.g., 'eta-f') or full path to checkpoint. If omitted, asks interactively.")
    parser.add_argument("--sb", type=float, default=50.0, help="Small Blind amount (default: 50.0)")
    parser.add_argument("--bb", type=float, default=100.0, help="Big Blind amount (default: 100.0)")
    parser.add_argument("--chips", type=float, default=10000.0, help="Starting chips (default: 10000.0)")
    args = parser.parse_args()
    
    checkpoint_path = args.model
    
    # Logic to resolve checkpoint
    if checkpoint_path is None:
        # Interactive mode
        exp_name = select_experiment_interactively()
        print(f"Selected experiment: {exp_name}")
        checkpoint_path = find_latest_checkpoint(exp_name)
        
        if not checkpoint_path:
             print(f"Error: No checkpoints found in experiment '{exp_name}'")
             sys.exit(1)
        print(f"Found latest checkpoint: {checkpoint_path}")
        
    elif not os.path.exists(checkpoint_path):
        # Treat args.model as experiment name if it's not a valid path
        print(f"Searching for latest checkpoint (Filter: {args.model})...")
        found_path = find_latest_checkpoint(args.model)
        
        if found_path:
            print(f"Found latest checkpoint: {found_path}")
            checkpoint_path = found_path
        else:
            print(f"Error: No checkpoints found for experiment '{args.model}'")
            sys.exit(1)
            
    play(checkpoint_path, sb=args.sb, bb=args.bb, chips=args.chips)
