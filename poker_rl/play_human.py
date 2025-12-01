import argparse
import numpy as np
import sys
import os
import time
import re
import msvcrt

# ==========================================
# 1. Setup Path for POKERENGINE
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
# play_human.py is in poker_rl/, so POKERENGINE is in ../POKERENGINE
poker_engine_path = os.path.join(current_dir, '..', 'POKERENGINE')

if os.path.exists(poker_engine_path):
    if poker_engine_path not in sys.path:
        sys.path.append(poker_engine_path)
else:
    # Fallback: maybe running from root
    root_path = os.path.join(current_dir, '..')
    if root_path not in sys.path:
        sys.path.append(root_path)

# ==========================================
# 2. Import POKERENGINE Modules
# ==========================================
try:
    # Try direct import (if POKERENGINE is in path)
    from poker_engine.game import PokerGame, Street
    from poker_engine.cards import Card
    from poker_engine.actions import Action, ActionType
except ImportError:
    try:
        # Try package import (if running from root)
        from POKERENGINE.poker_engine.game import PokerGame, Street
        from POKERENGINE.poker_engine.cards import Card
        from POKERENGINE.poker_engine.actions import Action, ActionType
    except ImportError as e:
        print(f"Critical Error: Could not import POKERENGINE. {e}")
        print(f"Sys Path: {sys.path}")
        sys.exit(1)

# ==========================================
# 3. Import Agents
# ==========================================
from poker_rl.agents.random_agent import RandomAgent
from poker_rl.agents.call_station import CallStationAgent

# ==========================================
# 4. Helper Functions
# ==========================================
def print_game_state(game, human_player_id):
    print("\n" + "="*60)
    print(f"Street: {game.street.value.upper()}")
    
    # Use get_pot_size() for accurate current pot
    current_pot = game.get_pot_size()
    print(f"Pot: {current_pot:.1f}")
    
    # Use community_cards instead of board
    board_cards = game.community_cards
    board_str = ' '.join([c.pretty_str(color=True) for c in board_cards]) if board_cards else "None"
    print(f"Community Cards: {board_str}")
    print("-" * 60)
    
    for pid, player in enumerate(game.players):
        role = "YOU" if pid == human_player_id else "BOT"
        status = "FOLDED" if player.is_folded() else "ACTIVE"
        if player.is_all_in(): status = "ALL-IN"
        
        cards_str = "XX XX"
        if pid == human_player_id or game.street == Street.SHOWDOWN:
             if player.hand:
                cards_str = ' '.join([c.pretty_str(color=True) for c in player.hand])
        
        # Use player.player_id for display if available, else index
        display_pid = getattr(player, 'player_id', pid)
        
        print(f"{role:<4} (P{display_pid}): Chips={player.chips:<8.1f} Bet={player.bet_this_round:<8.1f} Status={status}")
        print(f"      Hand: {cards_str}")
    print("="*60 + "\n")

def get_realtime_input(prompt: str, current_bet: float, my_bet: float, 
                       min_raise: float, my_chips: float, to_call: float,
                       has_raise: bool) -> str:
    """
    Get user input with real-time cost preview for raise/bet actions
    Updates the Stack info line in real-time
    """
    input_str = ""
    
    # Build the Stack info line template
    min_raise_to = current_bet + min_raise if has_raise else 0
    
    def render_stack_line(cost_info: str = ""):
        """Render the Stack info line with optional cost info"""
        base_line = f"Stack: {my_chips:.1f}"
        if cost_info:
            base_line += cost_info
        base_line += f" | To Call: {to_call:.1f}"
        if has_raise:
            base_line += f" | Min Raise To: {min_raise_to:.1f}"
        return base_line
    
    def render_line():
        """Render the complete input line and update Stack line above"""
        # Calculate cost message if applicable
        cost_info = ""
        
        if input_str:
            # Try to parse command and amount from current input
            match = re.match(r"^([a-z]+)(\d+(?:\.\d+)?)", input_str)
            if match:
                cmd = match.group(1)
                try:
                    amount = float(match.group(2))
                    
                    if cmd in ['r', 'raise']:
                        # Raise: amount is "raise to" (total bet)
                        actual_cost = amount - my_bet
                        
                        if amount < min_raise_to and current_bet > 0:
                            cost_info = f' → ❌ -{actual_cost:.1f} (Min: {min_raise_to:.1f})'
                        elif actual_cost > my_chips:
                            cost_info = f' → ❌ -{actual_cost:.1f} (Not enough!)'
                        elif actual_cost < 0:
                            cost_info = f' → ❌ Invalid'
                        else:
                            cost_info = f' → 💵 -{actual_cost:.1f} chips'
                    
                    elif cmd in ['b', 'bet']:
                        if amount > my_chips:
                            cost_info = f' → ❌ -{amount:.1f} (Not enough!)'
                        elif current_bet > 0:
                            cost_info = f' → ❌ Use raise'
                        else:
                            cost_info = f' → 💵 -{amount:.1f} chips'
                
                except (ValueError, IndexError):
                    pass
        
        # Update Stack line (2 lines up)
        sys.stdout.write('\033[2A')  # Move up 2 lines
        sys.stdout.write('\r' + render_stack_line(cost_info) + '\033[K')  # Overwrite and clear
        sys.stdout.write('\033[2B')  # Move down 2 lines
        
        # Update input line
        sys.stdout.write(f'\r{prompt}{input_str}\033[K')
        sys.stdout.flush()
    
    # Initial render
    sys.stdout.write(prompt)
    sys.stdout.flush()
    
    try:
        while True:
            if msvcrt.kbhit():
                char = msvcrt.getch()
                
                # Enter key
                if char == b'\r':
                    print()  # New line after input
                    return input_str
                
                # Backspace
                elif char == b'\x08':
                    if input_str:
                        input_str = input_str[:-1]
                        render_line()
                
                # ESC to cancel
                elif char == b'\x1b':
                    print("\n[Cancelled]")
                    return ""
                
                # Printable characters
                elif 32 <= ord(char) <= 126:
                    char_str = char.decode()
                    input_str += char_str
                    render_line()
    except KeyboardInterrupt:
        print("\n\n[Game interrupted by user]")
        sys.exit(0)

def get_human_action(game, player_id):
    legal_actions = game.get_legal_actions(player_id)
    legal_str = ", ".join([a.value for a in legal_actions])
    
    current_bet = game.current_bet
    player = game.players[player_id]
    my_bet = player.bet_this_round
    to_call = current_bet - my_bet
    min_raise = game.min_raise
    
    print(f"\n[Your Turn]")
    # Initial Stack Line (will be updated by get_realtime_input)
    print(f"Stack: {player.chips:.1f} | To Call: {to_call:.1f}", end='')
    if ActionType.RAISE in legal_actions:
        print(f" | Min Raise To: {current_bet + min_raise:.1f}")
    else:
        print()
    
    while True:
        print(f"Legal: [{legal_str}]")
        
        # Use get_realtime_input instead of input()
        raw_input = get_realtime_input(
            prompt="Action > ",
            current_bet=current_bet,
            my_bet=my_bet,
            min_raise=min_raise,
            my_chips=player.chips,
            to_call=to_call,
            has_raise=(ActionType.RAISE in legal_actions)
        ).strip().lower()
        
        if not raw_input:
            continue
            
        # Parse input: Handle "r40", "b100", "raise 40"
        cmd = ""
        amount = 0.0
        
        # Check for concatenated format (e.g., r40, b100)
        match = re.match(r"([a-z]+)(\d+(\.\d+)?)", raw_input.split()[0])
        if match:
            cmd = match.group(1)
            amount = float(match.group(2))
        else:
            # Standard space-separated format
            parts = raw_input.split()
            cmd = parts[0]
            amount = float(parts[1]) if len(parts) > 1 else 0.0
        
        try:
            if cmd in ['fold', 'f']:
                return Action.fold()
            elif cmd in ['check', 'ch']:
                return Action.check()
            elif cmd in ['call', 'c']:
                # Context-sensitive 'c': Check if nothing to call, else Call
                if to_call == 0:
                    return Action.check()
                return Action.call(to_call)
            elif cmd in ['bet', 'b']:
                return Action.bet(amount)
            elif cmd in ['raise', 'r']:
                # Input: "raise 300" -> Raise TO 300
                return Action.raise_to(amount)
            elif cmd in ['all', 'allin', 'a']:
                return Action.all_in(player.chips)
            elif cmd in ['quit', 'q', 'exit']:
                sys.exit(0)
            else:
                print("Commands: fold(f), check(ch), call(c), bet(b) <amt>, raise(r) <total_amt>, allin(a)")
        except Exception as e:
            print(f"Error creating action: {e}")

def get_bot_action(agent, game, player_id):
    # Construct dummy observation and mask
    legal_actions = game.get_legal_actions(player_id)
    
    # Map legal actions to our 8 discrete actions for the agent
    mask = np.zeros(8, dtype=np.int8)
    
    # Simple mapping for benchmark bots
    if ActionType.FOLD in legal_actions: mask[0] = 1
    if ActionType.CHECK in legal_actions or ActionType.CALL in legal_actions: mask[1] = 1
    if ActionType.BET in legal_actions or ActionType.RAISE in legal_actions: mask[2:7] = 1 # Allow all bet sizes
    if ActionType.ALL_IN in legal_actions: mask[7] = 1
    
    # RandomAgent and CallStationAgent don't use the observation vector, so we pass None
    action_idx = agent.compute_action(None, mask)
    
    # Convert index back to engine Action
    player = game.players[player_id]
    pot = game.get_pot_size() # Use get_pot_size() here too
    
    if action_idx == 0:
        return Action.fold()
    elif action_idx == 1:
        to_call = game.current_bet - player.bet_this_round
        return Action.check() if to_call == 0 else Action.call(to_call)
    elif action_idx == 7:
        return Action.all_in(player.chips)
    else:
        # Bet actions (2-6)
        pcts = [0.33, 0.50, 0.75, 1.0, 1.5]
        if 2 <= action_idx <= 6:
            pct = pcts[action_idx - 2]
            amount = pot * pct
            
            # Ensure min raise / min bet
            if game.current_bet > 0:
                 target = game.current_bet + amount
                 min_raise = game.current_bet + game.min_raise
                 target = max(target, min_raise)
                 
                 # Check if target exceeds chips (cap to all-in)
                 if target > player.chips + player.bet_this_round:
                     return Action.all_in(player.chips)
                     
                 return Action.raise_to(target)
            else:
                 amount = max(amount, game.big_blind)
                 if amount > player.chips:
                     return Action.all_in(player.chips)
                 return Action.bet(amount)
        
    return Action.check() # Fallback

def main():
    parser = argparse.ArgumentParser(description="Play Poker against a Bot")
    parser.add_argument("--opponent", type=str, choices=['random', 'call_station'], default='random', help="Opponent type")
    parser.add_argument("--stack", type=float, default=2000.0, help="Starting stack size")
    parser.add_argument("--sb", type=float, default=50.0, help="Small Blind")
    parser.add_argument("--bb", type=float, default=100.0, help="Big Blind")
    args = parser.parse_args()
    
    print(f"Starting game against {args.opponent} bot...")
    
    if args.opponent == 'random':
        bot_agent = RandomAgent()
    else:
        bot_agent = CallStationAgent()
        
    # Initialize chips
    human_chips = args.stack
    bot_chips = args.stack
    
    human_id = 0
    bot_id = 1
    
    hand_count = 0
    
    while human_chips > 0 and bot_chips > 0:
        hand_count += 1
        print(f"\n\n=== HAND {hand_count} ===")
        
        game = PokerGame(small_blind=args.sb, big_blind=args.bb)
        
        # Alternate button
        button = hand_count % 2
        
        game.start_hand(
            players_info=[(human_id, human_chips), (bot_id, bot_chips)],
            button=button
        )
        
        # Use is_hand_over instead of hand_over
        while not game.is_hand_over:
            print_game_state(game, human_id)
            
            current_player = game.get_current_player()
            
            if current_player == human_id:
                action = get_human_action(game, human_id)
            else:
                print("Bot is thinking...")
                time.sleep(1) # Fake thinking time
                action = get_bot_action(bot_agent, game, bot_id)
                print(f"Bot did: {action}")
            
            try:
                success, error = game.process_action(current_player, action)
                if not success:
                    print(f"Invalid action: {error}")
                    continue
            except Exception as e:
                print(f"Error processing action: {e}")
                continue
        
        # Hand over
        print_game_state(game, human_id)
        print("Hand Over!")
        
        if game.winner is not None:
             winner_role = "YOU" if game.winner == human_id else "BOT"
             print(f"Winner: {winner_role} (P{game.winner})")
        else:
             print("Split Pot!")
        
        # Update chips
        human_chips = game.players[human_id].chips
        bot_chips = game.players[bot_id].chips
        
        print(f"Your Chips: {human_chips}")
        print(f"Bot Chips: {bot_chips}")
        
        if input("Play another hand? (y/n): ").lower() != 'y':
            break

if __name__ == "__main__":
    main()
