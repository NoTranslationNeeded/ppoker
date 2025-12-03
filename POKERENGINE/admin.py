"""
Master Admin Manual Play Script (God Mode)
Features:
- Configurable N players
- Configurable Blinds & Starting Chips
- Full visibility of all cards
- Manual control for ALL players
- Real-time raise input with cost preview
"""
import sys
import os
import re
import msvcrt

# Add POKERENGINE to path if needed
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from poker_engine import PokerGame, Action, ActionType

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def get_realtime_input(prompt: str, current_bet: float, my_bet: float, 
                       min_raise: float, my_chips: float, to_call: float,
                       has_raise: bool) -> str:
    """
    Get user input with real-time cost preview for raise/bet actions
    Updates the Stack info line in real-time
    
    Args:
        prompt: Display prompt
        current_bet: Current bet to match
        my_bet: Player's bet this round
        min_raise: Minimum raise amount
        my_chips: Player's remaining chips
        to_call: Amount needed to call
        has_raise: Whether RAISE action is legal
        
    Returns:
        Complete input string (e.g., "r300", "call", "fold")
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

def print_game_state(game: PokerGame):
    print("\n" + "="*100)
    print(f" STREET: {game.street.value.upper()} | POT: {game.get_pot_size():.1f} (Main: {game.pot:.1f})")
    print(f" BOARD:  {', '.join(c.pretty_str() for c in game.community_cards) if game.community_cards else 'None'}")
    print("-" * 100)
    print(f"{'ID':<4} {'Role':<8} {'Chips':<10} {'Bet':<10} {'Status':<16} {'Hand':<20}")
    print("-" * 100)
    
    current_actor = game.get_current_player()
    
    for i, p in enumerate(game.players):
        # Determine Role
        role = ""
        if i == game.button_position: role = "BTN"
        
        # Determine SB/BB positions based on player count
        num_players = len(game.players)
        if num_players == 2:
            sb_pos = game.button_position
            bb_pos = (game.button_position + 1) % num_players
        else:
            sb_pos = (game.button_position + 1) % num_players
            bb_pos = (game.button_position + 2) % num_players
            
        if i == sb_pos: role += " SB"
        if i == bb_pos: role += " BB"
        
        # Status (with last action)
        if i == current_actor and not game.is_hand_over:
            status = "🔴 ACTING"
        elif p.is_folded():
            status = "FOLDED"
        elif p.is_all_in():
            # Show last action for all-in, or default
            status = p.last_action if p.last_action else "ALL-IN"
        elif p.last_action:
            status = p.last_action
        else:
            status = "-"  # No action yet
            
        # Hand
        hand_str = f"[{', '.join(c.pretty_str() for c in p.hand)}]" if p.hand else "[]"
        if p.is_folded():
            hand_str = "[FOLDED]"
            
        print(f"P{p.player_id:<3} {role:<8} {p.chips:<10.1f} {p.bet_this_round:<10.1f} {status:<16} {hand_str:<20}")
        
    print("="*100)

def get_user_action(game: PokerGame, player_id: int) -> Action:
    # player_id passed here is the index in game.players list
    # We need to get the real player ID for display
    real_player_id = game.players[player_id].player_id
    
    legal_actions = game.get_legal_actions(player_id)
    legal_str = ", ".join([a.value for a in legal_actions])
    
    current_bet = game.current_bet
    my_bet = game.players[player_id].bet_this_round
    to_call = current_bet - my_bet
    min_raise = game.min_raise
    
    print(f"\n[Player {real_player_id} Turn]")
    print(f"Stack: {game.players[player_id].chips:.1f} | To Call: {to_call:.1f}", end='')
    if ActionType.RAISE in legal_actions:
        print(f" | Min Raise To: {current_bet + min_raise:.1f}")
    else:
        print()  # New line if no raise info
    
    while True:
        print(f"Legal: [{legal_str}]")
        try:
            raw_input = get_realtime_input(
                f"Action (P{real_player_id}) > ",
                current_bet=current_bet,
                my_bet=my_bet,
                min_raise=min_raise,
                my_chips=game.players[player_id].chips,
                to_call=to_call,
                has_raise=(ActionType.RAISE in legal_actions)
            ).strip().lower()
        except EOFError:
            sys.exit(0)
            
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
                return Action.all_in(game.players[player_id].chips)
            elif cmd in ['quit', 'q', 'exit']:
                sys.exit(0)
            else:
                print("Commands: fold(f), check(ch), call(c), bet(b) <amt>, raise(r) <total_amt>, allin(a)")
        except Exception as e:
            print(f"Error: {e}")

def get_valid_int(prompt: str, min_val: int) -> int:
    while True:
        try:
            val_str = input(f"{prompt}: ").strip()
            if not val_str:
                continue
            val = int(val_str)
            if val < min_val:
                print(f"Value must be at least {min_val}")
                continue
            return val
        except ValueError:
            print("Invalid integer")

def get_valid_float(prompt: str, min_val: float) -> float:
    while True:
        try:
            val_str = input(f"{prompt}: ").strip()
            if not val_str:
                continue
            val = float(val_str)
            if val < min_val:
                print(f"Value must be at least {min_val}")
                continue
            return val
        except ValueError:
            print("Invalid number")

def main():
    print("="*50)
    print(" MASTER ADMIN POKER (GOD MODE)")
    print("="*50)
    
    # Configuration
    num_players = get_valid_int("Number of players (min 2)", 2)
    start_chips = get_valid_float("Starting chips (min 100)", 100.0)
    sb_val = get_valid_float("Small Blind (min 1)", 1.0)
    bb_val = get_valid_float(f"Big Blind (min {sb_val})", sb_val)
    
    game = PokerGame(small_blind=sb_val, big_blind=bb_val)
    
    # Initialize chips for all players with IDs
    # List of tuples: (player_id, chips)
    active_players = [(i, start_chips) for i in range(num_players)]
    button = 0
    
    while True:
        # clear_screen()  <-- Removed to keep history
        print(f"\n--- NEW HAND --- (Button: P{button})")
        
        try:
            # Pass (id, chips) tuples to start_hand
            game.start_hand(active_players, button)
        except ValueError as e:
            print(f"Error starting hand: {e}")
            break
            
        while not game.is_hand_over:
            print_game_state(game)
            
            current_player = game.get_current_player()
            if current_player == -1:
                break
                
            action = get_user_action(game, current_player)
            
            success, error = game.process_action(current_player, action)
            if not success:
                print(f"\n❌ Invalid Action: {error}")
                input("Press Enter to continue...")
            else:
                print(f"\n✅ Action accepted: {action}")
        
        # Hand over
        print_game_state(game)
        
        if game.winner is not None:
             # Map internal index back to player ID for display
             winner_player = game.players[game.winner]
             print(f"\n🏆 Hand Over! Winner: Player {winner_player.player_id}")
        else:
             print(f"\n🤝 Hand Over! Split Pot or Showdown")
             
        print("Hand History:")
        for line in game.get_hand_history():
            print(f"  {line}")
            
        # Update chips for next hand and remove busted players
        new_active_players = []
        for p in game.players:
            if p.chips > 0:
                new_active_players.append((p.player_id, p.chips))
            else:
                print(f"\n💀 Player {p.player_id} eliminated!")
        
        active_players = new_active_players
        num_players = len(active_players)
            
        # Check for busted players
        if num_players < 2:
            print("\nGame Over! Only one player remaining.")
            break
            
        if input("\nPlay another hand? (y/n) > ").lower() != 'y':
            break
            
        # Move button
        # In N-player, button moves to next active player
        # Since we removed players, the indices shifted. 
        # Simple rotation is fine for random/manual play.
        button = (button + 1) % num_players

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[Game stopped by user - Goodbye!]")
        sys.exit(0)
