
import sys
import os
import numpy as np

# Setup paths
current_dir = os.path.dirname(os.path.abspath(__file__))
poker_engine_path = os.path.join(current_dir, 'POKERENGINE')
if os.path.exists(poker_engine_path) and poker_engine_path not in sys.path:
    sys.path.append(poker_engine_path)
    
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    from poker_engine import PokerGame, Action, ActionType
    from poker_rl.utils.obs_builder import ObservationBuilder
    
    def print_mask_status(game, player_id, scenario_name):
        mask = ObservationBuilder._get_legal_actions_mask(game, player_id)
        min_bet_available = mask[2] == 1
        
        player = game.players[player_id]
        current_bet = game.current_bet
        min_raise = game.min_raise
        
        print(f"\n=== {scenario_name} ===")
        print(f"Stack: {player.chips}, Current Bet: {current_bet}, Min Raise: {min_raise}")
        
        if current_bet > 0:
            target = current_bet + min_raise
            print(f"Required for Min Raise: {target}")
        else:
            print(f"Required for Min Bet: {game.big_blind}")
            
        print(f"Action 2 (Min Bet/Raise) Available: {min_bet_available}")
        
        # Verification Logic
        if current_bet > 0:
            req = current_bet + min_raise
            has_chips = (player.chips + player.bet_this_round) >= req
            if has_chips != min_bet_available:
                print("❌ MISMATCH: Mask does not match chip logic!")
            else:
                print("✅ Logic Verified")
        else:
            req = game.big_blind
            has_chips = player.chips >= req
            if has_chips != min_bet_available:
                print("❌ MISMATCH: Mask does not match chip logic!")
            else:
                print("✅ Logic Verified")

    # 1. Normal Case (Deep Stack)
    game = PokerGame(small_blind=50, big_blind=100)
    game.start_hand([(0, 1000), (1, 1000)], 0)
    print_mask_status(game, 0, "Preflop SB (Deep Stack)")
    
    # 2. Short Stack (< BB)
    # SB has 40 chips (posted 40/50? No, posted 40, 0 left? No, start with 40)
    # If start with 40, posts 40 (All in).
    # Let's try start with 80. Posts 50. Left 30.
    # BB is 100. To call is 50.
    # Min Raise to 200. Need 150 more. Have 30.
    game = PokerGame(small_blind=50, big_blind=100)
    game.start_hand([(0, 80), (1, 1000)], 0)
    print_mask_status(game, 0, "Preflop SB (Short Stack 80)")
    
    # 3. Exact Stack (= Min Raise)
    # SB posts 50. Needs to call 50 (total 100).
    # Min Raise is to 200. Needs 150 more.
    # Start with 200. Posts 50. Left 150.
    # Total chips 200. Min Raise Target 200.
    game = PokerGame(small_blind=50, big_blind=100)
    game.start_hand([(0, 200), (1, 1000)], 0)
    print_mask_status(game, 0, "Preflop SB (Exact Stack 200)")

    # 4. Postflop Min Bet
    # Pot 200. P0 acts.
    # Min Bet 100.
    # P0 has 50.
    game = PokerGame(small_blind=50, big_blind=100)
    game.start_hand([(0, 1000), (1, 1000)], 0)
    game.process_action(0, Action.call(50)) # SB calls
    game.process_action(1, Action.check())  # BB checks
    # Flop. P1 (BB) acts first? No, SB (P0) acts first if button=0.
    # Let's force P0 chips to 50.
    game.players[0].chips = 50.0
    print_mask_status(game, 0, "Flop SB (Stack 50, Min Bet 100)")

except Exception as e:
    import traceback
    traceback.print_exc()

