
import sys
import os
import traceback
import numpy as np

print("Starting reproduction script...")

try:
    # ==========================================
    # 1. Setup Path for POKERENGINE
    # ==========================================
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # We are in playground/glacial-supernova/
    # POKERENGINE is in ./POKERENGINE
    poker_engine_path = os.path.join(current_dir, 'POKERENGINE')
    
    if os.path.exists(poker_engine_path):
        if poker_engine_path not in sys.path:
            sys.path.append(poker_engine_path)
            print(f"Added POKERENGINE path: {poker_engine_path}")
    else:
        print(f"Warning: POKERENGINE path not found at {poker_engine_path}")

    # ==========================================
    # 2. Import POKERENGINE Modules
    # ==========================================
    try:
        from poker_engine import PokerGame, Action, ActionType
        print("Imported poker_engine successfully.")
    except ImportError:
        try:
            from POKERENGINE.poker_engine import PokerGame, Action, ActionType
            print("Imported POKERENGINE.poker_engine successfully.")
        except ImportError as e:
            print(f"Critical Error: Could not import POKERENGINE. {e}")
            sys.exit(1)

    # Import ObservationBuilder
    # We need to add current_dir to path to import poker_rl
    if current_dir not in sys.path:
        sys.path.append(current_dir)
        
    from poker_rl.utils.obs_builder import ObservationBuilder
    print("Imported ObservationBuilder successfully.")

    def print_mask(mask, title):
        print(f"\n--- {title} ---")
        actions = [
            "Fold", "Check/Call", 
            "Min Raise Only", # Action 2
            "Bet 10%", "Bet 25%", "Bet 33%", "Bet 50%", "Bet 75%", "Bet 100%", 
            "Bet 125%", "Bet 150%", "Bet 200%", "Bet 300%", "All-in"
        ]
        
        # Scenario 3: Limped Pot Flop
        print("\nScenario 3: Limped Pot Flop (Pot=200)")
        game.process_action(1, Action.check())
        # Flop dealt.
        current = game.get_current_player()
        print(f"Flop Current Player: {current}")
        
        mask_flop = ObservationBuilder._get_legal_actions_mask(game, current)
        print_mask(mask_flop, "Flop Actions (Limped Pot)")

    test_masking()
    print("\nTest completed successfully.")

except Exception:
    traceback.print_exc()
