
import numpy as np
from poker_engine.game import PokerGame, Action, ActionType
from .equity_calculator import get_8_features

class ObservationBuilder:
    def __init__(self):
        pass

    @staticmethod
    def get_observation(game: PokerGame, player_id: int, action_history: dict) -> dict:
        obs_vec = np.zeros(150, dtype=np.float32)
        
        # 1. Cards (0-118)
        # Hole cards
        for i, card in enumerate(game.players[player_id].hand[:2]):
            obs_vec[i*17:(i+1)*17] = ObservationBuilder._encode_card_onehot(card)
        # Community cards
        for i, card in enumerate(game.community_cards):
            obs_vec[34+i*17:34+(i+1)*17] = ObservationBuilder._encode_card_onehot(card)
            
        # 2. Game State (119-134)
        player = game.players[player_id]
        opponent = game.players[1 - player_id]
        pot = game.get_pot_size()
        to_call = game.current_bet - player.bet_this_round
        
        bb = game.big_blind
        max_bb = 500.0 # Normalization constant
        
        # Street mapping
        street_val = 0.0
        if game.street.value == 'flop': street_val = 0.33
        elif game.street.value == 'turn': street_val = 0.66
        elif game.street.value == 'river': street_val = 1.0
        
        obs_vec[119:135] = [
            (player.chips / bb) / max_bb,
            (opponent.chips / bb) / max_bb,
            (pot / bb) / max_bb,
            (game.current_bet / bb) / max_bb,
            (player.bet_this_round / bb) / max_bb,
            (to_call / bb) / max_bb,
            1.0 if game.button_position == player_id else 0.0,
            street_val,
            to_call / (pot + to_call) if to_call > 0 and pot > 0 else 0.0,
            np.clip((player.chips / pot) / 10.0, 0, 1.0) if pot > 0 else 1.0,
            0.0, # Hand count / max hands (not used in single hand episode)
            len(game.community_cards) / 5.0,
            (game.min_raise / bb) / max_bb,
            (opponent.bet_this_round / bb) / max_bb,
            (opponent.bet_this_hand / bb) / max_bb,
            bb / 100.0, # Relative blind size
        ]
        
        # 3. Advanced Hand Evaluation Features (135-142)
        # Get 8 features: [hs/equity, ppot, npot, hand_index, is_pre, is_flop, is_turn, is_river]
        hole_cards = player.hand
        board = game.community_cards
        street_name = game.street.value  # 'preflop', 'flop', 'turn', 'river'
        
        advanced_features = get_8_features(hole_cards, board, street_name)
        obs_vec[135:143] = advanced_features
        
        # 4. Remaining padding (143-149)
        obs_vec[143:150] = [0, 0, 0, 0, 0, 0, 0]
        
        # 5. Action History (Removed for LSTM)
            
        mask = ObservationBuilder._get_legal_actions_mask(game, player_id).astype(np.float32)
        
        return {
            "observations": obs_vec.astype(np.float32),
            "action_mask": mask
        }

    @staticmethod
    def _encode_card_onehot(card) -> np.ndarray:
        encoding = np.zeros(17, dtype=np.float32)
        # Rank 0-12
        ranks = "23456789TJQKA"
        try:
            rank_idx = ranks.index(str(card.rank))
        except:
            # Handle 10 represented as 'T' or '10'
            if str(card.rank) == '10': rank_idx = 8
            else: rank_idx = 0 # Fallback
            
        encoding[rank_idx] = 1.0
        
        # Suit 13-16
        # Assuming Card.suit is 's', 'h', 'd', 'c' or similar
        suit_map = {'s': 0, 'h': 1, 'd': 2, 'c': 3, '♠': 0, '♥': 1, '♦': 2, '♣': 3}
        suit_idx = suit_map.get(str(card.suit).lower(), 0)
        encoding[13 + suit_idx] = 1.0
        
        return encoding

    @staticmethod
    def _get_legal_actions_mask(game: PokerGame, player_id: int) -> np.ndarray:
        legal = game.get_legal_actions(player_id)
        mask = np.zeros(14, dtype=np.int8)
        
        # Map Engine ActionTypes to our Discrete(14)
        if ActionType.FOLD in legal:
            # Prevent Open-Folding: If we can check, we should never fold.
            if ActionType.CHECK in legal:
                mask[0] = 0
            else:
                mask[0] = 1
                
        if ActionType.CHECK in legal or ActionType.CALL in legal: mask[1] = 1
        
        # Smart Masking for Bet/Raise
        if ActionType.BET in legal or ActionType.RAISE in legal:
            pot = game.get_pot_size()
            player = game.players[player_id]
            current_bet = game.current_bet
            min_raise = game.min_raise
            
            # Action 2: Min Raise ONLY
            # Mask OUT if current_bet == 0 (No Min-Bet allowed)
            # Enable if current_bet > 0 (Facing bet)
            if current_bet > 0:
                min_target = current_bet + min_raise
                if player.chips + player.bet_this_round >= min_target:
                    mask[2] = 1
            else:
                # Open Pot: Min-Bet is BANNED per user request.
                mask[2] = 0

            # Percentage Bets (Indices 3-12)
            # 3: 10%, 4: 25%, 5: 33%, 6: 50%, 7: 75%, 8: 100%, 9: 125%, 10: 150%, 11: 200%, 12: 300%
            pcts = [0.10, 0.25, 0.33, 0.50, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]
            for i, pct in enumerate(pcts):
                idx = 3 + i
                amount = pot * pct
                
                # Logic from _map_action (simplified for validation)
                if current_bet > 0:
                    # Raise
                    target = current_bet + amount
                    min_target = current_bet + min_raise
                    
                    # Prevent Redundant All-In: If raise amount >= stack, use All-In action instead
                    max_bet = player.chips + player.bet_this_round
                    if target >= max_bet:
                        continue

                    # Allow if target >= min_target OR if we have enough chips to cover min_target?
                    # Actually, if calculated amount < min_raise, we usually mask it.
                    # BUT, user wants 10% raise to be valid if it auto-corrects?
                    # "Min-Bet is banned... Min-Raise is kept."
                    # "Small Pots: 10% Bet should auto-correct to Min-Bet."
                    # What about 10% Raise?
                    # If 10% Raise < Min Raise, should it auto-correct to Min Raise?
                    # User said "Min-Raise is kept".
                    # If 10% Raise < Min Raise, and we allow it, it becomes Min Raise.
                    # That duplicates Action 2.
                    # But Action 2 is explicit Min Raise.
                    # So maybe we should mask 10% Raise if it's < Min Raise, to force use of Action 2?
                    # Or just allow it.
                    # Let's stick to standard logic for Raises: Must be >= Min Raise.
                    # Because Action 2 exists for exactly that purpose.
                    if target >= min_target:
                        mask[idx] = 1
                    # Min bet is big blind
                    # Strict Masking: If amount < big_blind, it is MASKED.
                    # No auto-correction allowed.
                    if amount >= game.big_blind:
                        mask[idx] = 1
                        
        if ActionType.ALL_IN in legal: mask[13] = 1
        
        return mask
