
import numpy as np
from poker_engine.game import PokerGame, Action, ActionType
from .equity_calculator import get_8_features

RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']

class ObservationBuilder:
    def __init__(self):
        pass

    @staticmethod
    def canonicalize_suits(hole_cards, board):
        """
        Normalize suits by first appearance order
        
        Returns:
            List of (rank_str, canonical_suit_int) tuples
        """
        suit_map = {}
        next_suit_id = 0
        canonical = []
        
        for card in (hole_cards + board):
            # Assign ID to new suit
            if card.suit not in suit_map:
                suit_map[card.suit] = next_suit_id
                next_suit_id += 1
            
            canonical.append((card.rank, suit_map[card.suit]))
        
        return canonical

    @staticmethod
    def normalize_chips_log(chips_in_bb):
        """Log scale normalization for chip amounts"""
        return np.log1p(chips_in_bb) / np.log1p(500.0)

    @staticmethod
    def _encode_card_onehot(canonical_card):
        """
        (rank_str, canonical_suit_int) → 17-dim one-hot
        """
        rank, suit = canonical_card
        encoding = np.zeros(17, dtype=np.float32)
        
        rank_idx = RANKS.index(rank)
        encoding[rank_idx] = 1.0
        encoding[13 + suit] = 1.0  # suit is already 0-3
        
        return encoding

    @staticmethod
    def _get_street_context_features(game, player_id, action_history):
        """
        Street-wise summary statistics: 4 streets × 4 features = 16 dimensions
        """
        features = np.zeros(16, dtype=np.float32)
        streets = ['preflop', 'flop', 'turn', 'river']
        
        for i, street in enumerate(streets):
            base_idx = i * 4
            actions = action_history.get(street, [])
            
            # [0] Number of raises (0-10, normalized)
            raises = sum(1 for (action_idx, _, _, _) in actions 
                        if action_idx in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])  # All-In included!
            features[base_idx + 0] = min(raises, 10) / 10.0
            
            # [1] Aggressor (0=none, 0.5=me, 1.0=opponent)
            last_aggressor = None
            for (action_idx, pid, _, _) in actions:
                if action_idx in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]:  # All-In included!
                    last_aggressor = pid
            
            if last_aggressor is None:
                features[base_idx + 1] = 0.0
            elif last_aggressor == player_id:
                features[base_idx + 1] = 0.5
            else:
                features[base_idx + 1] = 1.0
            
            # [2] My total investment this street (ACCURATE with actual amounts)
            # env.py stores actual bet_amount in tuple: (action_idx, player_id, bet_ratio, bet_amount)
            my_actions_amounts = [amount for (_, pid, _, amount) in actions if pid == player_id]
            total_invested_absolute = sum(my_actions_amounts)
            features[base_idx + 2] = ObservationBuilder.normalize_chips_log(total_invested_absolute / game.big_blind)
            
            # [3] Was 3-bet or higher? (binary)
            features[base_idx + 3] = 1.0 if raises >= 2 else 0.0
        
        return features

    @staticmethod
    def _get_current_street_context(game, player_id, action_history):
        """Current street action patterns: 6 dimensions"""
        features = np.zeros(6, dtype=np.float32)
        
        current_street = game.street.value
        actions = action_history.get(current_street, [])
        
        # [0] Actions count this street
        features[0] = min(len(actions), 10) / 10.0
        
        # [1] I raised this street
        features[1] = 1.0 if any(pid == player_id and action_idx in [2,3,4,5,6,7,8,9,10,11,12,13]  # All-In included!
                                 for (action_idx, pid, _, _) in actions) else 0.0
        
        # [2] Opponent raised this street
        features[2] = 1.0 if any(pid != player_id and action_idx in [2,3,4,5,6,7,8,9,10,11,12,13]  # All-In included!
                                 for (action_idx, pid, _, _) in actions) else 0.0
        
        # [3] Passive-to-Aggressive transition (improved check-raise detection)
        # Simplified from exact check-raise to "passive→aggressive" pattern
        passive_to_aggressive = False
        my_was_passive = False
        for (action_idx, pid, _, _) in actions:
            if pid == player_id:
                if action_idx in [0, 1]:  # Fold/Check/Call - passive
                    my_was_passive = True
                elif action_idx in [2,3,4,5,6,7,8,9,10,11,12,13] and my_was_passive:  # aggressive
                    passive_to_aggressive = True
                    break
        features[3] = 1.0 if passive_to_aggressive else 0.0
        
        # [4] Donk-bet happened (OOP bets into preflop aggressor)
        # Simplified: not implemented yet
        features[4] = 0.0
        
        # [5] Last action was aggressive
        if actions:
            last_action = actions[-1][0]
            features[5] = 1.0 if last_action in [2,3,4,5,6,7,8,9,10,11,12,13] else 0.0
        
        return features

    @staticmethod
    def _get_investment_features(game, player_id, start_stacks):
        """Investment info: 2 dimensions"""
        features = np.zeros(2, dtype=np.float32)
        
        my_total_invested = game.players[player_id].bet_this_hand
        starting_stack = start_stacks[player_id] if start_stacks else 10000.0  # Fallback
        
        # [0] Total investment (log scale)
        features[0] = ObservationBuilder.normalize_chips_log(my_total_invested / game.big_blind)
        
        # [1] Investment ratio (0-1)
        investment_ratio = my_total_invested / starting_stack if starting_stack > 0 else 0.0
        features[1] = min(investment_ratio, 1.0)
        
        return features

    @staticmethod
    def _get_position_features(game, player_id):
        """Position-related explicit info: 2 dimensions"""
        features = np.zeros(2, dtype=np.float32)
        
        is_button = (game.button_position == player_id)
        is_preflop = (game.street.value == 'preflop')
        
        # [0] Position Value (0.0=OOP, 1.0=IP)
        if is_preflop:
            features[0] = 0.0 if is_button else 1.0  # Non-button is IP preflop
        else:
            features[0] = 1.0 if is_button else 0.0  # Button is IP postflop
        
        # [1] Permanent Position Advantage (postflop only)
        if is_preflop:
            features[1] = 0.5  # Neutral
        else:
            features[1] = 1.0 if is_button else 0.0
        
        return features

    @staticmethod
    def get_observation(game: PokerGame, player_id: int, action_history: dict, start_stacks=None) -> dict:
        """
        Build observation vector with all improvements
        
        Args:
            start_stacks: List of starting chips for each player (for investment ratio)
        """
        obs_vec = np.zeros(176, dtype=np.float32)
        
        # CRITICAL: Preserve original Card objects for equity calculation
        hole_cards_original = game.players[player_id].hand
        board_original = game.community_cards
        
        # 1. Canonicalize suits (for encoding only)
        canonical = ObservationBuilder.canonicalize_suits(hole_cards_original, board_original)
        
        # 2. Encode cards (0-118) - Use canonical
        for i, card in enumerate(canonical[:7]):
            obs_vec[i*17:(i+1)*17] = ObservationBuilder._encode_card_onehot(card)
        
        # 3. Game State (119-134) - Apply LOG scale normalization
        player = game.players[player_id]
        opponent = game.players[1 - player_id]
        pot = game.get_pot_size()
        to_call = game.current_bet - player.bet_this_round
        
        bb = game.big_blind
        
        # Street mapping
        street_val = 0.0
        if game.street.value == 'flop': street_val = 0.33
        elif game.street.value == 'turn': street_val = 0.66
        elif game.street.value == 'river': street_val = 1.0
        
        obs_vec[119:135] = [
            ObservationBuilder.normalize_chips_log(player.chips / bb),
            ObservationBuilder.normalize_chips_log(opponent.chips / bb),
            ObservationBuilder.normalize_chips_log(pot / bb),
            ObservationBuilder.normalize_chips_log(game.current_bet / bb),
            ObservationBuilder.normalize_chips_log(player.bet_this_round / bb),
            ObservationBuilder.normalize_chips_log(to_call / bb),
            1.0 if game.button_position == player_id else 0.0,
            street_val,
            to_call / (pot + to_call) if to_call > 0 and pot > 0 else 0.0,  # Pot odds (relative)
            np.clip((player.chips / pot) / 10.0, 0, 1.0) if pot > 0 else 1.0,  # SPR (relative)
            0.0,  # Hand count / max hands (not used)
            len(game.community_cards) / 5.0,  # Card count (relative)
            ObservationBuilder.normalize_chips_log(game.min_raise / bb),
            ObservationBuilder.normalize_chips_log(opponent.bet_this_round / bb),
            ObservationBuilder.normalize_chips_log(opponent.bet_this_hand / bb),
            bb / 100.0,  # Blind size (relative)
        ]
        
        # 4. Expert features (135-142) - Use ORIGINAL Card objects!
        advanced_features = get_8_features(
            hole_cards_original,  # Card objects (NOT canonicalized!)
            board_original,       # Card objects (NOT canonicalized!)
            game.street.value
        )
        obs_vec[135:143] = advanced_features
        
        # 5. Padding (143-149) - Keep as 0 for future flexibility
        # Note: Maintained for backward compatibility and hand_index_pos=138 hard-coded
        obs_vec[143:150] = 0.0
        
        # 6. Street history (150-165)
        obs_vec[150:166] = ObservationBuilder._get_street_context_features(game, player_id, action_history)
        
        # 7. Current street context (166-171)
        obs_vec[166:172] = ObservationBuilder._get_current_street_context(game, player_id, action_history)
        
        # 8. Investment info (172-173)
        obs_vec[172:174] = ObservationBuilder._get_investment_features(game, player_id, start_stacks)
        
        # 9. Position info (174-175)
        obs_vec[174:176] = ObservationBuilder._get_position_features(game, player_id)
        
        return {
            "observations": obs_vec.astype(np.float32),  # Explicit float32 conversion
            "action_mask": ObservationBuilder._get_legal_actions_mask(game, player_id).astype(np.float32)
        }

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
            if current_bet > 0:
                min_target = current_bet + min_raise
                if player.chips + player.bet_this_round >= min_target:
                    mask[2] = 1
            else:
                mask[2] = 0
                
            # Percentage Bets (Indices 3-12)
            pcts = [0.10, 0.25, 0.33, 0.50, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]
            for i, pct in enumerate(pcts):
                idx = 3 + i
                amount = pot * pct
                
                if current_bet > 0:
                    target = current_bet + amount
                    min_target = current_bet + min_raise
                    max_bet = player.chips + player.bet_this_round
                    if target >= max_bet:
                        continue
                    if target >= min_target:
                        mask[idx] = 1
                    if amount >= game.big_blind:
                        mask[idx] = 1
                        
        if ActionType.ALL_IN in legal: mask[13] = 1
        
        return mask
