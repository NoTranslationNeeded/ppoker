"""
Equity Calculator for Advanced Hand Evaluation Features

Provides functions to calculate:
- Preflop Equity (from pre-computed table)
- Hand Strength (HS) - normalized score
- Positive Potential (PPot) - improvement probability
- Negative Potential (NPot) - deterioration probability
- Hand Index (0-168 preflop, 169-178 postflop)
"""
import json
import os
import random
from typing import List, Tuple, Union
from functools import lru_cache

import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
poker_engine_path = os.path.join(current_dir, '../../POKERENGINE')
sys.path.append(poker_engine_path)

from poker_engine import Card, Deck
from poker_engine.evaluator import HandEvaluator

# Rank mapping: 0=2, 1=3, ..., 12=A
RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
HAND_RANK_NAMES = ['HC', 'Pair', '2P', '3oK', 'Str', 'Flush', 'FH', '4oK', 'SF', 'RF']

# Load preflop equity table (cached)
_EQUITY_TABLE = None

def _load_equity_table():
    """Load preflop equity table from JSON file (cached)"""
    global _EQUITY_TABLE
    if _EQUITY_TABLE is None:
        equity_file = os.path.join(os.path.dirname(__file__), 'preflop_equity_table.json')
        with open(equity_file, 'r') as f:
            _EQUITY_TABLE = json.load(f)
            # Convert string keys to integers
            _EQUITY_TABLE = {int(k): v for k, v in _EQUITY_TABLE.items()}
    return _EQUITY_TABLE

def rank_to_int(rank_char: str) -> int:
    """Convert rank character to int (0-12)"""
    return RANKS.index(rank_char)

def get_hand_index_preflop(card1: Card, card2: Card) -> int:
    """
    Map hole cards to 0-168 index using 13x13 matrix
    
    Args:
        card1, card2: Hole cards
    
    Returns:
        Index 0-168
    """
    r1 = rank_to_int(card1.rank)
    r2 = rank_to_int(card2.rank)
    is_suited = (card1.suit == card2.suit)
    
    # Make r1 >= r2
    r1, r2 = max(r1, r2), min(r1, r2)
    
    if r1 == r2:  # Pair
        index = r1 * 13 + r2
    elif is_suited:  # Suited (higher rank as row)
        index = r1 * 13 + r2
    else:  # Offsuit (swap to make r1 < r2)
        index = r2 * 13 + r1
    
    return index

def get_hand_index_postflop(score: int) -> int:
    """
    Map hand rank to 169-178 index
    
    Args:
        score: Hand evaluation score
    
    Returns:
        Index 169-178
    """
    rank_category = score >> 20  # Extract category (0-9)
    return 169 + rank_category

def get_hand_name_from_index(hand_index: int) -> str:
    """
    Convert hand index to readable hand name for logging
    
    Args:
        hand_index: 0-168 (preflop) or 169-178 (postflop)
    
    Returns:
        Hand name string (e.g., "AA", "AKs", "Flush")
    """
    if 0 <= hand_index <= 168:
        # Preflop: convert matrix index to hand name
        r1 = hand_index // 13
        r2 = hand_index % 13
        
        if r1 == r2:
            # Pair
            return f"{RANKS[r1]}{RANKS[r2]}"
        elif r1 > r2:
            # Suited (higher rank as row)
            return f"{RANKS[r1]}{RANKS[r2]}s"
        else:
            # Offsuit
            return f"{RANKS[r2]}{RANKS[r1]}o"
    elif 169 <= hand_index <= 178:
        # Postflop: rank category name
        category = hand_index - 169
        return HAND_RANK_NAMES[category] if category < len(HAND_RANK_NAMES) else "?"
    else:
        return "?"


def get_preflop_equity(card1: Card, card2: Card) -> float:
    """
    Get preflop equity from pre-computed table
    
    Args:
        card1, card2: Hole cards
    
    Returns:
        Equity (0-1)
    """
    equity_table = _load_equity_table()
    index = get_hand_index_preflop(card1, card2)
    return equity_table.get(index, 0.5)  # Default 0.5 if not found

def get_hand_strength_postflop(hole_cards: List[Card], board: List[Card]) -> float:
    """
    Calculate current hand strength (normalized score)
    
    Args:
        hole_cards: 2 hole cards
        board: Community cards (3-5 cards)
    
    Returns:
        Hand strength normalized to 0-1 (score / 10,000,000), clipped to [0, 1]
    """
    score = HandEvaluator.evaluate(hole_cards, board)
    # Normalize: max theoretical score is ~10M
    # Safety: clip to [0, 1] in case score exceeds 10M
    return min(1.0, max(0.0, score / 10_000_000.0))

def calculate_ppot_npot(hole_cards: List[Card], board: List[Card], 
                       num_opponent_samples: int = 50) -> Tuple[float, float]:
    """
    Calculate Positive and Negative Potential using Monte Carlo sampling
    
    Strategy:
    - Sample 50 random opponent hands (uncertainty estimation)
    - For each opponent hand, simulate all 46 possible next cards (full enumeration)
    
    Args:
        hole_cards: 2 hole cards
        board: Community cards (3-4 cards, not 5)
        num_opponent_samples: Number of opponent hands to sample (default 50)
    
    Returns:
        (ppot, npot) - both in range 0-1
    """
    if len(board) >= 5:
        # Already river, no future cards
        return 0.0, 0.0
    
    # Current hand strength
    my_score_now = HandEvaluator.evaluate(hole_cards, board)
    
    # Create deck and remove known cards
    used_cards = set()
    for card in hole_cards + board:
        used_cards.add((card.suit, card.rank))
    
    # Build remaining deck
    remaining_deck = []
    for suit in ['S', 'H', 'D', 'C']:
        for rank in RANKS:
            if (suit, rank) not in used_cards:
                remaining_deck.append(Card(suit, rank))
    
    # Sample opponent hands
    behind_improved = 0
    behind_total = 0
    ahead_worsened = 0
    ahead_total = 0
    
    # Sample num_opponent_samples random opponent hands
    if len(remaining_deck) < 2:
        return 0.0, 0.0
    
    for _ in range(num_opponent_samples):
        # Sample 2 cards for opponent
        opp_hand = random.sample(remaining_deck, 2)
        
        # Evaluate current state
        opp_score_now = HandEvaluator.evaluate(opp_hand, board)
        
        # Determine current rank
        if my_score_now > opp_score_now:
            current_rank = 'AHEAD'
        elif my_score_now < opp_score_now:
            current_rank = 'BEHIND'
        else:
            current_rank = 'TIED'
        
        # Create deck without opponent cards for next card simulation
        next_card_deck = [c for c in remaining_deck 
                         if not (c.suit == opp_hand[0].suit and c.rank == opp_hand[0].rank)
                         and not (c.suit == opp_hand[1].suit and c.rank == opp_hand[1].rank)]
        
        # Simulate all possible next cards
        for next_card in next_card_deck:
            future_board = board + [next_card]
            
            # Evaluate future state
            my_score_future = HandEvaluator.evaluate(hole_cards, future_board)
            opp_score_future = HandEvaluator.evaluate(opp_hand, future_board)
            
            # Determine future rank
            if my_score_future > opp_score_future:
                future_rank = 'AHEAD'
            elif my_score_future < opp_score_future:
                future_rank = 'BEHIND'
            else:
                future_rank = 'TIED'
            
            # Update PPot counters
            if current_rank in ['BEHIND', 'TIED']:
                behind_total += 1
                if future_rank == 'AHEAD':
                    behind_improved += 1
            
            # Update NPot counters
            if current_rank == 'AHEAD':
                ahead_total += 1
                if future_rank == 'BEHIND':
                    ahead_worsened += 1
    
    # Calculate probabilities
    ppot = behind_improved / behind_total if behind_total > 0 else 0.0
    npot = ahead_worsened / ahead_total if ahead_total > 0 else 0.0
    
    return ppot, npot

def get_8_features(hole_cards: List[Card], board: List[Card], 
                  street: Union[str, int]) -> List[float]:
    """
    Get all 8 features for observation vector
    
    Args:
        hole_cards: 2 hole cards
        board: Community cards (0-5 cards)
        street: 'preflop', 'flop', 'turn', 'river', or int 0-3
    
    Returns:
        List of 8 floats:
        [hs/equity, ppot/0, npot/0, hand_index_norm, is_pre, is_flop, is_turn, is_river]
    """
    # Normalize street to string if it's an integer or enum
    # Handle float (e.g. 0.0) by converting to int
    if isinstance(street, float):
        street = int(street)

    if isinstance(street, int):
        street_map = {0: 'preflop', 1: 'flop', 2: 'turn', 3: 'river'}
        street = street_map.get(street, 'preflop')
    else:
        # Ensure lowercase string
        street = str(street).lower()
        if 'preflop' in street: street = 'preflop'
        elif 'flop' in street: street = 'flop'
        elif 'turn' in street: street = 'turn'
        elif 'river' in street or 'showdown' in street: street = 'river'

    # Street one-hot (mutually exclusive using if-elif)
    if street == 'preflop':
        is_preflop, is_flop, is_turn, is_river = 1.0, 0.0, 0.0, 0.0
    elif street == 'flop':
        is_preflop, is_flop, is_turn, is_river = 0.0, 1.0, 0.0, 0.0
    elif street == 'turn':
        is_preflop, is_flop, is_turn, is_river = 0.0, 0.0, 1.0, 0.0
    elif street == 'river':
        is_preflop, is_flop, is_turn, is_river = 0.0, 0.0, 0.0, 1.0
    else:
        # Safety: default to preflop if unknown street
        is_preflop, is_flop, is_turn, is_river = 1.0, 0.0, 0.0, 0.0
    
    # Preflop-specific logic
    if street == 'preflop':
        # Preflop: [equity, 0, 0, hand_index_int, street_onehot]
        equity = get_preflop_equity(hole_cards[0], hole_cards[1])
        hand_index = get_hand_index_preflop(hole_cards[0], hole_cards[1])
        # Return integer ID (0-168) as float for embedding layer
        hand_index_float = float(hand_index)
        
        # Validation: Preflop constraints
        assert 0 <= hand_index <= 168, f"Preflop hand_index {hand_index} out of range [0, 168]"
        assert is_preflop == 1.0, f"Preflop flag mismatch: is_preflop={is_preflop}"
        
        # NaN/Inf check for equity
        import math
        if math.isnan(equity) or math.isinf(equity):
            equity = 0.5  # Default fallback
        equity = max(0.0, min(1.0, equity))  # Extra safety clamp
        
        features = [equity, 0.0, 0.0, hand_index_float, 
                    is_preflop, is_flop, is_turn, is_river]
        
        # Final validation: PPot and NPot must be 0.0 for preflop
        assert features[1] == 0.0, f"Preflop PPot must be 0.0, got {features[1]}"
        assert features[2] == 0.0, f"Preflop NPot must be 0.0, got {features[2]}"
        
        return features
    
    else:
        # Postflop: [hs, ppot, npot, hand_index_int, street_onehot]
        hs = get_hand_strength_postflop(hole_cards, board)
        
        # Calculate PPot/NPot only if not river
        if street == 'river':
            ppot, npot = 0.0, 0.0
        else:
            ppot, npot = calculate_ppot_npot(hole_cards, board)
        
        # Hand index based on category
        score = HandEvaluator.evaluate(hole_cards, board)
        hand_index = get_hand_index_postflop(score)
        # Return integer ID (169-178) as float for embedding layer
        hand_index_float = float(hand_index)
        
        # NaN/Inf checks for all probability values
        import math
        if math.isnan(hs) or math.isinf(hs):
            hs = 0.0  # Default fallback
        if math.isnan(ppot) or math.isinf(ppot):
            ppot = 0.0
        if math.isnan(npot) or math.isinf(npot):
            npot = 0.0
        
        # Extra safety clamp
        hs = max(0.0, min(1.0, hs))
        ppot = max(0.0, min(1.0, ppot))
        npot = max(0.0, min(1.0, npot))
        
        # Validation: Postflop constraints
        assert 169 <= hand_index <= 178, f"Postflop hand_index {hand_index} out of range [169, 178]"
        assert is_preflop == 0.0, f"Postflop flag mismatch: is_preflop={is_preflop}"
        assert 0.0 <= ppot <= 1.0, f"PPot {ppot} out of range [0, 1]"
        assert 0.0 <= npot <= 1.0, f"NPot {npot} out of range [0, 1]"
        assert 0.0 <= hs <= 1.0, f"HS {hs} out of range [0, 1]"
        
        features = [hs, ppot, npot, hand_index_float,
                    is_preflop, is_flop, is_turn, is_river]
        
        return features

@lru_cache(maxsize=200000)
def get_8_features_cached(hole_tuple, board_tuple, street):
    """
    Cached version of get_8_features to speed up training.
    Inputs must be tuples of (suit, rank) to be hashable.
    """
    # Convert tuples back to Card objects
    hole_cards = [Card(suit, rank) for (suit, rank) in hole_tuple]
    board = [Card(suit, rank) for (suit, rank) in board_tuple]
    
    return get_8_features(hole_cards, board, street)
