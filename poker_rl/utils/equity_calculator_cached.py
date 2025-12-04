"""
Cached version function
"""
from functools import lru_cache

@lru_cache(maxsize=10000)
def get_8_features_cached(hole_tuple, board_tuple, street):
    """
    Cached version of get_8_features to speed up training.
    Inputs must be tuples of (suit, rank) to be hashable.
    """
    # Convert tuples back to Card objects
    hole_cards = [Card(suit, rank) for suit, rank in hole_tuple]
    board = [Card(suit, rank) for suit, rank in board_tuple]
    
    return get_8_features(hole_cards, board, street)
