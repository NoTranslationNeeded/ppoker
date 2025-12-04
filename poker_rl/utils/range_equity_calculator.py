"""
Optimized Equity Calculator for Delta-Equity Reward

Provides Range-based Equity calculation:
- Preflop: O(1) table lookup
- Postflop: Lightweight Monte Carlo (n=100)
"""

import json
import os
from typing import List, Tuple
from poker_engine import Card

class PreflopEquityCache:
    """
    Singleton for Preflop Equity Table
    
    Loads preflop_equity_table.json once and provides O(1) lookup
    """
    _instance = None
    _equity_table = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._load_table()
        return cls._instance
    
    @classmethod
    def _load_table(cls):
        """Load preflop equity table (called once at initialization)"""
        if cls._equity_table is not None:
            return
        
        table_path = os.path.join(
            os.path.dirname(__file__),
            'preflop_equity_table.json'
        )
        
        with open(table_path, 'r') as f:
            cls._equity_table = json.load(f)
    
    @classmethod
    def get_equity(cls, hand_index: int) -> float:
        """
        Get preflop equity for given hand index
        
        Args:
            hand_index: 0-168 (AA=0, KK=12, ..., 72o=168)
        
        Returns:
            Equity vs random range (0.0-1.0)
        """
        if cls._equity_table is None:
            cls._load_table()
        
        return float(cls._equity_table[str(hand_index)])


class FastEquityCalculator:
    """
    Unified Range-based Equity Calculator
    
    Optimizations:
    - Preflop: Table lookup (1000x faster)
    - Postflop: Monte Carlo n=100 (10x faster than n=1000)
    - Caching: Avoid redundant calculations
    """
    
    @staticmethod
    def calculate_equity_vs_range(
        hole_cards: List[Card],
        board: List[Card],
        street: str
    ) -> float:
        """
        Calculate equity vs random opponent range
        
        Args:
            hole_cards: 2 hole cards
            board: 0-5 community cards
            street: 'preflop', 'flop', 'turn', 'river'
        
        Returns:
            Equity (0.0-1.0) vs random range
        """
        if street == 'preflop':
            return FastEquityCalculator._preflop_equity(hole_cards)
        else:
            return FastEquityCalculator._postflop_equity(
                hole_cards, board, street
            )
    
    @staticmethod
    def _preflop_equity(hole_cards: List[Card]) -> float:
        """
        Preflop: O(1) table lookup
        
        Returns equity vs random opponent
        """
        from poker_rl.utils.equity_calculator import get_hand_index_preflop
        
        # Get hand index (0-168)
        hand_index = get_hand_index_preflop(hole_cards[0], hole_cards[1])
        
        # Table lookup
        equity = PreflopEquityCache.get_equity(hand_index)
        
        return equity
    
    @staticmethod
    def _postflop_equity(
        hole_cards: List[Card],
        board: List[Card],
        street: str
    ) -> float:
        """
        Postflop: Monte Carlo with n=100
        
        Uses existing get_8_features which already calculates
        Hand Strength as equity vs random opponent
        """
        from poker_rl.utils.equity_calculator import get_8_features_cached
        
        # Convert to tuples for caching
        hole_tuple = tuple((c.suit, c.rank) for c in hole_cards)
        board_tuple = tuple((c.suit, c.rank) for c in board)
        
        # get_8_features returns:
        # [0]: Hand Strength (equity-like)
        # [4]: Effective Hand Strength (HS + PPot - NPot)
        features = get_8_features_cached(hole_tuple, board_tuple, street)
        
        # Use Hand Strength as equity
        # This is already calculated vs random opponent
        equity = features[0]
        
        return equity
