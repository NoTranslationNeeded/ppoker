"""
Potential State for Delta-Equity Reward

Manages state information and calculates Potential Function Φ(s)
"""

from typing import Tuple
from poker_engine import Card, PokerGame
from poker_rl.utils.range_equity_calculator import FastEquityCalculator


class PotentialState:
    """
    Tracks game state for Potential-based Reward Shaping
    
    Φ(s) = Equity(s) × Pot(s) - ChipsInvested(s)
    
    Optimizations:
    - Equity caching (recalculate only when board changes)
    - Lazy evaluation
    - Range-based equity (vs random opponent)
    """
    
    def __init__(self, game: PokerGame, player_id: int, big_blind: float, hand_start_stack: float = None):
        """
        Initialize Potential State
        
        Args:
            game: Current poker game state
            player_id: 0 or 1
            big_blind: Big blind size for normalization
            hand_start_stack: Starting stack for this hand (optional, will use current chips if None)
        """
        self.player_id = player_id
        self.big_blind = big_blind
        
        # Current game state
        self.hole_cards = game.players[player_id].hand
        self.board = game.community_cards
        self.pot = game.get_pot_size()
        self.street = game.street.value  # 'preflop', 'flop', etc.
        
        # Calculate chips invested this hand
        player = game.players[player_id]
        if hand_start_stack is not None:
            self.invested_this_hand = hand_start_stack - player.chips
            self.hand_start_stack = hand_start_stack
        else:
            # If no hand_start_stack provided, assume no investment yet
            self.invested_this_hand = 0.0
            self.hand_start_stack = player.chips
        
        # Total Chips (will be set from env)
        # For now, approximate as 2x this player's starting stack
        # This will be properly set when called from env
        self.total_chips = None
        
        # Equity caching
        self._cached_equity = None
        self._board_hash = None
    
    def set_total_chips(self, total_chips: float):
        """Set total chips for normalization"""
        self.total_chips = total_chips
    
    def get_equity(self) -> float:
        """
        Get Range-based Equity with caching
        
        Recalculates only when board changes
        
        Returns:
            Equity vs random opponent (0.0-1.0)
        """
        # Calculate board hash
        board_hash = hash(tuple((c.suit, c.rank) for c in self.board))
        
        # Check cache
        if self._cached_equity is not None and self._board_hash == board_hash:
            # Cache hit - return immediately
            return self._cached_equity
        
        # Cache miss or board changed - recalculate
        self._cached_equity = FastEquityCalculator.calculate_equity_vs_range(
            hole_cards=self.hole_cards,
            board=self.board,
            street=self.street
        )
        self._board_hash = board_hash
        
        return self._cached_equity
    
    def calculate_potential(self) -> float:
        """
        Calculate Potential Function Φ(s)
        
        Φ(s) = Equity × Pot - ChipsInvested
        
        Changes from original design:
        - NO street weights (all w=1.0)
        - Range-based equity (opponent = random)
        - Total Chips normalization (Problem #2 SOLVED!)
        
        Returns:
            Potential value (normalized)
        """
        # Get equity (cached if possible)
        equity = self.get_equity()
        
        # Expected Value = Equity × Pot
        expected_value = equity * self.pot
        
        # Risk-adjusted = EV - Investment
        risk_adjusted = expected_value - self.invested_this_hand
        
        # Normalize by Total Chips (consistent with Terminal Reward)
        # This ensures no Deep Stack Bias and PPO-friendly scale
        if self.total_chips is not None and self.total_chips > 0:
            normalized = risk_adjusted / self.total_chips
        else:
            # Fallback: use big_blind (should not happen in normal flow)
            normalized = risk_adjusted / self.big_blind / 100.0
        
        return normalized
    
    def __repr__(self) -> str:
        """Debug representation"""
        equity = self.get_equity()
        potential = self.calculate_potential()
        
        return (
            f"PotentialState(player={self.player_id}, street={self.street}, "
            f"pot={self.pot:.1f}, invested={self.invested_this_hand:.1f}, "
            f"equity={equity:.3f}, Phi={potential:.4f})"
        )
