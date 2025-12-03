"""
Player state management
"""
from typing import List, Optional
from .cards import Card


class PlayerState:
    """Player states during a hand"""
    ACTIVE = "active"          # Still in the hand
    FOLDED = "folded"          # Folded
    ALL_IN = "all_in"          # All-in (no more chips)


class Player:
    """Represents a poker player"""
    
    def __init__(self, player_id: int, chips: float):
        """
        Initialize a player
        
        Args:
            player_id: Player identifier (0 or 1 for heads-up)
            chips: Starting chip count
        """
        self.player_id = player_id
        self.chips = float(chips)  # Total chips (not in pot)
        
        # Hand state
        self.hand: List[Card] = []
        self.state = PlayerState.ACTIVE
        
        # Betting state
        self.bet_this_round = 0.0  # Amount bet in current betting round
        self.bet_this_hand = 0.0   # Total amount bet in current hand
        
        # Action tracking
        self.last_action: Optional[str] = None  # Last action taken this round
        
    def deal_hand(self, cards: List[Card]):
        """Deal hole cards to player"""
        self.hand = cards
        
    def reset_for_new_hand(self):
        """Reset player state for a new hand"""
        self.hand = []
        self.state = PlayerState.ACTIVE
        self.bet_this_round = 0.0
        self.bet_this_hand = 0.0
        self.last_action = None
    
    def reset_for_new_round(self):
        """Reset bet for new betting round (flop, turn, river)"""
        self.bet_this_round = 0.0
        self.last_action = None
    
    def place_bet(self, amount: float) -> float:
        """
        Place a bet (removes chips from stack)
        
        Args:
            amount: Amount to bet
            
        Returns:
            Actual amount bet (may be less if all-in)
        """
        # Ensure amount doesn't exceed chips
        actual_amount = min(amount, self.chips)
        
        self.chips -= actual_amount
        self.bet_this_round += actual_amount
        self.bet_this_hand += actual_amount
        
        # Check if all-in
        if self.chips <= 0:
            self.state = PlayerState.ALL_IN
            
        return actual_amount

    def post_blind(self, amount: float) -> float:
        """Post a blind (same as place_bet but semantically distinct)"""
        return self.place_bet(amount)
    
    def fold(self):
        """Fold hand"""
        self.state = PlayerState.FOLDED
    
    def is_active(self) -> bool:
        """Check if player is still active in hand"""
        return self.state == PlayerState.ACTIVE
    
    def is_all_in(self) -> bool:
        """Check if player is all-in"""
        return self.state == PlayerState.ALL_IN
    
    def is_folded(self) -> bool:
        """Check if player has folded"""
        return self.state == PlayerState.FOLDED
    
    def can_bet(self) -> bool:
        """Check if player can still bet (not folded, not all-in)"""
        return self.is_active() and self.chips > 0
    
    def __str__(self) -> str:
        cards_str = ', '.join(str(c) for c in self.hand) if self.hand else 'No cards'
        return f"Player {self.player_id}: {self.chips:.0f} chips, {cards_str}, {self.state}"
    
    def __repr__(self) -> str:
        return f"Player(id={self.player_id}, chips={self.chips}, state={self.state})"
