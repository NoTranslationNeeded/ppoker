"""
Action types and validation
"""
from enum import Enum
from typing import Optional


class ActionType(Enum):
    """Possible poker actions"""
    FOLD = "fold"
    CHECK = "check"
    CALL = "call"
    BET = "bet"
    RAISE = "raise"
    ALL_IN = "all_in"


class Action:
    """Represents a poker action"""
    
    def __init__(self, action_type: ActionType, amount: float = 0.0):
        """
        Initialize an action
        
        Args:
            action_type: Type of action
            amount: Bet/raise amount (0 for fold/check/call)
        """
        self.action_type = action_type
        self.amount = amount
    
    @classmethod
    def fold(cls) -> 'Action':
        """Create a fold action"""
        return cls(ActionType.FOLD, 0.0)
    
    @classmethod
    def check(cls) -> 'Action':
        """Create a check action"""
        return cls(ActionType.CHECK, 0.0)
    
    @classmethod
    def call(cls, amount: float) -> 'Action':
        """Create a call action"""
        return cls(ActionType.CALL, amount)
    
    @classmethod
    def bet(cls, amount: float) -> 'Action':
        """Create a bet action"""
        return cls(ActionType.BET, amount)
    
    @classmethod
    def raise_to(cls, amount: float) -> 'Action':
        """Create a raise action"""
        return cls(ActionType.RAISE, amount)
    
    @classmethod
    def all_in(cls, amount: float) -> 'Action':
        """Create an all-in action"""
        return cls(ActionType.ALL_IN, amount)
    
    def is_aggressive(self) -> bool:
        """Check if action is aggressive (bet/raise/all-in)"""
        return self.action_type in [ActionType.BET, ActionType.RAISE, ActionType.ALL_IN]
    
    def __str__(self) -> str:
        if self.amount > 0:
            return f"{self.action_type.value}({self.amount:.0f})"
        return self.action_type.value
    
    def __repr__(self) -> str:
        return f"Action({self.action_type.value}, {self.amount})"


class ActionValidator:
    """Validates poker actions according to TDA rules"""
    
    @staticmethod
    def validate_action(action: Action, 
                       player_chips: float,
                       current_bet: float,
                       player_bet_this_round: float,
                       min_raise: float,
                       pot_size: float) -> tuple[bool, Optional[str]]:
        """
        Validate if an action is legal
        
        Args:
            action: Action to validate
            player_chips: Player's remaining chips
            current_bet: Current bet to match
            player_bet_this_round: Amount player has already bet this round
            min_raise: Minimum raise amount
            pot_size: Current pot size
            
        Returns:
            (is_valid, error_message)
        """
        amount_to_call = current_bet - player_bet_this_round
        
        if action.action_type == ActionType.FOLD:
            return (True, None)
        
        elif action.action_type == ActionType.CHECK:
            if amount_to_call > 0:
                return (False, "Cannot check when facing a bet")
            return (True, None)
        
        elif action.action_type == ActionType.CALL:
            if amount_to_call <= 0:
                return (False, "Nothing to call")
            if player_chips < amount_to_call:
                return (False, "Not enough chips to call (should be all-in)")
            return (True, None)
        
        elif action.action_type == ActionType.BET:
            if current_bet > 0:
                return (False, "Cannot bet when facing a bet (use raise)")
            if action.amount > player_chips:
                return (False, "Bet exceeds chip count")
            # In no-limit, minimum bet is typically BB (we'll check in game logic)
            return (True, None)
        
        elif action.action_type == ActionType.RAISE:
            if current_bet == 0:
                return (False, "Cannot raise when no bet exists (use bet)")
            
            # Minimum raise = last raise amount
            # action.amount should be the TOTAL bet, not the raise increment
            total_bet = action.amount
            raise_increment = total_bet - current_bet
            
            # Simple rule: raise increment must meet minimum
            # No 50% rule needed for online poker (no ambiguous actions)
            if raise_increment < min_raise:
                return (False, f"Raise must be at least {min_raise:.0f}")
            
            # Check if player has enough chips for this raise
            if total_bet > player_chips + player_bet_this_round:
                return (False, "Not enough chips (use all-in instead)")
            
            return (True, None)
        
        elif action.action_type == ActionType.ALL_IN:
            # All-in is always valid if player has chips
            if player_chips <= 0:
                return (False, "No chips to go all-in with")
            return (True, None)
        
        return (False, "Unknown action type")
    
    @staticmethod
    def get_legal_actions(player_chips: float,
                          current_bet: float,
                          player_bet_this_round: float,
                          min_raise: float,
                          big_blind: float) -> list[ActionType]:
        """
        Get list of legal action types for current situation
        
        Args:
            player_chips: Player's remaining chips
            current_bet: Current bet to match
            player_bet_this_round: Amount player has already bet this round
            min_raise: Minimum raise amount
            big_blind: Big blind amount
            
        Returns:
            List of legal ActionTypes
        """
        legal_actions = []
        amount_to_call = current_bet - player_bet_this_round
        
        # Fold is always legal (even when can check)
        legal_actions.append(ActionType.FOLD)
        
        # Check or Call
        if amount_to_call == 0:
            legal_actions.append(ActionType.CHECK)
        else:
            if player_chips >= amount_to_call:
                legal_actions.append(ActionType.CALL)
        
        # Bet or Raise
        if player_chips > amount_to_call:
            if current_bet == 0:
                # Can bet
                legal_actions.append(ActionType.BET)
            else:
                # Can raise if have enough for min raise
                if player_chips - amount_to_call >= min_raise:
                    legal_actions.append(ActionType.RAISE)
        
        # All-in (if have chips)
        if player_chips > 0:
            legal_actions.append(ActionType.ALL_IN)
        
        return legal_actions
