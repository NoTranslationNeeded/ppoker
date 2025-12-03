"""
Card and Deck implementation for poker engine
"""
import random
from typing import List


class Card:
    """Represents a single playing card"""
    
    # Suits
    SPADES = 'S'
    HEARTS = 'H'
    DIAMONDS = 'D'
    CLUBS = 'C'
    
    SUITS = [SPADES, HEARTS, DIAMONDS, CLUBS]
    
    # Ranks (2-9, T, J, Q, K, A)
    RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
    
    # Rank values for comparison
    RANK_VALUES = {rank: i for i, rank in enumerate(RANKS)}
    
    def __init__(self, suit: str, rank: str):
        """
        Initialize a card
        
        Args:
            suit: One of SPADES, HEARTS, DIAMONDS, CLUBS
            rank: One of 2-9, T, J, Q, K, A
        """
        if suit not in self.SUITS:
            raise ValueError(f"Invalid suit: {suit}")
        if rank not in self.RANKS:
            raise ValueError(f"Invalid rank: {rank}")
            
        self.suit = suit
        self.rank = rank
    
    def __str__(self) -> str:
        """String representation (RLCard compatible format: SA, H5, etc.)"""
        return f"{self.suit}{self.rank}"
    
    def pretty_str(self, color: bool = False) -> str:
        """
        Pretty string with Unicode suit symbols for display
        
        Args:
            color: If True, add ANSI color codes
        """
        # Unicode suit symbols
        suit_symbols = {
            'S': '♠',  # Spades
            'H': '♥',  # Hearts
            'D': '♦',  # Diamonds
            'C': '♣'   # Clubs
        }
        
        s = f"{suit_symbols[self.suit]}{self.rank}"
        
        if color:
            # ANSI Color codes
            RED = "\033[91m"
            BLUE = "\033[94m"
            GREEN = "\033[92m"
            RESET = "\033[0m"
            
            if self.suit in ['H', 'D']:
                return f"{RED}{s}{RESET}"
            elif self.suit == 'S':
                return f"{BLUE}{s}{RESET}"
            else:
                return f"{GREEN}{s}{RESET}"
                
        return s
    
    def __repr__(self) -> str:
        return f"Card('{self.suit}{self.rank}')"
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, Card):
            return False
        return self.suit == other.suit and self.rank == other.rank
    
    def __hash__(self) -> int:
        return hash((self.suit, self.rank))
        
    def to_int(self) -> int:
        """
        Convert to integer index (0-51)
        Order: 2S, 2H, 2D, 2C, 3S...
        """
        rank_idx = self.RANKS.index(self.rank)
        suit_idx = self.SUITS.index(self.suit)
        return rank_idx * 4 + suit_idx
    
    @classmethod
    def from_string(cls, card_str: str) -> 'Card':
        """
        Create a card from string (e.g., 'SA', 'H5', 'DK')
        
        Args:
            card_str: Two-character string (suit + rank)
        """
        if len(card_str) != 2:
            raise ValueError(f"Card string must be 2 characters, got: {card_str}")
        return cls(suit=card_str[0], rank=card_str[1])


class Deck:
    """Standard 52-card deck"""
    
    def __init__(self):
        """Initialize a fresh shuffled deck"""
        self.cards: List[Card] = []
        self.reset()
    
    def reset(self):
        """Reset deck to all 52 cards and shuffle"""
        self.cards = [
            Card(suit, rank)
            for suit in Card.SUITS
            for rank in Card.RANKS
        ]
        self.shuffle()
    
    def shuffle(self):
        """Shuffle the deck in place"""
        random.shuffle(self.cards)
    
    def deal(self, n: int = 1) -> List[Card]:
        """
        Deal n cards from the top of the deck
        
        Args:
            n: Number of cards to deal
            
        Returns:
            List of dealt cards
            
        Raises:
            ValueError: If not enough cards in deck
        """
        if n > len(self.cards):
            raise ValueError(f"Not enough cards in deck. Requested: {n}, Available: {len(self.cards)}")
        
        dealt = self.cards[:n]
        self.cards = self.cards[n:]
        return dealt
    
    def deal_one(self) -> Card:
        """Deal a single card"""
        return self.deal(1)[0]
    
    def remaining(self) -> int:
        """Return number of cards remaining in deck"""
        return len(self.cards)
    
    def __len__(self) -> int:
        return len(self.cards)
    
    def __str__(self) -> str:
        return f"Deck({len(self.cards)} cards)"
