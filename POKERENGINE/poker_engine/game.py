"""
Core Texas Hold'em Game Logic
TDA 2024 Compliant - With Side Pot Support
"""
from typing import List, Optional, Tuple, Dict
from enum import Enum
import copy

from .cards import Card, Deck
from .player import Player, PlayerState
from .actions import Action, ActionType, ActionValidator
from .evaluator import HandEvaluator


class Street(Enum):
    """Betting rounds"""
    PREFLOP = "preflop"
    FLOP = "flop"
    TURN = "turn"
    RIVER = "river"
    SHOWDOWN = "showdown"


class Pot:
    """
    Represents a pot (main or side) in poker
    
    TDA Rule 21: Side pot distribution
    """
    def __init__(self, amount: float, eligible_players: List[int]):
        """
        Initialize a pot
        
        Args:
            amount: Pot size in chips
            eligible_players: List of player IDs who can win this pot
        """
        self.amount = amount
        self.eligible_players = eligible_players
    
    def __repr__(self) -> str:
        return f"Pot({self.amount:.0f}, eligible={self.eligible_players})"


class PokerGame:
    """
    Heads-up No-Limit Texas Hold'em Game
    TDA 2024 Compliant
    """
    
    def __init__(self, small_blind: float, big_blind: float):
        """
        Initialize poker game
        
        Args:
            small_blind: Small blind amount
            big_blind: Big blind amount
        """
        self.small_blind = small_blind
        self.big_blind = big_blind
        
        # Game state
        self.players: List[Player] = []
        self.deck = Deck()
        self.community_cards: List[Card] = []
        self.street = Street.PREFLOP
        
        # Pot management
        self.pot = 0.0
        self.current_bet = 0.0
        self.last_raise_amount = big_blind
        
        # Action tracking
        self.actions_this_round = 0
        self.bb_has_option = False  # Pre-flop BB option
        
        # TDA Rule 34-B: Heads-up: SB = Button
        self.button_position = 0
        
        # Tracking
        self.hand_number = 0
        self.is_hand_over = False
        self.winner: Optional[int] = None
        self.hand_history: List[str] = []
        self.current_actor = -1
        
    def start_hand(self, players_info: list, button: int = 0):
        """
        Start a new hand with N players
        
        Args:
            players_info: List of chip counts (float) OR List of (id, chips) tuples
            button: Button position index
        """
        self.hand_number += 1
        self.button_position = button
        self.is_hand_over = False
        self.winner = None
        self.hand_history = []
        
        num_players = len(players_info)
        if num_players < 2:
            raise ValueError("Need at least 2 players")
            
        # Initialize players
        self.players = []
        for i, info in enumerate(players_info):
            if isinstance(info, (int, float)):
                # Old style: just chips, auto-assign ID based on index
                player = Player(i, float(info))
            else:
                # New style: (id, chips) tuple/list
                pid, chips = info
                player = Player(pid, float(chips))
            self.players.append(player)
            
        # Reset deck and deal hands
        self.deck.reset()
        for player in self.players:
            player.hand = self.deck.deal(2)
            
        self.community_cards = []
        self.street = Street.PREFLOP
        self.pot = 0.0
        self.current_bet = 0.0
        self.min_raise = self.big_blind
        self.last_raise_amount = self.big_blind
        self.actions_this_round = 0
        self.bb_has_option = False
        
        # Post Blinds
        sb_amount = self.small_blind
        bb_amount = self.big_blind
        
        if num_players == 2:
            # Heads-up: Button is SB, Other is BB
            sb_pos = self.button_position
            bb_pos = (self.button_position + 1) % num_players
        else:
            # Normal: SB is left of Button, BB is left of SB
            sb_pos = (self.button_position + 1) % num_players
            bb_pos = (self.button_position + 2) % num_players
            
        # Post SB
        sb_player = self.players[sb_pos]
        actual_sb = sb_player.post_blind(sb_amount)
        sb_player.last_action = "Post SB"
        # self.pot += actual_sb  <-- REMOVED: Will be collected at end of round
        
        # Post BB
        bb_player = self.players[bb_pos]
        actual_bb = bb_player.post_blind(bb_amount)
        bb_player.last_action = "Post BB"
        # self.pot += actual_bb  <-- REMOVED: Will be collected at end of round
        
        self.current_bet = bb_amount
        
        # BB Option logic
        if actual_bb >= bb_amount: # Full BB posted
            self.bb_has_option = True
            
        self.hand_history.append(f"Hand #{self.hand_number} started with {num_players} players")
        self.hand_history.append(f"Button: P{self.button_position}, SB: P{sb_pos}, BB: P{bb_pos}")
        
        # Set initial actor
        # Pre-flop: Player after BB acts first
        self.current_actor = (bb_pos + 1) % num_players

    def get_current_player(self) -> int:
        """
        Get current player to act
        
        Returns:
            Player ID, or -1 if betting round is over
        """
        return self.current_actor
    
    def _advance_actor(self):
        """
        Move current_actor to the next eligible player.
        If round is over, set current_actor to -1.
        """
        if self._is_betting_round_over():
            self.current_actor = -1
            return

        num_players = len(self.players)
        next_idx = (self.current_actor + 1) % num_players
        
        # Loop to find next eligible player
        # Safety break to prevent infinite loop
        for _ in range(num_players):
            player = self.players[next_idx]
            if not player.is_folded() and not player.is_all_in():
                self.current_actor = next_idx
                return
            next_idx = (next_idx + 1) % num_players
            
        # Should not happen if _is_betting_round_over() is correct
        self.current_actor = -1

    def process_action(self, player_id: int, action: Action) -> Tuple[bool, Optional[str]]:
        """
        Process a player action
        
        Args:
            player_id: Player making the action
            action: Action to process
            
        Returns:
            (success, error_message)
        """
        if player_id != self.current_actor:
            return (False, "Not your turn")
            
        player = self.players[player_id]
        
        # Validate action
        is_valid, error = ActionValidator.validate_action(
            action,
            player.chips,
            self.current_bet,
            player.bet_this_round,
            self.min_raise,
            self.get_pot_size()
        )
        
        if not is_valid:
            return (False, error)
        
        # Track action
        self.actions_this_round += 1
        
        # Clear BB option if BB acts
        # BB option is only relevant if no one raised.
        # If BB checks, option used. If BB raises, option used.
        if self.street == Street.PREFLOP and self.bb_has_option:
             # If BB acts, option is consumed
             bb_pos = (self.button_position + 2) % len(self.players) if len(self.players) > 2 else (1 - self.button_position)
             if player_id == bb_pos:
                 self.bb_has_option = False
        
        # Process action logic
        if action.action_type == ActionType.FOLD:
            player.fold()
            player.last_action = "FOLDED"
            self.hand_history.append(f"P{player_id} folds")
            self._check_hand_over()
            if self.is_hand_over:
                self.current_actor = -1
                return (True, None)
            
        elif action.action_type == ActionType.CHECK:
            player.last_action = "Check"
            self.hand_history.append(f"P{player_id} checks")
            
        elif action.action_type == ActionType.CALL:
            amount_to_call = self.current_bet - player.bet_this_round
            actual_amount = player.place_bet(amount_to_call)
            player.last_action = f"Call {actual_amount:.0f}"
            self.hand_history.append(f"P{player_id} calls {actual_amount:.0f}")
            
        elif action.action_type == ActionType.BET:
            actual_amount = player.place_bet(action.amount)
            self.current_bet = player.bet_this_round
            self.last_raise_amount = action.amount
            self.min_raise = action.amount
            player.last_action = f"Bet {actual_amount:.0f}"
            self.hand_history.append(f"P{player_id} bets {actual_amount:.0f}")
            
        elif action.action_type == ActionType.RAISE:
            amount_to_add = action.amount - player.bet_this_round
            actual_amount = player.place_bet(amount_to_add)
            
            old_bet = self.current_bet
            self.current_bet = player.bet_this_round
            self.last_raise_amount = self.current_bet - old_bet
            self.min_raise = self.last_raise_amount
            
            player.last_action = f"Raise To {self.current_bet:.0f}"
            self.hand_history.append(f"P{player_id} raises to {self.current_bet:.0f}")
            
        elif action.action_type == ActionType.ALL_IN:
            actual_amount = player.place_bet(player.chips)
            
            if player.bet_this_round > self.current_bet:
                old_bet = self.current_bet
                self.current_bet = player.bet_this_round
                raise_amount = self.current_bet - old_bet
                
                if raise_amount >= self.min_raise:
                    self.last_raise_amount = raise_amount
                    self.min_raise = raise_amount
            
            player.last_action = f"All-in {actual_amount:.0f}"
            self.hand_history.append(f"P{player_id} all-in for {actual_amount:.0f}")
        
        # Check if betting round is over
        if self._is_betting_round_over():
            self._advance_street()
        else:
            self._advance_actor()
        
        return (True, None)
    
    def _is_betting_round_over(self) -> bool:
        """Check if current betting round is over"""
        active_players = [p for p in self.players if not p.is_folded()]
        num_active = len(active_players)
        
        if num_active <= 1:
            return True
            
        # Pre-flop: BB option logic
        # If BB has option, round not over until BB acts
        if self.street == Street.PREFLOP and self.bb_has_option:
            return False
            
        # If everyone checked (current_bet == 0)
        if self.current_bet == 0:
            if self.actions_this_round < num_active:
                return False
            return True
            
        # If there is a bet
        # Round over if everyone active (not all-in) has matched the bet
        for p in active_players:
            if not p.is_all_in() and p.bet_this_round < self.current_bet:
                return False
                
        return True
    
    def _advance_street(self):
        """Advance to next street"""
        # Collect bets to pot
        for player in self.players:
            self.pot += player.bet_this_round
            player.reset_for_new_round()
        
        self.current_bet = 0.0
        self.min_raise = self.big_blind
        self.last_raise_amount = self.big_blind
        self.actions_this_round = 0
        
        # Deal community cards
        if self.street == Street.PREFLOP:
            self.deck.deal_one()  # Burn card
            self.community_cards.extend(self.deck.deal(3))
            self.street = Street.FLOP
            self.hand_history.append(f"Flop: {', '.join(c.pretty_str() for c in self.community_cards)}")
            
        elif self.street == Street.FLOP:
            self.deck.deal_one()  # Burn card
            self.community_cards.append(self.deck.deal_one())
            self.street = Street.TURN
            self.hand_history.append(f"Turn: {self.community_cards[-1].pretty_str()}")
            
        elif self.street == Street.TURN:
            self.deck.deal_one()  # Burn card
            self.community_cards.append(self.deck.deal_one())
            self.street = Street.RIVER
            self.hand_history.append(f"River: {self.community_cards[-1].pretty_str()}")
            
        elif self.street == Street.RIVER:
            self._showdown()
            return  # Stop recursion at showdown
            
        # Reset actor for new street
        # Post-flop: SB (Button+1) acts first
        if self.street != Street.SHOWDOWN:
            num_players = len(self.players)
            start_pos = (self.button_position + 1) % num_players
            self.current_actor = start_pos
            
            # Find first eligible actor
            found_actor = False
            curr = start_pos
            for _ in range(num_players):
                p = self.players[curr]
                if not p.is_folded() and not p.is_all_in():
                    self.current_actor = curr
                    found_actor = True
                    break
                curr = (curr + 1) % num_players
            
            if not found_actor:
                pass

        # Check if we should auto-advance (all-in scenario)
        if self.street != Street.SHOWDOWN and not self.is_hand_over:
            if self._should_auto_advance():
                self._advance_street()

    def _should_auto_advance(self) -> bool:
        """
        Check if game should auto-advance
        (e.g., all players all-in, or only 1 player with chips against all-ins)
        """
        active_players = [p for p in self.players if not p.is_folded()]
        players_with_chips = [p for p in active_players if p.chips > 0]
        
        # If 0 or 1 player has chips, and we have > 1 active players, we just run it out
        if len(players_with_chips) <= 1 and len(active_players) > 1:
            return True
        return False

    
    def _calculate_pots(self) -> List[Pot]:
        """
        Calculate main pot and side pots
        
        TDA Rule 21: Side pot distribution
        
        Returns:
            List of Pot objects, ordered from main pot to side pots
            
        Algorithm:
        1. Collect all player bets with IDs
        2. Sort by bet amount
        3. Create pots for each betting level
        4. Players all-in at lower levels are removed from higher pots
        """
        # Collect bets with player info
        # Collect bets with player info from ALL players (including folded)
        player_bets = []
        for i, player in enumerate(self.players):
            total_bet = player.bet_this_hand
            if total_bet > 0:
                player_bets.append((i, total_bet))
        
        # Sort by bet amount (ascending)
        player_bets.sort(key=lambda x: x[1])
        
        pots = []
        previous_level = 0
        remaining_players = set([pid for pid, _ in player_bets])
        
        for player_id, bet_amount in player_bets:
            if bet_amount > previous_level:
                # Calculate pot for this level
                level_contribution = bet_amount - previous_level
                pot_size = level_contribution * len(remaining_players)
                
                # Create pot
                # Only active (non-folded) players are eligible to win
                eligible_players = [pid for pid in remaining_players if not self.players[pid].is_folded()]
                
                if eligible_players:
                    pots.append(Pot(pot_size, eligible_players))
                else:
                    # If no one is eligible (everyone folded?), add to previous pot or handle as dead money
                    # In a showdown scenario, at least one person must be active.
                    # If this happens, it means this chunk of money belongs to the last survivor.
                    # But _calculate_pots is usually called when >1 active players.
                    # If we have pots, add to the last one (main pot or side pot)
                    if pots:
                        pots[-1].amount += pot_size
                
                previous_level = bet_amount
            
            # Remove this player from future pots (they're all-in or folded/capped)
            remaining_players.discard(player_id)
        
        return pots
    
    def _showdown(self):
        """Determine winner at showdown with side pot support"""
        self.street = Street.SHOWDOWN
        
        # Collect final bets
        for player in self.players:
            self.pot += player.bet_this_round
        
        active_players = [(i, p) for i, p in enumerate(self.players) if not p.is_folded()]
        
        if len(active_players) == 1:
            # Only one player left (opponent folded)
            self.winner = active_players[0][0]
            self.players[self.winner].chips += self.pot
            self.hand_history.append(f"P{self.winner} wins {self.pot:.0f} (opponent folded)")
        else:
            # Calculate pots (main + side pots)
            pots = self._calculate_pots()
            
            # Evaluate hands
            hand_values = []
            for player_id, player in active_players:
                value = HandEvaluator.evaluate(player.hand, self.community_cards)
                hand_name = HandEvaluator.get_hand_name(player.hand, self.community_cards)
                hand_values.append((player_id, value, hand_name))
            
            # Display showdown info
            self.hand_history.append(f"Showdown:")
            for pid, val, name in hand_values:
                self.hand_history.append(f"  P{pid}: {name}")
            
            # Award each pot to the best eligible hand
            total_winnings = {}
            for pot_idx, pot in enumerate(pots):
                # Filter hands eligible for this pot
                eligible_hands = [(pid, val, name) for pid, val, name in hand_values 
                                 if pid in pot.eligible_players]
                
                if not eligible_hands:
                    # Should not happen, but safety check
                    continue
                
                # Find winner(s) - highest value wins
                eligible_hands.sort(key=lambda x: x[1], reverse=True)
                best_hand_value = eligible_hands[0][1]
                
                # Check for ties
                winners = [pid for pid, val, name in eligible_hands if val == best_hand_value]
        for player_id, bet_amount in player_bets:
            if bet_amount > previous_level:
                # Calculate pot for this level
                level_contribution = bet_amount - previous_level
                pot_size = level_contribution * len(remaining_players)
                
                # Create pot
                pots.append(Pot(pot_size, list(remaining_players)))
                
                previous_level = bet_amount
            
            # Remove this player from future pots (they're all-in)
            remaining_players.discard(player_id)
        
        return pots
    
    def _showdown(self):
        """Determine winner at showdown with side pot support"""
        self.street = Street.SHOWDOWN
        
        # Collect final bets
        for player in self.players:
            self.pot += player.bet_this_round
        
        active_players = [(i, p) for i, p in enumerate(self.players) if not p.is_folded()]
        
        if len(active_players) == 1:
            # Only one player left (opponent folded)
            self.winner = active_players[0][0]
            self.players[self.winner].chips += self.pot
            self.hand_history.append(f"P{self.winner} wins {self.pot:.0f} (opponent folded)")
        else:
            # Calculate pots (main + side pots)
            pots = self._calculate_pots()
            
            # Evaluate hands
            hand_values = []
            for player_id, player in active_players:
                value = HandEvaluator.evaluate(player.hand, self.community_cards)
                hand_name = HandEvaluator.get_hand_name(player.hand, self.community_cards)
                best_five = HandEvaluator.get_best_five(player.hand, self.community_cards)
                hand_values.append((player_id, value, hand_name, best_five))
            
            # Display showdown info
            self.hand_history.append(f"Showdown:")
            for pid, val, name, best_five in hand_values:
                # Format: P0: Pair [♠A, ♣A, ♥K, ♦Q, ♥J]
                best_five_str = ', '.join([c.pretty_str() for c in best_five])
                self.hand_history.append(f"  P{pid}: {name} [{best_five_str}]")
            
            # Award each pot to the best eligible hand
            total_winnings = {}
            for pot_idx, pot in enumerate(pots):
                # Filter hands eligible for this pot
                eligible_hands = [(pid, val, name, best_five) for pid, val, name, best_five in hand_values 
                                 if pid in pot.eligible_players]
                
                if not eligible_hands:
                    # Should not happen, but safety check
                    continue
                
                # Find winner(s) - highest value wins
                eligible_hands.sort(key=lambda x: x[1], reverse=True)
                best_hand_value = eligible_hands[0][1]
                
                # Check for ties
                winners = [pid for pid, val, name, best_five in eligible_hands if val == best_hand_value]
                
                # Split pot among winners
                pot_share = pot.amount / len(winners)
                
                for winner_id in winners:
                    self.players[winner_id].chips += pot_share
                    total_winnings[winner_id] = total_winnings.get(winner_id, 0) + pot_share
                
                # Log pot award
                pot_type = "Main pot" if pot_idx == 0 else f"Side pot {pot_idx}"
                if len(winners) == 1:
                    self.hand_history.append(f"{pot_type}: P{winners[0]} wins {pot.amount:.0f}")
                else:
                    winners_str = ", ".join([f"P{w}" for w in winners])
                    self.hand_history.append(f"{pot_type}: {winners_str} split {pot.amount:.0f} ({pot_share:.0f} each)")
            
            # Set primary winner (player who won the most)
            if total_winnings:
                self.winner = max(total_winnings.items(), key=lambda x: x[1])[0]
        
        self.is_hand_over = True
    
    def _check_hand_over(self):
        """Check if hand is over (someone folded)"""
        active_players = [p for p in self.players if not p.is_folded()]
        
        if len(active_players) == 1:
            # Find winner
            for i, p in enumerate(self.players):
                if not p.is_folded():
                    self.winner = i
                    break
            
            # Collect all bets to pot
            for player in self.players:
                self.pot += player.bet_this_round
                player.bet_this_round = 0.0  # Reset bets after collecting
            
            # Award pot
            self.players[self.winner].chips += self.pot
            
            self.is_hand_over = True
            self.street = Street.SHOWDOWN
    
    def get_pot_size(self) -> float:
        """Get current pot size including bets this round"""
        return self.pot + sum(p.bet_this_round for p in self.players)
    
    def get_legal_actions(self, player_id: int) -> List[ActionType]:
        """Get legal actions for player"""
        player = self.players[player_id]
        
        legal_actions = ActionValidator.get_legal_actions(
            player.chips,
            self.current_bet,
            player.bet_this_round,
            self.min_raise,
            self.big_blind
        )
        
        # TDA Rule 47: Re-opening the bet
        # If facing a bet/raise smaller than min_raise (short all-in),
        # and we have already acted (implied by facing a raise), we cannot raise.
        # In Heads-up, if amount_to_call < min_raise and amount_to_call > 0,
        # it means opponent went all-in with less than a full raise.
        # Exception: If we haven't acted yet? No, in HU if we face a raise, we acted.
        # Exception: If the short all-in was the FIRST bet?
        # If current_bet < big_blind? (Possible if short stack posts BB?)
        
        amount_to_call = self.current_bet - player.bet_this_round
        
        # Exception for Preflop SB (who hasn't acted on the big blind yet)
        # Calculate SB position correctly based on player count
        num_players = len(self.players)
        if num_players == 2:
            sb_pos = self.button_position
        else:
            sb_pos = (self.button_position + 1) % num_players
            
        is_preflop_sb = (self.street == Street.PREFLOP and player_id == sb_pos)
        
        if not is_preflop_sb and 0 < amount_to_call < self.min_raise:
            # Facing a short all-in raise -> Cannot re-raise
            if ActionType.RAISE in legal_actions:
                legal_actions.remove(ActionType.RAISE)
                
        return legal_actions
    
    def get_hand_history(self) -> List[str]:
        """Get hand history"""
        return self.hand_history.copy()
