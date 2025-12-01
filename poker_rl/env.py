
import os
import sys
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv

# Add POKERENGINE to path
current_dir = os.path.dirname(os.path.abspath(__file__))
poker_engine_path = os.path.join(current_dir, '..', 'POKERENGINE')
sys.path.append(poker_engine_path)

try:
    from poker_engine import PokerGame, Action, ActionType, Card
except ImportError:
    # Fallback for different directory structures or if POKERENGINE is already in path
    try:
        from POKERENGINE.poker_engine import PokerGame, Action, ActionType, Card
    except ImportError:
        raise ImportError("Could not import poker_engine. Please ensure POKERENGINE is in the Python path.")

from poker_rl.utils.obs_builder import ObservationBuilder

class PokerMultiAgentEnv(MultiAgentEnv):
    """
    2-player Heads-up No-Limit Texas Hold'em Multi-Agent Environment
    
    Compatible with:
    - RLlib multi-agent training
    - Self-play
    - NumPy-based observations
    """
    
    metadata = {'render_modes': ['human'], 'render_fps': 1}
    
    def __init__(self, config=None):
        super().__init__()
        
        config = config or {}
        
        # Game constants
        self.small_blind = 50.0
        self.big_blind = 100.0
        self.starting_chips = 10000.0 # Default, will be overridden by stack distribution
        
        # Stack depth distribution (BB units)
        self.stack_distribution = {
            'standard': (80, 120, 0.40),   # 100BB ±20, 40%
            'middle': (20, 50, 0.30),      # 20-50BB, 30%
            'short': (5, 20, 0.20),        # 5-20BB, 20%
            'deep': (150, 250, 0.10)       # 150-250BB, 10%
        }
        
        # Observation Space (338 floats)
        # 119: Card One-Hot (7 cards * 17 dims) -> Wait, 2 hole + 5 community = 7 cards. 7 * 17 = 119.
        # 31: Game State
        # 160: Action History (4 streets * 4 actions * 10 dims)
        # Total: 119 + 31 + 160 = 310?
        # Plan says 338. Let's re-calculate based on plan.
        # Plan: 
        # 119: Card one-hot
        # 20: Game State + Mask? No, Plan says "20: Game State + Legal Actions Mask" then "20: Street Context".
        # Let's stick to the detailed breakdown in Plan:
        # Obs[0:119]: Cards (2 hole + 5 community) * 17 = 119
        # Obs[119:150]: Game State (31 floats)
        # Obs[150:310]: Action History (4 streets * 4 actions * 10 dims = 160)
        # Total: 119 + 31 + 160 = 310.
        # The plan mentioned 338 in one place but 310 in another. I will go with 310 as it sums up correctly.
        
        # Observation Space
        # Dict space for Action Masking
        self.observation_space = spaces.Dict({
            "observations": spaces.Box(
                low=0.0,
                high=2.5,
                shape=(150,),
                dtype=np.float32
            ),
            "action_mask": spaces.Box(
                low=0.0,
                high=1.0,
                shape=(8,),
                dtype=np.float32
            )
        })
        
        # Action Space: Discrete(8)
        # 0: Fold
        # 1: Check/Call
        # 2: Bet 33%
        # 3: Bet 50%
        # 4: Bet 75%
        # 5: Bet 100%
        # 6: Bet 150%
        # 7: All-in
        self.action_space = spaces.Discrete(8)
        
        self._agent_ids = {"player_0", "player_1"}
        
        # Initialize game state
        self.game = None
        self.chips = [0.0, 0.0]
        self.button = 0
        self.action_history = {}
        self.hand_count = 0
        self.max_hands = 1000 # Prevent infinite loops
        
    def _sample_stack_depth(self) -> float:
        """Sample stack depth based on distribution"""
        categories = ['standard', 'middle', 'short', 'deep']
        probs = [0.40, 0.30, 0.20, 0.10]
        
        category = np.random.choice(categories, p=probs)
        min_bb, max_bb, _ = self.stack_distribution[category]
        
        stack_bb = np.random.uniform(min_bb, max_bb)
        return stack_bb * self.big_blind

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
            
        self.hand_count = 0
        
        # Sample stacks
        self.chips = [
            self._sample_stack_depth(),
            self._sample_stack_depth()
        ]
        self.hand_start_stacks = list(self.chips)
        
        # Randomize button
        self.button = np.random.randint(0, 2)
        
        # Start game
        self.game = PokerGame(
            small_blind=self.small_blind,
            big_blind=self.big_blind
        )
        self.game.start_hand(
            players_info=[(0, self.chips[0]), (1, self.chips[1])],
            button=self.button
        )
        
        # Reset history
        self.action_history = {
            'preflop': [],
            'flop': [],
            'turn': [],
            'turn': [],
            'river': []
        }
        
        # Logging for debug
        self.hand_logs = []
        self.hand_logs.append(f"\n=== Hand Start (Dealer: P{self.button}) ===")
        self.hand_logs.append(f"Stacks: P0={self.chips[0]:.1f}, P1={self.chips[1]:.1f}")
        self.hand_logs.append(f"Hole Cards P0: {[str(c) for c in self.game.players[0].hand]}")
        self.hand_logs.append(f"Hole Cards P1: {[str(c) for c in self.game.players[1].hand]}")
        
        current_player = self.game.get_current_player()
        
        # Get obs (now returns Dict)
        # CRITICAL: Return obs for BOTH players so RLlib initializes both agents.
        # Even if only one acts, the other needs to be "in" the episode to get rewards.
        obs_dict = {
            "player_0": ObservationBuilder.get_observation(self.game, 0, self.action_history),
            "player_1": ObservationBuilder.get_observation(self.game, 1, self.action_history)
        }
        info_dict = {}
        
        return obs_dict, info_dict

    def step(self, action_dict):
        # Only one agent acts at a time in this turn-based game
        current_player = self.game.get_current_player()
        agent_id = f"player_{current_player}"
        
        if agent_id not in action_dict:
            # Should not happen in normal flow
            return {}, {}, {}, {}, {}
            
        action_idx = action_dict[agent_id]
        
        # Map and execute action
        engine_action = self._map_action(action_idx, current_player)
        
        # Record for history (before processing, to get pot size before action?)
        # Actually we want to record WHAT happened.
        # If it's a bet/raise, we need the amount.
        pot_before = self.game.get_pot_size()
        street_before = self.game.street.value
        
        # Process action
        success, error = self.game.process_action(current_player, engine_action)
        
        if not success:
            # This should be prevented by action masking, but if it happens,
            # we might need to fallback or terminate.
            # For now, treat as check/fold to prevent crash, or raise error.
            # Ideally, mask prevents this.
            print(f"WARNING: Illegal action {action_idx} by {agent_id}: {error}")
            # Fallback to Check/Fold
            if self.game.current_bet > self.game.players[current_player].bet_this_round:
                 self.game.process_action(current_player, Action.fold())
            else:
                 self.game.process_action(current_player, Action.check())
        
        # Record action in history
        bet_amount = engine_action.amount if engine_action.amount > 0 else 0
        # If it was a call, amount is diff.
        # But for history, we care about the "intent" or the "move".
        # Let's record the abstract action index and the ratio.
        
        # Calculate actual bet amount for ratio
        if engine_action.action_type in [ActionType.BET, ActionType.RAISE, ActionType.ALL_IN, ActionType.CALL]:
             # For call, amount is implicit in engine but explicit in Action creation?
             # Action.call(amount)
             pass
        
        # Simplified history recording:
        # We use the action_idx directly as it represents the "intent" (e.g. Bet 33%)
        # For ratio, we calculate it.
        real_bet = 0.0
        if engine_action.amount > 0:
            real_bet = engine_action.amount
        
        self._record_action(action_idx, current_player, real_bet, pot_before, street_before)
        
        # Log action
        action_str = f"Street: {street_before}, P{current_player} {engine_action} (Pot: {pot_before:.1f})"
        self.hand_logs.append(action_str)
        
        # Check if hand is over
        if self.game.is_hand_over:
            return self._handle_hand_over()
        
        # Prepare next step
        next_player = self.game.get_current_player()
        
        obs_dict = {f"player_{next_player}": ObservationBuilder.get_observation(self.game, next_player, self.action_history)}
        reward_dict = {f"player_{next_player}": 0.0} # No intermediate rewards
        terminated_dict = {"__all__": False}
        truncated_dict = {"__all__": False}
        info_dict = {
            f"player_{next_player}": {}
        }
        
        return obs_dict, reward_dict, terminated_dict, truncated_dict, info_dict

    def _handle_hand_over(self):
        # Calculate rewards
        reward_dict = {}
        
        # Update chips from game state
        self.chips[0] = self.game.players[0].chips
        self.chips[1] = self.game.players[1].chips
        
        # P0 Reward
        stack_before_p0 = self.hand_start_stacks[0]
        stack_after_p0 = self.chips[0]
        chip_change = stack_after_p0 - stack_before_p0
        bb_change = chip_change / self.big_blind
        p0_reward = bb_change / 100.0
        
        reward_dict["player_0"] = float(p0_reward)
        reward_dict["player_1"] = float(-p0_reward) # Zero-sum
        
        # Occasional logging
        # train_batch_size=32768 steps. Avg steps/hand ~4-5. -> ~8000 hands/batch.
        # To log once per 4000 hands, prob should be 1/4000 = 0.00025
        if np.random.random() < 0.00025:
            self.hand_logs.append(f"Community Cards: {[str(c) for c in self.game.community_cards]}")
            self.hand_logs.append(f"Result: P0 {p0_reward:.4f}, P1 {-p0_reward:.4f}")
            self.hand_logs.append("=== Hand End ===")
            # Use ' || ' separator to keep log as one atomic line in Ray output
            print(" || ".join(self.hand_logs))
        
        # Terminate episode (one hand per episode for now, or continue?)
        # Plan says "Tournament style: Episode = Tournament".
        # But for now, let's do "One Hand per Episode" for simplicity in Phase 1,
        # OR implement the reset logic for next hand if not busted.
        
        # If we want multi-hand episodes:
        # Check if anyone is busted
        if self.chips[0] <= 0 or self.chips[1] <= 0:
            terminated_dict = {"__all__": True}
            obs_dict = {} # No observation needed for terminal state
        else:
            # Start new hand
            terminated_dict = {"__all__": True} # For Phase 1, let's just end episode after 1 hand to keep it simple and match "One Hand" logic first.
            # Wait, Plan says "Tournament style".
            # If I set terminated=True, RLlib resets the env.
            # So "One Hand per Episode" is easier for now.
            # But the Plan explicitly says "Episode = Tournament".
            # Let's stick to One Hand per Episode for Phase 1 to verify environment works.
            # We can expand to Tournament in Phase 2/3.
            pass
            
        # CRITICAL FIX: Must return observations for ALL agents at termination
        # otherwise RLlib might drop the reward for the agent that didn't act in the last step.
        obs_dict = {
            "player_0": ObservationBuilder.get_observation(self.game, 0, self.action_history),
            "player_1": ObservationBuilder.get_observation(self.game, 1, self.action_history)
        }
        truncated_dict = {"__all__": False}
        info_dict = {}
        
        return obs_dict, reward_dict, terminated_dict, truncated_dict, info_dict

    # _get_observation, _encode_card_onehot removed (DRY)
    # Using ObservationBuilder instead

    def _map_action(self, action_idx: int, player_id: int) -> Action:
        player = self.game.players[player_id]
        pot = self.game.get_pot_size()
        to_call = self.game.current_bet - player.bet_this_round
        
        if action_idx == 0:
            return Action.fold()
        elif action_idx == 1:
            return Action.check() if to_call == 0 else Action.call(to_call)
        elif action_idx == 7:
            return Action.all_in(player.chips)
        else:
            # Percentage bets
            pcts = [0.33, 0.50, 0.75, 1.0, 1.5]
            pct = pcts[action_idx - 2]
            bet_amount = pot * pct
            
            if self.game.current_bet > 0:
                # Raise
                target = self.game.current_bet + bet_amount
                # Add epsilon to avoid floating point issues where target < min_raise by tiny amount
                min_raise_target = self.game.current_bet + self.game.min_raise + 1e-5
                target = max(target, min_raise_target)
                
                # Cap at chips
                max_bet = player.chips + player.bet_this_round
                if target >= max_bet:
                    return Action.all_in(player.chips)
                return Action.raise_to(target)
            else:
                # Bet
                bet_amount = max(bet_amount, self.big_blind)
                if bet_amount >= player.chips:
                    return Action.all_in(player.chips)
                return Action.bet(bet_amount)

    # _get_legal_actions_mask removed (DRY)

    def _record_action(self, action_idx: int, player_id: int, bet_amount: float, pot_before: float, street: str):
        if pot_before > 0:
            bet_ratio = bet_amount / pot_before
        else:
            bet_ratio = 0.0
        bet_ratio = np.clip(bet_ratio, 0.0, 2.5)
        
        if street in self.action_history:
            self.action_history[street].append((action_idx, player_id, bet_ratio))

