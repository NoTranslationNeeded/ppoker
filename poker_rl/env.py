
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
        
        # =================================================================
        # OBSERVATION SPACE: 176 Dimensions (+ 14 Action Mask)
        # =================================================================
        # STRUCTURE BREAKDOWN:
        #
        # [0-118]   Cards (7 cards × 17 one-hot)           = 119 dims
        #           - 2 hole cards + 5 community cards
        #           - Each card: 13 ranks + 4 suits
        #
        # [119-134] Game State (normalized)                = 16 dims
        #           - My chips, Opp chips, Pot, Current bet
        #           - Bets this round, Call amount
        #           - Button position, Street (0/0.33/0.66/1.0)
        #           - Pot odds, SPR, ...
        #
        # [135-142] Hand Strength Features                 = 8 dims
        #           - HS/Equity, PPot, NPot, Hand Index
        #           - Street one-hot (preflop/flop/turn/river)
        #
        # [143-149] Padding (reserved for future)          = 7 dims
        #
        # [150-165] Street History Context                 = 16 dims
        #           - Per-street summary (4 streets × 4 features)
        #           - Raises count, Aggressor, Investment, 3-bet flag
        #
        # [166-171] Current Street Context                 = 6 dims
        #           - Actions count, I raised, Opp raised
        #           - Passive→Aggressive, Donk bet, Last action
        #
        # [172-173] Investment Features                    = 2 dims
        #           - Total investment (log-scaled)
        #           - Investment ratio (0-1)
        #
        # [174-175] Position Features                      = 2 dims
        #           - Position value (0=OOP, 1=IP)
        #           - Permanent advantage (postflop)
        #
        # TOTAL: 119 + 16 + 8 + 7 + 16 + 6 + 2 + 2 = 176 ✓
        # =================================================================
        
        # Observation Space: Dict for Action Masking
        self.observation_space = spaces.Dict({
            "observations": spaces.Box(
                low=np.float32(0.0),
                high=np.float32(200.0),
                shape=(176,),
                dtype=np.float32
            ),
            "action_mask": spaces.Box(
                low=np.float32(0.0),
                high=np.float32(1.0),
                shape=(14,),
                dtype=np.float32
            )
        })
        
        # Action Space: Discrete(14)
        # 0: Fold
        # 1: Check/Call
        # 2: Min Bet/Raise (NEW)
        # 3: Bet 10%
        # 4: Bet 25%
        # 5: Bet 33%
        # 6: Bet 50%
        # 7: Bet 75%
        # 8: Bet 100%
        # 9: Bet 125%
        # 10: Bet 150%
        # 11: Bet 200%
        # 12: Bet 300%
        # 13: All-in
        self.action_space = spaces.Discrete(14)
        
        self._agent_ids = {"player_0", "player_1"}
        
        # Initialize game state
        self.game = None
        self.chips = [0.0, 0.0]
        self.button = 0
        self.action_history = {}
        self.hand_count = 0
        self.max_hands = 1000 # Prevent infinite loops
        
        # Delta-Equity Reward state
        self.potential_states = {}  # {player_id: PotentialState}
        self.gamma = 0.99  # Discount factor for intermediate rewards
        
    def _sample_stack_depth(self) -> float:
        """Sample stack depth based on distribution"""
        categories = ['standard', 'middle', 'short', 'deep']
        probs = [0.40, 0.30, 0.20, 0.10]
        
        category = np.random.choice(categories, p=probs)
        min_bb, max_bb, _ = self.stack_distribution[category]
        
        stack_bb = np.random.uniform(min_bb, max_bb)
        return stack_bb * self.big_blind
    
    def _sample_stacks(self):
        """
        Effective Stack-based Sampling
        
        Strategy:
        - Sample effective stack (determines actual strategy)
        - 50% chance: Equal stacks (Cash Game style)
        - 50% chance: Asymmetric stacks (Tournament style)
        
        Benefits:
        - Learns all effective stack depths (5BB ~ 200BB)
        - Maintains asymmetric situation capability (Chip Leader bullying)
        - Reduces redundant sampling (50% less duplicate effective stacks)
        
        Returns:
            [stack_p0, stack_p1]
        """
        # 1. Sample effective stack (the strategic reference point)
        eff_stack = self._sample_stack_depth()
        
        # 2. 50% symmetric, 50% asymmetric
        if np.random.random() < 0.5:
            # Cash Game style: Equal stacks
            stack_p0 = eff_stack
            stack_p1 = eff_stack
        else:
            # Tournament style: Deep vs Short
            # One player has effective stack, other has 1.5-5x more
            # (Allows learning Chip Leader pressure tactics)
            deep_stack = eff_stack * np.random.uniform(1.5, 5.0)
            
            # Randomly assign who gets deep stack
            if np.random.random() < 0.5:
                stack_p0, stack_p1 = eff_stack, deep_stack
            else:
                stack_p0, stack_p1 = deep_stack, eff_stack
        
        return [stack_p0, stack_p1]

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
            
        self.hand_count = 0
        
        # Sample stacks (Effective Stack-based)
        self.chips = self._sample_stacks()
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
            'river': []
        }
        
        # Logging for debug
        # Decide logging status at start of hand (prob 1/10000)
        self.should_log_this_hand = np.random.random() < 0.0001
        self.hand_logs = []
        
        # Reward tracking per hand
        self.hand_rewards = {
            0: {"intermediate": 0.0, "terminal": 0.0, "total": 0.0},
            1: {"intermediate": 0.0, "terminal": 0.0, "total": 0.0}
        }
        
        if self.should_log_this_hand:
            self.hand_logs.append(f"\n=== Hand Start (Dealer: P{self.button}) ===")
            self.hand_logs.append(f"Stacks: P0={self.chips[0]:.1f}, P1={self.chips[1]:.1f}")
            self.hand_logs.append(f"Hole Cards P0: {[str(c) for c in self.game.players[0].hand]}")
            self.hand_logs.append(f"Hole Cards P1: {[str(c) for c in self.game.players[1].hand]}")
        
        # Initialize PotentialState for both players
        from poker_rl.potential_state import PotentialState
        effective_stack = min(self.hand_start_stacks[0], self.hand_start_stacks[1])
        self.potential_states = {
            0: PotentialState(self.game, 0, self.big_blind, self.hand_start_stacks[0]),
            1: PotentialState(self.game, 1, self.big_blind, self.hand_start_stacks[1])
        }
        # Set effective stack for normalization
        self.potential_states[0].set_effective_stack(effective_stack)
        self.potential_states[1].set_effective_stack(effective_stack)
        
        # ===== PBRS: Store initial Φ (CRITICAL for zero-sum) =====
        # Initial state has non-zero Φ due to blinds already posted
        # We must compensate for this in terminal reward to maintain zero-sum
        self.initial_phi = {
            0: self.potential_states[0].calculate_potential(),
            1: self.potential_states[1].calculate_potential()
        }
        
        # ===== PBRS: Track previous Φ for dual reward update =====
        # Both actor AND observer need rewards when state changes
        # This prevents "observer reward missing" bug
        self.prev_potentials = {
            0: self.initial_phi[0],
            1: self.initial_phi[1]
        }
        
        current_player = self.game.get_current_player()
        
        # Get obs (now returns Dict)
        # CRITICAL: Return obs for BOTH players so RLlib initializes both agents.
        # Even if only one acts, the other needs to be "in" the episode to get rewards.
        obs_dict = {
            "player_0": ObservationBuilder.get_observation(self.game, 0, self.action_history, self.hand_start_stacks),
            "player_1": ObservationBuilder.get_observation(self.game, 1, self.action_history, self.hand_start_stacks)
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
        
        # ===== DELTA-EQUITY: Calculate Potential BEFORE action =====
        from poker_rl.potential_state import PotentialState
        phi_before = self.potential_states[current_player].calculate_potential()
        
        # Map and execute action
        engine_action = self._map_action(action_idx, current_player)
        
        # Record for history
        pot_before = self.game.get_pot_size()
        street_before = self.game.street.value
        
        # Process action
        success, error = self.game.process_action(current_player, engine_action)
        
        if not success:
            print(f"WARNING: Illegal action {action_idx} by {agent_id}: {error}")
            # Fallback to Check/Fold
            if self.game.current_bet > self.game.players[current_player].bet_this_round:
                 self.game.process_action(current_player, Action.fold())
            else:
                 self.game.process_action(current_player, Action.check())
        
        # Record action in history
        bet_amount = engine_action.amount if engine_action.amount > 0 else 0
        real_bet = 0.0
        if engine_action.amount > 0:
            real_bet = engine_action.amount
        
        self._record_action(action_idx, current_player, real_bet, pot_before, street_before)
        
        # Log action
        action_str = f"Street: {street_before}, P{current_player} {engine_action} [Action: {action_idx}] (Pot: {pot_before:.1f})"
        
        # ===== PBRS: Calculate Intermediate Rewards BEFORE hand_over check =====
        # CRITICAL: Must calculate for ALL actions, including last action (fold/showdown)
        # Old bug: Skipped intermediate calculation when is_hand_over=True
        from poker_rl.potential_state import PotentialState
        effective_stack = min(self.hand_start_stacks[0], self.hand_start_stacks[1])
        
        # Actor's reward
        self.potential_states[current_player] = PotentialState(
            self.game, current_player, self.big_blind, self.hand_start_stacks[current_player]
        )
        self.potential_states[current_player].set_effective_stack(effective_stack)
        phi_after = self.potential_states[current_player].calculate_potential()
        intermediate_reward = self.gamma * phi_after - phi_before
        
        # Observer's reward (dual update)
        opponent = 1 - current_player
        self.potential_states[opponent] = PotentialState(
            self.game, opponent, self.big_blind, self.hand_start_stacks[opponent]
        )
        self.potential_states[opponent].set_effective_stack(effective_stack)
        phi_before_opponent = self.prev_potentials[opponent]
        phi_after_opponent = self.potential_states[opponent].calculate_potential()
        intermediate_reward_opponent = self.gamma * phi_after_opponent - phi_before_opponent
        
        # Track both rewards
        self.hand_rewards[current_player]["intermediate"] += intermediate_reward
        self.hand_rewards[opponent]["intermediate"] += intermediate_reward_opponent
        
        # Update prev_potentials
        self.prev_potentials[current_player] = phi_after
        self.prev_potentials[opponent] = phi_after_opponent
        
        # Logging (optional)
        if self.should_log_this_hand:
            from poker_rl.utils.equity_calculator import get_8_features
            player_hand = self.game.players[current_player].hand
            feats = get_8_features(player_hand, self.game.community_cards, street_before)
            hs, ppot, npot, hidx = feats[:4]
            action_str += f" | P{current_player}_Feat: [HS={hs:.2f} PPot={ppot:.2f} NPot={npot:.2f} HIdx={int(hidx)}]"
            action_str += f" | Phi: {phi_before:.4f}->{phi_after:.4f} | Reward={intermediate_reward:+.4f}"
            self.hand_logs.append(action_str)
        
        # ===== Check if hand is over (AFTER calculating intermediate rewards) =====
        if self.game.is_hand_over:
            # Get terminal rewards
            obs_dict, terminal_reward_dict, terminated_dict, truncated_dict, info_dict = self._handle_hand_over()
            
            # CRITICAL: Merge intermediate + terminal rewards
            # This ensures last action's Φ change is included
            final_reward_dict = {}
            for agent_id in terminal_reward_dict:
                player_idx = int(agent_id.split("_")[1])
                intermediate = intermediate_reward if player_idx == current_player else intermediate_reward_opponent
                final_reward_dict[agent_id] = terminal_reward_dict[agent_id] + intermediate
            
            return obs_dict, final_reward_dict, terminated_dict, truncated_dict, info_dict
        
        # ===== Hand continues =====
        next_player = self.game.get_current_player()
        
        obs_dict = {f"player_{next_player}": ObservationBuilder.get_observation(
            self.game, next_player, self.action_history, self.hand_start_stacks
        )}
        
        # Return intermediate rewards for both players
        reward_dict = {
            f"player_{current_player}": float(intermediate_reward),
            f"player_{opponent}": float(intermediate_reward_opponent)
        }
        
        terminated_dict = {"__all__": False}
        truncated_dict = {"__all__": False}
        info_dict = {f"player_{next_player}": {}}
        
        return obs_dict, reward_dict, terminated_dict, truncated_dict, info_dict

    def _handle_hand_over(self):
        # Calculate rewards
        reward_dict = {}
        
        # Update chips from game state
        self.chips[0] = self.game.players[0].chips
        self.chips[1] = self.game.players[1].chips
        
        # ===== DELTA-EQUITY: Terminal Reward (Effective Stack Normalization) =====
        # Calculate chip changes
        chip_change_p0 = self.chips[0] - self.hand_start_stacks[0]
        chip_change_p1 = self.chips[1] - self.hand_start_stacks[1]
        
        # Effective Stack (핸드 시작 시 작은 쪽 스택)
        # This is the strategic reference point in poker - all decisions are made relative to effective stack
        effective_stack = min(self.hand_start_stacks[0], self.hand_start_stacks[1])
        
        # ===== PBRS: Calculate final Φ for both players ===== 
        # CRITICAL: Must subtract final potential to satisfy PBRS theory
        # This ensures: Total Episode Reward = Real Chip Change (no "free lunch" from intermediate rewards)
        phi_final_p0 = self.potential_states[0].calculate_potential()
        phi_final_p1 = self.potential_states[1].calculate_potential()
        
        # Terminal Reward = Real Chip Reward - Final Potential + Initial Potential
        # This "repays the loan" from intermediate rewards AND compensates for non-zero initial Φ
        # Initial Φ ≠ 0 because blinds are already posted at hand start
        # Without +Φ₀ compensation, zero-sum would be violated (SB and BB have different Φ₀)
        terminal_reward_p0 = chip_change_p0 / effective_stack - phi_final_p0 + self.initial_phi[0]
        terminal_reward_p1 = chip_change_p1 / effective_stack - phi_final_p1 + self.initial_phi[1]
        
        reward_dict["player_0"] = float(terminal_reward_p0)
        reward_dict["player_1"] = float(terminal_reward_p1)
        
        # Track terminal rewards
        self.hand_rewards[0]["terminal"] = terminal_reward_p0
        self.hand_rewards[1]["terminal"] = terminal_reward_p1
        self.hand_rewards[0]["total"] = self.hand_rewards[0]["intermediate"] + terminal_reward_p0
        self.hand_rewards[1]["total"] = self.hand_rewards[1]["intermediate"] + terminal_reward_p1
        
        # ===== Zero-Sum Verification (CRITICAL SAFETY CHECK) =====
        # PBRS: Terminal reward may not be zero-sum (due to Φ subtraction)
        # BUT: Total episode reward MUST be zero-sum
        total_reward_p0 = self.hand_rewards[0]["total"]
        total_reward_p1 = self.hand_rewards[1]["total"]
        zero_sum_error = abs(total_reward_p0 + total_reward_p1)
        
        if zero_sum_error > 0.5:  # Relaxed tolerance for gamma effect (γ=0.99)
            # With γ < 1, perfect zero-sum is impossible due to discount factor
            # Expected violation: ~0.01 × sum(Φ_final)
            # Tolerance of 0.5 (50% of effective stack) provides maximum safety margin
            raise ValueError(
                f"CRITICAL: Zero-Sum Violation in TOTAL rewards! "
                f"P0_total={total_reward_p0:.6f}, P1_total={total_reward_p1:.6f}, "
                f"Diff={zero_sum_error:.10f}, "
                f"P0: intermediate={self.hand_rewards[0]['intermediate']:.4f} + terminal={terminal_reward_p0:.4f}, "
                f"P1: intermediate={self.hand_rewards[1]['intermediate']:.4f} + terminal={terminal_reward_p1:.4f}"
            )
        
        # Occasional logging
        # Use the flag decided at reset
        if self.should_log_this_hand:
            # Get 8 advanced features for both players
            from poker_rl.utils.equity_calculator import get_8_features, get_hand_name_from_index
            
            p0_features = get_8_features(
                self.game.players[0].hand,
                self.game.community_cards,
                self.game.street.value
            )
            p1_features = get_8_features(
                self.game.players[1].hand,
                self.game.community_cards,
                self.game.street.value
            )
            
            # Extract feature values
            p0_hs, p0_ppot, p0_npot, p0_hidx = p0_features[:4]
            p1_hs, p1_ppot, p1_npot, p1_hidx = p1_features[:4]
            
            # Get hand names
            p0_hand = get_hand_name_from_index(int(p0_hidx))
            p1_hand = get_hand_name_from_index(int(p1_hidx))
            
            # Determine street
            street_name = self.game.street.value.capitalize()
            
            # Build feature log
            self.hand_logs.append(f"P0({p0_hand}): HS={p0_hs:.2f} PPot={p0_ppot:.2f} NPot={p0_npot:.2f} HIdx={int(p0_hidx)} [{street_name}]")
            self.hand_logs.append(f"P1({p1_hand}): HS={p1_hs:.2f} PPot={p1_ppot:.2f} NPot={p1_npot:.2f} HIdx={int(p1_hidx)} [{street_name}]")
            
            self.hand_logs.append(f"Community Cards: {[str(c) for c in self.game.community_cards]}")
            
            # Original result line (chip change based)
            chip_change_p0 = self.chips[0] - self.hand_start_stacks[0]
            self.hand_logs.append(f"Result: P0 {chip_change_p0:.1f} chips, P1 {-chip_change_p0:.1f} chips")
            
            # ===== REWARD SUMMARY =====
            self.hand_logs.append("--- Reward Summary ---")
            self.hand_logs.append(
                f"P0: Intermediate={self.hand_rewards[0]['intermediate']:.4f}, "
                f"Terminal={self.hand_rewards[0]['terminal']:.4f}, "
                f"TOTAL={self.hand_rewards[0]['total']:.4f}"
            )
            self.hand_logs.append(
                f"P1: Intermediate={self.hand_rewards[1]['intermediate']:.4f}, "
                f"Terminal={self.hand_rewards[1]['terminal']:.4f}, "
                f"TOTAL={self.hand_rewards[1]['total']:.4f}"
            )
            self.hand_logs.append(
                f"Zero-Sum Check: P0+P1={self.hand_rewards[0]['total'] + self.hand_rewards[1]['total']:.6f}"
            )
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
            "player_0": ObservationBuilder.get_observation(self.game, 0, self.action_history, self.hand_start_stacks),
            "player_1": ObservationBuilder.get_observation(self.game, 1, self.action_history, self.hand_start_stacks)
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
        elif action_idx == 13: # All-in is now 13
            return Action.all_in(player.chips)
        elif action_idx == 2:
            # Min Raise ONLY (No Min Bet)
            if self.game.current_bet > 0:
                # Raise to min raise
                # Add epsilon to ensure we meet the "at least" requirement despite float precision
                target = self.game.current_bet + self.game.min_raise + 1e-5
                
                # Cap at chips
                max_bet = player.chips + player.bet_this_round
                if target >= max_bet:
                    return Action.all_in(player.chips)
                return Action.raise_to(target)
            else:
                # Min-Bet is BANNED.
                # If this action is selected (despite masking), fallback to Check.
                # This prevents "Min-Bet" behavior if the mask is ignored.
                return Action.check()
        else:
            # Percentage bets
            # 3: 10%, 4: 25%, 5: 33%, 6: 50%, 7: 75%, 8: 100%, 9: 125%, 10: 150%, 11: 200%, 12: 300%
            pcts = [0.10, 0.25, 0.33, 0.50, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]
            pct = pcts[action_idx - 3] # Shifted by 3
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
        """Record action with both ratio and actual amount for accurate tracking"""
        if pot_before > 0:
            bet_ratio = bet_amount / pot_before
        else:
            bet_ratio = 0.0
        bet_ratio = np.clip(bet_ratio, 0.0, 2.5)
        
        if street in self.action_history:
            # Tuple structure: (action_idx, player_id, bet_ratio, bet_amount)
            # Extended from 3 to 4 elements for accurate investment tracking
            self.action_history[street].append((action_idx, player_id, bet_ratio, bet_amount))

