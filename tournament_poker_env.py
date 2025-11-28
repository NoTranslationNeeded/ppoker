"""
Tournament-style Poker Environment
Multiple hands until one player's stack reaches 0
With blind level escalation and randomized starting stacks

Pure Dense Reward: (chip_payoff / BB) / 250.0
Ultra-Fast Equity: ompeval C++ multithreaded Monte Carlo
"""
import rlcard
import numpy as np
import ompeval  # Ultra-fast C++ Monte Carlo equity calculation

class TournamentPokerEnv:
    """
    Tournament poker where players play multiple hands
    until one player loses all chips
    """
    
    def __init__(self, starting_chips=100, small_blind=1, big_blind=2, randomize_stacks=True, 
                 reward_type='icm_survival', reward_config=None):
        self.base_starting_chips = starting_chips
        self.base_small_blind = small_blind
        self.base_big_blind = big_blind
        self.randomize_stacks = randomize_stacks
        
        # Pure Dense Reward: No complex reward functions needed
        # reward_type and reward_config parameters kept for backward compatibility
        
        # Blind level structure: Fixed at 125/250 for Deep Stack play
        # The user requested BB to be fixed at 250.
        self.blind_levels = [
            (125, 250),
        ]
        self.hands_per_level = 999999  # Effectively infinite, blinds do not increase
        
        # RLCard base environment
        self.base_env = None
        self.chips = [starting_chips, starting_chips]
        self.hand_count = 0
        self.current_blind_level = 0
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.current_dealer_id = 0
        self.tournament_over = False
        
        # Action mapping
        # 0: Fold
        # 1: Check/Call
        # 2: Bet 33% Pot (New)
        # 3: Bet 75% Pot (Was Raise Half Pot)
        # 4: Bet 100% Pot (Was Raise Pot)
        # 5: Bet 150% Pot (New)
        # 6: All-in
        self.rlcard_to_agent_action = {
            0: 0, # Fold
            1: 1, # Check/Call
            2: 3, # Raise -> Bet 75% (Default raise in RLCard is usually half pot, mapping to 75% slot)
            3: 6, # All-in -> All-in
            # Note: RLCard's default 'raise' is limited. 
            # We will handle custom bet sizes by modifying the environment or action interpretation
            # For now, we map available RLCard actions to our discrete space
        }
        
        # Action space size
        self.action_space_size = 7
        
        # Observation space
        # 54 (RLCard default) + 1 (Equity) + 2 (Chip stacks) + 1 (Chip ratio) + 1 (Pot ratio) + 1 (Blind level)
        self.observation_space_size = 60
        self.current_state = None
        self.current_player = 0

    def step(self, action):
        """Execute one action in the tournament"""
        # Map Agent Action -> RLCard Action
        # Reverse of rlcard_to_agent_action
        # 0->0, 1->1, 3->2, 6->3
        # Note: If we want to support other bet sizes (2, 4, 5), we need custom logic.
        # For now, we only map the ones that exist in RLCard's legal actions.
        rlcard_action = action
        if action == 3:
            rlcard_action = 2
        elif action == 6:
            rlcard_action = 3
            
        # Execute action in current hand
        next_state, next_player = self.base_env.step(rlcard_action)
        self.current_state = next_state  # Store state for legal actions
        self.current_player = next_player
        
        # Check if hand is over
        if self.base_env.is_over():
            # Hand finished - update chips
            payoffs = self.base_env.get_payoffs()
            
            self.chips[0] += payoffs[0]
            self.chips[1] += payoffs[1]
            
            # Check if tournament is over
            if self.chips[0] <= 0 or self.chips[1] <= 0:
                self.tournament_over = True
                
                # Pure Dense Reward: Even at tournament end, only reward final chip change
                # This maintains consistency with the chip EV maximization principle
                # Formula: (chip_payoff / big_blind) / 250.0
                final_reward_0 = (payoffs[0] / self.big_blind) / 250.0
                final_reward_1 = (payoffs[1] / self.big_blind) / 250.0
                
                obs = np.zeros(self.observation_space_size, dtype=np.float32)
                done = True
                info = {
                    'tournament_winner': 0 if self.chips[0] > 0 else 1,
                    'hands_played': self.hand_count,
                    'final_chips': self.chips.copy(),
                    'starting_chips': self.starting_chips.copy()
                }
                
                return obs, final_reward_0, done, info
            else:
                # Tournament continues - start new hand
                # Dense Reward: Calculate reward for this hand (BB normalization)
                # Formula: (chip_payoff / big_blind) / 250.0
                bb_payoff_0 = payoffs[0] / self.big_blind
                bb_payoff_1 = payoffs[1] / self.big_blind
                
                hand_reward_0 = bb_payoff_0 / 250.0
                hand_reward_1 = bb_payoff_1 / 250.0
                
                # Extract card information from the ended hand
                card_info = self._extract_card_info(next_state)
                
                obs = self._start_new_hand()
                done = False
                info = {
                    'hand_ended': True,
                    'hand_payoffs': payoffs,
                    'current_chips': self.chips.copy(),
                    'blind_level': self.current_blind_level,
                    'cards': card_info
                }
                
                return obs, hand_reward_0, done, info
        else:
            # Hand continues
            obs = self._process_state(next_state, next_player)
            reward = 0.0
            done = False
            info = {}
            
            return obs, reward, done, info

    def reset(self):
        """Reset tournament - both players start with randomized chips"""
        if self.randomize_stacks:
            # Random Stack Depth: 1 to 250 BB
            bb_depth = np.random.randint(1, 251)
            
            # Random Big Blind: 250 to 5000
            # Ensure even number for integer Small Blind
            new_bb = np.random.randint(250, 5001)
            if new_bb % 2 != 0:
                new_bb += 1
            new_sb = int(new_bb / 2)
            
            # Calculate Starting Chips
            start_chips = bb_depth * new_bb
            
            self.chips = [float(start_chips), float(start_chips)]
            self.starting_chips = self.chips.copy()
            
            # Update blind levels for this episode (Fixed for the episode)
            self.blind_levels = [(new_sb, new_bb)]
            self.current_blind_level = 0
            self.small_blind = new_sb
            self.big_blind = new_bb
            
        else:
            # Default fixed deep stack (25000 chips, 100/200 blinds)
            self.chips = [25000.0, 25000.0]
            self.starting_chips = self.chips.copy()
            self.current_blind_level = 0
            self.small_blind = self.base_small_blind
            self.big_blind = self.base_big_blind
        
        # Reset tournament state
        self.hand_count = 0
        self.tournament_over = False
        
        # Start first hand and return initial observation
        self.current_dealer_id = 0  # First hand starts with player 0 as dealer
        return self._start_new_hand()

    def _start_new_hand(self):
        """Start a new hand within the tournament"""
        self.hand_count += 1
        
        # Update blind level every hands_per_level hands
        new_blind_level = min(
            (self.hand_count - 1) // self.hands_per_level,
            len(self.blind_levels) - 1
        )
        
        if new_blind_level != self.current_blind_level:
            self.current_blind_level = new_blind_level
            self.small_blind, self.big_blind = self.blind_levels[self.current_blind_level]
            print(f" Blind level increased to {self.small_blind}/{self.big_blind} (Level {self.current_blind_level + 1})")
        
        # Toggle dealer for subsequent hands (Hand 1 uses the one set in reset)
        if self.hand_count > 1:
            self.current_dealer_id = 1 - self.current_dealer_id

        # Create new RLCard environment for this hand
        config = {
            'seed': np.random.randint(0, 100000),
            'game_num_players': 2,
            'dealer_id': self.current_dealer_id
        }
        self.base_env = rlcard.make('no-limit-holdem', config=config)
        
        # [NEW] Configure blinds explicitly
        if hasattr(self.base_env.game, 'small_blind'):
            self.base_env.game.small_blind = self.small_blind
        if hasattr(self.base_env.game, 'big_blind'):
            self.base_env.game.big_blind = self.big_blind
        
        # Get initial state
        state, player_id = self.base_env.reset()
        
        # [NEW] Synchronize chip stacks
        # rlcard's reset() automatically posts blinds, so we must account for them
        for i in range(2):
            player = self.base_env.game.players[i]
            # Calculate remaining chips: Tournament Stack - Chips already put in pot (Blinds)
            # Cast to int to prevent rlcard TypeError (float64 -> int64 casting error)
            player.remained_chips = int(max(0, self.chips[i] - player.in_chips))
            
        self.current_state = state  # Store state for legal actions
        self.current_player = player_id
        
        # Process observation with chip context
        obs = self._process_state(state, player_id)
        
        return obs

    def _process_state(self, state, player_id):
        """
        Convert RLCard state to observation vector
        Adds chip information and blind level
        """
        # Base observation (54 elements)
        base_obs = state['obs']
        
        # Calculate hand equity using Monte Carlo simulation
        equity = self._calculate_equity(state, player_id)

        
        # Chip information (normalized)
        # Normalize by max possible starting chips (50000) or current pot context
        MAX_CHIPS = 50000.0
        my_chips = self.chips[player_id] / MAX_CHIPS
        opp_chips = self.chips[1 - player_id] / MAX_CHIPS
        chip_ratio = my_chips / (my_chips + opp_chips) if (my_chips + opp_chips) > 0 else 0.5
        pot_ratio = (self.small_blind + self.big_blind) / self.chips[player_id] if self.chips[player_id] > 0 else 0.0
        
        # Blind level (normalized 0-1)
        if len(self.blind_levels) > 1:
            blind_level_normalized = self.current_blind_level / (len(self.blind_levels) - 1)
        else:
            blind_level_normalized = 0.0  # Fixed blind level
        
        # Combine into observation
        obs = np.concatenate([
            base_obs.flatten(),
            [equity],
            [my_chips],
            [opp_chips],
            [chip_ratio],
            [pot_ratio],
            [blind_level_normalized]
        ]).astype(np.float32)
        
        return obs

    def _calculate_equity(self, state, player_id):
        """Ultra-fast C++ Monte Carlo equity with ompeval (10-30x faster than Python loop)"""
        # Get hand and community cards
        raw_obs = state['raw_obs']
        hand = raw_obs['hand']
        public_cards = raw_obs['public_cards']
        
        # Convert RLCard format ('SA') to ompeval format ('As')
        hand_str = "".join([self._card_to_ompeval(c) for c in hand])
        board_str = "".join([self._card_to_ompeval(c) for c in public_cards])
        
        # Create hand ranges
        ranges = [
            ompeval.CardRange(hand_str),      # Our hand
            ompeval.CardRange("random"),      # vs random opponent
        ]
        
        # Convert board to bitmask (C++ native format)
        board_mask = ompeval.CardRange.getCardMask(text=board_str)
        dead_mask = ompeval.CardRange.getCardMask(text="")  # No dead cards
        
        # C++ multithreaded equity calculator
        eq_calc = ompeval.EquityCalculator()
        eq_calc.set_hand_limit(100)  # Monte Carlo trials (fast enough with C++)
        
        # Start C++ calculation (multithreaded!)
        eq_calc.start(
            hand_ranges=ranges,
            board_cards=board_mask,
            dead_cards=dead_mask,
            enumerate_all=False  # Monte Carlo mode
        )
        
        # Wait for completion
        eq_calc.wait()
        
        # Get result
        result = eq_calc.get_results()
        return result.equity[0]  # Our hand's equity
    
    def _extract_card_info(self, state):
        """Extract card information from RLCard state"""
        try:
            raw_obs = state['raw_obs']
            
            # Get hands for both players (if available)
            hands = raw_obs.get('all_hands', [raw_obs.get('hand', [])])
            
            # Get community cards
            public_cards = raw_obs.get('public_cards', [])
            
            # Convert card strings to readable format
            def format_card(card_str):
                """Convert RLCard format to readable (e.g., 'SA' -> 'As')"""
                suit_map = {'S': 's', 'H': 'h', 'D': 'd', 'C': 'c'}
                if len(card_str) >= 2:
                    suit = suit_map.get(card_str[0], card_str[0])
                    rank = card_str[1]
                    return f"{rank}{suit}"
                return card_str
            
            return {
                'player_0_hand': [format_card(c) for c in (hands[0] if len(hands) > 0 else [])],
                'player_1_hand': [format_card(c) for c in (hands[1] if len(hands) > 1 else [])],
                'community': [format_card(c) for c in public_cards]
            }
        except Exception as e:
            return {
                'player_0_hand': [],
                'player_1_hand': [],
                'community': []
            }
    
    def _card_to_ompeval(self, card_str):
        """Convert RLCard card string (e.g. 'SA') to ompeval format (e.g. 'As')"""
        # RLCard format: 'SA' (Suit + Rank)
        # ompeval format: 'As' (Rank + suit lowercase)
        suit_map = {'S': 's', 'H': 'h', 'D': 'd', 'C': 'c'}
        
        suit = suit_map[card_str[0]]
        rank = card_str[1]  # T, J, Q, K, A or 2-9
        
        return f"{rank}{suit}"
    
    def get_legal_actions(self):
        """Get currently legal actions from stored state"""
        if self.current_state and 'legal_actions' in self.current_state:
            rlcard_legal = list(self.current_state['legal_actions'].keys())
            
            # Filter and Map to Agent Actions
            agent_legal = []
            
            # Robustness: Try to get game state for physical checks
            current_pot = 0
            my_stack = 0
            check_chips = False
            
            try:
                if self.base_env and hasattr(self.base_env, 'game'):
                    current_pot = self.base_env.game.dealer.pot
                    my_stack = self.base_env.game.players[self.current_player].remained_chips
                    check_chips = True
            except Exception:
                # If we can't access game state, fall back to allowing all actions
                # This prevents crashes during initialization or edge cases
                check_chips = False
            
            for action_id in rlcard_legal:
                if action_id in self.rlcard_to_agent_action:
                    agent_action = self.rlcard_to_agent_action[action_id]
                    
                    # [Physical Safety] Check if we have enough chips for the bet
                    # 0(Fold), 1(Call), 6(All-in) are always allowed if RLCard says so
                    if check_chips and agent_action in [2, 3, 4, 5]:
                        # Bet ratios: 33%, 75%, 100%, 150%
                        ratios = {2: 0.33, 3: 0.75, 4: 1.0, 5: 1.5}
                        amount_needed = current_pot * ratios[agent_action]
                        
                        # If bet amount >= stack, we should use All-in instead
                        # So disable this specific bet size button
                        if amount_needed >= my_stack:
                            continue
                    
                    agent_legal.append(agent_action)
            
            return sorted(agent_legal)
        return []


if __name__ == "__main__":
    # Test the tournament environment
    env = TournamentPokerEnv(randomize_stacks=True)
    
    print("=" * 80)
    print("Testing Tournament Poker Environment (Random Agent)")
    print("=" * 80)
    
    # Action name map for display
    action_names = {
        0: "Fold",
        1: "Check/Call",
        2: "Bet 33% Pot",
        3: "Bet 75% Pot",
        4: "Bet 100% Pot",
        5: "Bet 150% Pot",
        6: "All-in"
    }
    
    for episode in range(1): # Run 1 episode as requested
        print(f"\n Episode {episode + 1}")
        obs = env.reset()
        print(f"Starting chips - Player 0: {env.chips[0]:.1f}, Player 1: {env.chips[1]:.1f}")
        print(f"Initial blinds: {env.small_blind}/{env.big_blind} (Depth: {env.chips[0]/env.big_blind:.1f} BB)\n")
        
        done = False
        hand_count = 0
        
        while not done and hand_count < 100:
            legal_actions = env.get_legal_actions()
            if not legal_actions:
                break
            
            # Pick random action
            action = np.random.choice(legal_actions)
            action_name = action_names.get(action, "Unknown")
            
            print(f"  Hand {env.hand_count} | Player {env.current_player} acts: {action_name} (Action {action})")
            
            obs, reward, done, info = env.step(action)
            
            if info.get('hand_ended'):
                print(f"  -- Hand Ended --")
                print(f"     Result: {info['hand_payoffs']}")
                print(f"     Chips: P0={env.chips[0]:.1f}, P1={env.chips[1]:.1f}")
                print(f"     Cards: {info['cards']}")
                print("-" * 40)
                hand_count += 1
            
            if done:
                print(f"\n Tournament Over!")
                print(f"Winner: Player {info['tournament_winner']}")
                print(f"Total hands: {info['hands_played']}")
