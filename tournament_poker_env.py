"""
Tournament-style Poker Environment
Multiple hands until one player's stack reaches 0
With blind level escalation and randomized starting stacks

Pure Dense Reward: (chip_payoff / BB) / 250.0
Ultra-Fast Equity: ompeval C++ multithreaded Monte Carlo
"""
import rlcard
import numpy as np
# import ompeval  # Moved to __init__ to avoid pickling issues with Ray

class TournamentPokerEnv:
    """
    Tournament poker where players play multiple hands
    until one player loses all chips
    """
    
    def __init__(self, starting_chips=100, small_blind=1, big_blind=2, randomize_stacks=True, 
                 reward_type='icm_survival', reward_config=None, max_hands=1000):
        # Lazy import ompeval here to ensure it's loaded in the worker process
        global ompeval
        import ompeval
        
        self.base_starting_chips = starting_chips
        self.base_small_blind = small_blind
        self.base_big_blind = big_blind
        self.randomize_stacks = randomize_stacks
        self.max_hands = max_hands
        
        # Initialize ompeval EquityCalculator once
        self.equity_calc = ompeval.EquityCalculator()
        
        # Pure Dense Reward: No complex reward functions needed
        # reward_type and reward_config parameters kept for backward compatibility
        
        # Blind level structure: Fixed at 125/250 for Deep Stack play
        # The user requested BB to be fixed at 250.
        self.blind_levels = [
            (125, 250),
        ]
        self.hands_per_level = 20  # Blinds increase every 20 hands
        
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
        # 0. Get Game Info
        try:
            game = self.base_env.game
            current_player = game.players[self.current_player]
            dealer = game.dealer
            pot_size = dealer.pot
            legal_actions = list(self.base_env.get_legal_actions().keys())
        except AttributeError:
             # Safety fallback
             return self._process_state(self.current_state, self.current_player), 0, False, {}

        # 1. Interpret Action & Calculate Amount
        rlcard_action = 1 # Default: Check/Call
        
        # Bet Ratios
        # Action 2=33%, 3=75%, 4=100%, 5=150%
        bet_ratios = {2: 0.33, 3: 0.75, 4: 1.0, 5: 1.5}

        # Ensure action is int
        if hasattr(action, 'item'):
            action = int(action.item())
        else:
            action = int(action)

        if action == 0:   # Fold
            rlcard_action = 0
        elif action == 1: # Check/Call
            rlcard_action = 1
        elif action == 6: # All-in
            rlcard_action = 3 # RLCard All-in ID
        
        elif action in [2, 3, 4, 5]: # Custom Bet Sizes
            # Calculate target amount
            ratio = bet_ratios.get(action, 0.5)
            additional_bet = int(pot_size * ratio)
            
            # Validation & Clamping
            # min_raise might be on game or we assume 1 (or big_blind?)
            # Usually game.min_raise exists.
            min_raise = getattr(game, 'min_raise', self.big_blind) 
            my_stack = current_player.remained_chips
            
            # Clamp to min_raise
            if additional_bet < min_raise:
                additional_bet = min_raise
            
            # Check Stack (All-in Logic)
            if additional_bet >= my_stack:
                rlcard_action = 3 # Switch to All-in
            else:
                # Injection
                rlcard_action = 2 # Raise
                
                try:
                    # Inject raise_amount
                    # RLCard usually expects float for amounts in some versions, or int.
                    # We'll use float as per user suggestion/safety.
                    self.base_env.game.raise_amount = float(additional_bet)
                    
                    # DEBUG: Verify injection
                    # print(f"DEBUG: Injected raise_amount={self.base_env.game.raise_amount} for Action {action}")
                except Exception as e:
                    print(f"[Injection Error] Failed to inject amount: {e}")
                    rlcard_action = 1 # Fallback to Call

        # 2. Safety Fallback (Masking)
        if rlcard_action not in legal_actions:
            # If mapped action is illegal, try fallback
            if 1 in legal_actions: rlcard_action = 1
            elif 0 in legal_actions: rlcard_action = 0
            else: rlcard_action = legal_actions[0]

        # 3. Execute Step
        try:
            next_state, next_player = self.base_env.step(rlcard_action)
        except Exception as e:
            print(f"[Critical] Engine Error on action {action} (mapped to {rlcard_action}): {e}")
            # Force Call/Check
            next_state, next_player = self.base_env.step(1)
        self.current_state = next_state  # Store state for legal actions
        self.current_player = next_player
        
        # Check if hand is over
        if self.base_env.is_over():
            # Hand finished - update chips
            payoffs = self.base_env.get_payoffs()
            
            self.chips[0] += payoffs[0]
            self.chips[1] += payoffs[1]
            
            # Check if tournament is over
            if self.chips[0] <= 0 or self.chips[1] <= 0 or self.hand_count >= self.max_hands:
                self.tournament_over = True
                
                # Determine winner (chip count tie-breaker for max_hands limit)
                if self.chips[0] > self.chips[1]:
                    winner = 0
                elif self.chips[1] > self.chips[0]:
                    winner = 1
                else:
                    winner = -1 # Tie
                
                # Pure Dense Reward: BB-normalized chip change
                # Formula: (chip_payoff / BB) / 250.0 (already properly normalized to ~[-1, 1])
                final_reward_0 = (payoffs[0] / self.big_blind) / 250.0
                final_reward_1 = (payoffs[1] / self.big_blind) / 250.0
                
                obs = np.zeros(self.observation_space_size, dtype=np.float32)
                done = True
                info = {
                    'tournament_winner': winner,
                    'hands_played': self.hand_count,
                    'final_chips': self.chips.copy(),
                    'starting_chips': self.starting_chips.copy()
                }
                
                return obs, final_reward_0, done, info
            else:
                # Tournament continues - start new hand
                # Dense Reward: BB-normalized chip change per hand
                # Formula: (chip_payoff / BB) / 250.0 (already properly normalized to ~[-1, 1])
                hand_reward_0 = (payoffs[0] / self.big_blind) / 250.0
                hand_reward_1 = (payoffs[1] / self.big_blind) / 250.0
                
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
        """Reset tournament - Dynamic Hyper-Turbo Mode"""
        if self.randomize_stacks:
            # 1. Existing random stack/blind logic
            # Stack depth: 10 ~ 50 BB (Reduced for faster games)
            bb_depth = np.random.randint(10, 51)
            
            # Random starting blind (100 ~ 1000)
            new_bb = np.random.randint(100, 1001)
            if new_bb % 2 != 0: new_bb += 1
            new_sb = int(new_bb / 2)
            
            # Set starting chips
            start_chips = bb_depth * new_bb
            self.chips = [float(start_chips), float(start_chips)]
            self.starting_chips = self.chips.copy()
            
            # [Core Modification] Dynamic Blind Schedule
            # Generate doubling blinds starting from random new_sb/new_bb
            self.blind_levels = []
            curr_sb, curr_bb = new_sb, new_bb
            
            # Generate 10 levels (2^10 = 1024x increase, sufficient)
            for _ in range(10):
                self.blind_levels.append((curr_sb, curr_bb))
                curr_sb *= 2
                curr_bb *= 2
            
            self.current_blind_level = 0
            self.small_blind = new_sb
            self.big_blind = new_bb
            
            # [Core Modification] Level up every 5 hands (Hyper-Turbo)
            self.hands_per_level = 5
            
        else:
            # Fixed stack mode (for testing)
            self.chips = [2000.0, 2000.0]
            self.starting_chips = self.chips.copy()
            # Ensure blinds increase in fixed mode too
            self.blind_levels = [(10, 20), (20, 40), (40, 80), (80, 160), (160, 320)]
            self.current_blind_level = 0
            self.small_blind = 10
            self.big_blind = 20
            self.hands_per_level = 5 # 5 hands here too

        # Reset tournament state
        self.hand_count = 0
        self.tournament_over = False
        self.current_dealer_id = 0
        
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
        ALL VALUES NORMALIZED TO 0-1 RANGE (MAX_CHIPS = 50000)
        """
        # Base observation (54 elements)
        base_obs = state['obs']
        
        # Calculate hand equity using Monte Carlo simulation
        equity = self._calculate_equity(state, player_id)

        
        # Chip information (normalized by MAX_CHIPS = 50000)
        MAX_CHIPS = 50000.0
        my_chips = self.chips[player_id] / MAX_CHIPS
        opp_chips = self.chips[1 - player_id] / MAX_CHIPS
        chip_ratio = my_chips / (my_chips + opp_chips) if (my_chips + opp_chips) > 0 else 0.5
        
        # Pot size normalized (blinds are chips, so divide by MAX_CHIPS)
        pot_size = (self.small_blind + self.big_blind) / MAX_CHIPS
        
        # Current blinds normalized
        small_blind_normalized = self.small_blind / MAX_CHIPS
        big_blind_normalized = self.big_blind / MAX_CHIPS
        
        # Blind level (normalized 0-1)
        if len(self.blind_levels) > 1:
            blind_level_normalized = self.current_blind_level / (len(self.blind_levels) - 1)
        else:
            blind_level_normalized = 0.0  # Fixed blind level
        
        # Combine into observation (60 dimensions)
        obs = np.concatenate([
            base_obs.flatten(),       # 54 dims
            [equity],                 # 1 dim
            [my_chips],               # 1 dim
            [opp_chips],              # 1 dim
            [chip_ratio],             # 1 dim
            [pot_size],               # 1 dim (instead of pot_ratio)
            [blind_level_normalized]  # 1 dim
        ]).astype(np.float32)
        
        # Safety check: clip all values to [0, 1] range
        obs = np.clip(obs, 0.0, 1.0)
        
        return obs

    def _calculate_equity(self, state, player_id):
        """Calculate hand equity using ompeval C++ EquityCalculator (Ultra-Fast)"""
        # Get hand and community cards
        raw_obs = state['raw_obs']
        hand = raw_obs['hand']
        public_cards = raw_obs['public_cards']
        
        # If no hand, return 0.5
        if not hand:
            return 0.5
        
        # Convert to ompeval strings
        try:
            hero_hand_str = "".join([self._card_to_ompeval_str(c) for c in hand])
            board_str = "".join([self._card_to_ompeval_str(c) for c in public_cards])
            
            hero_range = ompeval.CardRange(hero_hand_str)
            opp_range = ompeval.CardRange("random")
            board_mask = ompeval.CardRange.getCardMask(board_str)
            
            # Use persistent calculator if available, else create new
            if not hasattr(self, 'equity_calc'):
                self.equity_calc = ompeval.EquityCalculator()
                
            self.equity_calc.start(
                [hero_range, opp_range],
                board_mask,
                0,      # dead cards
                False,  # enumerate
                0,      # stdev target
                None,   # callback
                0.002,  # update interval
                1       # threads
            )
            self.equity_calc.set_time_limit(0.002) # 2ms time limit
            self.equity_calc.wait()
            results = self.equity_calc.get_results()
            
            return results.equity[0]
            
        except Exception as e:
            # print(f"Equity Error: {e}")
            return 0.5  # Safety fallback
    
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
    
    def _card_to_ompeval_str(self, card_str):
        """Convert RLCard card string (e.g. 'SA') to ompeval string (e.g. 'As')"""
        # RLCard format: 'SA' (Suit + Rank)
        # ompeval format: 'As' (Rank + suit lowercase)
        suit_map = {'S': 's', 'H': 'h', 'D': 'd', 'C': 'c'}
        
        if len(card_str) < 2: return ""
        
        suit = suit_map.get(card_str[0], 's')
        rank = card_str[1]  # T, J, Q, K, A or 2-9
        
        return f"{rank}{suit}"
    
    def get_legal_actions(self):
        """
        Get currently legal actions with 'Smart Masking'
        Only enables bet sizes that are valid (>= min_raise)
        """
        if not self.current_state or 'legal_actions' not in self.current_state:
            return [0, 1] # Fallback to Fold/Call

        # 1. RLCard legal actions
        rlcard_legal = list(self.current_state['legal_actions'].keys())
        print(f"DEBUG: RLCard Legal: {rlcard_legal}") # Uncomment to debug upstream
        agent_legal = []

        # 2. Get Game State (for physical checks)
        try:
            game = self.base_env.game
            current_player = game.players[self.current_player]
            pot_size = game.dealer.pot
            min_raise = getattr(game, 'min_raise', 0) # Minimum raise amount
            my_stack = current_player.remained_chips
        except AttributeError:
            # Safety fallback if game state is inaccessible
            for action_id in rlcard_legal:
                if action_id in self.rlcard_to_agent_action:
                    agent_legal.append(self.rlcard_to_agent_action[action_id])
            return sorted(list(set(agent_legal)))

        # 3. Precise Validation for each button
        
        # (1) Fold(0), Check/Call(1) are always valid if RLCard says so
        if 0 in rlcard_legal: agent_legal.append(0)
        if 1 in rlcard_legal: agent_legal.append(1)
        
        # (2) Bet Options (Action 2~5)
        # 33%(2), 75%(3), 100%(4), 150%(5)
        bet_ratios = {2: 0.33, 3: 0.75, 4: 1.0, 5: 1.5}
        
        # Can we Raise or All-in?
        # RLCard: 2=Raise Half Pot (Hijacked), 3=Raise Pot, 4=All-in
        can_raise_or_allin = (2 in rlcard_legal) or (3 in rlcard_legal) or (4 in rlcard_legal)
        
        if can_raise_or_allin:
            for action_id, ratio in bet_ratios.items():
                # Calculate expected additional bet
                calc_amount = int(pot_size * ratio)
                
                # Logic: If calculated amount < min_raise, we clamp to min_raise (in step function).
                # So we should check if the *clamped* amount is valid.
                effective_amount = max(calc_amount, min_raise)
                
                # Condition: Must be affordable (strictly less than stack)
                # If equal or greater, it falls under All-in (Action 6)
                is_affordable = (effective_amount < my_stack)
                
                if is_affordable:
                    agent_legal.append(action_id)
                    
        # (3) All-in(6) Check
        # If Raise or All-in is possible in RLCard, All-in is always an option
        if can_raise_or_allin:
            agent_legal.append(6)

        return sorted(agent_legal)


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
            # legal_actions = env.get_legal_actions() # Redundant
            # print(f"DEBUG: Legal Actions: {legal_actions}") # Uncomment to see available options
            
            action = np.random.choice(legal_actions)
            action_name = action_names.get(action, "Unknown")
            
            print(f"  Hand {env.hand_count} | Player {env.current_player} acts: {action_name} (Action {action}) [Legal: {legal_actions}]")
            
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
