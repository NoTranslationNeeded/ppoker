"""
PettingZoo ParallelEnv wrapper for Tournament Poker Environment
Enables RLlib to train on tournament-style poker
"""
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from gymnasium import spaces
import numpy as np
from tournament_poker_env import TournamentPokerEnv

class TournamentPokerParallelEnv(MultiAgentEnv):
    metadata = {
        "render_modes": ["human"],
        "name": "tournament_poker_v1"
    }

    def __init__(self, starting_chips=100, randomize_stacks=True, render_mode=None,
                 reward_type='icm_survival', reward_config=None, max_hands=1000):
        super().__init__()
        
        self.env = TournamentPokerEnv(
            starting_chips=starting_chips,
            small_blind=1,
            big_blind=2,
            randomize_stacks=randomize_stacks,
            reward_type=reward_type,
            reward_config=reward_config,
            max_hands=max_hands
        )
        self.render_mode = render_mode
        
        # Define agents (Ray expects _agent_ids set, but we manage dynamically)
        self.possible_agents = ["player_0", "player_1"]
        self.agents = self.possible_agents[:]
        self._agent_ids = set(self.possible_agents)
        
        # Define spaces - 60 dimensions with blind level info
        obs_space = spaces.Box(low=0, high=1, shape=(60,), dtype=np.float32)
        act_space = spaces.Discrete(self.env.action_space_size)
        
        self.observation_spaces = {agent: obs_space for agent in self.possible_agents}
        self.action_spaces = {agent: act_space for agent in self.possible_agents}
        
        # Ray expects these to be directly accessible as dicts for MultiAgentEnv
        self.observation_space = self.observation_spaces
        self.action_space = self.action_spaces
        
    def reset(self, *, seed=None, options=None):
        """Reset the environment"""
        if seed is not None:
            np.random.seed(seed)
            
        # Reset underlying env
        self.agents = self.possible_agents[:]
        obs = self.env.reset()
        
        # Return observations - only current player observes
        current_player = f"player_{self.env.current_player}"
        observations = {current_player: obs}
        
        # Return infos - ONLY for the agent receiving an observation
        # Ray requires info keys to be a subset of observation keys
        infos = {current_player: {}}
        
        return observations, infos
    
    def step(self, action_dict):
        """
        Execute actions for agents who are acting this turn
        Ray passes a dictionary: {agent_id: action}
        """
        # Get current acting agent
        current_player = f"player_{self.env.current_player}"
        
        # Execute action if provided
        if current_player in action_dict:
            action = action_dict[current_player]
            next_obs, reward, done, info = self.env.step(action)
            
            # Prepare returns
            observations = {}
            rewards = {}
            terminations = {}
            truncations = {}
            infos = {}
            
            if done:
                # Tournament over - get final payoffs
                # Normalize by MAX_CHIPS (consistent with environment)
                MAX_CHIPS = 50000.0
                final_chips = info['final_chips']
                starting_chips = info['starting_chips']
                
                chip_diff_0 = final_chips[0] - starting_chips[0]
                chip_diff_1 = final_chips[1] - starting_chips[1]
                
                # Assign rewards (normalized by MAX_CHIPS)
                rewards["player_0"] = float(chip_diff_0 / MAX_CHIPS)
                rewards["player_1"] = float(chip_diff_1 / MAX_CHIPS)
                
                # Mark both as terminated
                terminations["player_0"] = True
                terminations["player_1"] = True
                truncations["player_0"] = False
                truncations["player_1"] = False
                
                # Empty observations (game over)
                observations = {}
                
                # Infos - can be empty or contain final info if needed, but no obs means no info usually in Ray
                # But for termination, we might want to pass info. 
                # However, to be safe with "subset of obs" rule, we keep it empty or only if we returned obs.
                # Actually, Ray allows info for agents not in obs IF they are in the returned info dict?
                # The error said: "Key set for infos must be a subset of obs"
                # So if obs is empty, infos MUST be empty.
                infos = {}
                
                # Ray 2.x: Add __all__ to terminations/truncations
                terminations["__all__"] = True
                truncations["__all__"] = False
                
            else:
                # Game continues
                next_agent = f"player_{self.env.current_player}"
                observations = {next_agent: next_obs}
                
                # Reward is for the agent who acted
                rewards = {current_player: float(reward)}
                
                terminations = {agent: False for agent in self.possible_agents}
                truncations = {agent: False for agent in self.possible_agents}
                
                # Ray 2.x: Add __all__
                terminations["__all__"] = False
                truncations["__all__"] = False
                
                # Infos - ONLY for the next agent who gets an observation
                infos = {next_agent: info}
                
            return observations, rewards, terminations, truncations, infos
        else:
            # No action for current player (should not happen in Ray loop if configured correctly)
            return {}, {}, {"__all__": False}, {"__all__": False}, {}

if __name__ == "__main__":
    # Test the PettingZoo wrapper
    print("=" * 80)
    print("Testing Tournament Poker PettingZoo Wrapper")
    print("=" * 80)
    
    env = TournamentPokerParallelEnv(randomize_stacks=True)
    
    for episode in range(2):
        print(f"\n🎮 Episode {episode + 1}")
        observations, infos = env.reset()
        print(f"Starting chips: {env.env.chips}")
        print(f"Observation shape: {list(observations.values())[0].shape}")
        
        done = False
        step_count = 0
        max_steps = 500
        
        while not done and step_count < max_steps:
            # Get current agent
            current_agent = list(observations.keys())[0]
            
            # Random action
            legal_actions = env.env.get_legal_actions()
            action = np.random.choice(legal_actions)
            
            # Step
            observations, rewards, terminations, truncations, infos = env.step({current_agent: action})
            
            # Check if done
            done = any(terminations.values())
            step_count += 1
            
            if done:
                print(f"\n✅ Episode complete after {step_count} steps")
                # Info might be empty now due to our fix, so be careful accessing it for print
                # We can't easily print winner info here if info is empty.
                # But for training it's fine.
                print("Tournament Over")
        
        if not done:
            print(f"\n⚠️ Max steps reached ({max_steps})")
    
    print("\n" + "=" * 80)
    print("✅ Wrapper test complete!")
