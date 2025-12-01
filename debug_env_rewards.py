
import numpy as np
from poker_rl.env import PokerMultiAgentEnv

def test_rewards():
    env = PokerMultiAgentEnv()
    obs, info = env.reset()
    
    print(f"Initial Stacks: {env.hand_start_stacks}")
    
    done = False
    step_count = 0
    
    while not done:
        # Random actions
        actions = {
            "player_0": env.action_space.sample(),
            "player_1": env.action_space.sample()
        }
        
        # Only pass action for current player
        current_player = env.game.get_current_player()
        agent_id = f"player_{current_player}"
        actual_actions = {agent_id: actions[agent_id]}
        
        print(f"Step {step_count}: Player {current_player} acts {actions[agent_id]}")
        
        obs, rewards, terminated, truncated, info = env.step(actual_actions)
        step_count += 1
        
        if "player_0" in rewards and rewards["player_0"] != 0:
            print(f"Rewards received: {rewards}")
            
        done = terminated["__all__"] or truncated["__all__"]
        
    print(f"Final Chips: {env.chips}")
    print(f"Start Chips: {env.hand_start_stacks}")
    print(f"Chip Change P0: {env.chips[0] - env.hand_start_stacks[0]}")
    print(f"Chip Change P1: {env.chips[1] - env.hand_start_stacks[1]}")

if __name__ == "__main__":
    test_rewards()
