import pandas as pd
import sys
import os

log_path = r"c:\Users\99san\.gemini\antigravity\playground\glacial-supernova\experiments\logs\eta-f\PPO_poker_env_62fec_00000_0_2025-12-03_21-34-18\progress.csv"

try:
    df = pd.read_csv(log_path)
    
    if df.empty:
        print("Log file is empty.")
        sys.exit(0)

    last_row = df.iloc[-1]
    
    print("--- Optimized Run (eta-f) Speed Report ---")
    
    # Throughput columns
    throughput_cols = [c for c in df.columns if 'throughput' in c]
    for col in throughput_cols:
        print(f"{col}: {last_row[col]:.2f}")

    if 'timesteps_total' in df.columns and 'time_total_s' in df.columns:
        total_steps = last_row['timesteps_total']
        total_time = last_row['time_total_s']
        avg_speed = total_steps / total_time if total_time > 0 else 0
        print(f"Average Speed (Total): {avg_speed:.2f} steps/s")
        print(f"Total Steps: {total_steps}")
        print(f"Total Time: {total_time:.2f} s")
        
        # Recent speed (last 3 updates to see trend)
        if len(df) > 3:
            recent_df = df.iloc[-3:]
            delta_steps = recent_df['timesteps_total'].iloc[-1] - recent_df['timesteps_total'].iloc[0]
            delta_time = recent_df['time_total_s'].iloc[-1] - recent_df['time_total_s'].iloc[0]
            recent_speed = delta_steps / delta_time if delta_time > 0 else 0
            print(f"Recent Speed (Last 3 updates): {recent_speed:.2f} steps/s")

except Exception as e:
    print(f"Error reading log: {e}")
