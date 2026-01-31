import gymnasium as gym
import numpy as np

def run_fixed_eval(body, connections, env_name):
    # 1. Setup Environment (Same environment as before)
    env = gym.make(env_name, body=body, connections=connections)
    obs, _ = env.reset()
    total_fitness = 0
    
    # 2. Run for exactly 500 steps (One Episode)
    for t in range(500):
        # --- THE FIXED CONTROLLER FORMULA ---
        # Frequency = 20 steps, Amplitude = 0.5, Neutral = 1.1
        # This makes every actuator expand and contract in a rhythm
        signal = 1.1 + 0.5 * np.sin(2 * np.pi * t / 20)
        
        # Apply the same signal to all actuators
        num_actuators = env.action_space.shape[0]
        action = np.full(num_actuators, signal)
        
        # 3. Step the physics
        obs, reward, terminated, truncated, info = env.step(action)
        
        # 4. SUM THE REWARD (The Fitness calculation happens here)
        total_fitness += reward
        
        if terminated or truncated:
            break
            
    env.close()
    return total_fitness