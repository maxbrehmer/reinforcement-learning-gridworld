# Plot values
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
import numpy as np
import pickle
from myminigrid import num_episodes, alpha, gamma, epsilon, epsilon_decay, min_epsilon
from GW_env import NUM_AGENTS, WIDTH, HEIGHT, GOAL_REWARD, STEP_COST, CRASH_COST

with open('moving_averages.pkl', 'rb') as f:
    moving_averages = pickle.load(f)

# Extract episodes and moving average values
episodes = list(moving_averages.keys())
average_rewards = list(moving_averages.values())

# String of parameter values
parameters_str = f"""Num Episodes: {num_episodes}
Alpha: {alpha}
Gamma: {gamma}
Epsilon: {epsilon}
Epsilon Decay: {epsilon_decay}
Min Epsilon: {min_epsilon}
Num Agents: {NUM_AGENTS}
Width: {WIDTH}
Height: {HEIGHT}
Goal Reward: {GOAL_REWARD}
Step Cost: {STEP_COST}
Crash Cost: {CRASH_COST}"""

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(episodes, average_rewards, color='blue', label='Moving Average of Total Rewards')

# Labeling the plot
plt.xlabel('Episode')
plt.ylabel('Average Total Reward')
plt.title('Moving Average of Total Rewards Over Episodes')
plt.legend()

# Display parameters in a neat box
plt.figtext(0.7, 0.2, parameters_str, bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="black", lw=1))

# Display the plot
plt.grid(True)
plt.show()
