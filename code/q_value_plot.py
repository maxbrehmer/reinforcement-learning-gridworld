import matplotlib.pyplot as plt
import numpy as np
import pickle
from GW_env import WIDTH, HEIGHT

with open('q_tables.pkl', 'rb') as f:
    q_tables = pickle.load(f)

# Calculate the maximum Q-value for each state
max_q_values = np.max(q_tables, axis=2)

fig, ax = plt.subplots(figsize=(10, 10))

# Create a color map to represent Q-values
cmap = plt.cm.get_cmap('viridis')

# Normalize the Q-values for color mapping
norm = plt.Normalize(np.min(max_q_values), np.max(max_q_values))

# Plotting each state with its corresponding max Q-value
for x in range(0, 2*WIDTH - 1):
    for y in range(0, 2*HEIGHT - 1):
        if x < WIDTH and y < HEIGHT:  # No conversion needed
            i, j = x, y
        elif x < WIDTH and y >= HEIGHT:  # Convert y to negative
            i, j = x, (y-HEIGHT+1) * (-1)
        elif x >= WIDTH and y < HEIGHT:  # Convert x to negative
            i, j = (x-WIDTH+1) * (-1), y
        else:  # Convert x and y to negative
            i, j = (x-WIDTH+1) * (-1), (y-HEIGHT+1) * (-1)

        # Color intensity based on the max Q-value
        color = cmap(norm(max_q_values[x, y]))
        rect = plt.Rectangle((i-0.5, j-0.5), 1, 1, color=color)
        ax.add_patch(rect)

# Adjust the axis limits and aspect ratio
ax.set_xlim([-WIDTH + 0.5, WIDTH - 0.5])
ax.set_ylim([-HEIGHT + 0.5, HEIGHT - 0.5])
ax.set_aspect('equal')

# Customizing the grid
ax.set_xticks(np.arange(-WIDTH + 1, WIDTH, 1))
ax.set_yticks(np.arange(-HEIGHT + 1, HEIGHT, 1))
ax.grid(True, which='both', linestyle='--', color='grey')

ax.set_xlabel('X Distance to Goal')
ax.set_ylabel('Y Distance to Goal')
ax.set_title('Maximum Q-values Representation Based on Distance to Goal')

# Adding a color bar to understand the values
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
plt.colorbar(sm, ax=ax, label='Max Q-value')

plt.show()