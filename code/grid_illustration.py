import matplotlib.pyplot as plt
import numpy as np

WIDTH, HEIGHT = 5, 5

# Adjusted ranges to include from -4 to +4, aligning with observation space
x_range = np.arange(-WIDTH + 1, WIDTH)  # From -4 to 4
y_range = np.arange(-HEIGHT + 1, HEIGHT)  # From -4 to 4
xx, yy = np.meshgrid(x_range, y_range)
distance = np.sqrt(xx**2 + yy**2)

# Normalize distances for color intensity
max_distance = np.sqrt((WIDTH - 1)**2 + (HEIGHT - 1)**2)
norm_distance = 1 - (distance / max_distance)

fig, ax = plt.subplots(figsize=(10, 10))
for i in range(len(x_range)):
    for j in range(len(y_range)):
        # Color intensity based on distance
        color_value = norm_distance[j, i]
        rect = plt.Rectangle((x_range[i]-0.5, y_range[j]-0.5), 1, 1, color=(color_value, 0, 0, 0.5))  # Semi-transparent
        ax.add_patch(rect)

# Adjust the axis limits and aspect ratio
ax.set_xlim([-WIDTH + 0.5, WIDTH - 1.5+1])  # From -4 to 4
ax.set_ylim([-HEIGHT + 0.5, HEIGHT - 1.5+1])  # From -4 to 4
ax.set_aspect('equal')

# Customizing the grid
ax.set_xticks(np.arange(-WIDTH + 1, WIDTH, 1))
ax.set_yticks(np.arange(-HEIGHT + 1, HEIGHT, 1))
ax.grid(True, which='both', linestyle='--', color='grey')

ax.set_xlabel('X Distance to Goal')
ax.set_ylabel('Y Distance to Goal')
ax.set_title('State Representation Based on Distance to Goal')

plt.show()
