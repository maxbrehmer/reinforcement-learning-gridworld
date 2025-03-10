import GW_env
import numpy as np
import pickle

# Environment setup
env = GW_env.GridWorld()
num_episodes = 10000    # Number of episodes
alpha = 0.05            # Learning rate
gamma = 0.995           # Discount factor
epsilon = 0.2           # Exploration rate
epsilon_decay = 0.01    # Exploration decay rate
min_epsilon = 0.01      # Minimum exploration rate

# Q-table setup
# Shared Q-table for all agents with one element for each distance-to-goal combination and action
q_tables = np.zeros((
    (env.width)*2, (env.height)*2, 5 # 2 directions on 2 axes, 5 actions
    ))

# Function to select a random index from the maximum values in an array
def select_random_max(array):
    max_value = np.max(array)
    max_indices = np.where(array == max_value)[0]
    random_index = np.random.choice(max_indices)
    return random_index

# Initialize total rewards and moving averages
total_rewards = {episode: 0 for episode in range(num_episodes)}
moving_averages = {episode: 0 for episode in range(num_episodes)}

def main(env=env, num_episodes=num_episodes, alpha=alpha, gamma=gamma, epsilon=epsilon, epsilon_decay=epsilon_decay, min_epsilon=min_epsilon, q_tables=q_tables, total_rewards=total_rewards, moving_averages=moving_averages):
    # Q-learning algorithm
    for episode in range(num_episodes):
        terminate = False
        total_reward = 0
        env.reset()
        #print(f"Agent positions: {env.agent_positions}")

        while not terminate:
            action_dict = {}
            for agent in env.agents:
                state = env.distance_to_goal[agent] # Current state
                if state[0] < 0:
                    state[0] = (env.width-1) + np.abs(state[0])
                if state[1] < 0:
                    state[1] = (env.height-1) + np.abs(state[1])

                if np.random.uniform(0, 1) > epsilon and np.sum(np.abs(q_tables[state[0], state[1]])) > 0:
                    #action = np.argmax(q_tables[state[0], state[1]]) # Exploit
                    action = select_random_max(q_tables[state[0], state[1]]) # Exploit
                else:
                    action = np.random.choice([0, 1, 2, 3, 4]) # Explore
                action_dict[agent] = action
                current_dist = env.distance_to_goal[agent] # Current distance to goal

            # Take action
            next_state, reward, done, _ = env.step(action_dict)

            # Update Q-table
            for agent in env.agents:
                next_dist = next_state[agent] # Next distance to goal
                if current_dist[0] < 0:
                    current_dist[0] = (env.width-1) + np.abs(current_dist[0])
                if current_dist[1] < 0:
                    current_dist[1] = (env.height-1) + np.abs(current_dist[1])
                if next_dist[0] < 0:
                    next_dist[0] = (env.width-1) + np.abs(next_dist[0])
                if next_dist[1] < 0:
                    next_dist[1] = (env.height-1) + np.abs(next_dist[1])

                old_q_value = q_tables[current_dist[0], current_dist[1], action_dict[agent]]       # Previous Q-value
                td_target = reward[agent] + gamma * np.max(q_tables[next_dist[0], next_dist[1]])   # TD target value
                td_error = td_target - old_q_value                                                 # TD error
                q_tables[current_dist[0], current_dist[1], action_dict[agent]] += alpha * td_error # Update Q-value using Q-learning update rule

            # Update total reward
            total_reward += sum(reward.values())

            # Check if episode is done
            terminate = all(done.values()) or env.timestep >= 1000

            # Print positions and actions for each agent as well as the Q-values and the total reward
            #print(f"Agent actions: {action_dict}")
            #print(f"Agent positions: {next_state}")
            #print(f"Agent Q-values: \n{q_tables}")
            #print(f"Total reward: {total_reward}")

        total_rewards[episode] = total_reward
        # Moving average of total reward over the last 100 episodes or the available episodes
        moving_average = np.mean(list(total_rewards.values())[max(0, episode-100):episode+1])

        # Store moving average of total reward
        moving_averages[episode] = moving_average

        # Decay exploration rate
        epsilon = max(min_epsilon, epsilon * np.exp(-epsilon_decay * episode))
        #epsilon = max(min_epsilon, epsilon * epsilon_decay)

        # Print episode results and moving average of total reward
        print(f"Episode {episode+1}/{num_episodes} - Average total reward: {moving_average:.1f} - Epsilon: {epsilon:.2f}")
        #print()

        # Save moving averages and Q-tables
        with open('moving_averages.pkl', 'wb') as f:
            pickle.dump(moving_averages, f)

        with open('q_tables.pkl', 'wb') as f:
            pickle.dump(q_tables, f)

if __name__ == "__main__":
    main()
