from pettingzoo import ParallelEnv
from gymnasium import spaces
import numpy as np

NUM_AGENTS = 1
WIDTH = 15
HEIGHT = 15
GOAL_REWARD = WIDTH + HEIGHT
STEP_COST = 0.1
CRASH_COST = 1

class GridWorld(ParallelEnv):
    metadata = {"render.modes": ["human"], "name": "GridWorld_v0"}

    def __init__(self, render_mode=None):
        super().__init__()

        # Define agents
        self.possible_agents = ["car_" + str(i) for i in range(NUM_AGENTS)]
        # Mapping between agent name and ID
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )

        self.width = WIDTH
        self.height = HEIGHT
        self.goals = {agent: np.array([np.random.randint(0, self.width-1), np.random.randint(0, self.height-1)]) for agent in self.possible_agents}     # Random goal positions
        #self.goals = {agent: np.array([np.floor(self.width/2).astype(int), np.floor(self.height/2).astype(int)]) for agent in self.possible_agents}    # Fixed goal positions
        self.render_mode = render_mode
        self.action_space = {agent: spaces.Discrete(5) for agent in self.possible_agents} # 0=wait, 1=north, 2=east, 3=south, 4=west
        self.observation_space = spaces.Dict({
            "distance_to_goal": spaces.Tuple((spaces.Discrete(self.width), spaces.Discrete(self.height))), # x, y coordinates distance to goal
        })
    
    def reset(self):
        self.agents = self.possible_agents.copy() # Reset agents
        self.timestep = 0
        
        # Reset agent positions
        self.agent_positions = {agent: np.array([
            np.random.randint(0, self.width-1), np.random.randint(0, self.height-1)
            ]) for agent in self.agents} 
        #self.goals = self.goals                                                                                    # Goal positions set at initialization
        self.goals = {agent: np.array([np.random.randint(0, self.width-1), 
                                       np.random.randint(0, self.height-1)]) for agent in self.agents}              # Random goal positions

        self.distance_to_goal = {agent: np.array([self.goals[agent][0] - self.agent_positions[agent][0], 
                                                  self.goals[agent][1] - self.agent_positions[agent][1]]) for agent in self.agents}
        self.rewards = {agent: 0 for agent in self.agents}      # Reset rewards
        self.dones = {agent: False for agent in self.agents}    # Reset dones
        self.infos = {agent: {} for agent in self.agents}       # No additional information
    
    def step(self, actions):
        self.rewards = {agent: 0 for agent in self.agents} # Reset rewards
        tentative_positions = {}
        collisions = set()

        # Check if agents have reached their goals already
        for agent in self.agents:
            if np.array_equal(self.distance_to_goal[agent], np.array([0, 0])) and not self.dones[agent]:
                self.rewards[agent] += GOAL_REWARD
                self.dones[agent] = True

        # Step 1: Determine tentative next positions, ignoring done agents
        for agent in self.agents:
            if self.dones[agent]:  # Skip agents that are done
                continue
            action = actions[agent]
            current_position = self.agent_positions[agent]
            # Determine next position based on action
            if action == 1:  # North
                next_position = (current_position[0], min(current_position[1] + 1, self.height - 1))
            elif action == 2:  # East
                next_position = (min(current_position[0] + 1, self.width - 1), current_position[1])
            elif action == 3:  # South
                next_position = (current_position[0], max(current_position[1] - 1, 0))
            elif action == 4:  # West
                next_position = (max(current_position[0] - 1, 0), current_position[1])
            else:              # Wait
                next_position = current_position
            tentative_positions[agent] = next_position
        
        # Step 2: Detect collisions among non-done agents
        for agent, position in tentative_positions.items():
            # Convert position to a tuple for comparison
            position_tuple = tuple(position)
            if sum(1 for pos in tentative_positions.values() if tuple(pos) == position_tuple) > 1:
                collisions.add(agent)
                self.rewards[agent] -= CRASH_COST
                # Reset to current position due to collision
                tentative_positions[agent] = self.agent_positions[agent]
        
        # Step 3: Apply valid moves and update states
        for agent in self.agents:
            if self.dones[agent]:  # Do not update position or rewards for done agents
                continue

            if agent not in collisions:
                self.agent_positions[agent] = tentative_positions[agent]  # Apply move
            # Update distance to goal
            self.distance_to_goal[agent] = np.array([self.goals[agent][0] - self.agent_positions[agent][0], 
                                                     self.goals[agent][1] - self.agent_positions[agent][1]])

            # Agents entering the goal receive a reward and are set to done
            if np.array_equal(self.distance_to_goal[agent], np.array([0, 0])):
                self.rewards[agent] += GOAL_REWARD
                self.dones[agent] = True
            
            self.rewards[agent] -= STEP_COST  # Apply step cost for all agents that are not done

        self.timestep += 1
        return self.distance_to_goal, self.rewards, self.dones, self.infos