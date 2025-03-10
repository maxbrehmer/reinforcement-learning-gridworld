from pettingzoo import ParallelEnv
from gym import spaces
import numpy as np

class GridWorld(ParallelEnv):
    metadata = {'render.modes': ['human'], "name": "GridWorld_v0"}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode

        self.width = 8
        self.height = 8
        self.r_step = 1
        self.r_collide = 10
        self.r_goal = self.width * self.height
        self.agents = ["agent_1", "agent_2", "agent_3"]
        self.possible_agents = self.agents[:] # For ParrallelEnv wrapper
        self.current_step = 0
        self.max_steps = 500

        # Action space is discrete, 5 possible actions (0=wait, 1=north, 2=east, 3=south, 4=west)
        self.action_spaces = {agent: spaces.Discrete(5) for agent in self.agents}

        # State space is ... (only works for width = height)
        self.state_spaces = {agent: spaces.Dict({
            "agent_positions": spaces.Box(low=0, high=self.width-1, shape=(2,), dtype=np.int64), # Possible positions for agent
            "target_positions": spaces.Box(low=0, high=self.width-1, shape=(2,), dtype=np.int64) # Possible positions for target (takes into account all targets, not just the one for the agent)
        }) for agent in self.agents}

        self.agent_positions = {agent: np.array([np.random.randint(0, self.width), np.random.randint(0, self.height)]) for agent in self.agents}
        # Agent goal: [n+(-1)**n, n+(-1)**n - min(n-np.round(self.height/2), np.round(np.random.randint(0, n)/2))] for agent in self.agents
        self.goals = {agent: np.array([n+(-1)**n, n+(-1)**n - min(n-np.round(self.height/2), np.round(self.height/2))]) for n, agent in enumerate(self.agents)}
        self.rewards = {agent: 0 for agent in self.agents}

    def reset(self):
        # Reset agent positions
        self.agent_positions = {agent: np.array([np.random.randint(0, self.width), np.random.randint(0, self.height)]) for agent in self.agents}
        self.rewards = {agent: 0 for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        return {agent: self.state(agent) for agent in self.agents}, infos
    
    def state(self, agent):
        all_agent_positions = {all_agents: np.array([self.agent_positions[all_agents]]) for all_agents in self.agents} # All agent positions
        return {
            "agent_positions": all_agent_positions, # All agent positions
            "target_positions": np.array([self.goals[agent]]) # Target position for agent
        }

    def action_space(self, agent_id):
        return self.action_spaces[agent_id] # Return action space for agent
    
    def check_terminal_condition(self, agent):
        return np.array_equal(self.agent_positions[agent], self.goals[agent])

    def step(self, actions):
        next_positions = {}
        dones = {agent: False for agent in self.agents}

        for agent, action in actions.items():
            # Calculate next position based on state and action
            current_x, current_y = self.agent_positions[agent]
            if action == 1: # North
                next_positions[agent] = [current_x, min(current_y+1, self.height-1)]
            elif action == 2: # East
                next_positions[agent] = [min(current_x+1, self.width-1), current_y]
            elif action == 3: # South
                next_positions[agent] = [current_x, max(current_y-1, 0)]
            elif action == 4: # West
                next_positions[agent] = [max(current_x-1, 0), current_y]
            else: # Wait
                next_positions[agent] = [current_x, current_y]
        
        new_states = {}
        for agent in self.agents:
            # Check for collisions
            colliding_agents = [other_agent for other_agent in self.agents if other_agent != agent and np.array_equal(next_positions[agent], next_positions[other_agent])]
            if colliding_agents:
                next_positions[agent] = self.agent_positions[agent] # If collision, stay in place
                for colliding_agent in colliding_agents:
                    next_positions[colliding_agent] = self.agent_positions[colliding_agent] # If collision, stay in place for other agents involved
                    self.rewards[colliding_agent] -= self.r_collide # Negative reward for collision for other agents involved
                self.rewards[agent] -= self.r_collide # Negative reward for collision for 'agent'
            else:
                self.agent_positions[agent] = next_positions[agent] # Move agent to new position
                # Check for agent reaching goal
                if self.agent_positions[agent] == self.goals[agent].tolist():
                    self.rewards[agent] += self.r_goal # Positive reward for reaching goal
            self.rewards[agent] -= self.r_step # Negative reward for each step
            new_states[agent] = self.state(agent)

        all_agents_done = all([self.check_terminal_condition(agent) for agent in self.agents]) # Check if all agents are done
        if self.current_step >= self.max_steps or all_agents_done:
            dones = {agent: True for agent in self.agents}  # All agents are done

        self.current_step += 1
        
        infos = {agent: {} for agent in self.agents}
        return self.state(self.agents), self.rewards, dones, infos
    
    def render(self, mode='human'):
        # Print agent positions and goals
        if self.render_mode == "human": # Print to console
            print("Agent positions: ", self.agent_positions)
            print("Agent goals: ", self.goals)
            print("Agent rewards: ", self.rewards)
        else:
            super().render(mode=mode)