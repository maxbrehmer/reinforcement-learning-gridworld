from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector
from gymnasium import spaces
import numpy as np

class GridWorldEnv(AECEnv):
    metadata = {'render.modes': ['human']}
    
    def _init_(self, size=5):
      super()._init_()
      self.size = size
      self.action_space = spaces.Discrete(5)  # 0: up, 1: right, 2: down, 3: left 4: wait
      self.observation_space = spaces.Box(low=0, high=size-1, shape=(2,), dtype=int)
      
      self.agents = ['car_1', 'car_2']
      self.possible_agents = self.agents.copy()
      # mapping for goal
      self.agent_name_mapping = dict(zip(self.possible_agents, list(range(len(self.possible_agents)))))
      
      self._agent_selector = agent_selector(self.agents)
  
      # Initialize rewards, dones, and infos dictionaries
      self.rewards = {agent: 0.0 for agent in self.agents}
      self.dones = {agent: False for agent in self.agents}
      self.infos = {agent: {} for agent in self.agents}
      
      self.reset()


    def reset(self):
        self.agent_pos = {agent: np.array([0, 0]) for agent in self.agents}
        self.goal_pos = {agent: np.array([self.size-1, self.size-1]) for agent in self.agents}
        #Agent goal: [n+(-1)*n, n+(-1)*n - min(n-np.round(self.height/2), np.round(np.random.randint(0, n)/2))] for agent in self.agents
        self.agent_selection = self._agent_selector.reset()
    
        # Reset rewards, dones, and infos for each agent
        self.rewards = {agent: 0.0 for agent in self.agents}
        self.dones = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

    # Below is unknown why it was implemented
    # def _clear_rewards_done_infos(self):
    #       self.rewards = {agent: 0.0 for agent in self.agents}
    #       self.dones = {agent: False for agent in self.agents}
    #       self.infos = {agent: {} for agent in self.agents}
      
    def step(self, action,agentdone):
        
        agent = self.agent_selection
        if action == 0:  # up
            self.agent_pos[agent][1] = min(self.size - 1, self.agent_pos[agent][1] + 1)
        elif action == 1:  # right
            self.agent_pos[agent][0] = min(self.agent_pos[agent][0] + 1, self.size - 1)
        elif action == 2:  # down
            self.agent_pos[agent][1] = max(self.agent_pos[agent][1] - 1, 0)
        elif action == 3:  # left
            self.agent_pos[agent][0] = max(self.agent_pos[agent][0] - 1, 0)
        elif action == 4:
            pass
        
        # Check if the agent reached its goal
        reached_goal = np.array_equal(self.agent_pos[agent], self.goal_pos[agent])
        
        if not agentdone:
          # Prepare reward
          reward = 10 if reached_goal else -1  # Reward for reaching the goal, small penalty otherwise
        
        else:
          reward = 0
        
        # Done flag
        done = reached_goal
        
        # Info
        info = {}
        
        self.rewards[agent] = reward
        self.dones[agent] = done
        self.infos[agent] = info
        
        #self._clear_rewards_done_infos()
        self.agent_selection = self._agent_selector.next()
      

import numpy as np

# Environment setup
env = GridWorldEnv(size=5)
num_episodes = 30
alpha = 0.1  # Learning rate
gamma = 0.99  # Discount factor
epsilon = 0.5  # Exploration rate
epsilon_decay = 0.995  # Decay rate for exploration
min_epsilon = 0.01

# Q-Table initialization: For simplicity, we assume states are just positions on the grid, and actions are the four directions.
num_states = env.size * env.size
num_actions = 5
q_table = np.zeros((num_states, num_actions))

# Convert position to a single integer state
def pos_to_state(pos):
    return pos[0] * env.size + pos[1] #change (leads to colisions)

i = 0
for episode in range(num_episodes):
    done = np.array([False,False])
    total_reward = 0
    env.reset()
    while not all(done):
        action_dict = {}
        for agent in env.agents:
            
            state = pos_to_state(env.agent_pos[agent])
            if not done[env.agent_name_mapping[agent]]:
                if np.random.rand() < epsilon:
                    action = np.random.choice(num_actions)  # Explore
                else:
                    action = np.argmax(q_table[state])  # Exploit
            else:
                action = 4
            #print(str(action))
                
        
        # Step in the environment for each agent
        for agent, action in action_dict.items():
            
            env.step(action,done[env.agent_name_mapping[agent]]) # Changes agent position
            next_state = pos_to_state(env.agent_pos[agent]) # Checks next state for new position
            reward = env.rewards[agent]
            done[env.agent_name_mapping[agent]] = env.dones[agent] # array of dones
            print(agent + ": " + str(done[env.agent_name_mapping[agent]]))
            # Q-Learning Update
            max_next_q = np.max(q_table[next_state])
            q_table[state, action] = q_table[state, action] + alpha * (reward + gamma * max_next_q - q_table[state, action])
            
            total_reward += reward
            
        # Decay epsilon
        epsilon = max(min_epsilon, epsilon * epsilon_decay)
    print(f"Episode {episode + 1}: Total Reward: {total_reward}, Epsilon: {epsilon}")