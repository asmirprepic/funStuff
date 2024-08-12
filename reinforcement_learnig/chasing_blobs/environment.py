from blob import Blob
import numpy as np
import matplotlib.pyplot as plt
import cv2

class Environment:
  def __init__(self, size, num_actions, learning_rate, discount, episodes, move_penalty, enemy_penalty, food_reward):
    self.size = size
    self.blobs = {}
    self.num_actions = num_actions
    self.learning_rate = learning_rate
    self.discount = discount 
    self.move_penalty = move_penalty
    self.enemy_penalty = enemy_penalty
    self.food_reward = food_reward
    self.q_table_player = {}
    self.q_table_enemy = {}
    self.episodes = episodes
    self.caught_count = 0
    self.evaded_count = 0
    self.caught_counts = []
    self.evaded_counts = []
    self.player_rewards = []
    self.enemy_rewards = []
    self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 8))  # Two subplots



  def create_blob(self,blob_name):
    if blob_name in self.blobs:
      self.blobs[blob_name].reset_position()
      #raise ValueError(f"A blob named {blob_name} already exists")
    
    self.blobs[blob_name] = Blob(self.size)
    
  #def initialize_q_table(self):
  #  q_table = {}
    
  def update_q_table(self,q_table,obs,action,reward,new_obs):
    if obs not in q_table:
        q_table[obs] = [np.random.uniform(-5, 0) for _ in range(self.num_actions)]

    # Likewise for new_obs
    if new_obs not in q_table:
        q_table[new_obs] = [np.random.uniform(-5, 0) for _ in range(self.num_actions)]

    current_q = q_table[obs][action]
    max_future_q = np.max(q_table[new_obs])

    # Calculate the new Q-value
    new_q = (1 - self.learning_rate) * current_q + self.learning_rate * (reward + self.discount * max_future_q)

    # Update the Q-table
    q_table[obs][action] = new_q

  def get_q_value(self,q_table,state,action):
    if state not in self.q_table:
      
      q_table[state] = [np.random.uniform(-5,0) for _ in range (self.num_actions)]
  
    return q_table[state][action]

  def epsilon_greedy_action(self,state,q_table,epsilon):
    if np.random.random() < epsilon:
      
      return np.random.randint(0,self.num_actions)
    else:
      
      return self.greedy_action(state)
  
  def greedy_action(self,state):
    #print(state)
    #player,enemy = state
    dx,dy = state
    
    if abs(dx) > abs(dy):
      return 3 if dx > 0 else 1
    else:
      return 0 if dy > 0 else 2

  def calculate_reward_player(self,player,enemy):
    if player.x == enemy.x and player.y == enemy.y:
      return -self.enemy_penalty
    else: 
      distance = np.sqrt((player.x-enemy.x)**2 + (player.y-enemy.y)**2)
      distance_reward = distance/self.size
      proximity_penalty = 10 / (distance + 1)
      survival_reward = 10*(distance/self.size)
      return   survival_reward #- proximity_penalty

  def calculate_reward_enemy(self,player,enemy):
    if player.x == enemy.x and player.y == enemy.y:
      return self.enemy_penalty
    else:
      distance = np.sqrt((player.x-enemy.x)**2 + (player.y-enemy.y)**2)
      distance_reward = -distance/self.size
      chase_reward = 20 * ((self.size - distance) / self.size)
      return distance_reward + chase_reward

  def choose_action(self,state,q_table,epsilon):
    if np.random.random() < epsilon:
      return np.random.randint(0,self.num_actions)
    
    else:
      return np.argmax(q_table.get(state,[0]*self.num_actions))

  def train(self, epsilon, show_every):
    episode_rewards_player = []
    episode_rewards_enemy = []
    self.create_blob('player')
    self.create_blob("enemy")
    player = self.blobs["player"]
    enemy = self.blobs["enemy"]

    for episode in range(self.episodes):
        
      self.blobs['player'].reset_position()
      self.blobs['enemy'].reset_position()

      episode_reward_player = 0
      episode_reward_enemy = 0
      caught = False

    
      for i in range(200):  # Number of steps in an episode
        obs_player = (player-enemy)
        obs_enemy = (enemy-player)


        #if np.random.random() > epsilon:
        #    action = np.argmax([self.get_q_value(obs, a) for a in range(self.num_actions)])
        #else:
        #    action = np.random.randint(0, self.num_actions)

        action_player = self.choose_action(obs_player,self.q_table_player,epsilon)
        action_enemy = self.choose_action(obs_enemy,self.q_table_enemy,epsilon)

        
        player.action(action_player)
        enemy.action(action_enemy)
        
        reward_player = self.calculate_reward_player(player, enemy)
        reward_enemy = self.calculate_reward_enemy(player,enemy)
        
        episode_reward_player += reward_player
        episode_reward_enemy += reward_enemy

        new_obs_player = ( player-enemy)
        new_obs_enemy = (enemy-player)
        #print(f"new_obs: {new_obs}, actions: {[self.get_q_value(new_obs, a) for a in range(self.num_actions)]}")
        
        self.update_q_table(self.q_table_player,obs_player,action_player,reward_player,new_obs_player)
        self.update_q_table(self.q_table_enemy,obs_enemy,action_enemy,reward_enemy,new_obs_enemy)

        if episode % show_every == 0:
          show = True
        else:
          show = False

        if show:
          self.render()

        if player.x == enemy.x and player.y == enemy.y:
          caught = True
          self.caught_count += 1
          break
      
      if not caught:
        self.evaded_count += 1

      self.player_rewards.append(episode_reward_player)
      self.enemy_rewards.append(episode_reward_enemy)
      self.caught_counts.append(self.caught_count)
      self.evaded_counts.append(self.evaded_count)


      if episode % show_every == 0:
        print(f"on #{episode}, epsilon is {epsilon}")
        print(f"Player was caught {self.caught_count} times.")
        print(f"Player evaded {self.evaded_count} times.")
        self.update_plot(episode)

      episode_rewards_player.append(episode_reward_player)
      episode_rewards_enemy.append(episode_reward_enemy)

      # Decay epsilon
      if epsilon > 0.1:
         epsilon *= 0.99995
          
    
    return episode_rewards_player, episode_rewards_enemy
  
  def update_plot(self, episode, window_size=100):
    self.ax1.clear()
    self.ax2.clear()

    if len(self.player_rewards) >= window_size:
        ma_player_rewards = self.moving_average(self.player_rewards, window_size)
        ma_enemy_rewards = self.moving_average(self.enemy_rewards, window_size)
        self.ax1.plot(ma_player_rewards, label='Player Rewards (MA)')
        self.ax1.plot(ma_enemy_rewards, label='Enemy Rewards (MA)')
    else:
        self.ax1.plot(self.player_rewards, label='Player Rewards')
        self.ax1.plot(self.enemy_rewards, label='Enemy Rewards')

    self.ax1.legend()
    self.ax1.set_title(f'Rewards (Episode: {episode})')
    self.ax1.set_xlabel('Episode')
    self.ax1.set_ylabel('Reward')

    # Calculate percentages
    total = self.caught_count + self.evaded_count
    if total > 0:
        caught_percentage = (self.caught_count / total) * 100
        evaded_percentage = (self.evaded_count / total) * 100
    else:
        caught_percentage = evaded_percentage = 0

    # Plotting caught counts as a bar chart
    self.ax2.bar(['Caught', 'Evaded'], [caught_percentage, evaded_percentage], color=['red', 'green'])
    self.ax2.set_title('Caught and Evaded Counts as Percentage')
    self.ax2.set_xlabel('Outcome')
    self.ax2.set_ylabel('Percentage')

    plt.draw()
    plt.pause(0.01)

  def moving_average(self,data,window_size):
    return np.convolve(data,np.ones(window_size)/window_size,mode='valid')
  
  def render(self):
    env = np.zeros((self.size, self.size, 3), dtype=np.uint8)
    player = self.blobs['player']
    enemy = self.blobs['enemy']

    env[player.y][player.x] = [255, 0, 0]
    env[enemy.y][enemy.x] = [0, 0, 255]

    img = cv2.resize(env, (300, 300), interpolation=cv2.INTER_NEAREST)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, 'Player', (player.x * 300 // self.size, player.y * 300 // self.size + 15),
                font, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, 'Enemy', (enemy.x * 300 // self.size, enemy.y * 300 // self.size + 15),
                font, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    if player.x == enemy.x and player.y == enemy.y:
        cv2.putText(img, 'Player Caught!', (150, 150), font, 0.6, (255, 255, 0), 2, cv2.LINE_AA, bottomLeftOrigin=False)
    
        
    cv2.imshow("Environment", img)
    cv2.waitKey(1)

  



