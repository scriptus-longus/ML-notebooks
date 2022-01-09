"""
the following is an implementation of dual q learing architecture
for the Atari game SpaceInvaders.
The code also demonstrates how mutlithreading can be used in ML
and RL tasks to speed up training
"""
import torch
import gym
import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
from threading import *
from time import sleep

device = torch.device("cuda")
print("cuda available: " + str(torch.cuda.is_available))

# standard replay memory
class ReplayMemory():
  def __init__(self, capacity=3000):
    self.memory = []
    self.capacity = capacity

  def __len__(self):
    return len(self.memory)

  def add_transition(self, state, policy, reward, new_state, is_final):
    transition = [state, policy, reward, new_state, is_final]
    if len(self.memory) >= self.capacity:
      self.memory.pop(random.randrange(len(self.memory)))
    self.memory.append(transition)

  def get_batch(self, batch_size):
    if batch_size > len(self.memory):
      return self.memory
    return random.sample(self.memory, k=batch_size)


class Game():
  def __init__(self):
    self.env = gym.make("SpaceInvaders-v0")
    self.observation_history = []
        
  def get_state(self, state_length=10):
    if len(self.observation_history) < state_length:
      ret = [np.zeros((210, 160, 3))]*(state_length-len(self.observation_history))
      ret += self.observation_history
    else:
      ret = self.observation_history[-state_length:]
    ret = np.concatenate(ret, axis=2)
    return ret
    
  def play(self, model=None, render=True, debug_log=False):
    observation = env.reset()
    self.observation_history.append(observation)
       
    while True:
      if render:
        env.render()
                
        state = self.get_state()
                
        if model != None:
          state = torch.as_tensor(state, dtype=torch.float)
          state = state.permute(2, 0, 1).unsqueeze(0)
          action = int(torch.argmax(model(state.to(device))))
        else:
          action = env.action_space.sample()
                
        observation, reward, done, _ = env.step(action)
        self.observation_history.append(observation)
            
        if done:
          observation = env.reset()
          break

    env.close()

class DQN(torch.nn.Module):
  def __init__(self, state_length):
    super(DQN, self).__init__()
    self.conv1 = torch.nn.Conv2d(10*3, 50, kernel_size=3, stride=2)
    self.conv2 = torch.nn.Conv2d(50, 50, kernel_size=3, stride=1)
    self.conv3 = torch.nn.Conv2d(50, 40, kernel_size=3, stride=1)
    self.conv4 = torch.nn.Conv2d(40, 30, kernel_size=3, stride=1)
      
    self.relu = torch.nn.functional.relu
    self.pooling = torch.nn.MaxPool2d(2,2)
        
    self.action_value_l1 = torch.nn.Linear(360, 100)
    self.action_value_l2 = torch.nn.Linear(100, 1)
      
    self.policy_l1 = torch.nn.Linear(360, 100)
    self.policy_l2 = torch.nn.Linear(100, 6)
        
  def forward(self, state):
    #print(state.shape)
    state = self.relu(self.conv1(state))
    state = self.pooling(state)
    state = self.relu(self.conv2(state))
    state = self.pooling(state)
    state = self.relu(self.conv3(state))
    state = self.pooling(state)
    state = self.relu(self.conv4(state))
    state = self.pooling(state)
                
    #print(state.shape)
            
    state = state.view(-1, 360)
        
    a = self.relu(self.action_value_l1(state))
    a = self.action_value_l2(a)       # action value "channel"
   
    p = self.relu(self.policy_l1(state))
    p = self.policy_l2(p)             # policy "channel"
 
    q = a + p - p.mean()
    return q

class Trainer(object):
  def __init__(self, model, batch_size=30, learning_rate=0.001):
    self.batch_size = batch_size
    self.memory =  ReplayMemory()
    self.model = model
    self.state_length = 10
    
    self.prev_model = DQN(self.state_length)
    self.prev_model.apply(self.init_weights) 
    self.prev_model.to(device) 
 
    self.criterion = torch.nn.MSELoss()
    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

    self.playing = False
    self.losses = []

  def init_weights(self, m):
    if type(m) == torch.nn.Linear:
      torch.nn.init.xavier_uniform_(m.weight)
      m.bias.data.fill_(0.01)

  def optimize(self, epochs,  gamma=0.999):
    #torch.autograd.set_detect_anomaly(True)
    self.playing = True

    play_thread = Thread(target=self.play)           # we play the game in a second thread to speed up training
    play_thread.start()
    gamma = torch.tensor(gamma).to(device)
     
    # wait for training data
    while True:
      if len(self.memory) > self.batch_size:
        break

    for e in tqdm(range(epochs)):
      batch = self.memory.get_batch(batch_size=self.batch_size)
 
      q_values = torch.zeros(len(batch))
      prev_q_values = torch.zeros(len(batch))
      rewards = torch.zeros(len(batch))
  
  
      for i in range(self.batch_size):
        q_values[i] =  self.model(batch[i][0].to(device)).flatten()[batch[i][1]]

        if batch[i][3] != None:
          prev_q_values[i] = torch.max(self.prev_model(batch[i][3].to(device)))

        rewards[i] = batch[i][2]
      
      q_values = q_values.to(device)
      prev_q_values = prev_q_values.to(device)
      rewards = rewards.to(device)
    
      prev_q_values = prev_q_values * gamma + rewards

      loss = self.criterion(q_values, prev_q_values)
      self.optimizer.zero_grad()
      loss.backward()
      self.optimizer.step()
      
      if e % 5 == 0:
        self.prev_model.load_state_dict(self.model.state_dict())    

    self.playing = False
    return 
  
  def play(self, epsilon=0.7, decay=0.998):
    game = Game()
    e = 0
    while True:
      observation = game.env.reset()
      game.observation_history.append(observation)
      
      for i in range(200):
        use_model = random.random() > epsilon
        
        state = game.get_state()
        state = torch.as_tensor(state, dtype=torch.float)
        state = state.permute(2, 0, 1).unsqueeze(0)    
 
        if use_model:
          q_values = self.model(state.to(device))
          action = int(torch.argmax(q_values))
        else:
          action = random.randint(0,5)

        observation, reward, done, _  = game.env.step(action)
        game.observation_history.append(observation)

        if done:
          new_state = None
        else:
          new_state = game.get_state()
          new_state = torch.as_tensor(new_state, dtype=torch.float)
          new_state = new_state.permute(2,0,1).unsqueeze(0)

        self.memory.add_transition(state, action, reward, new_state, done)
        
        if done:
          break

      e += 1
      if self.playing == False:
        break
      
      epsilon *= decay
      game.env.reset()
      game.observation_history = []
    game.env.close()
    return
   
if __name__ == "__main__" :
  model = DQN(state_length=10).to(device)
  trainer = Trainer(model)
  trainer.optimize(epochs=200)

  torch.save(model.state_dict(), "model01.pt")
