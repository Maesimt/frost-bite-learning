import sys
from collections import deque
import random
import numpy as np
import tensorflow as tf
import os.path
from os import path
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.activations import relu, linear
from tensorflow.keras.optimizers import SGD, Adam

from agents.agent import Agent
from helpers import showProgress, meanOfLast

def QNetwork(obs_size, num_actions, nhidden, lr):

    model = Sequential()
    model.add(Dense(nhidden, input_dim=obs_size, activation=relu))
    model.add(Dense(nhidden, activation=relu))
    model.add(Dense(num_actions, activation=linear))
    model.compile(loss=tf.keras.losses.Huber(), optimizer=Adam(lr=lr))
    
    return model

class DQNWatchAgent(Agent):
    """Deep Q-Learning agent"""

    def __init__(self, actions, obs_size, **kwargs):
        super(DQNWatchAgent, self).__init__(obs_size, actions)

        # Taille de S
        self.obs_size = obs_size.shape[0]
        self.actions = actions
        self.episodes_not_saved = 0
        
        # Epsilon
        self.epsilon = kwargs.get('epsilon', .01)       
        # Si epsilon = 1, décroissance progressive
        if self.epsilon == 1:
            self.epsilon_decay = True
        else:
            self.epsilon_decay = False
        
        # Facteur de dévaluation
        self.gamma = kwargs.get('gamma', .99)
        
        # Hyperparamètres des réseaux de neurones (modèle et cible)
        self.batch_size = kwargs.get('batch_size', 64)
        self.epoch_length = kwargs.get('epoch_length', 100)
        self.lr = kwargs.get('learning_rate', .0001)
        self.tau = kwargs.get('tau', .05)
        self.nhidden = kwargs.get('nhidden', 150)
        
        # Instanciation des réseaux de neurones (modèle et cible)
        self.model_network = QNetwork(self.obs_size, self.num_actions, kwargs.get('nhidden', 150), self.lr)
        # Load existing model.

        self.model_network.load_weights("/home/guillaumecummings/Desktop/weights_550.h5")

        self.target_network = QNetwork(self.obs_size, self.num_actions, kwargs.get('nhidden', 150), self.lr)
        self.target_network.set_weights(self.model_network.get_weights()) 

        # Mémoire pour replay
        self.memory = deque(maxlen=kwargs.get('mem_size', 1000000))
    
        self.step_counter = 0
    
    def act(self, state):    

        i = np.argmax(self.model_network.predict(state.reshape(1, state.shape[0]))[0])
                     
        return i

    def learn(a,b,c,d,e,f):
        return
    
    def printName(self):
        print('+ Agent: DQN Watch                  +')

    def printParameters(self):
        print('+ epsilon: ' + str(self.epsilon))
        print('+ obs_size: ' + str(self.obs_size))
        print('+ gamma: ' + str(self.gamma))
        print('+ batch_size: ' + str(self.batch_size))
        print('+ epoch_length: ' + str(self.epoch_length))
        print('+ learning_rate: ' + str(self.lr))
        print('+ tau: ' + str(self.tau))
        print('+ nHidden: ' + str(self.nhidden))

class DQNWatchExperiment(object):
    def run_qlearning(self, env, agent, max_number_of_episodes=100, interactive = False, display_frequency=1):

        episodes_completed = []
        episodes_reward = []
        episodes_mean = []
        # repeat for each episode
        for episode_number in range(max_number_of_episodes):
            
            # initialize state
            state = env.reset()
            
            done = False # used to indicate terminal state
            R = 0 # used to display accumulated rewards for an episode
            t = 0 # used to display accumulated steps for an episode i.e episode length
            
            # repeat for each step of episode, until state is terminal
            while not done:
                
                t += 1 # increase step counter - for display
                
                # choose action from state using policy derived from Q
                action = agent.act(state)
                
                # take action, observe reward and next state
                next_state, reward, done, _ = env.step(action)
                
                # agent learn (Q-Learning update)
                agent.learn(state, action, reward, next_state, done)
                
                # state <- next state
                state = next_state
                
                R += reward # accumulate reward - for display

                env.render()
            
            
            episodes_completed.append(episode_number)
            episodes_reward.append(R)
            episodes_mean.append(meanOfLast(episodes_completed, episodes_reward, 50))
            showProgress(agent, episodes_completed, episodes_reward, episodes_mean, 50)

        env.close()