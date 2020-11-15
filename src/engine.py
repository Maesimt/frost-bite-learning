import gym
import time
import numpy as np
import termplotlib as tpl
import os

from agents.sarsaAgent import SarsaAgent
from agents.reinforce import REINFORCEAgent

def showProgress(agent, x, y, meanOfN):
    os.system('clear')
    print('+-------------------------------------+')
    agent.printName()
    print('+-------------------------------------+')
    agent.printParameters()
    print('+-------------------------------------+')
    print('+ Episode ' + str(len(x)) + '              score: '+ str(y[len(y)-1]))
    divider = meanOfN if meanOfN < len(x) else len(x)
    print('+ Mean of last ' + str(meanOfN) + ' = ' + str(np.sum(y[-divider:]) / divider) + '   Highest Score: ' + str(np.max(y)))
    print('+-------------------------------------+')
    fig = tpl.figure()
    fig.plot(episodes_completed, episodes_reward, width=100, height=30)
    fig.show()
    
def run():
    env = gym.make('Frostbite-ram-v0')

    actions = env.action_space

    # agents
    # agent = SarsaAgent(range(actions.n))
    #agentSmith = SarsaAgent(env.observation_space,action_space, epsilon=0.1, alpha=0.01, gamma=0.1)
    agentSmith = REINFORCEAgent(env.observation_space, env.action_space,
        learning_rate = 0.001,
        gamma = 0.99,
        hidden1 = 32,
        hidden2 = 32)

    episodes_completed = []
    episodes_reward = []

    # repeat for each episode
    for episode_number in range(1000):
        
        # initialize state
        state = env.reset()

        done = False # used to indicate terminal state
        R = 0 # used to display accumulated rewards for an episode
        t = 0 # used to display accumulated steps for an episode i.e episode length
        
        # choose action from state using policy derived from Q
        action = agentSmith.act(state)

        # repeat for each step of episode, until state is terminal
        while not done:

            t += 1 # increase step counter - for display
            
            # take action, observe reward and next state
            next_state, reward, done, _ = env.step(action)
            
            # choose next action from next state using policy derived from Q
            next_action = agentSmith.act(next_state)
            
            # agent learn
            agentSmith.learn(state, action, reward, next_state, next_action)
            
            # state <- next state, action <- next_action
            state = next_state
            action = next_action

            R += reward # accumulate reward - for display  
        
        episodes_completed.append(episode_number)
        episodes_reward.append(R)
        showProgress(agentSmith, episodes_completed, episodes_reward, 50)
        #env.render()

    env.close()

