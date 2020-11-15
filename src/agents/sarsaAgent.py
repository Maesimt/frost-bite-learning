import numpy as np
from agents.agent import Agent
from helpers import showProgress

class SarsaAgent(Agent):
    
    def __init__(self, observation_space, actions, epsilon=0.01, alpha=0.5, gamma=1):
        super(SarsaAgent, self).__init__(observation_space,actions)
        
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.Q = {}
        # {etat1: [0, -1, 55, 14]}
        
    def stateToString(self, state):
        # mystring = ""
        # if np.isscalar(state):
        #     mystring = str(state)
        # else:
        #     for digit in state:
        #         mystring += str(digit)
        # raise NotImplementedError
        return str(state)    
    
    def act(self, state):
        stateStr = self.stateToString(state)   
        
        # Si l'etat n'a jamais ete rencontrer 
        # Initialiser les tableaux pour noter les recompenses et le nb.fois 
        # que les actions ont ete realiser a partir de cette etat.
        if stateStr not in self.Q:
            self.Q[stateStr] = np.zeros(self.num_actions, dtype = np.longdouble)
        
        # Prendre une action aleatoire selon epsilon.
        if np.random.binomial(1, self.epsilon) == 1:
            return np.random.randint(0, self.num_actions)
        
        # Sinon choisir l'action gloutonne.
        ind = np.where(self.Q[stateStr] == np.max(self.Q[stateStr]))
        return np.random.choice(ind[0])

    def learn(self, state1, action1, reward, state2, action2):
        state1Str = self.stateToString(state1)
        state2Str = self.stateToString(state2)
        
        if state2Str not in self.Q:
            self.Q[state2Str][action2] = 0
        
        QSA = self.Q[state1Str][action1]
        QSAP = self.Q[state2Str][action2]
        
        td_target = reward + self.gamma * QSAP
        td_delta = td_target - QSA
        self.Q[state1Str][action1] = QSA + self.alpha * td_delta

    def printName(self):
        print('+ Agent: Sarsa                        +')

    def printParameters(self):
        print('+ epsilon: ' + str(self.epsilon))
        print('+ alpha: ' + str(self.alpha))
        print('+ gamma: ' + str(self.gamma))

class SarsaExperiment(object):
    def run(self, agent, env, episodes):
        episodes_completed = []
        episodes_reward = []

        # repeat for each episode
        for episode_number in range(episodes):
            
            # initialize state
            state = env.reset()
            done = False # used to indicate terminal state
            R = 0 # used to display accumulated rewards for an episode
            t = 0 # used to display accumulated steps for an episode i.e episode length
            
            # choose action from state using policy derived from Q
            action = agent.act(state)

            # repeat for each step of episode, until state is terminal
            while not done:

                t += 1 # increase step counter - for display
                
                # take action, observe reward and next state
                next_state, reward, done, _ = env.step(action)
                
                # choose next action from next state using policy derived from Q
                next_action = agent.act(next_state)
                
                # agent learn
                agent.learn(state, action, reward, next_state, next_action)
                
                # state <- next state, action <- next_action
                state = next_state
                action = next_action

                R += reward # accumulate reward - for display  
            
            episodes_completed.append(episode_number)
            episodes_reward.append(R)
            showProgress(agent, episodes_completed, episodes_reward, 50)
        
        env.close()