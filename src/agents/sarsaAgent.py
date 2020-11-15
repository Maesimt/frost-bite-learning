import numpy as np
from agents.agent import Agent

class SarsaAgent(Agent):
    
    def __init__(self, actions, epsilon=0.01, alpha=0.5, gamma=1):
        super(SarsaAgent, self).__init__(actions)
        
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

