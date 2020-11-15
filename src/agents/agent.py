class Agent(object):  
        
    def __init__(self,observation_space, action_space):
        self.observation_space = observation_space
        self.state_size = observation_space.shape[0]
        self.action_space = action_space
        self.num_actions = action_space.n

    def act(self, state):
        raise NotImplementedError

    def printName():
        raise NotImplementedError

    def printParameters():
        raise NotImplementedError

    def act(self, state):
        raise NotImplementedError