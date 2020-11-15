class Agent(object):  
        
    def __init__(self, actions):
        self.actions = actions
        self.num_actions = len(actions)

    def printName():
        raise NotImplementedError

    def printParameters():
        raise NotImplementedError

    def act(self, state):
        raise NotImplementedError