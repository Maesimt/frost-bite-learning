import gym

from agents.sarsaAgent import SarsaAgent, SarsaExperiment
from agents.reinforce import REINFORCEAgent, ReinforceExperiment

def run(algo):
    env = gym.make('Frostbite-ram-v0')

    if algo == 'sarsa':
        agent = SarsaAgent(env.observation_space, env.action_space, epsilon=0.1, alpha=0.01, gamma=0.1)
        SarsaExperiment().run(agent, env, 1000)
    elif algo == 'reinforce':
        agent = REINFORCEAgent(
            observation_space=env.observation_space,
            actions_space=env.action_space,
            learning_rate = 0.001,
            gamma = 0.99,
            hidden1 = 36,
            hidden2 = 36)
        ReinforceExperiment(env,agent, stop_criterion=1000).run()

run('reinforce')