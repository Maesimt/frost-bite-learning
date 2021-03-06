import gym
import numpy as np

from agents.sarsaAgent import SarsaAgent, SarsaExperiment
from agents.reinforce import REINFORCEAgent, ReinforceExperiment
from agents.dqn import DQNAgent, DQNExperiment
from agents.dqnWatch import DQNWatchAgent, DQNWatchExperiment
from agents.actorCritic import ActorCriticAgent, ActorCriticExperiment

def run(algo):
    env = gym.make('Frostbite-ram-v0')

    if algo == 'sarsa':
        agent = SarsaAgent(env.observation_space, env.action_space, epsilon=0.1, alpha=0.01, gamma=0.1)
        SarsaExperiment().run(agent, env, 100000)
    elif algo == 'reinforce':
        agent = REINFORCEAgent(
            observation_space=env.observation_space,
            actions_space=env.action_space,
            learning_rate = 0.001,
            gamma = 0.99,
            hidden1 = 128,
            hidden2 = 18,
            hidden3 = 18)
        ReinforceExperiment(env,agent, stop_criterion=10000, EPISODES=100000).run()
    elif algo == 'dqn':
        agent = DQNAgent(
            gym.spaces.Discrete(10),
            obs_size=env.observation_space,
            epsilon=1,
            epoch_length=50,
            nhidden=202,
            learning_rate=0.0002,
            gamma=0.945,
            tau=0.75
            )
        DQNExperiment().run_qlearning(env, agent, 100000, True)
    elif algo == 'watch dqn':
        agent = DQNWatchAgent(
            env.action_space,
            obs_size=env.observation_space,
            epsilon=0.01,
            epoch_length=100,
            nhidden=256,
            learning_rate=0.0001,
            gamma=0.9,
            tau=0.1
            )
        DQNWatchExperiment().run_qlearning(env, agent, 100000, True)
    elif algo == 'actorCritic':
        agent = ActorCriticAgent(
            observation_space=env.observation_space,
            actions_space=env.action_space,
            alpha = 0.0001,
            beta = 0.0001,
            gamma = 0.9,
            hidden1 = 18,
            hidden2 = 150,
        )
        # agent = ActorCriticAgent(
        #     observation_space=env.observation_space,
        #     actions_space=env.action_space,
        #     alpha = 0.0001,
        #     beta = 0.0001,
        #     gamma = 0.9995,
        #     hidden1 = 128,
        #     hidden2 = 72,
        # )
        ActorCriticExperiment(env, agent, EPISODES=100000).run_actorcritic()
run('dqn') 