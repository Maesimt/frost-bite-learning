import gym
import time
import numpy as np

from agents.sarsaAgent import SarsaAgent

env = gym.make('Frostbite-ram-v0')

actions = env.action_space

agentSmith = SarsaAgent(range(actions.n))

# repeat for each episode
for episode_number in range(1000000):
    
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

        env.render()

        if episode_number > 300000:
            time.sleep(0.1)
    
    print('------------------------------')
    print('Episode ' + str(episode_number))
    print('Episode length = ' + str(t))
    print('Episode reward = ' + str(R))


# if not interactive display, show graph at the end
# if not interactive:
self.fig.clf()
stats = plotting.EpisodeStats(
    episode_lengths=self.episode_length,
    episode_rewards=self.episode_reward,
    episode_running_variance=np.zeros(max_number_of_episodes))
plotting.plot_episode_stats(stats, display_frequency)

env.close()