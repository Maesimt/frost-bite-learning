import gym
import tensorflow as tf
from tensorflow.keras.layers import Dense, Activation, Input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
import numpy as np

from helpers import showProgress, meanOfLast

class Agent(object):  
        
    def __init__(self, observation_space, action_space):
        self.observation_space = observation_space
        self.state_size = observation_space.shape[0]
        self.action_space = action_space
        self.num_actions = action_space.n

    def act(self, state):
        raise NotImplementedError

class ActorCriticAgent(Agent):
    def __init__(self, observation_space, actions_space, alpha, beta, gamma=0.99,
                 hidden1=1024, hidden2=512):
        super(ActorCriticAgent, self).__init__(observation_space, actions_space)
        
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.hidden1 = hidden1
        self.hidden2 = hidden2

        self.actor, self.critic, self.policy = self.build_actor_critic_network()
        
        self.render = False

    def build_actor_critic_network(self):
        inpt = Input(shape=(self.state_size,))
        delta = Input(shape=[1])
        dense1 = Dense(self.hidden1, activation='relu')(inpt)
        dense2 = Dense(self.hidden2, activation='relu')(dense1)
        probs = Dense(self.num_actions, activation='softmax')(dense2)
        values = Dense(1, activation='linear')(dense2)

        def custom_loss(y_true, y_pred):
            out = tf.keras.backend.clip(y_pred, 1e-8, 1-1e-8)
            log_likelihood = y_true*tf.keras.backend.log(out)

            return tf.keras.backend.sum(-log_likelihood*delta)

        actor = Model(inputs=[inpt, delta], outputs=[probs])
        actor.compile(optimizer=Adam(lr=self.alpha), loss=custom_loss)

        critic = Model(inputs=[inpt], outputs=[values])
        critic.compile(optimizer=Adam(lr=self.beta), loss='mean_squared_error')
        
        policy = Model(inputs=[inpt], outputs=[probs])

        return actor, critic, policy

    def act(self, observation):
        state = observation[np.newaxis, :]
        probabilities = self.policy.predict(state)[0]
        action = np.random.choice(self.num_actions, p=probabilities)

        return action

    def learn(self, state, action, reward, state_, done):

        # 1. Verifier shape de state et state_ et ajuster si besoin
        state  =  state[np.newaxis, :]
        state_ = state_[np.newaxis, :] 

        # 2. Effectuer predictions sur state et state_. Utilise le critic pour faire les predictions
        critic_value = self.critic.predict(state_)
        critic = self.critic.predict(state)

        # 3. Calculer le target: reward _ gamma * critic_value_ * (1-done)
        target = reward  + self.gamma * critic_value * (1 -int(done))

        # 4. Calculer le delta (ou avantage): target - critic_value
        delta = target - critic_value

        # 5. Encoder les actions sous forme de label
        actions = np.zeros([1, self.num_actions])
        actions[np.arange(1), action] = 1

        # 6. Entrainer l'actor
        self.actor.fit([state, delta], actions, verbose=0)

        # 7. Entrainer le critic
        self.critic.fit(state, target, verbose=0)

    def printName(self):
        print('+ Agent: Actor Critic                    +')

    def printParameters(self):
        print('+ gamma: ' + str(self.gamma))
        print('+ alpha: ' + str(self.alpha))
        print('+ beta: ' + str(self.beta))
        print('+ hidden1: ' + str(self.hidden1))
        print('+ hidden2: ' + str(self.hidden2))

class ActorCriticExperiment(object):
    def __init__(self, env, agent, EPISODES=1000, training=True, episode_max_length=None, mean_episodes=10, stop_criterion=100):
        self.env = env
        self.agent = agent
        self.EPISODES = EPISODES
        self.training = training
        self.episode_max_length = episode_max_length
        self.mean_episodes = mean_episodes
        self.stop_criterion = stop_criterion

    def run_actorcritic(self):
        
        # Tableaux utiles pour l'affichage
        scores, mean, episodes = [], [], []
        episodes_completed = []
        episodes_reward = []
        episodes_mean = []

        for i in range(self.EPISODES):
            done = False
            score = 0
            state = self.env.reset()

            counter = 0
            while not done:
                counter +=1

                # Afficher l'environnement
                if self.agent.render:
                    self.env.render()

                # Obtient l'action pour l'état courant
                action = self.agent.act(state)

                # Effectue l'action
                next_state, reward, done, _ = self.env.step(action)

                self.agent.learn(state, action, reward, next_state, done )

                # Mise à jour de l'état
                state = next_state

                # Accumulation des récompenses
                score += reward

                # Arrête l'épisode après 'episode_max_length' instants
                if self.episode_max_length != None and counter > self.episode_max_length:
                    done = True

            # Arrête l'entraînement lorsque la moyenne des récompense sur 'mean_episodes' épisodes est supérieure à 
            if np.mean(scores[-self.mean_episodes:]) > self.stop_criterion:
                break

            # Sauvegarde du modèle (poids) tous les 50 épisodes
            if self.training and i % 50 == 0:
                self.agent.actor.save(f"./{self.env.spec.id}_actor.h5")    
                self.agent.critic.save(f"./{self.env.spec.id}_critic.h5")    
            
            # Affichage des récompenses obtenues
            if self.training == True:
                episodes_completed.append(i)
                episodes_reward.append(score)
                episodes_mean.append(meanOfLast(episodes_completed, episodes_reward, 50))
                showProgress(agent, episodes_completed, episodes_reward, episodes_mean, 50)


