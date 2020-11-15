import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from agents.agent import Agent
from helpers import showProgress

class REINFORCEAgent(Agent):
    def __init__(self, observation_space, actions_space, learning_rate = 0.001, gamma = 0.99, hidden1=64, hidden2=64):
        super(REINFORCEAgent, self).__init__(observation_space, actions_space)

        # Hyperparamètres du policy gradient
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.hidden1, self.hidden2 = hidden1, hidden2

        # Création du modèle de la politique
        self.policy, self.predict = self.policy_network()

        # Mémoire de la trajectoire
        self.states_memory, self.actions_memory, self.rewards_memory = [], [], []
        
        self.render = False

    def policy_network(self):
        """
        La politique est modélisée par une réseau de neurones
        Entrée: état
        Sortie: probabilité de chaque action
        """
        inpt = Input(shape=(self.state_size,))
        advantages = Input(shape=[1])
        dense1 = Dense(self.hidden1, activation='relu')(inpt)
        dense2 = Dense(self.hidden2, activation='relu')(dense1)
        probs = Dense(self.num_actions, activation='softmax')(dense2)

        def custom_loss(y_true, y_pred):
            
            out = tf.keras.backend.clip(y_pred, 1e-8, 1-1e-8)
            log_likelihood = y_true*tf.keras.backend.log(out)

            return tf.keras.backend.sum(-log_likelihood * advantages)
        
        policy = Model(inputs=[inpt, advantages], outputs=[probs])
        policy.compile(optimizer=Adam(lr=self.learning_rate), loss=custom_loss)

        predict = Model(inputs=[inpt], outputs=[probs])

        return policy, predict
    

    def act(self, state):
        """
        Sélection d'une action suivant la sortie du réseau de neurones
        """
        
        # Compléter le code ci-dessous ~ 2 lignes
        
        state = state[np.newaxis, :]
        probabilities = self.predict.predict(state, batch_size=1)[0]
        print('Swag -> probabilities', probabilities)
        print('Swag -> np.random.choice', np.random.choice(self.num_actions, 1, p=probabilities))
        return np.random.choice(self.num_actions, 1, p=probabilities)[0]

    
    def discount_rewards(self, rewards):
        """
        La politique est évaluée à partir des gains dévalués
        """
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, len(rewards))):
            running_add = running_add * self.gamma + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards

    
    def remember(self, state, action, reward):
        """
        Sauvegarde <s, a ,r> pour chaque instant
        """
        
        # Compléter le code ci-dessous ~ 3 lignes
        
        self.states_memory.append(state)
        self.rewards_memory.append(reward)
        self.actions_memory.append(action)


    def learn(self):
        """
        Mise à jour du "policy network" à chaque épisode
        """
        states_memory = np.array(self.states_memory)
        actions_memory = np.array(self.actions_memory)
        rewards_memory = np.array(self.rewards_memory)

        actions = np.zeros([len(actions_memory), self.num_actions])
        print('actions_memory', actions_memory)
        print(' self.num_actions',  self.num_actions)
        actions[np.arange(len(actions_memory)), actions_memory] = 1

        discounted_rewards = self.discount_rewards(self.rewards_memory)
        discounted_rewards -= np.mean(discounted_rewards)
        discounted_rewards /= np.std(discounted_rewards)

        self.policy.train_on_batch([states_memory, discounted_rewards], actions)
        self.states_memory, self.actions_memory, self.rewards_memory = [], [], []

    def printName(self):
        print('+ Agent: Reinforce                    +')

    def printParameters(self):
        print('+ learning_rate: ' + str(self.learning_rate))
        print('+ gamma: ' + str(self.gamma))
        print('+ hidden1: ' + str(self.hidden1))
        print('+ hidden2: ' + str(self.hidden2))

class ReinforceExperiment(object):
    def __init__(self, env, agent, EPISODES=1000, training=True, episode_max_length=None, mean_episodes=10, stop_criterion=100):
        self.env = env
        self.agent = agent
        self.EPISODES = EPISODES
        self.training = training
        self.episode_max_length = episode_max_length
        self.mean_episodes = mean_episodes
        self.stop_criterion = stop_criterion

    def run(self):
        
        # Tableaux utiles pour l'affichage
        scores, mean, episodes = [], [], []
    
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

                # Sauvegarde la transition <s, a, r> dans la mémoire
                self.agent.remember(state, action, reward)

                # Mise à jour de l'état
                state = next_state

                # Accumulation des récompenses
                score += reward

                # Arrête l'épisode après 'episode_max_length' instants
                if self.episode_max_length != None and counter > self.episode_max_length:
                    done = True

            # Lance l'apprentissage de la politique
            if self.training == True:
                self.agent.learn()

            # Arrête l'entraînement lorsque la moyenne des récompense sur 'mean_episodes' épisodes est supérieure à 
            if np.mean(scores[-self.mean_episodes:]) > self.stop_criterion:
                break

            # Sauvegarde du modèle (poids) tous les 50 épisodes
            if self.training and i % 50 == 0:
                self.agent.predict.save(f"./{self.env.spec.id}_reinforce.h5")    
            
            # Affichage des récompenses obtenues
            if self.training == True:
                scores.append(score)
                episodes.append(i)
                showProgress(self.agent, episodes, scores, 50)