import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

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
        
        state = state[np.newaxis, :]
        probabilities = self.predict.predict(state, batch_size=1)[0]
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
        actions[np.arange(len(actions_memory)), actions_memory] = 1

        discounted_rewards = self.discount_rewards(self.rewards_memory)
        discounted_rewards -= np.mean(discounted_rewards)
        discounted_rewards /= np.std(discounted_rewards)

        self.policy.train_on_batch([states_memory, discounted_rewards], actions)
        self.states_memory, self.actions_memory, self.rewards_memory = [], [], []

    def getName(self):
        return "Reinforce"