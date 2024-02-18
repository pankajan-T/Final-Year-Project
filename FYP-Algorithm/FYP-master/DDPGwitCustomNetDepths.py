import os

import numpy as np
import tensorflow as tf

import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomUniform
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LayerNormalization
from tensorflow.keras.losses import MSE

from collections.abc import Iterable

# =============================== REPLAY BUFFER ===============================
class ReplayBuffer:
    def __init__(self, max_size, state_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0

        self.state_memory     = np.zeros((self.mem_size, *state_shape))
        self.action_memory    = np.zeros((self.mem_size, n_actions))
        self.reward_memory    = np.zeros(self.mem_size)
        self.new_state_memory = np.zeros((self.mem_size, *state_shape))
        self.terminal_memory  = np.zeros(self.mem_size, dtype=np.bool_) # using np.bool is really useful when pytorch is used.

    def store_transition(self, state, action, reward, new_state, done):
        index = self.mem_cntr % self.mem_size # implement a queue

        self.state_memory[index]     = state
        self.action_memory[index]    = action
        self.reward_memory[index]    = reward
        self.new_state_memory[index] = new_state
        self.terminal_memory[index]  = done # problematic !!!

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, batch_size, replace=False) # replace = False -> in a single batch, no element gets sampled more than once. 

        states     = self.state_memory[batch]
        actions    = self.action_memory[batch]
        rewards    = self.reward_memory[batch]
        new_states = self.new_state_memory[batch]
        dones      = self.terminal_memory[batch]

        return states, actions, rewards, new_states, dones



# =============================== CRITIC NETWORK ===============================
class CriticNetwork(keras.Model):
    def __init__(
            self,
            name, # model name (required by tf.keras.Model)
            network_dims, # an array consisting of three sub-integer-arrays specifying the number and the sizes of the hidden fully-connected layers of the critic network
            chkpt_dir='tmp/ddpg/'
    ):
        super(CriticNetwork, self).__init__()

        # verifying the format of critic network dims
        if not isinstance(network_dims, Iterable) or len(network_dims) != 3:
            raise Exception(f"network dims of the critic network must be an array of 3 sub arrays of integers: given {network_dims}")
        for i in range(3):
            if not isinstance(network_dims[i], Iterable):
                raise Exception(f"network dims of the critic network must consist of sub arrays of integers: found {network_dims[i]} at index {i}")

        self.model_name = name # do not use 'self.model'; it is a reserved variable name in tf
        self.no_of_action_layers = len(network_dims[0])
        self.no_of_state_layers  = len(network_dims[1])
        self.no_of_common_layers = len(network_dims[2])
        self.checkpoint_dir  = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, self.model_name+'_ddpg.h5') 
        # extensions for saving keras models: legacy '.h5' -> TF 1.X, '.tf' -> TF 2.X

        # ------------------------------------------- action layers ------------------------------------------
        self.action_layers = []
        for i in range(self.no_of_action_layers):
            weight_initializer = RandomUniform(minval=-1/np.sqrt(network_dims[0][i]), maxval=1/np.sqrt(network_dims[0][i]))
            layer = Dense(
                units = network_dims[0][i],
                activation = 'relu',
                kernel_initializer = weight_initializer,
                bias_initializer   = weight_initializer,
                name = f"critic_hidden_action_{i+1}"
            )
            self.action_layers.append(layer)

        # ------------------------------------------- state layers -------------------------------------------
        self.state_layers = []
        for i in range(self.no_of_state_layers):
            weight_initializer = RandomUniform(minval=-1/np.sqrt(network_dims[1][i]), maxval=1/np.sqrt(network_dims[1][i]))
            layer = Dense(
                units = network_dims[1][i],
                activation = 'relu',
                kernel_initializer = weight_initializer,
                bias_initializer   = weight_initializer,
                name = f"critic_hidden_state_{i+1}"
            )
            self.state_layers.append(layer)

        # ------------------------------------------- common layers ------------------------------------------
        self.common_layers = []
        for i in range(self.no_of_common_layers):
            weight_initializer = RandomUniform(minval=-1/np.sqrt(network_dims[2][i]), maxval=1/np.sqrt(network_dims[2][i]))
            layer = Dense(
                units = network_dims[2][i],
                activation = 'relu',
                kernel_initializer = weight_initializer,
                bias_initializer   = weight_initializer,
                name = f"critic_hidden_common_{i+1}"
            )
            self.common_layers.append(layer)

        # ------------------------------------------- final Q layer -------------------------------------------
        final_layer_initializer = RandomUniform(minval=-3*10**-4, maxval=3*10**-4)
        self.q = Dense(
            units = 1,
            activation = None,
            kernel_initializer = final_layer_initializer,
            bias_initializer   = final_layer_initializer,
            name = "q_value"
        )

        # ------------------------------------- layer normalization layers -------------------------------------
        # here, we perform layer normalization at two places: for the input states and at the concatenation of action and state feature vectors
        self.state_normalizer = LayerNormalization(epsilon=10**-4, center=False, scale=False)
        self.combining_normalizer = LayerNormalization(epsilon=10**-4, center=False, scale=False)

    def call(self, state, action):

        # obtain the action features
        # NOTE:- actions are not typically required to be normalized since action is already bounded between [-1, +1]. 
        #        in a case where action is bounded in the range [0, 1], normalizing may even corrupt the information of the action vector.
        action_features = action 
        for i in range(self.no_of_action_layers):
            action_features = self.action_layers[i](action_features)

        # normalize the state input
        state_features = self.state_normalizer(state)

        # obtain the state features
        for i in range(self.no_of_state_layers):
            state_features = self.state_layers[i](state_features)

        # combine the two feature vectors
        state_action_features = tf.concat([action_features, state_features], axis=1)
        # normalize the combined features
        combined_features = self.combining_normalizer(state_action_features)

        # process the combined features
        for i in range(self.no_of_common_layers):
            combined_features = self.common_layers[i](combined_features)

        # get the critic estimation
        q_value = self.q(combined_features)

        return q_value

# ================================ ACTOR NETWORK ===============================
class ActorNetwork(keras.Model):
    def __init__(
            self,
            name, # model name (required by tf.keras.Model)
            n_actions, # action shape (dimenisonality of action space)
            action_activation, # activation function for action ('tanh', 'sigmoid)
            network_dims, # a list of integers representing the sizes of the fully-connected layers of the actor network
            chkpt_dir='tmp/ddpg/'
    ):
        super(ActorNetwork, self).__init__()

        self.model_name = name # do not use 'self.model'; it is a reserved variable name in tf
        self.no_of_layers = len(network_dims)
        self.checkpoint_dir  = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, self.model_name+'_ddpg.h5') 

        # verifying the format of actor network dims
        if not isinstance(network_dims, Iterable) or len(network_dims) == 0:
            raise Exception(f"network dims of the actor network must be a non-empty array of integers: given {network_dims}")
        
        # ------------------------------- hidden dense (fully-connected) layers ------------------------------
        self.hidden_layers = []
        for i in range(self.no_of_layers):
            weight_initializer = RandomUniform(minval=-1/np.sqrt(network_dims[i]), maxval=1/np.sqrt(network_dims[i]))
            layer = Dense(
                units = network_dims[i],
                activation = 'relu',
                kernel_initializer = weight_initializer,
                bias_initializer   = weight_initializer,
                name = f"actor_hidden_{i+1}"
            )
            self.hidden_layers.append(layer)

        # ---------------------------------------- final action layer ----------------------------------------
        final_layer_initializer = RandomUniform(minval=-3*10**-4, maxval=3*10**-4)
        self.mu = Dense(
            units = n_actions,
            activation = action_activation, # limit the action in the range [-1, 1] -> 'tanh' or [0, 1] -> 'sigmoid'
            kernel_initializer = final_layer_initializer,
            bias_initializer   = final_layer_initializer,
            name = 'action'
        )

        # ------------------------------------ layer normalization layers ------------------------------------
        # here, we only apply layer normalization on the input state.
        self.state_normalizer = LayerNormalization(epsilon=10**-4, center=False, scale=False)

    def call(self, state):

        # normalize the state input
        state_features = self.state_normalizer(state)

        # process the state features
        for i in range(self.no_of_layers):
            state_features = self.hidden_layers[i](state_features)

        # get the action prediction
        action = self.mu(state_features)

        return action



# ================================== DDPG AGENT =================================
class DDPGAgentwithCustomNetworkDepths:
    def __init__(
            self,
            input_dims, # state shape
            n_actions,  # dimensionality of actions
            # env,        # gymnasium env
            alpha,      # learning rate of actor
            beta,       # learning rate of critic
            gamma,      # discounting factor
            tau,        # soft target update factor
            critic_dims,
            actor_dims,
            batch_size,
            buffer_size,
            noise,
            action_activation # activation function for the action ('tanh', 'sigmoid')
    ):
        """
        Creates a DDPG agent.

        :param critic_dims: an array consisting of three sub-integer-arrays specifying the number and the sizes of the hidden fully-connected layers of the critic network. \
        the first array defines the layers that only the input action passes through, and the second array specifies the layers that only the state passes through. \
        the final array determines the final fully-connected layers (excluding the final ``q`` layer) that the concatenation of the action and state representation vectors propagates. 
        :param actor_dims: a list of integers representing the number and the sizes of the hidden fully-connected layers of the actor network
        :param noise: the standard deviation of the exploration noise that gets added to the action vector predicted by the actor network while training. \
        accepts a float value or a list with the format ``[reg_value, (index, noise_std_dev), ...]``. \
        here, the ``reg_value`` is taken as the standard deviation for the noise for all the action elements except the ones specified by the indices, ``index``, in the 2-element tuples given in the list. \
        for the action elements given by those indices, the second element of those tuples is taken as the standard deviation of the exploration noise. 
        :param action_activation: the activation function for the last layer of the actor network; can take the values 'tanh' or 'sigmoid'.
        """

        # set the class attributes
        self.tau = tau
        self.n_actions = n_actions
        self.noise = noise
        self.batch_size = batch_size
        self.gamma = gamma
        self.action_activation = action_activation

        # instantiate replay buffer
        self.memory = ReplayBuffer(buffer_size, state_shape=input_dims, n_actions=n_actions)

        # instantiate the networks
        self.actor  = ActorNetwork(name="actor", n_actions=n_actions, network_dims=actor_dims, action_activation=self.action_activation)
        self.critic = CriticNetwork(name="critic", network_dims=critic_dims)
        self.target_actor  = ActorNetwork(name="target_actor", n_actions=n_actions, network_dims=actor_dims, action_activation=self.action_activation)
        self.target_critic = CriticNetwork(name="target_critic", network_dims=critic_dims)

        # compile networks
        self.actor.compile(optimizer=Adam(learning_rate=alpha))
        self.critic.compile(optimizer=Adam(learning_rate=beta))
        # target networks do not require an optimizer or a learning rate since they are learned through soft updates.
        # but, to use the networks in TF2, we have to compile them with an optimizer and a learning rate. 
        self.target_actor.compile(optimizer=Adam(learning_rate=alpha))
        self.target_critic.compile(optimizer=Adam(learning_rate=beta))

        # load identical weights to target networks
        self.update_target_network_parameters(tau=1)

    def update_target_network_parameters(self, tau=None):
        if tau == None:
            tau = self.tau

        target_actor_weights = self.target_actor.weights
        for i, actor_weights in enumerate(self.actor.weights):
            target_actor_weights[i] = tau * actor_weights + (1-tau) * target_actor_weights[i]
        self.target_actor.set_weights(target_actor_weights)

        target_critic_weights = self.target_critic.weights
        for i, critic_weights in enumerate(self.critic.weights):
            target_critic_weights[i] = tau * critic_weights + (1-tau) * target_critic_weights[i]
        self.target_critic.set_weights(target_critic_weights)

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def save_models(self, checkpoint_file_name, checkpoint_dir):
        print("..... saving models .....")
        if not os.path.exists(checkpoint_dir):
            raise Exception(f"the provided checkpoint directory doesn't exist: given '{checkpoint_dir}'")
        self.actor.save_weights(checkpoint_dir + checkpoint_file_name + '_actor_ddpg.h5')
        self.critic.save_weights(checkpoint_dir + checkpoint_file_name + '_critic_ddpg.h5')
        self.target_actor.save_weights(checkpoint_dir + checkpoint_file_name + '_target_actor_ddpg.h5')
        self.target_critic.save_weights(checkpoint_dir + checkpoint_file_name + '_target_critic_ddpg.h5')

    def load_models(self, checkpoint_file_name, checkpoint_dir):
        print("..... loading models .....")
        if not os.path.exists(checkpoint_dir):
            raise Exception(f"the provided checkpoint directory doesn't exist: given '{checkpoint_dir}'")
        self.actor.load_weights(checkpoint_dir + checkpoint_file_name + '_actor_ddpg.h5')
        self.critic.load_weights(checkpoint_dir + checkpoint_file_name + '_critic_ddpg.h5')
        self.target_actor.load_weights(checkpoint_dir + checkpoint_file_name + '_target_actor_ddpg.h5')
        self.target_critic.load_weights(checkpoint_dir + checkpoint_file_name + '_target_critic_ddpg.h5')

    def choose_action(self, observation, noise_schedule=False, step_size=1_000, factor=1/2, evaluate=False):
        state = tf.convert_to_tensor([observation], dtype=tf.float32) # introducing the batch dimension
        action = self.actor(state) # 'action' would also have a batch dimension 

        if not evaluate:
            # while training the agent, introduce an exploration noise
            # here, the exploration noise is sampled from a normal distribution with zero mean and specified std deviation. 

            # process the noise element
            if type(self.noise) == list:
                noise = np.ones([self.n_actions]) * self.noise[0] # reg_value
                for (i, std_dev) in self.noise[1: ]: # (index, noise_std_dev)
                    noise[i] = std_dev
                self.noise = tf.convert_to_tensor(noise.astype(np.float32))

            if noise_schedule and self.memory.mem_cntr % step_size == 0 and self.memory.mem_cntr != 0:
                self.noise = self.noise * factor
                print(f"step: {self.memory.mem_cntr} - noise was set to {self.noise}")

            action += tf.random.normal(shape=[self.n_actions], mean=0.0, stddev=self.noise)
            # when added the noise, the action can go beyond the action space limits; so, clip the actions.
            if self.action_activation == 'tanh':
                clip_min = -1
            elif self.action_activation == 'sigmoid':
                clip_min = 0
            action = tf.clip_by_value(action, clip_value_max=1.0, clip_value_min=clip_min)

        return action[0] # get rid of the batch dimension
    
    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        
        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)

        states     = tf.convert_to_tensor(state, dtype=tf.float32)
        actions    = tf.convert_to_tensor(action, dtype=tf.float32)
        rewards    = tf.convert_to_tensor(reward, dtype=tf.float32)
        new_states = tf.convert_to_tensor(new_state, dtype=tf.float32)

        # update the critic
        with tf.GradientTape() as tape:
            target_actions = self.target_actor(new_states)
            next_step_critic_values = tf.squeeze(self.target_critic(new_states, target_actions), axis=1)
            critic_values = tf.squeeze(self.critic(states, actions), axis=1)
            targets = rewards + self.gamma * next_step_critic_values * (1-done) # y_i
            critic_loss = MSE(targets, critic_values)
        
        critic_network_gradients = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic.optimizer.apply_gradients(zip(critic_network_gradients, self.critic.trainable_variables))

        # update the actor
        with tf.GradientTape() as tape:
            new_policy_actions = self.actor(states)
            critic_values_ = -self.critic(states, new_policy_actions) # why (-) ? gradient ascent
            actor_loss = tf.math.reduce_mean(critic_values_)

        actor_network_gradients = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(actor_network_gradients, self.actor.trainable_variables))

        # soft target updates
        self.update_target_network_parameters()
