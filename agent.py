import tensorflow as tf


class Agent:
    def __init__(self, session, action_size, width, height, states_size,
                 optimizer=tf.train.AdamOptimizer(1e-4), eta=0.5, beta=0.01):
        self.layers = {}
        self.action_size = action_size
        self.optimizer = optimizer
        self.session = session

        self.width = width
        self.height = height
        self.states_size = states_size

        # beta is the entropy strength regularization term, a bigger entropy means higher emphasis on exploration
        self.beta = beta
        # eta regularizes the value to give more emphasis on the action taken, rather than the current states
        self.eta = eta

        with tf.device('/cpu:0'):
            with tf.variable_scope('network'):
                self.action = tf.placeholder('int32', [None], name='action')
                self.target_value = tf.placeholder('float32', [None], name='target_value')

                self.state, self.policy, self.value = self.build_model(self.width, self.height,
                                                                       self.states_size)

                self.weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='network')
                self.advantages = tf.placeholder('float32', [None], name='advantages')

            with tf.variable_scope('optimizer'):
                # Compute the one hot vectors for each action given.
                action_one_hot = tf.one_hot(self.action, self.action_size, 1.0, 0.0)

                # There are some issues when taking the log of the policy when it is exactly 1 or 0
                min_policy = 1e-8
                max_policy = 1-min_policy
                # log policy is the expected log probability of arriving in the current states
                self.log_policy = tf.log(tf.clip_by_value(self.policy, min_policy, max_policy))

                # log pi for action is the expected log probability of arriving in the current states given the
                # action taken
                self.log_pi_for_action = tf.reduce_sum(tf.multiply(self.log_policy, action_one_hot), axis=1)

                # We want to perform gradient ascent to maximize the discounted rewards, tf automatically tries to
                # reduce the loss, therefore we feed it the negative log policy multiplied by the estimate of the
                # advantage given by taking the current action in the current states
                self.policy_loss = -tf.reduce_mean(self.log_pi_for_action * self.advantages)

                # The value loss is just the squared deference between the current states's value and the desired value
                self.value_loss = tf.reduce_mean(tf.square(self.target_value - self.value))

                # The entropy improves exploration by discouraging premature convergence to suboptimal deterministic
                # policies, in other words, to penalize a small entropy ( which means that the probability distribution
                # is concentrated in one action ) we subtract the entropy from the loss
                self.entropy = tf.reduce_sum(tf.multiply(self.policy, -self.log_policy))

                # We try to minimize the loss such that the best actions are chosen more often
                self.loss = self.eta * self.value_loss + self.policy_loss - self.entropy * self.beta

                # Create a list of tuples of gradients and their respective weights
                grads = tf.gradients(self.loss, self.weights)
                # clip by global norm reduces the chances of gradients exploding
                grads, _ = tf.clip_by_global_norm(grads, 40.0)
                grads_vars = list(zip(grads, self.weights))

                # Create an operator to apply the gradients using the optimizer.
                self.train_op = optimizer.apply_gradients(grads_vars)

    def build_model(self, height, width, states_per_action):
        with tf.device('/gpu:0'):
            state = tf.placeholder('float32', shape=(None, height, width, states_per_action), name='states')
            self.layers['state'] = state

            # First convolutional layer
            with tf.variable_scope('conv1'):
                conv1 = tf.contrib.layers.convolution2d(inputs=state,
                                                        num_outputs=16, kernel_size=[8, 8], stride=[4, 4],
                                                        padding="VALID",
                                                        activation_fn=tf.nn.relu,
                                                        weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                                        biases_initializer=tf.zeros_initializer())
                self.layers['conv1'] = conv1

            # Second convolutional layer
            with tf.variable_scope('conv2'):
                conv2 = tf.contrib.layers.convolution2d(inputs=conv1, num_outputs=32,
                                                        kernel_size=[4, 4], stride=[2, 2], padding="VALID",
                                                        activation_fn=tf.nn.relu,
                                                        weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                                        biases_initializer=tf.zeros_initializer())
                self.layers['conv2'] = conv2

            # Flatten the network 
            with tf.variable_scope('flatten'):
                flatten = tf.contrib.layers.flatten(inputs=conv2)
                self.layers['flatten'] = flatten

            # Fully connected layer with 256 hidden units
            with tf.variable_scope('fc1'):
                fc1 = tf.contrib.layers.fully_connected(inputs=flatten, num_outputs=256,
                                                        activation_fn=tf.nn.relu,
                                                        weights_initializer=tf.contrib.layers.xavier_initializer(),
                                                        biases_initializer=tf.zeros_initializer())
                self.layers['fc1'] = fc1

            # The policy output
            with tf.variable_scope('policy'):
                policy = tf.contrib.layers.fully_connected(inputs=fc1,
                                                           num_outputs=self.action_size, activation_fn=tf.nn.softmax,
                                                           weights_initializer=tf.contrib.layers.xavier_initializer(),
                                                           biases_initializer=None)
                self.layers['policy'] = policy

            # The value output
            with tf.variable_scope('value'):
                value = tf.contrib.layers.fully_connected(inputs=fc1, num_outputs=1,
                                                          activation_fn=None,
                                                          weights_initializer=tf.contrib.layers.xavier_initializer(),
                                                          biases_initializer=None)
                self.layers['value'] = value

        return state, policy, value

    def get_value(self, state):
        return self.session.run(self.value, {self.state: state}).flatten()

    def get_policy_and_value(self, state):
        policy, value = self.session.run([self.policy, self.value], {self.state: state})
        return policy.flatten(), value.flatten()

    def train(self, states, actions, target_values, advantages):
        self.session.run(self.train_op, feed_dict={
            self.state: states,
            self.action: actions,
            self.target_value: target_values,
            self.advantages: advantages

        })
