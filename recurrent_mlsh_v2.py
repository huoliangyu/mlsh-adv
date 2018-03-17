import tensorflow.contrib.rnn as rnn

from pg import *


class RecurrentMLSHV2(PolicyGradient):
    def policy_network(self, mlp_input, output_size, scope,
                       size=config.baseline_layer_size,
                       n_layers=config.n_layers, output_activation=None):

        if str(config.env_name).startswith("Fourrooms"):
            self.state_embedding = tf.tile(
                tf.one_hot(indices=tf.cast(mlp_input, dtype=tf.int32),
                           depth=self.env.nS), [1, config.num_sub_policies, 1])
            num_actions = self.env.action_space.n

        else:
            self.state_embedding = tf.tile(tf.expand_dims(mlp_input, axis=1),
                                           [1, config.num_sub_policies, 1])
            num_actions = self.env.action_space.shape[0]

        rnn_cell = rnn.BasicRNNCell(num_units=num_actions)

        self.sub_policies, states = tf.nn.dynamic_rnn(cell=rnn_cell,
                                                      inputs=self.state_embedding,
                                                      dtype=tf.float32,
                                                      scope='subpolicy')

        lstm_cell = rnn.BasicLSTMCell(num_units=config.num_sub_policies)

        if config.freeze_sub_policy:
            self.proposed_sub_policies = tf.stop_gradient(self.sub_policies,
                                                          'stop')
        else:
            self.proposed_sub_policies = self.sub_policies

        concatenated = tf.concat(
            [self.proposed_sub_policies, self.state_embedding], axis=2)

        self.out, _ = tf.nn.dynamic_rnn(cell=lstm_cell, inputs=concatenated,
                                        dtype=tf.float32, scope='master')
        last_output = self.out[:, -1, :]

        self.chosen_index = tf.argmax(last_output, axis=1)
        max_output = tf.reduce_max(last_output, axis=1, keep_dims=True)
        tmp = tf.nn.relu(last_output - max_output + 1e-17)
        self.weights = tmp / tf.reduce_sum(tmp, axis=1, keep_dims=True)

        final_policy = tf.reduce_sum(
            tf.expand_dims(self.weights, axis=2) * self.proposed_sub_policies,
            axis=1)

        if config.sub_policy_index > -1:
            final_policy = self.proposed_sub_policies[:,
                           config.sub_policy_index, :]

        return final_policy


if __name__ == "__main__":
    if not os.path.exists('./plots'):
        os.makedirs('./plots')

    env = gym.make(config.env_name)
    config = config('RecurrentMLSH-v2')
    model = RecurrentMLSHV2(env, config)
    model.run()
