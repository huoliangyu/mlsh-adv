import gym
import tensorflow as tf
import tensorflow.contrib.rnn as rnn

import baselines.common.tf_util as U
from baselines.common.distributions import make_pdtype
from baselines.common.mpi_running_mean_std import RunningMeanStd
from config import config


class RecurrentMLSHV9(object):
    recurrent = True

    def __init__(self, name, *args, **kwargs):
        print('initializing')
        if 'gen' in name:
            self.name = name[:-4]
            self.batch_size = 1
        else:
            self.name = name
            self.batch_size = config.batch_size

        with tf.variable_scope(self.name):
            self._init(*args, **kwargs)
            self.scope = tf.get_variable_scope().name

    def single_cell(self, num_units, cell_type, name):
        if cell_type == 'RNN':
            rnn_cell = rnn.BasicRNNCell(num_units=num_units, name=name)
            return rnn_cell
        elif cell_type == 'LSTM':
            lstm_cell = rnn.BasicLSTMCell(num_units=num_units, name=name)
            return lstm_cell
        elif cell_type == 'GRU':
            gru_cell = rnn.GRUCell(num_units=num_units, name=name)
            return gru_cell
        else:
            raise Exception()

    def dynamic_rnn(self, cell, inputs, batch_size):
        def rnn_forward(out, hidden, alive, count_length):
            new_out, new_hidden = cell(inputs, hidden)
            new_out = tf.nn.batch_normalization(new_out, 0, 1, offset=None,
                                                scale=None,
                                                variance_epsilon=1e-14)
            new_out = tf.expand_dims(new_out, axis=1)
            new_out = tf.concat([out, new_out], axis=1)
            alive_now = tf.cast(out[:, -1, -1] > 0, tf.int32)
            alive *= alive_now
            return [new_out, new_hidden, alive, count_length + alive_now]

        def stop(out, hidden, alive, count_length):
            return tf.reduce_sum(alive) == 0

        hidden = cell.zero_state(batch_size, tf.float32)
        out, hidden = cell(inputs, hidden)
        out = tf.nn.batch_normalization(out, 0, 1, offset=None, scale=None,
                                        variance_epsilon=1e-14)
        out = tf.expand_dims(out, axis=1)
        count_length = tf.constant(1, shape=[batch_size])

        alive = tf.constant(1, shape=[batch_size])
        condition = stop

        if config.sub_policy_network == 'LSTM':
            raw_shape = list(cell.state_size)
            for i in range(len(raw_shape)):
                raw_shape[i] = rnn.LSTMStateTuple(hidden[i][0].get_shape(),
                                                  hidden[i][1].get_shape())
        elif config.sub_policy_network == 'GRU':
            raw_shape = tuple([hid.get_shape() for hid in hidden])

        return tf.while_loop(condition, rnn_forward,
                             loop_vars=[out, hidden, alive, count_length],
                             maximum_iterations=config.max_num_sub_policies - 1,
                             shape_invariants=[tf.TensorShape(
                                 [batch_size, None, out.get_shape()[-1]]),
                                 tuple(raw_shape), tf.TensorShape(batch_size),
                                 tf.TensorShape(batch_size)])

    def policy_network(self, input, scope, size=config.baseline_layer_size,
                       n_layers=config.n_layers, output_activation=None,
                       batch_size=config.batch_size):

        self.num_actions_plus_1 = self.action_dim + 1

        subpolicy_multi_cell = rnn.MultiRNNCell([self.single_cell(
            self.num_actions_plus_1, config.sub_policy_network, 'sub') for i in
                                                 range(
                                                     config.num_sub_policy_layers)],
                                                state_is_tuple=True)

        self.sub_policies, states, alive, length = self.dynamic_rnn(
            subpolicy_multi_cell, inputs=input, batch_size=batch_size)

        self.sub_policies = self.sub_policies[:, :, :self.action_dim]
        # self.sub_policies => (batch size, max_length, action_dim)
        # length => (batch size)

        master_multi_cell = rnn.MultiRNNCell([self.single_cell(
            num_units=config.max_num_sub_policies,
            cell_type=config.master_network, name='master') for i in
                                              range(config.num_master_layers)],
                                             state_is_tuple=True)

        max_length = tf.reduce_max(length)

        concatenated = tf.concat([self.sub_policies,
                                  tf.tile(tf.expand_dims(input, axis=1),
                                      [1, max_length, 1])], axis=2)

        # concatenated => (batch size, max_length, action_dim + state_space)

        if config.freeze_sub_policy:
            concatenated = tf.stop_gradient(concatenated, name='stop')

        self.out, states = tf.nn.dynamic_rnn(cell=master_multi_cell,
                                             inputs=concatenated,
                                             sequence_length=length,
                                             dtype=tf.float32, scope='master')

        # self.out => (batch size, max_length, max_num_sub)

        ranges = tf.expand_dims(
            tf.tile(tf.expand_dims(tf.range(max_length), axis=0),
                    [batch_size, 1]), axis=2)
        # ranges => (batch size, max_length, 1)
        mask = (tf.equal(ranges,
                         tf.expand_dims(tf.expand_dims(length - 1, axis=1),
                                        axis=2)))
        # mask => (batch size, max_length, 1)

        candidate_logits = tf.reduce_sum(self.out * tf.cast(mask, tf.float32),
                                         axis=1)
        # candidate logits => (batch size, max_num_sub)

        logit_ranges = tf.tile(
            tf.expand_dims(tf.range(config.max_num_sub_policies), axis=0),
            [batch_size, 1])
        # logit ranges => (batch size, max_num_sub)

        logit_mask = (logit_ranges < tf.expand_dims(length - 1, axis=1))
        negative_logit_mask = (
            logit_ranges >= tf.expand_dims(length - 1, axis=1))
        # negative_logit_mask, logit mask => (batch size, max_num_sub)

        mins = tf.reduce_min(candidate_logits, axis=1, keepdims=True)
        # mins => (batch size, 1)

        reset_logits = candidate_logits * tf.cast(logit_mask, tf.float32)
        # reset_logits => (batch size, max_num_sub)

        offset = tf.cast(negative_logit_mask, tf.float32) * mins
        # offset => (batch size, max_num_sub)

        logits = reset_logits + offset

        self.chosen_index = tf.cast(tf.argmax(logits, axis=1), tf.int32)

        if config.weight_average:
            self.weights = tf.nn.softmax(logits=logits, axis=1)
        else:
            tf.Assert(tf.reduce_sum(
                tf.cast(self.chosen_index < length, tf.int32)) == batch_size,
                      [self.chosen_index, length])
            # max_output = tf.reduce_max(logits, axis=1, keepdims=True)
            # tmp = tf.nn.relu(logits - max_output + 1e-17)
            # self.weights = tmp / tf.reduce_sum(tmp, axis=1, keepdims=True)
            self.weights = tf.one_hot(indices=self.chosen_index,
                                     depth=config.num_sub_policies)

        final_policy = tf.reduce_sum(
            tf.expand_dims(self.weights[:, :max_length],
                           axis=2) * self.sub_policies, axis=1)

        if config.sub_policy_index > -1:
            final_policy = self.sub_policies[:, config.sub_policy_index, :]

        print('finish building network!')
        print('hidden layer size', self.num_actions_plus_1)

        return final_policy

    def _init(self, ob_space, ac_space, hid_size, num_hid_layers, config,
              gaussian_fixed_var=True):
        assert isinstance(ob_space, gym.spaces.Box)

        self.pdtype = pdtype = make_pdtype(ac_space)
        self.action_dim = self.pdtype.param_shape()[0] // 2
        sequence_length = None

        ob = U.get_placeholder(name="ob", dtype=tf.float32,
                               shape=[sequence_length] + list(ob_space.shape))

        with tf.variable_scope("obfilter"):
            self.ob_rms = RunningMeanStd(shape=ob_space.shape)

        with tf.variable_scope('vf'):
            obz = tf.clip_by_value((ob - self.ob_rms.mean) / self.ob_rms.std,
                                   -5.0, 5.0)
            last_out = obz
            for i in range(num_hid_layers):
                last_out = tf.nn.tanh(
                    tf.layers.dense(last_out, hid_size, name="fc%i" % (i + 1),
                                    kernel_initializer=U.normc_initializer(
                                        1.0)))
            self.vpred = tf.layers.dense(last_out, 1, name='final',
                                         kernel_initializer=U.normc_initializer(
                                             1.0))[:, 0]

        with tf.variable_scope('pol'):
            if gaussian_fixed_var and isinstance(ac_space, gym.spaces.Box):
                mean = self.policy_network(obz, config.algorithm,
                                           batch_size=self.batch_size)

                logstd = tf.get_variable(name="logstd", shape=[1,
                                                               pdtype.param_shape()[
                                                                   0] // 2],
                                         initializer=tf.zeros_initializer())
                pdparam = tf.concat([mean, mean * 0.0 + logstd], axis=1)
            else:
                pdparam = self.policy_network(obz, config.algorithm,
                                              batch_size=self.batch_size)

        self.pd = pdtype.pdfromflat(pdparam)

        self.state_in = []
        self.state_out = []

        stochastic = tf.placeholder(dtype=tf.bool, shape=())
        ac = U.switch(stochastic, self.pd.sample(), self.pd.mode())
        self._act = U.function([stochastic, ob], [ac, self.vpred])

    def act(self, stochastic, ob):
        ac1, vpred1 = self._act(stochastic, ob[None])
        return ac1[0], vpred1[0]

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    def get_initial_state(self):
        return []