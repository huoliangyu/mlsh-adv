import gym
import tensorflow as tf
import tensorflow.contrib.rnn as rnn

import baselines.common.tf_util as U
from baselines.common.distributions import make_pdtype
from baselines.common.mpi_running_mean_std import RunningMeanStd
from config import config


class LSTMFCPolicy(object):
    recurrent = True

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

    def __init__(self, name, *args, **kwargs):
        with tf.variable_scope(name):
            self._init(*args, **kwargs)
            self.scope = tf.get_variable_scope().name

    def _init(self, ob_space, ac_space, hid_size, num_hid_layers, config,
              gaussian_fixed_var=True):
        assert isinstance(ob_space, gym.spaces.Box)

        self.pdtype = pdtype = make_pdtype(ac_space)
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
            last_out = obz

            def sub_pol(input_m, scope):
                state_embedding = tf.tile(tf.expand_dims(input_m, axis=1),
                                          [1, config.num_sub_policies, 1])
                rnn_cells = [self.single_cell(pdtype.param_shape()[0] // 2,
                                              config.sub_policy_network, 'sub')
                             for i in range(config.num_sub_policy_layers)]

                subpolicy_multi_cell = rnn.MultiRNNCell(rnn_cells)

                last_out, states = tf.nn.dynamic_rnn(cell=subpolicy_multi_cell,
                                                     inputs=state_embedding,
                                                     dtype=tf.float32,
                                                     scope=scope)
                return last_out[:, -1, :]

            ppsl = []
            for i in range(config.num_sub_policies):
                ppsl.append(sub_pol(obz, 'pol' + '/' + str(i)))
            last_out = tf.concat(ppsl, axis=1)

            if gaussian_fixed_var and isinstance(ac_space, gym.spaces.Box):
                mean = tf.layers.dense(last_out, pdtype.param_shape()[0] // 2,
                                       name='final',
                                       kernel_initializer=U.normc_initializer(
                                           0.01))
                logstd = tf.get_variable(name="logstd", shape=[1,
                                                               pdtype.param_shape()[
                                                                   0] // 2],
                                         initializer=tf.zeros_initializer())
                pdparam = tf.concat([mean, mean * 0.0 + logstd], axis=1)
            else:
                pdparam = tf.layers.dense(last_out, pdtype.param_shape()[0],
                                          name='final',
                                          kernel_initializer=U.normc_initializer(
                                              0.01))

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
