import copy

import matplotlib.pyplot as plt
import tensorflow.contrib.rnn as rnn

from pg import *
from visualize import visualize_fourrooms_multi_policy


class RecurrentMLSHV2META(PolicyGradient):
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

    def add_placeholders_op(self):
        self.at_master_timescale_placeholder = tf.placeholder(tf.bool, shape=(),
                                                              name='time_scale')
        self.observation_placeholder = tf.placeholder(tf.float32, shape=[None,
                                                                         self.observation_dim])
        if self.discrete:
            self.action_placeholder = tf.placeholder(tf.int64, shape=None)
        else:
            self.action_placeholder = tf.placeholder(tf.float32, shape=[None,
                                                                        self.action_dim])

        # Define a placeholder for advantages
        self.advantage_placeholder = tf.placeholder(tf.float32, shape=None)
        self.master_advantage_placeholder = tf.placeholder(tf.float32,
                                                           shape=None)

    def build(self):
        # self.last_chosen_index = tf.constant(0)
        self.last_chosen_one_hot = tf.one_hot(indices=0,
                                              depth=self.config.num_sub_policies)

        self.add_placeholders_op()
        self.build_policy_network_op()
        self.add_loss_op()
        self.add_optimizer_op()

        if self.config.use_baseline:
            self.add_baseline_op()

    def sub_policies_act(self, mlp_input):
        if str(config.env_name).startswith("Fourrooms"):
            self.state_embedding = tf.tile(
                tf.one_hot(indices=tf.cast(mlp_input, dtype=tf.int32),
                           depth=self.env.nS), [1, config.num_sub_policies, 1])
            num_actions = self.env.action_space.n

        else:
            self.state_embedding = tf.tile(tf.expand_dims(mlp_input, axis=1),
                                           [1, config.num_sub_policies, 1])
            num_actions = self.env.action_space.shape[0]

        subpolicy_multi_cell = rnn.MultiRNNCell(
            [self.single_cell(num_actions, config.sub_policy_network, 'sub') for
             i in range(config.num_sub_policy_layers)], state_is_tuple=True)

        self.sub_policies, states = tf.nn.dynamic_rnn(cell=subpolicy_multi_cell,
                                                      inputs=self.state_embedding,
                                                      dtype=tf.float32,
                                                      scope='subpolicy')
        return self.sub_policies

    def master_policy_act(self, sub_policies):

        lstm_cell = rnn.BasicLSTMCell(num_units=config.num_sub_policies)

        self.proposed_sub_policies = sub_policies

        concatenated = tf.concat(
            [self.proposed_sub_policies, self.state_embedding], axis=2)

        self.out, _ = tf.nn.dynamic_rnn(cell=lstm_cell, inputs=concatenated,
                                        dtype=tf.float32, scope='master')
        self.master_policy_action_logits = last_output = self.out[:, -1, :]

        # self.last_chosen_index = self.chosen_index = tf.argmax(last_output,
        #  axis=1)
        # self.master_chosen_sub_policy_index = tf.argmax(last_output, axis=1)

        max_output = tf.reduce_max(last_output, axis=1, keep_dims=True)
        tmp = tf.nn.relu(last_output - max_output + 1e-17)
        self.weights = tmp / tf.reduce_sum(tmp, axis=1, keep_dims=True)
        self.last_chosen_one_hot = chosen_one_hot = tf.expand_dims(self.weights,
                                                                   axis=2)
        return chosen_one_hot

    def build_policy_network_op(self, scope="policy_network"):
        self.proposed_sub_policies = self.sub_policies_act(
            self.observation_placeholder)

        # master_timescale_mask = tf.cast(self.at_master_timescale_placeholder,
        #                                 tf.float32)
        # self.master_chosen_one_hot = master_timescale_mask * \
        #                              self.master_policy_act(
        #     self.proposed_sub_policies) + (
        #                                       1 - master_timescale_mask) * \
        #                                   tf.identity(
        #     self.last_chosen_one_hot)
        # self.master_chosen_one_hot = self.master_policy_act(
        # self.proposed_sub_policies)

        # self.master_chosen_sub_policy_index = tf.argmax(
        # self.master_policy_action_logits, axis=1)
        self.master_chosen_sub_policy_index = tf.placeholder(tf.int32, [None])

        self.master_chosen_one_hot = tf.expand_dims(
            tf.one_hot(self.master_chosen_sub_policy_index, depth=2), axis=2)

        self.final_policy = tf.reduce_sum(
            self.master_chosen_one_hot * self.proposed_sub_policies, axis=1)

        if self.discrete:
            self.action_logits = self.final_policy
            self.sampled_action = tf.squeeze(
                tf.multinomial(self.action_logits, 1), axis=1)
            self.logprob = -tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self.action_placeholder, logits=self.action_logits)

            # self.master_logprob = \
            #     -tf.nn.sparse_softmax_cross_entropy_with_logits(
            #     labels=self.master_chosen_sub_policy_index,
            #     logits=self.master_policy_action_logits)
        else:
            action_means = self.final_policy
            log_std = tf.get_variable('log_std', shape=[self.action_dim],
                                      trainable=True)
            action_std = tf.exp(log_std)
            multivariate = tfd.MultivariateNormalDiag(loc=action_means,
                                                      scale_diag=action_std)
            self.sampled_action = tf.random_normal(
                [self.action_dim]) * action_std + action_means
            self.logprob = multivariate.log_prob(self.action_placeholder)

    def add_loss_op(self):
        self.subpolicy_loss = -tf.reduce_mean(
            self.logprob * self.advantage_placeholder)

        # self.master_loss = -tf.reduce_mean(
        #     self.master_logprob * self.master_advantage_placeholder)

    # extract adv at every N timestep
    def calculate_master_advantage(self, adv):
        master_adv = []
        for i in xrange(len(adv)):
            if i % self.config.master_timescale == 0:
                master_adv.append(adv[i])
            else:
                master_adv.append(0)

        return np.array(master_adv)

    def add_optimizer_op(self):
        self.master_adam = tf.train.AdamOptimizer(learning_rate=self.lr)
        # self.master_train_op = self.master_adam.minimize(self.master_loss,
        #
        # var_list=tf.get_collection(
        #
        # tf.GraphKeys.TRAINABLE_VARIABLES,
        #                                                      scope='master'))

        self.subpolicy_adam = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.subpolicy_train_op = self.subpolicy_adam.minimize(
            self.subpolicy_loss,
            var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                       scope='subpolicy'))

    def sample_path(self, env, ti, num_episodes=None):
        episode = 0
        episode_rewards = []
        paths = []
        t = 0
        self.indices = []

        while num_episodes or t < self.config.batch_size:
            index = np.random.randint(0, 2)
            if index == 0:
                state = self.env.reset(seed={
                    'fixedstart+goal:start': (env.nS - env.ncol - 2),
                    'fixedstart+goal:goal': 2
                })
            else:
                state = self.env.reset(seed={
                    'fixedstart+goal:start': (1 + env.ncol),
                    'fixedstart+goal:goal': (env.nS - 3)
                })

            # print 'task start %s, goal %s' % (self.env.start, self.env.goal)

            states, actions, rewards = [], [], []
            episode_reward = 0
            rooms = []

            for step in range(self.config.max_ep_len):
                states.append(state)
                self.indices.append(index)

                is_at_master_timescale = False
                if step % self.config.master_timescale == 0:
                    is_at_master_timescale = True

                if str(config.env_name).startswith("Fourrooms"):
                    room = self.get_room_by_state(state)
                    rooms.append(room)

                    action = self.sess.run(self.sampled_action, feed_dict={
                        self.at_master_timescale_placeholder:
                            is_at_master_timescale,
                        self.observation_placeholder: [[states[-1]]],
                        self.master_chosen_sub_policy_index: [index]
                    })
                    action = action[0]
                else:
                    action = self.sess.run(self.sampled_action, feed_dict={
                        self.observation_placeholder: states[-1][None]
                    })
                    action = action[0]

                action = self.epsilon_greedy(action=action,
                                             eps=self.get_epsilon(ti))
                if self.config.render:
                    env.render()

                state, reward, done, info = env.step(action)
                actions.append(action)
                rewards.append(reward)
                episode_reward += reward
                t += 1
                if done or step == self.config.max_ep_len - 1:
                    episode_rewards.append(episode_reward)
                    break
                if (not num_episodes) and t == self.config.batch_size:
                    break

            if str(config.env_name).startswith(
                "Fourrooms") and self.config.render:
                print(
                    [(states[room], rooms[room]) for room in range(len(rooms))])
                print(Counter(rooms))
                print(sorted(Counter(rooms), key=lambda i: i[1]))
                exit()

            path = {
                "observation": np.array(states), "reward": np.array(rewards),
                "action": np.array(actions)
            }
            paths.append(path)
            episode += 1
            if num_episodes and episode >= num_episodes:
                break

        return paths, episode_rewards

    def train(self):
        print '===================== in RecurrentMLSHV2META.train ' \
              '====================='

        last_record = 0

        self.init_averages()
        scores_eval = []
        self.plot = {
            'room' + str(i): {j: [] for j in range(config.num_sub_policies)} for
            i in range(4)}

        num_tasks = 1
        if self.config.do_meta_learning:
            num_tasks = self.config.num_meta_learning_training_tasks

        # old = self.sess.run(tf.get_collection(
        #     tf.GraphKeys.TRAINABLE_VARIABLES, scope='subpolicy'))

        print 'self.config.do_meta_learning = %s' % self.config.do_meta_learning
        print 'num_tasks = %s' % num_tasks

        for taski in xrange(num_tasks):
            print '===================== task #%s =====================' % taski

            for t in range(self.config.num_batches):
                print 'train iter #%s: (epsilon: %s)' % (t, self.get_epsilon(t))
                paths, total_rewards = self.sample_path(env=self.env, ti=t)

                scores_eval += total_rewards

                if str(config.env_name).startswith("Fourrooms"):
                    observations = np.expand_dims(
                        np.concatenate([path["observation"] for path in paths]),
                        axis=1)
                else:
                    observations = np.concatenate(
                        [path["observation"] for path in paths])

                actions = np.concatenate([path["action"] for path in paths])
                rewards = np.concatenate([path["reward"] for path in paths])
                returns = self.get_returns(paths)
                advantages = self.calculate_advantage(returns, observations)
                # master_advantages = self.calculate_master_advantage(
                # advantages)

                if self.config.use_baseline:
                    self.update_baseline(returns, observations)

                # self.sess.run(self.master_train_op, feed_dict={
                #     self.at_master_timescale_placeholder: True,
                #     self.observation_placeholder: observations,
                #     self.action_placeholder: actions,
                #     self.master_advantage_placeholder: master_advantages,
                #     self.master_chosen_sub_policy_index: indices
                #     # self.master_advantage_placeholder: advantages,
                # })

                # if t >= self.config.warmup:
                self.sess.run(self.subpolicy_train_op, feed_dict={
                    self.at_master_timescale_placeholder: True,
                    self.observation_placeholder: observations,
                    self.action_placeholder: actions,
                    self.advantage_placeholder: advantages,
                    self.master_chosen_sub_policy_index: self.indices
                })

                if t % self.config.summary_freq == 0:
                    self.update_averages(total_rewards, scores_eval)
                    self.record_summary(self.batch_counter)

                self.batch_counter += 1

                avg_reward = np.mean(total_rewards)
                sigma_reward = np.sqrt(
                    np.var(total_rewards) / len(total_rewards))
                msg = "Average reward: {:04.2f} +/- {:04.2f}".format(avg_reward,
                                                                     sigma_reward)
                self.logger.info(msg)

                last_record += 1
                if self.config.record and (
                        last_record > self.config.record_freq):
                    self.logger.info("Recording...")
                    last_record = 0
                    self.record()

                # TODO: Message for Jiayu: This is the subpolicy viz code

                if t == config.num_batches - 1 and \
                    config.visualize_sub_policies:
                    logits = self.sess.run(self.sub_policies, feed_dict={
                        self.observation_placeholder: np.expand_dims(
                            np.arange(81), axis=1)
                    })
                    plt.clf()

                    actions = []
                    envs = []
                    self.env.reset(seed={
                        'fixedstart+goal:start': (env.nS - env.ncol - 2),
                        'fixedstart+goal:goal': 2
                    })
                    envs.append(copy.deepcopy(self.env))
                    self.env.reset(seed={
                        'fixedstart+goal:start': (1 + env.ncol),
                        'fixedstart+goal:goal': (env.nS - 3)
                    })
                    envs.append(copy.deepcopy(self.env))

                    for sub in range(config.num_sub_policies):
                        actions.append(np.argmax(logits[:, sub, :], axis=1))

                    visualize_fourrooms_multi_policy(envs, actions)

                    plt.tight_layout()
                    plt.savefig('plots/subpolicies_%s.png' % get_timestamp())

                    self.save_model_checkpoint(self.sess, self.saver,
                                               os.path.join(
                                                   self.config.output_path,

                                                   'model.ckpt'), t)

        self.logger.info("- Training done.")
        export_plot(scores_eval, "Score", config.env_name,
                    self.config.plot_output)
        export_plot(scores_eval, "Score", config.env_name,
                    "plots/score_%s.png" % get_timestamp())

        if str(config.env_name).startswith(
            "Fourrooms") and config.visualize_master_policy:
            plt.rcParams["figure.figsize"] = [12, 12]
            f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col',
                                                       sharey='row')
            axes = {'room0': ax1, 'room1': ax2, 'room2': ax3, 'room3': ax4}
            for room in self.plot:
                axes[room].set_title(room, size=20)
                for sub in range(config.num_sub_policies):
                    prob_list = self.plot[room][sub]
                    axes[room].plot(range(len(prob_list)), prob_list,
                                    linewidth=5)
                axes[room].legend(['subpolicy' + str(sub) for sub in
                                   range(config.num_sub_policies)],
                                  loc='upper left', prop={'size': 20})
            plt.tight_layout()
            # plt.savefig('Rooms and Subs ' + str(np.random.randint(0, 10000)),
            #             dpi=300)
            plt.savefig('plots/action_logits_per_room_%s.png' % get_timestamp(),
                        dpi=300)


if __name__ == "__main__":
    if not os.path.exists('./plots'):
        os.makedirs('./plots')

    env = gym.make(config.env_name)
    # env.reset(seed={'fixedstart+goal:start':(1 + env.ncol),
    # 'fixedstart+goal:goal':(env.nS -3)})

    config = config('RecurrentMLSH-v2')
    model = RecurrentMLSHV2META(env, config)
    model.run()
