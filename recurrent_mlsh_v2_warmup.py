import copy

import matplotlib.pyplot as plt
import tensorflow.contrib.rnn as rnn

from pg import *
from visualize import visualize_fourrooms_master_policy, \
    visualize_fourrooms_sub_policy


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
        self.choices_placeholder = tf.placeholder(tf.int32, shape=[None, 1],
                                                  name='time_scale')
        self.observation_placeholder = tf.placeholder(tf.float32, shape=[None,
                                                                         self.observation_dim])
        if self.discrete:
            self.action_placeholder = tf.placeholder(tf.int64, shape=None)
        else:
            self.action_placeholder = tf.placeholder(tf.float32, shape=[None,
                                                                        self.action_dim])

        self.advantage_placeholder = tf.placeholder(tf.float32, shape=None)

    def build(self):
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

        # concatenated = tf.concat(
        #     [self.proposed_sub_policies, self.state_embedding], axis=2)
        concatenated = self.state_embedding

        self.out, _ = tf.nn.dynamic_rnn(cell=lstm_cell, inputs=concatenated,
                                        dtype=tf.float32, scope='master')
        self.master_policy_action_logits = last_output = self.out[:, -1, :]

        self.master_policy_action_logits = tf.gather_nd(
            params=self.master_policy_action_logits,
            indices=self.choices_placeholder)

        # max_output = tf.reduce_max(last_output, axis=1, keep_dims=True)
        # tmp = tf.nn.relu(last_output - max_output + 1e-6)
        # self.weights = tmp / tf.reduce_sum(tmp, axis=1, keep_dims=True)
        self.weights = tf.nn.softmax(self.master_policy_action_logits, axis=1)

        return self.weights

    def build_policy_network_op(self, scope="policy_network"):
        self.proposed_sub_policies = self.sub_policies_act(
            self.observation_placeholder)

        self.master_chosen_one_hot = self.master_policy_act(
            self.proposed_sub_policies)

        self.master_chosen_sub_policy_index = tf.argmax(
            self.master_policy_action_logits, axis=1)

        self.final_policy = tf.reduce_sum(
            tf.expand_dims(self.master_chosen_one_hot,
                           axis=2) * self.proposed_sub_policies, axis=1)

        self.action_logits = self.final_policy

        self.sampled_action = tf.squeeze(tf.multinomial(self.action_logits, 1),
                                         axis=1)

        self.action_logits_tmp = tf.placeholder(tf.float32,
                                                [None, self.action_dim])
        self.sampled_action_tmp = tf.squeeze(
            tf.multinomial(self.action_logits_tmp, 1), axis=1)
        self.logprob = -tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.action_placeholder, logits=self.action_logits)

    def add_loss_op(self):
        self.loss = -tf.reduce_mean(self.logprob * self.advantage_placeholder)

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
        self.train_op = self.master_adam.minimize(self.loss,
                                                  var_list=tf.get_collection(
                                                      tf.GraphKeys.TRAINABLE_VARIABLES,
                                                      scope='master'))

    def sample_path(self, env, ti, num_episodes=None):
        episode = 0
        episode_rewards = []
        paths = []
        t = 0
        at_time_scales = []
        sub_policy_choice = None

        while num_episodes or t < self.config.batch_size:
            state = self.env.reset()
            states, actions, rewards = [], [], []
            episode_reward = 0
            rooms = []

            for step in range(self.config.max_ep_len):
                states.append(state)

                is_at_master_timescale = 0
                if step % self.config.master_timescale == 0:
                    is_at_master_timescale = 1

                at_time_scales.append(is_at_master_timescale)

                room = self.get_room_by_state(state)
                rooms.append(room)

                if is_at_master_timescale == 1:
                    chosen_sub, action = self.sess.run(
                        [self.master_chosen_one_hot, self.sampled_action],
                        feed_dict={
                            self.choices_placeholder: [[0]],
                            self.observation_placeholder: [[states[-1]]]
                        })
                    action = action[0]
                    chosen_sub = chosen_sub[0]
                    sub_policy_choice = np.argmax(chosen_sub)
                elif is_at_master_timescale == 0:
                    chosen_sub = sub_policy_choice
                    action_logits = self.sess.run(self.proposed_sub_policies,
                                                  feed_dict={
                                                      self.observation_placeholder: [
                                                          [states[-1]]]
                                                  })[:, chosen_sub, :]

                    action = self.sess.run(self.sampled_action_tmp, feed_dict={
                        self.action_logits_tmp: action_logits
                    })[0]

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

        indices = range(config.batch_size)

        last_index = [0]

        self.choices = []

        assert at_time_scales[0] == 1

        for i in range(config.batch_size):
            if at_time_scales[i] == 1:
                choice = [indices[i]]
                last_index = choice
            else:
                choice = last_index

            self.choices.append(choice)

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

        print 'self.config.do_meta_learning = %s' % self.config.do_meta_learning
        print 'num_tasks = %s' % num_tasks

        master_policies = {}

        for taski in xrange(num_tasks):
            self.sess.run(self.init)

            if config.recover_checkpoint_path:
                print("Recovering model...")
                self.recover_model_checkpoint(self.sess, self.saver,
                                              config.recover_checkpoint_path)

            index = np.random.randint(0, 2)
            if index == 0:
                self.env.reset(seed={
                    'fixedstart+goal:start': (env.nS - env.ncol - 1),
                    'fixedstart+goal:goal': 2
                })
            else:
                self.env.reset(seed={
                    'fixedstart+goal:start': (1 + env.ncol),
                    'fixedstart+goal:goal': (env.nS - 3)
                })

            for t in range(self.config.num_batches):
                print 'train iter #%s:' % t
                print(t, self.get_epsilon(t), self.env.goal)
                paths, total_rewards = self.sample_path(env=self.env, ti=t)

                scores_eval += total_rewards

                if str(config.env_name).startswith("Fourrooms"):
                    observations = np.expand_dims(
                        np.concatenate([path["observation"] for path in paths]),
                        axis=1)
                else:
                    observations = np.concatenate(
                        [path["observation"] for path in paths])

                sub_policy_actions = np.concatenate(
                    [path["action"] for path in paths])
                rewards = np.concatenate([path["reward"] for path in paths])
                returns = self.get_returns(paths)
                advantages = self.calculate_advantage(returns, observations)

                if self.config.use_baseline:
                    self.update_baseline(returns, observations)

                ret, _ = self.sess.run(
                    [self.master_chosen_one_hot, self.train_op], feed_dict={
                        self.choices_placeholder: self.choices,
                        self.observation_placeholder: observations,
                        self.action_placeholder: sub_policy_actions,
                        self.advantage_placeholder: advantages
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

                if t == config.num_batches - 1 and config.viz_sub_policies:
                    master_actions, logits = self.sess.run(
                        [self.master_chosen_sub_policy_index,
                         self.sub_policies], feed_dict={
                            self.observation_placeholder: np.expand_dims(
                                np.arange(81), axis=1),
                            self.choices_placeholder: [[i] for i in range(81)]
                        })

                    master_policies[index] = master_actions

                    plt.clf()
                    sub_policy_actions = []
                    envs = []

                    self.env.reset(seed={
                        'fixedstart+goal:start': (env.nS - env.ncol - 1),
                        'fixedstart+goal:goal': 2
                    })
                    envs.append(copy.deepcopy(self.env))
                    self.env.reset(seed={
                        'fixedstart+goal:start': (1 + env.ncol),
                        'fixedstart+goal:goal': (env.nS - 3)
                    })
                    envs.append(copy.deepcopy(self.env))

                    for sub in range(config.num_sub_policies):
                        sub_policy_actions.append(
                            np.argmax(logits[:, sub, :], axis=1))

                    visualize_fourrooms_sub_policy(envs, sub_policy_actions)

                    plt.tight_layout()
                    plt.savefig('plots/subpolicies_%s.png' % get_timestamp())

                    if 0 in master_policies and 1 in master_policies:
                        plt.clf()
                        visualize_fourrooms_master_policy(envs,
                                                          [master_policies[0],
                                                           master_policies[1]])
                        plt.tight_layout()
                        plt.savefig('plots/master_%s.png' % get_timestamp())

                    self.save_model_checkpoint(self.sess, self.saver,
                                               os.path.join(
                                                   self.config.output_path,

                                                   'model.ckpt'), t)

                    self.logger.info("- Training done.")
                    export_plot(scores_eval, "Score", config.env_name,
                                self.config.plot_output)
                    export_plot(scores_eval, "Score", config.env_name,
                                "plots/score_%s.png" % get_timestamp())

                    # if str(config.env_name).startswith(
                    #     "Fourrooms") and config.visualize_master_policy:
                    #     plt.rcParams["figure.figsize"] = [12, 12]
                    #     f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2,
                    # sharex='col',
                    #
                    # sharey='row')
                    #     axes = {'room0': ax1, 'room1': ax2, 'room2': ax3,
                    # 'room3': ax4}
                    #     for room in self.plot:
                    #         axes[room].set_title(room, size=20)
                    #         for sub in range(config.num_sub_policies):
                    #             prob_list = self.plot[room][sub]
                    #             axes[room].plot(range(len(prob_list)),
                    # prob_list,
                    #                             linewidth=5)
                    #         axes[room].legend(['subpolicy' + str(sub) for
                    # sub in
                    #                            range(
                    # config.num_sub_policies)],
                    #                           loc='upper left',
                    # prop={'size': 20})
                    #     plt.tight_layout()
                    #     plt.savefig('plots/action_logits_per_room_%s.png' %
                    # get_timestamp(),
                    #                 dpi=300)


if __name__ == "__main__":
    if not os.path.exists('./plots'):
        os.makedirs('./plots')

    env = gym.make(config.env_name)
    config = config('RecurrentMLSH-v2')
    model = RecurrentMLSHV2META(env, config)
    model.run()
