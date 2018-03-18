import tensorflow.contrib.rnn as rnn

from pg import *


class RecurrentMLSHV2META(PolicyGradient):
    def sub_policies(self, mlp_input):
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
        return self.sub_policies

    def master_policy(self, sub_policies):

        lstm_cell = rnn.BasicLSTMCell(num_units=config.num_sub_policies)

        self.proposed_sub_policies = sub_policies

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


    def sample_path(self, env, num_episodes=None):
        episode = 0
        episode_rewards = []
        paths = []
        t = 0
        rooms_and_sub_policies = {}

        while num_episodes or t < self.config.batch_size:
            state = env.reset()
            states, actions, rewards = [], [], []
            episode_reward = 0
            rooms = []

            for step in range(self.config.max_ep_len):
                states.append(state)

                if str(config.env_name).startswith("Fourrooms"):
                    room = self.get_room_by_state(state)
                    rooms.append(room)

                    chosen_sub_policy, action = self.sess.run(
                        [self.chosen_index, self.sampled_action], feed_dict={
                            self.observation_placeholder: [[states[-1]]]
                        })
                    action = action[0]
                    chosen_sub_policy = chosen_sub_policy[0]
                    if room not in rooms_and_sub_policies:
                        rooms_and_sub_policies[room] = []
                    rooms_and_sub_policies[room].append(chosen_sub_policy)
                else:
                    action = self.sess.run(self.sampled_action, feed_dict={
                        self.observation_placeholder: states[-1][None]
                    })
                    action = action[0]

                action = self.epsilon_greedy(action=action,
                                             eps=self.get_epsilon(t))
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

        if str(config.env_name).startswith("Fourrooms"):
            counter_by_room = {}
            for room in rooms_and_sub_policies:
                counter = Counter(rooms_and_sub_policies[room])
                s = sum([counter[sub] for sub in counter])
                for sub in range(config.num_sub_policies):
                    counter[sub] = counter[sub] * 1.0 / s if sub in counter \
                        else \
                        0.0
                    self.plot[room][sub].append(counter[sub])
                counter_by_room[room] = counter
            print(counter_by_room)

        return paths, episode_rewards

if __name__ == "__main__":
    if not os.path.exists('./plots'):
        os.makedirs('./plots')

    env = gym.make(config.env_name)
    config = config('RecurrentMLSH-v2')
    model = RecurrentMLSHV2META(env, config)
    model.run()
