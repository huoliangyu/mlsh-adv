import numpy as np
import math
import gym
import os
import random
import time
import datetime
import matplotlib.pyplot as plt
from matplotlib import colors
from test_env import *

def eps_greedy_policy(Q, eps, s, nA):
  if random.random() < eps:
    # print 'random'
    return random.choice(xrange(nA))
  else:
    # print 'picking max'
    return np.argmax(Q[s], axis=0)

def get_timestamp():
  return datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d%H%M%S')


def learn_Q_QLearning(env, num_episodes=5000, gamma=1.0, lr=0.1, e=0.8, decay_rate=0.99):
  """Learn state-action values using the Q-learning algorithm with epsilon-greedy exploration strategy.
  Update Q at the end of every episode.

  Parameters
  ----------
  env: gym.core.Environment
    Environment to compute Q function for. Must have nS, nA, and P as
    attributes.
  num_episodes: int 
    Number of episodes of training.
  gamma: float
    Discount factor. Number in range [0, 1)
  learning_rate: float
    Learning rate. Number in range [0, 1)
  e: float
    Epsilon value used in the epsilon-greedy method. 
  decay_rate: float
    Rate at which epsilon falls. Number in range [0, 1)

  Returns
  -------
  np.array
    An array of shape [env.nS x env.nA] representing state, action values
  """
  
  ############################
  # YOUR IMPLEMENTATION HERE #
  ############################
  # print 'starting learn_Q_QLearning'

  # Q = np.zeros((env.nS, env.nA))
  # Q = np.ones((env.nS, env.nA)) * 1.0
  Q = np.ones((env.nS, env.nA))
  eps = e
  total_score = 0
  episode_length = 0
  running_aves = []
  for episode_iter in xrange(num_episodes):
    # print 'eps = %s' % eps
    # print 'episode_iter = %s' % episode_iter

    s = env.reset()
    done = False
    t = 0
    while not done:
      a = eps_greedy_policy(Q, eps, s, env.nA)
      snext, r, done, _ = env.step(a)

      # print 's = %s, a = %s, snext = %s, r = %s, done = %s' % (s, a, snext, r, done)

      total_score += r

      # if r == 1:
      #   print 'win episode_iter = %s' % episode_iter

      # update Q
      if done:
        Q[s][a] += lr * (r - Q[s][a])
      else:
        Q[s][a] += lr * (r + gamma * np.max(Q[snext], axis=0) - Q[s][a])

      # print 'episode_iter = %s, t = %s:  s = %s, a = %s, r = %s, snext = %s, done = %s, Q[%s][%s] = %s' % (episode_iter, t, s, a, r, snext, done, s, a, Q[s][a])

      s = snext

      t += 1

    episode_length += t
    eps *= decay_rate

    running_ave = total_score / (episode_iter + 1)
    running_aves.append(running_ave)

    if episode_iter % 100 == 0:
      print 'total_score = %s, # episode = %s, ave = %s, epi_length = %s, epsilon = %s' % (
        total_score, episode_iter + 1, running_ave, episode_length / (episode_iter + 1), eps)


  plt.plot(range(1, num_episodes + 1), running_aves)
  plt.xlabel('Epoch')
  plt.ylabel('Average Reward')
  plt.xlim([0, num_episodes])
  plt.ylim([0, math.ceil(max(running_aves) * 10) / 10 * 1.1])
  plt.savefig('plots/fourrooms_tabq_scores_%s.png' % get_timestamp())

  return Q

def render_single_Q(env, Q):
  """Renders Q function once on environment. Watch your agent play!

    Parameters
    ----------
    env: gym.core.Environment
      Environment to play Q function on. Must have nS, nA, and P as
      attributes.
    Q: np.array of shape [env.nS x env.nA]
      state-action values.
  """

  episode_reward = 0
  state = env.reset()
  done = False
  while not done:
    env.render()
    time.sleep(0.5) # Seconds between frames. Modify as you wish.
    action = np.argmax(Q[state])
    state, reward, done, _ = env.step(action)
    episode_reward += reward

  print "Episode reward: %f" % episode_reward

def extract_policy(env, Q, gamma=1.0):
  extracted_policy = np.zeros(env.nS, dtype='int')
  for s in xrange(env.nS):
    qmax= -float('inf')
    for a in xrange(env.nA):
      # Q(s, a) = E[r_pi(s,a)] + gamma * sum_{s'}[P_pi(s'|s,a) * V(s')]
      #         = sum_{s'|s,a} { P_pi(s'|s,a) * r(s'|s, a) } + gamma * sum_{s'|s,a} { [P_pi(s'|s,a) * V(s')] }
      #         = sum_{s'|s,a} { P_pi(s'|s,a) * [ r(s'|s, a) + gamma * * V(s') ] }
      Q_s_a = sum([
        probability * (reward + gamma * Q[nextstate][a]) for probability, nextstate, reward, _ in env.P[s][a]
      ])

      if Q_s_a > qmax:
        extracted_policy[s] = a
        qmax = Q_s_a
      # elif Q_s_a == qmax:
      #     # tie breaker
      #     if np.random.uniform() < 0.5:
      #         improved_policy[s] = a

  return extracted_policy

def visualize_policy(env, policy):
  nrow, ncol = env.desc.shape

  # data = np.random.rand(10, 10) * 20
  # 0: left
  # 1: down
  # 2: right
  # 3: up
  l = 0
  d = 1
  r = 2
  u = 3
  # boundary
  b = -1
  # goal
  g = -2

          # "XXGXXXXXX",
          # "XOOOXOOOX",
          # "XOOOOOOOX",
          # "XOOOXOOOX",
          # "XXOXXXOXX",
          # "XOOOXOOOX",
          # "XOOOOOOOX",
          # "XOOOXOOOX",
          # "XXXXXXXXX",
  # data = [
  #     [b, b, g, b, b, b, b, b, b],
  #     [b, r, u, l, b, d, d, d, b],
  #     [b, r, u, l, l, l, l, l, b],
  #     [b, r, u, l, b, u, u, u, b],
  #     [b, b, u, b, b, b, u, b, b],
  #     [b, r, u, l, b, d, d, d, b],
  #     [b, r, u, l, l, l, l, l, b],
  #     [b, r, u, l, b, u, u, u, b],
  #     [b, b, b, b, b, b, b, b, b],
  # ]

  # print '-----------------------------'
  # print env.desc
  # print '-----------------------------'
  policy[env.goal] = g
  for s in xrange(env.nS):
    if env.desc[s / nrow][s % ncol] in b'X':
      policy[s] = -1

  data = np.array_split(policy, nrow)

  # print(data)

  # create discrete colormap
  cmap = colors.ListedColormap(['green', 'black', 'red', 'blue', 'pink', 'cyan'])
  bounds = [-2,-1,0,1,2,3,4]
  norm = colors.BoundaryNorm(bounds, cmap.N)

  fig, ax = plt.subplots()
  ax.imshow(data, cmap=cmap, norm=norm)

  # draw gridlines
  ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)
  ax.set_xticks(np.arange(-.5, ncol, 1));
  ax.set_yticks(np.arange(-.5, nrow, 1));
  # ax.set_xticks([]);
  # ax.set_yticks([]);
  # plt.axis('off')
  # https://stackoverflow.com/questions/2176424/hiding-axis-text-in-matplotlib-plots
  ax.xaxis.set_ticklabels([])
  ax.yaxis.set_ticklabels([])

  print 'data ='
  print data

  # plt.text(-0.5, -0.5, 'u', fontsize=12)
  # plt.text(2.5, 2.5, 'd', fontsize=12)
  # plt.text(9, 9, 'l', fontsize=12)
  direction = {-2:'g', -1:'b', 0:u'\u2190', 1:u'\u2193', 2:u'\u2192', 3:u'\u2191'}
  for r in xrange(nrow):
    for c in xrange(ncol):
      if env.desc[r][c] not in b'XG':
        plt.text(c - 0.2, r + 0.2, direction[data[r][c]], fontsize=12, color='white')

  # plt.show()
  plt.savefig('plots/fourrooms_tabq_subpolicies_%s.png' % get_timestamp())

# Feel free to run your own debug code in main!
def main():
  if not os.path.exists('./plots'):
    os.makedirs('./plots')
  env = gym.make('Fourrooms-small-v0')
  Q = learn_Q_QLearning(env, num_episodes=10000, e=1.0, gamma=1.0, lr=0.3, decay_rate=0.92)

  # env = gym.make('Fourrooms-medium-v0')
  # Q = learn_Q_QLearning(env, num_episodes=20000, e=1.0, gamma=1.0, lr=0.3, decay_rate=0.92)

  # env = gym.make('Fourrooms-large-v0')
  # Q = learn_Q_QLearning(env, num_episodes=100000, e=1.0, gamma=1.0, lr=0.3, decay_rate=0.92)

  print('Q =\n%s' % Q)
  # render_single_Q(env, Q)
  optimal_policy = extract_policy(env, Q)
  print('optimal_policy =\n%s' % optimal_policy)
  visualize_policy(env, optimal_policy)

if __name__ == '__main__':
    main()
