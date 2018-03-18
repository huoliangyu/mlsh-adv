# https://stackoverflow.com/questions/43971138/python-plotting-colored-grid-based-on-values

import time
import datetime
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np

def get_timestamp():
  return datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d%H%M%S')

def visualize_fourrooms_policy(env, policy):
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
  # print env.goal
  # print '-----------------------------'

  for s in xrange(env.nS):
    if env.desc[s / nrow][s % ncol] in b'X':
      policy[s] = -1

  policy[env.goal] = g

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
  direction = {-2:u'\u2606', -1:'b', 0:u'\u2190', 1:u'\u2193', 2:u'\u2192', 3:u'\u2191'}
  arrow_offset = 0.375 if nrow == 21 else 0.2
  start_offset = 0.03
  if nrow == 13:
    start_offset = 0.001
  elif nrow == 21:
    start_offset = 0

  for r in xrange(nrow):
    for c in xrange(ncol):
      s = r * ncol + c
      if s == env.start:
        # plt.text(c - arrow_offset, r + arrow_offset, u'\u9898', fontsize=12, color='white')
        plt.scatter(c - start_offset, r + start_offset, s=80, facecolors='none', edgecolors='w')

      if s == env.goal:
        plt.scatter(c - start_offset, r + start_offset, s=80, marker=(5, 1), facecolors='none', edgecolors='w')

      if env.desc[r][c] not in b'XGH' and s != env.goal:
        plt.text(c - arrow_offset, r + arrow_offset, direction[data[r][c]], fontsize=12, color='white')

  # plt.show()
  plt.savefig('plots/fourrooms_tabq_subpolicies_%s.png' % get_timestamp())



def visualize_fourrooms_multi_policy(envs, policies):
    fig, ax = plt.subplots(1, len(policies))
    for i in range(len(policies)):
        env = envs[i]
        nrow, ncol = env.desc.shape

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

        for s in xrange(env.nS):
            if env.desc[s / nrow][s % ncol] in b'X':
                policies[i][s] = -1

        policies[i][env.goal] = g

        data = np.array_split(policies[i], nrow)

        # print(data)

        # create discrete colormap
        cmap = colors.ListedColormap(
            ['green', 'black', 'red', 'blue', 'pink', 'cyan'])
        bounds = [-2, -1, 0, 1, 2, 3, 4]
        norm = colors.BoundaryNorm(bounds, cmap.N)

        ax[i].imshow(data, cmap=cmap, norm=norm)

        # draw gridlines
        ax[i].grid(which='major', axis='both', linestyle='-', color='k',
                   linewidth=2)
        ax[i].set_xticks(np.arange(-.5, ncol, 1));
        ax[i].set_yticks(np.arange(-.5, nrow, 1));
        ax[i].xaxis.set_ticklabels([])
        ax[i].yaxis.set_ticklabels([])

        # plt.text(-0.5, -0.5, 'u', fontsize=12)
        # plt.text(2.5, 2.5, 'd', fontsize=12)
        # plt.text(9, 9, 'l', fontsize=12)
        direction = {
            -2: u'\u2606', -1: 'b', 0: u'\u2190', 1: u'\u2193', 2: u'\u2192',
            3: u'\u2191'
        }
        arrow_offset = 0.375 if nrow == 21 else 0.2
        start_offset = 0.03
        if nrow == 13:
            start_offset = 0.001
        elif nrow == 21:
            start_offset = 0

        for r in xrange(nrow):
            for c in xrange(ncol):
                s = r * ncol + c
                if s == env.start:
                    # plt.text(c - arrow_offset, r + arrow_offset, u'\u9898',
                    # fontsize=12, color='white')
                    ax[i].scatter(c - start_offset, r + start_offset, s=80,
                                  facecolors='none', edgecolors='w')

                if s == env.goal:
                    ax[i].scatter(c - start_offset, r + start_offset, s=80,
                                  marker=(5, 1), facecolors='none',
                                  edgecolors='w')

                if env.desc[r][c] not in b'XGH' and s != env.goal:
                    ax[i].text(c - arrow_offset, r + arrow_offset,
                               direction[data[r][c]], fontsize=12,
                               color='white')

    plt.tight_layout()