# https://stackoverflow.com/questions/43971138/python-plotting-colored-grid
# -based-on-values
import datetime
import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors


def get_timestamp():
    return datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d%H%M%S')


def visualize_fourrooms_master_policy(envs, policies):
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
        cmap = colors.ListedColormap(['green', 'black', 'red', 'blue'])
        bounds = [-2, -1, 0, 1]
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
            -2: u'\u2606', -1: 'b', 0: '0', 1: '1'
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


def visualize_fourrooms_sub_policy(envs, policies):
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
