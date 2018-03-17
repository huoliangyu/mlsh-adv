"""
A classic four room problem described in intro to RL book chap 9.
Code adapted from frozen lake environment.
"""

import logging
import sys

import numpy as np
from gym import utils
from gym.envs.toy_text import discrete

logger = logging.getLogger(__name__)

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

MAPS = {
    "small": [
        "XXXXXXXXX",
        "XOOOXOOOX",
        "XOOOOOOOX",
        "XOOOXOOOX",
        "XXOXXXOXX",
        "XOOOXOOOX",
        "XOOOOOOOX",
        "XOOOXOOOX",
        "XXXXXXXXX",
    ],
    "medium": [
        "XXXXXXXXXXXXX",
        "XOOOOOXOOOOOX",
        "XOOOOOXOOOOOX",
        "XOOOOOOOOOOOX",
        "XOOOOOXOOOOOX",
        "XOOOOOXOOOOOX",
        "XXXOXXXXXOXXX",
        "XOOOOOXOOOOOX",
        "XOOOOOXOOOOOX",
        "XOOOOOOOOOOOX",
        "XOOOOOXOOOOOX",
        "XOOOOOXOOOOOX",
        "XXXXXXXXXXXXX",
    ],
    "large": [
        "XXXXXXXXXXXXXXXXXXXXX",
        "XOOOOOOOOOXOOOOOOOOOX",
        "XOOOOOOOOOXOOOOOOOOOX",
        "XOOOOOOOOOXOOOOOOOOOX",
        "XOOOOOOOOOXOOOOOOOOOX",
        "XOOOOOOOOOOOOOOOOOOOX",
        "XOOOOOOOOOXOOOOOOOOOX",
        "XOOOOOOOOOXOOOOOOOOOX",
        "XOOOOOOOOOXOOOOOOOOOX",
        "XOOOOOOOOOXOOOOOOOOOX",
        "XXXXXOXXXXXXXXXOXXXXX",
        "XOOOOOOOOOXOOOOOOOOOX",
        "XOOOOOOOOOXOOOOOOOOOX",
        "XOOOOOOOOOXOOOOOOOOOX",
        "XOOOOOOOOOXOOOOOOOOOX",
        "XOOOOOOOOOOOOOOOOOOOX",
        "XOOOOOOOOOXOOOOOOOOOX",
        "XOOOOOOOOOXOOOOOOOOOX",
        "XOOOOOOOOOXOOOOOOOOOX",
        "XOOOOOOOOOXOOOOOOOOOX",
        "XXXXXXXXXXXXXXXXXXXXX",
    ],
}

def sample_from_ones(dist):
    # print dist
    # print np.nonzero(dist)[0]
    return np.random.choice(np.nonzero(dist)[0])

class Fourrooms(discrete.DiscreteEnv):
    """
    The agent must go through the doors to exit.
    An example of a 9X9 world would be:

    XXGXXXXXX
    XOOOXOOOX
    XOOOOOOOX
    XOOOXOOOX
    XXOXXXOXX
    XOOOXOOOX
    XOOOOOOOX
    XOOOXOOOX
    XXXXXXXXX

    X : Walls
    G : goal
    O : Normal floor, any normal ground can be a starting point.

    """
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self, desc=None, map_name='small'):
        if desc is None and map_name is None:
            raise ValueError('Must provide either desc or map_name')
        elif desc is None:
            desc = MAPS[map_name]
        self.desc = desc = np.asarray(desc, dtype='c')
        self.nrow, self.ncol = nrow, ncol = desc.shape

        self.nA = nA = 4
        self.nS = nS = nrow * ncol

        self.do_not_randomly_reset = False

        self.goal = 2
        self.start = nS - self.ncol - 2

        # print desc
        # print 'self.goal = %s' % self.goal
        # print 'self.start = %s' % self.start
        # print 'self.nrow = %s' % nrow
        # print 'self.ncol = %s' % ncol
        # print 'self.start / nrow = %s' % (self.start / nrow)
        # print 'self.start / ncol = %s' % (self.start / ncol)


        self.isd_all = np.array(desc == b'O').astype('float64').ravel()
        self.gsd_edge = np.array(desc == b'X').astype('float64').ravel()
        # exclude goal on not-reachable-part
        self.gsd_edge[0] = 0
        self.gsd_edge[ncol - 1] = 0
        self.gsd_edge[nS - ncol] = 0
        self.gsd_edge[nS - 1] = 0
        self.gsd_edge[(ncol - 1) / 2] = 0
        self.gsd_edge[(nS - 1) / 2] = 0
        self.gsd_edge[ncol * (nrow - 1) / 2] = 0
        self.gsd_edge[ncol * (nrow - 1) / 2 + ncol - 1] = 0
        self.gsd_edge[nS - 1 - (ncol - 1) / 2] = 0

        self.gsd_all = np.ones_like(desc).astype('float64').ravel()
        # exclude goal on edge
        self.gsd_all[0] = 0
        self.gsd_all[ncol - 1] = 0
        self.gsd_all[nS - ncol] = 0
        self.gsd_all[nS - 1] = 0
        self.gsd_all[(ncol - 1) / 2] = 0
        self.gsd_all[(nS - 1) / 2] = 0
        self.gsd_all[ncol * (nrow - 1) / 2] = 0
        self.gsd_all[ncol * (nrow - 1) / 2 + ncol - 1] = 0
        self.gsd_all[nS - 1 - (ncol - 1) / 2] = 0

        # desc[self.start / nrow][self.start % ncol] = 'S'
        # desc[self.goal / nrow][self.goal % ncol] = 'G'

        # self.isd = np.array(desc == b'S').astype('float64').ravel()
        self.isd = np.zeros(desc.shape).astype('float64').ravel()
        self.isd[self.start] = 1.0

        # print '-------------------'
        # print desc
        # print '-------------------'
        # print np.array(desc == b'S')
        # print '-------------------'
        # print np.array(desc == b'S').astype('float64')
        # print '-------------------'
        # print np.array(desc == b'S').astype('float64').ravel()
        # print '-------------------'
        # print self.isd
        # print '-------------------'
        # print np.array_split(self.isd, nrow)
        # print '-------------------'
        # print np.array_split(self.isd_all, nrow)
        # print '-------------------'
        # print np.array_split(self.gsd_edge, nrow)
        # print '-------------------'
        # print np.array_split(self.gsd_all, nrow)
        # print '-------------------'

        P = self.generate_transitions()

        super(Fourrooms, self).__init__(nS, nA, P, self.isd)

    def to_s(self, row, col):
        return row * self.ncol + col

    def inc(self, row, col, a):
        orig_row = row
        orig_col = col
        if a == 0:  # left
            col = max(col - 1, 0)
        elif a == 1:  # down
            row = min(row + 1, self.nrow - 1)
        elif a == 2:  # right
            col = min(col + 1, self.ncol - 1)
        elif a == 3:  # up
            row = max(row - 1, 0)
        is_wall = (self.desc[row][col] == b'X' and self.to_s(row, col) != self.goal)
        if is_wall:
            return (orig_row, orig_col)
        return (row, col)

    def generate_transitions(self):
        P = {s: {a: [] for a in range(self.nA)} for s in range(self.nS)}
        for row in range(self.nrow):
            for col in range(self.ncol):
                s = self.to_s(row, col)
                for a in range(self.nA):
                    li = P[s][a]

                    if s == self.goal:
                        li.append((1.0, s, 0, True))
                    else:
                        # TODO(yejiayu): Add stochastic case.
                        newrow, newcol = self.inc(row, col, a)
                        newstate = self.to_s(newrow, newcol)
                        done = (newstate == self.goal)
                        rew = (newstate == self.goal) * self.nrow * self.ncol - 1
                        li.append((1.0, newstate, rew, done))


        # print '-------------------'
        # print P
        # print '-------------------'

        return P

    # Hack: use seed as a map of code
    def reset(self, seed=None):
        # print 'seed=%s' % seed
        if seed is None:
            if self.do_not_randomly_reset:
                self.s = self.start
                self.lastaction=None
                print '---------------------------------'
                print 'setting start = %s' % self.start
                print 'setting goal = %s' % self.goal
                print '---------------------------------'
                return self.s

            return super(Fourrooms, self).reset()

        elif 'fixedgoal' in seed:

            self.goal = seed['fixedgoal']
            self.P = self.generate_transitions()

            self.start = sample_from_ones(self.isd_all)
            while self.start == self.goal:
                self.start = sample_from_ones(self.isd_all)
            self.lastaction=None

            print '---------------------------------'
            print 'setting start = %s' % self.start
            print 'setting goal = %s' % self.goal
            print '---------------------------------'
            self.s = self.start
            self.do_not_randomly_reset = True
            return self.s

        elif 'fixedstart' in seed:

            self.start = seed['fixedstart']
            self.lastaction=None

            print '---------------------------------'
            print 'setting start = %s' % self.start
            print 'setting goal = %s' % self.goal
            print '---------------------------------'
            self.s = self.start
            self.do_not_randomly_reset = True
            return self.s

        elif 'fixedstart+goal:start' in seed:

            self.start = seed['fixedstart+goal:start']
            self.goal = seed['fixedstart+goal:goal']
            self.P = self.generate_transitions()
            self.lastaction=None

            print '---------------------------------'
            print 'setting start = %s' % self.start
            print 'setting goal = %s' % self.goal
            print '---------------------------------'
            self.s = self.start
            self.do_not_randomly_reset = True
            return self.s

        elif 'start' in seed:
            np.random.seed(seed['start'])

            self.start = sample_from_ones(self.isd_all)
            while self.start == self.goal:
                self.start = sample_from_ones(self.isd_all)
            self.lastaction=None

            print '---------------------------------'
            print 'using seed = %s' % seed['start']
            print 'setting start = %s' % self.start
            print 'setting goal = %s' % self.goal
            print '---------------------------------'
            self.s = self.start
            self.do_not_randomly_reset = True
            return self.s

        elif 'goal-on-edge' in seed:
            np.random.seed(seed['goal-on-edge'])

            self.start = sample_from_ones(self.isd, self)
            self.lastaction=None
            self.goal = sample_from_ones(self.gsd_edge)
            # not needed
            # while self.goal == self.start:
            #     self.goal = sample_from_ones(self.gsd_edge)
            self.P = self.generate_transitions()

            print '---------------------------------'
            print 'using seed = %s' % seed['goal-on-edge']
            print 'setting start = %s' % self.start
            print 'setting goal = %s' % self.goal
            print '---------------------------------'
            self.s = self.start
            self.do_not_randomly_reset = True
            return self.s

        elif 'goal-on-all' in seed:
            np.random.seed(seed['goal-on-all'])

            self.start = sample_from_ones(self.isd)
            self.lastaction=None
            self.goal = sample_from_ones(self.gsd_all)
            while self.goal == self.start:
                self.goal = sample_from_ones(self.gsd_edge)
            self.P = self.generate_transitions()

            print '---------------------------------'
            print 'using seed = %s' % seed['goal-on-all']
            print 'setting start = %s' % self.start
            print 'setting goal = %s' % self.goal
            print '---------------------------------'
            self.s = self.start
            self.do_not_randomly_reset = True
            return self.s

        elif 'start+goal-on-edge' in seed:
            np.random.seed(seed['start+goal-on-edge'])

            self.start = sample_from_ones(self.isd_all)
            self.lastaction=None
            self.goal = sample_from_ones(self.gsd_edge)
            # not needed
            # while self.goal == self.start:
            #     self.goal = sample_from_ones(self.gsd_edge)
            self.P = self.generate_transitions()

            print '---------------------------------'
            print 'using seed = %s' % seed['start+goal-on-edge']
            print 'setting start = %s' % self.start
            print 'setting goal = %s' % self.goal
            print '---------------------------------'
            self.s = self.start
            self.do_not_randomly_reset = True
            return self.s

        elif 'start+goal-on-all' in seed:
            np.random.seed(seed['start+goal-on-all'])

            self.start = sample_from_ones(self.isd_all)
            self.lastaction=None
            self.goal = sample_from_ones(self.gsd_all)
            while self.goal == self.start:
                self.goal = sample_from_ones(self.gsd_edge)
            self.P = self.generate_transitions()

            print '---------------------------------'
            print 'using seed = %s' % seed['start+goal-on-all']
            print 'setting start = %s' % self.start
            print 'setting goal = %s' % self.goal
            print '---------------------------------'
            self.s = self.start
            self.do_not_randomly_reset = True
            return self.s

        else:
            raise ValueError('invalid seed specifi`ion: %s' % seed)

    # def randomizeCorrect(self):
    #     # self.realgoal = np.random.choice([68, 69, 70, 71, 72, 78, 79, 80, 81, 82, 88, 89, 90, 91, 92, 93, 99, 100, 101, 102, 103])
    #     self.realgoal = np.random.choice([68, 80, 90, 103])
    #     self.realgoal = 103
    #     pass

    def _render(self, mode='human', close=False):
        if close:
            return
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        row, col = self.s // self.ncol, self.s % self.ncol
        desc = self.desc.tolist()
        desc = [[c.decode('utf-8') for c in line] for line in desc]
        desc[row][col] = utils.colorize(desc[row][col], "red", highlight=True)
        if self.lastaction is not None:
            outfile.write("  ({})\n".format(["Left", "Down", "Right",
                                             "Up"][self.lastaction]))
        else:
            outfile.write("\n")
        outfile.write("\n".join(''.join(line) for line in desc) + "\n")

        if mode != 'human':
            return outfile
