### MDP Value Iteration and Policy Iteratoin
import os
import numpy as np
import gym
from visualize import visualize_fourrooms_policy
from test_env import *

np.set_printoptions(precision=3)

def value_converged(V, new_V, tol=1e-3):
    # max/Inf norm of diff < tol.  https://piazza.com/class/jbb728cf5s84rv?cid=168
    return np.all(np.abs(V - new_V) < tol)

def policy_converged(policy, new_policy):
    # L1 norm of diff == 0
    return np.sum(np.abs(policy - new_policy)) == 0

def bellman_backup(V, P, nS, nA, gamma=0.9):
    """Perform one bellman backup operation on Value function V.

    Parameters
    ----------
    V:  np.ndarray
        Value function, the primary operand of this Bellman backup operation
    P: dictionary
        It is from gym.core.Environment
        P[state][action] is tuples with (probability, nextstate, reward, terminal)
    nS: int
        number of states
    nA: int
        number of actions
    gamma: float
        Discount factor. Number in range [0, 1)

    Returns
    -------
    new V: np.ndarray
        The resulting new value function after Bellman backup operation.
    update a: np.ndarray
        Which actions update the Value function (like a policy)
    """
    new_V = np.array([-float('inf') for s in xrange(nS)])
    update_a = np.array([np.random.choice(nA) for s in xrange(nS)])

    for s in xrange(nS):
        for a in xrange(nA):
            # Q(s, a) = E[r_pi(s,a)] + gamma * sum_{s'}[P_pi(s'|s,a) * V(s')]
            #         = sum_{s'|s,a} { P_pi(s'|s,a) * r(s'|s, a) } + gamma * sum_{s'|s,a} { [P_pi(s'|s,a) * V(s')] }
            #         = sum_{s'|s,a} { P_pi(s'|s,a) * [ r(s'|s, a) + gamma * * V(s') ] }
            Q_s_a = sum([
                probability * (reward + gamma * V[nextstate]) for probability, nextstate, reward, _ in P[s][a]
            ])

            if Q_s_a > new_V[s]:
                update_a[s] = a
                new_V[s] = Q_s_a
            # elif Q_s_a == new_V[s]:
            #     # tie breaker
            #     if np.random.uniform() < 0.5:
            #         update_a[s] = a

    return new_V, update_a

def value_iteration(P, nS, nA, gamma=0.9, max_iteration=1000, tol=1e-3):
    """
    Learn value function and policy by using value iteration method for a given
    gamma and environment.

    Parameters:
    ----------
    P: dictionary
        It is from gym.core.Environment
        P[state][action] is tuples with (probability, nextstate, reward, terminal)
    nS: int
        number of states
    nA: int
        number of actions
    gamma: float
        Discount factor. Number in range [0, 1)
    max_iteration: int
        The maximum number of iterations to run before stopping. Feel free to change it.
    tol: float
        Determines when value function has converged.
    Returns:
    ----------
    value function: np.ndarray
    policy: np.ndarray
    num iter: # of iterations before convergence
    """
    V = np.zeros(nS)
    policy = np.zeros(nS, dtype=int)
    ############################
    # YOUR IMPLEMENTATION HERE #
    ############################
    for num_iter in xrange(max_iteration):
        new_V, new_policy = bellman_backup(V, P, nS, nA, gamma=gamma)

        if value_converged(V, new_V, tol=tol):
            break

        V = new_V
        policy = new_policy

    _, policy = bellman_backup(V, P, nS, nA, gamma=gamma)

    return V, policy, (num_iter + 1)


def randomize_fourrooms_task(env):
  nrow, ncol = env.desc.shape

  # env.reset(seed={'start':234})
  # env.reset(seed={'fixedstart+goal:start':(1 + ncol), 'fixedstart+goal:goal':(env.nS -3)})
  # env.reset(seed={'goal-on-edge':234})
  # env.reset(seed={'goal-on-all':22})
  # env.reset(seed={'start+goal-on-edge':234})
  # env.reset(seed={'start+goal-on-all':234})
  env.reset(seed={'start+goal-on-all':234})


# Feel free to run your own debug code in main!
if __name__ == "__main__":    
  if not os.path.exists('./plots'):
    os.makedirs('./plots')

  # env = gym.make('Fourrooms-small-v0')
  # randomize_fourrooms_task(env)
  # V_vi, p_vi, niter_vi = value_iteration(env.P, env.nS, env.nA, gamma=1.0, max_iteration=1000, tol=1e-3)

  # env = gym.make('Fourrooms-medium-v0')
  # randomize_fourrooms_task(env)
  # V_vi, p_vi, niter_vi = value_iteration(env.P, env.nS, env.nA, gamma=1.0, max_iteration=1000, tol=1e-3)

  env = gym.make('Fourrooms-large-v0')
  randomize_fourrooms_task(env)
  V_vi, p_vi, niter_vi = value_iteration(env.P, env.nS, env.nA, gamma=1.0, max_iteration=1000, tol=1e-3)

  print('Value Iteration')
  print 'V_vi =\n%s' % np.resize(V_vi, (env.nrow, env.ncol))
  print 'p_vi =\n%s' % np.resize(p_vi, (env.nrow, env.ncol))

  visualize_fourrooms_policy(env, p_vi)
