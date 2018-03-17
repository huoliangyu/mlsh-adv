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

def policy_evaluation(P, nS, nA, policy, gamma=0.9, max_iteration=1000, tol=1e-3):
    """Evaluate the value function from a given policy.

    Parameters
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
    policy: np.array
        The policy to evaluate. Maps states to actions.
    max_iteration: int
        The maximum number of iterations to run before stopping. Feel free to change it.
    tol: float
        Determines when value function has converged.
    Returns
    -------
    value function: np.ndarray
        The value function from the given policy.
    """
    ############################
    # YOUR IMPLEMENTATION HERE #
    ############################
    # print P

    V = np.zeros(nS)
    # print 'V = %s' % V
    for num_iter in xrange(max_iteration):
        new_V = np.zeros(nS)
        for s in xrange(nS):
            # V(s) = E[r_pi(s)] + gamma * sum_{s'}[P_pi(s'|s) * V(s')]
            #      = sum_{a|s} { sum_{s'|s,a} P_pi(s'|s,a) * r(s'|s, a) } + gamma * sum_{a|s} { sum_{s'|s,a} [P_pi(s'|s,a) * V(s')] }
            #      = sum_{a|s} { sum_{s'|s,a} P_pi(s'|s,a) * [ r(s'|s, a) + gamma  * V(s') ] }
            new_V[s] = sum([
                probability * (reward + gamma * V[nextstate]) for probability, nextstate, reward, _ in P[s][policy[s]]
            ])

        if value_converged(V, new_V, tol=tol):
            break

        V = new_V

    print '# iter (policy_evaluation) = %s' % (num_iter + 1)
    return V

def policy_improvement(P, nS, nA, value_from_policy, policy, gamma=0.9):
    """Given the value function from policy improve the policy.

    Parameters
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
    value_from_policy: np.ndarray
        The value calculated from the policy
    policy: np.array
        The previous policy.

    Returns
    -------
    new policy: np.ndarray
        An array of integers. Each integer is the optimal action to take
        in that state according to the environment dynamics and the
        given value function.
    """
    ############################
    # YOUR IMPLEMENTATION HERE #
    ############################
    improved_policy = np.zeros(nS, dtype='int')

    for s in xrange(nS):
        qmax= -float('inf')
        for a in xrange(nA):
            # Q(s, a) = E[r_pi(s,a)] + gamma * sum_{s'}[P_pi(s'|s,a) * V(s')]
            #         = sum_{s'|s,a} { P_pi(s'|s,a) * r(s'|s, a) } + gamma * sum_{s'|s,a} { [P_pi(s'|s,a) * V(s')] }
            #         = sum_{s'|s,a} { P_pi(s'|s,a) * [ r(s'|s, a) + gamma * * V(s') ] }
            Q_s_a = sum([
                probability * (reward + gamma * value_from_policy[nextstate]) for probability, nextstate, reward, _ in P[s][a]
            ])

            if Q_s_a > qmax:
                improved_policy[s] = a
                qmax = Q_s_a
            # elif Q_s_a == qmax:
            #     # tie breaker
            #     if np.random.uniform() < 0.5:
            #         improved_policy[s] = a

    return improved_policy

def policy_iteration(P, nS, nA, gamma=0.9, max_iteration=20, tol=1e-3):
    """Runs policy iteration.

    You should use the policy_evaluation and policy_improvement methods to
    implement this method.

    Parameters
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
    # print 'P =\n%s' % P
    # print 'P[1] =\n%s' % P[1]
    # print 'P[0][action] = %s' % P[0][1]
    # print 'policy =\n%s' % policy
    # print 'nS = %s' % nS
    # print 'nA = %s' % nA

    # initialize policy randomly for all states
    for s in xrange(nS):
        policy[s] = np.random.choice(nA)

    for num_iter in xrange(max_iteration):
        V = policy_evaluation(P, nS, nA, policy, gamma=gamma, tol=tol)
        improved_policy = policy_improvement(P, nS, nA, V, policy, gamma=gamma)
        # improved_policy = policy_improvement_as_bellman(P, nS, nA, V, policy, gamma=gamma)

        if policy_converged(policy, improved_policy):
            break

        policy = improved_policy
        # print('policy_', num_iter, ' = ', policy)

    return V, policy, (num_iter + 1)


def randomize_fourrooms_task(env):
  nrow, ncol = env.desc.shape

  # env.reset(seed={'start':234})
  # env.reset(seed={'fixedstart+goal:start':(1 + ncol), 'fixedstart+goal:goal':(env.nS -3)})
  # env.reset(seed={'goal-on-edge':234})
  # env.reset(seed={'goal-on-all':22})
  # env.reset(seed={'start+goal-on-edge':234})
  env.reset(seed={'start+goal-on-all':234})

# Feel free to run your own debug code in main!

if __name__ == "__main__":    
  if not os.path.exists('./plots'):
    os.makedirs('./plots')

  env = gym.make('Fourrooms-small-v0')
  randomize_fourrooms_task(env)
  V_pi, p_pi, niter_pi = policy_iteration(env.P, env.nS, env.nA, gamma=1.0, max_iteration=1000, tol=1e-3)

  # env = gym.make('Fourrooms-medium-v0')
  # randomize_fourrooms_task(env)
  # V_pi, p_pi, niter_pi = policy_iteration(env.P, env.nS, env.nA, gamma=1.0, max_iteration=1000, tol=1e-3)

  # env = gym.make('Fourrooms-large-v0')
  # randomize_fourrooms_task(env)
  # V_pi, p_pi, niter_pi = policy_iteration(env.P, env.nS, env.nA, gamma=1.0, max_iteration=1000, tol=1e-3)

  print('Policy Iteration')
  # print 'V_pi =\n%s' % V_pi
  # print 'p_pi =\n%s' % p_pi
  print 'V_vi =\n%s' % np.resize(V_pi, (env.nrow, env.ncol))
  print 'p_vi =\n%s' % np.resize(p_pi, (env.nrow, env.ncol))

  visualize_fourrooms_policy(env, p_pi)
