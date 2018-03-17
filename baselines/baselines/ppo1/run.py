#!/usr/bin/env python3

import numpy as np

from baselines import logger
from baselines.common import tf_util as U
from baselines.common.cmd_util import make_mujoco_env
from config import config


def train(env_id, num_timesteps, seed, config):
    if config.algorithm == 'RecurrentMLSHV7':
        from baselines.ppo1.recurrent_mlsh_v7 import RecurrentMLSHV7 as policy
        from baselines.ppo1 import pposgd_simple_alt as pposgd_simple
    elif config.algorithm == 'RecurrentMLSHV8':
        from baselines.ppo1.lstm_fc_policy import LSTMFCPolicy as policy
        from baselines.ppo1 import pposgd_simple
    elif config.algorithm == 'mlp_policy':
        from mlp_policy import MlpPolicy as policy
        from baselines.ppo1 import pposgd_simple
    U.make_session(num_cpu=32).__enter__()

    def policy_fn(name, ob_space, ac_space):
        return policy(name=name, ob_space=ob_space, ac_space=ac_space,
                      hid_size=config.hid_size,
                      num_hid_layers=config.num_hid_layers, config=config)

    env = make_mujoco_env(env_id, seed)
    pposgd_simple.learn(env, policy_fn, max_timesteps=num_timesteps,
                        timesteps_per_actorbatch=2048, clip_param=0.2,
                        entcoeff=0.0, optim_epochs=10,
                        optim_stepsize=config.learning_rate,
                        optim_batchsize=config.batch_size, gamma=config.gamma,
                        lam=0.95, schedule='linear')
    env.close()


def main():
    c = config('RecurrentMLSHV7')
    logger.configure()
    logger.log(c.output_path)
    train(config.env_name, num_timesteps=3e6, seed=np.random.randint(0, 100000),
          config=c)


if __name__ == '__main__':
    main()
