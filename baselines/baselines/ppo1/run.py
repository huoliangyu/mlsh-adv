#!/usr/bin/env python3

from baselines import logger
from baselines.common import tf_util as U
from baselines.common.cmd_util import make_mujoco_env, mujoco_arg_parser
# from baselines.ppo1.recurrent_mlsh_v8 import LSTMFCPolicy as policy
from baselines.ppo1.recurrent_mlsh_v7 import RecurrentMLSHV7 as policy
# from mlp_policy import MlpPolicy as policy
from config import config


def train(env_id, num_timesteps, seed):
    # from baselines.ppo1 import pposgd_simple
    from baselines.ppo1 import pposgd_simple_alt as pposgd_simple
    U.make_session(num_cpu=32).__enter__()

    def policy_fn(name, ob_space, ac_space):
        return policy(name=name, ob_space=ob_space, ac_space=ac_space,
                      hid_size=64, num_hid_layers=2)

    env = make_mujoco_env(env_id, seed)
    pposgd_simple.learn(env, policy_fn, max_timesteps=num_timesteps,
                        timesteps_per_actorbatch=2048, clip_param=0.2,
                        entcoeff=0.0, optim_epochs=10, optim_stepsize=3e-4,
                        optim_batchsize=config.batch_size, gamma=0.99, lam=0.95,
                        schedule='linear')
    env.close()


def main():
    c = config('RecurrentMLSHV7')
    args = mujoco_arg_parser().parse_args()
    logger.configure()
    logger.log(c.output_path)
    train(config.env_name, num_timesteps=3e6, seed=args.seed)


if __name__ == '__main__':
    main()
