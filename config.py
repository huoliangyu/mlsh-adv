import tensorflow as tf


class config():
    # TODO(bohan): Make config argparse flags in the main python files.
    def __init__(self, algorithm=None):
        if not algorithm:
            raise Exception()
        self.algorithm = algorithm

        output_path = "results/%s-bs=%s-algo=%s-usebaseline=%s-lr=%s" \
                      "-baselinelayers=%sx%s-num_sub=%s-maxeps=%s-min" \
                      "eps=%s-sub_index=%s-freezesub=%s-uniquestr" \
                      "=%s-numsublayers=%s-nummasterlayers=%s-max_num_sub" \
                      "=%s-wa=%s-sub_net=%s" \
                      "-master_net=%s" \
                      "/" % (self.env_name, self.batch_size, self.algorithm,
                             self.use_baseline, self.learning_rate,
                             self.n_layers, self.baseline_layer_size,
                             self.num_sub_policies, self.max_epsilon,
                             self.min_epsilon, self.sub_policy_index,
                             self.freeze_sub_policy, self.unique_key,
                             self.num_sub_policy_layers, self.num_master_layers,
                             self.max_num_sub_policies, self.weight_average,
                             self.sub_policy_network, self.master_network)

        self.model_output = output_path + "model.weights/"
        self.log_path = output_path + "log.txt"
        self.plot_output = output_path + "scores.png"
        self.record_path = output_path
        self.output_path = output_path

        plot_output_dir = "./plots"

    #=================== HRL ===================
    do_meta_learning = False
    num_meta_learning_training_tasks = 20
    # TODO message: Jiayu, this is where you toggle doing viz or not
    visualize_master_policy = True
    visualize_sub_policies = True
    freeze_sub_policy = True
    sub_policy_index = -1  # -1 means activates master policy

    max_num_sub_policies = 4
    num_sub_policies = 2

    sub_policy_network = 'LSTM'
    master_network = 'LSTM'
    num_sub_policy_layers = 1
    num_master_layers = 1
    #=====================================================

    #=================== Meta-Learning ===================
    do_meta_learning = False
    num_meta_learning_training_tasks = 20

    # TODO: delte this, four rooms could be randomized.
    def get_env_name(self):
        # return "Fourrooms-v" + str(np.random.randint(0, 2))
        return "Fourrooms-small-v0"
    #=====================================================

    # env_name = "CartPole-v0"
    # env_name = 'BipedalWalker-v2'
    # env_name = "InvertedPendulum-v1"
    env_name = "Fourrooms-small-v0"
    # env_name = "Fourrooms-medium-v0"
    # env_name = "Fourrooms-large-v0"

    # env_name = "HalfCheetah-v1"
    # env_name = "Ant-v1"
    # env_name = "Ant-v2"

    batch_size_by_env = {
        "CartPole-v0": 64,
        "Fourrooms-small-v0": 1000,
        "Fourrooms-medium-v0": 1000,
        "Fourrooms-large-v0": 1000,
        "HalfCheetah-v1": 50000,
        "Ant-v1": 100000,
        "Ant-v2": 64,
        "BipedalWalker-v2": 256
    }

    lr_by_env = {
        "CartPole-v0": 3e-2,
        "Fourrooms-small-v0": 3e-2,
        "Fourrooms-medium-v0": 3e-2,
        "Fourrooms-large-v0": 3e-2,
        "HalfCheetah-v1": 3e-2,
        "Ant-v1": 3e-2,
        "Ant-v2": 3e-2,
        "BipedalWalker-v2": 3e-2
    }

    gamma_by_env = {
        "CartPole-v0": 1.0,
        "Fourrooms-small-v0": 1.0,
        "Fourrooms-medium-v0": 1.0,
        "Fourrooms-large-v0": 1.0,
        "HalfCheetah-v1": 0.9,
        "Ant-v1": 0.8,
        "Ant-v2": 0.8,
        "BipedalWalker-v2": 1.0
    }

    num_batches_by_env = {
        "CartPole-v0": 100,
        "Fourrooms-small-v0": 1000,
        "Fourrooms-medium-v0": 1000,
        "Fourrooms-large-v0": 5,
        "HalfCheetah-v1": 1000,
        "Ant-v1": 1000,
        "Ant-v2": 1000,
        "BipedalWalker-v2": 1000
    }


    # recover_checkpoint_path = None
    recover_checkpoint_path = "results/Fourrooms-large-v0-bs=1000-algo" \
                              "=RecurrentMLSH-v2-usebaseline=True-lr=0.03-baselinelayers=4x32-num_sub=2-maxeps=0.0-mineps=0.0-sub_index=-1-freezesub=False-uniquestr=d-numsublayers=1-nummasterlayers=1-max_num_sub=4-wa=False-sub_net=LSTM-master_net=LSTM/model.ckpt-4"

    record = False
    unique_key = "d"
    render = False
    max_epsilon = 0.0
    min_epsilon = 0.0
    num_batches = num_batches_by_env[env_name]
    batch_size = batch_size_by_env[env_name]
    max_ep_len = min(10000000, batch_size)
    learning_rate = lr_by_env[env_name]
    gamma = gamma_by_env[env_name]
    use_baseline = True
    normalize_advantage = True
    n_layers = 4
    baseline_layer_size = 32
    weight_average = False
    activation = tf.nn.relu

    assert max_ep_len <= batch_size
    if max_ep_len < 0:
        max_ep_len = batch_size

    record_freq = 25
    summary_freq = 1
