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
                      "=%s-weighted_avg=%s-sub_net=%s" \
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

    env_name = "CartPole-v0"
    # env_name = "InvertedPendulum-v1"
    # env_name = "Fourrooms-v1"
    # env_name = "HalfCheetah-v1"
    # env_name = "Ant-v1"

    batch_size_by_env = {
        "CartPole-v0": 1000, "Fourrooms-v1": 1000, "HalfCheetah-v1": 50000, \
        "Ant-v1": 60000
    }

    lr_by_env = {
        "CartPole-v0": 3e-2, "Fourrooms-v1": 3e-2, "HalfCheetah-v1": 3e-2,
        "Ant-v1": 3e-2
    }

    gamma_by_env = {
        "CartPole-v0": 1.0, "Fourrooms-v1": 1.0, "HalfCheetah-v1": 0.9,
        "Ant-v1": 0.8
    }

    num_batches_by_env = {
        "CartPole-v0": 1000, "Fourrooms-v1": 100, "HalfCheetah-v1": 1000,
        "Ant-v1": 1000
    }

    examine_master = False
    recover_checkpoint_path = None
    record = False
    unique_key = ""
    render = False
    max_epsilon = 0.0
    min_epsilon = 0.0
    freeze_sub_policy = False
    sub_policy_index = -1  # -1 means activates master policy
    num_batches = num_batches_by_env[env_name]
    batch_size = batch_size_by_env[env_name]
    max_ep_len = min(1000000, batch_size)
    learning_rate = lr_by_env[env_name]
    gamma = gamma_by_env[env_name]
    use_baseline = True
    normalize_advantage = True
    n_layers = 4
    baseline_layer_size = 32
    max_num_sub_policies = 20
    num_sub_policies = 4

    sub_policy_network = 'LSTM'
    master_network = 'LSTM'
    num_sub_policy_layers = 8
    num_master_layers = 8

    weight_average = False
    activation = tf.nn.relu

    assert max_ep_len <= batch_size
    if max_ep_len < 0:
        max_ep_len = batch_size

    record_freq = 25
    summary_freq = 1
