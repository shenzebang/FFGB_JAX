from utils.api import ServerHyperParams, Batch, FedAlgorithm, FFGBDistillServerState, Classifier, StaticFns
import jax
from utils.test import test_fn
from typing import List
from utils.data_split import data_split
import jax.numpy as jnp


# one round of functional federated learning.
def run_one_round(
        server_state: FFGBDistillServerState,
        fedalg: FedAlgorithm,
        data_train_tuple: List[Batch],
        data_distill: Batch,
        key: jax.random.PRNGKey,
        hyperparams: ServerHyperParams,
        static_fns: StaticFns
):
    """Runs one round of functional federated learning."""

    key, subkey = jax.random.split(key)

    client_indxs = fedalg.sampler(hyperparams.num_clients, hyperparams.num_sampled_clients, subkey)
    print("sampled clients: {}".format(client_indxs))

    print("making datasets for sampled clients.")
    sampled_x = jnp.stack([data_train_tuple[client_indx].x for client_indx in list(client_indxs)], axis=0)
    sampled_y = jnp.stack([data_train_tuple[client_indx].y for client_indx in list(client_indxs)], axis=0)
    sampled_dss = Batch(x=sampled_x, y=sampled_y)
    client_states = jax.vmap(fedalg.client_init, in_axes=[0, None])(sampled_dss, server_state)


    # params_list = server_classifier.params_list
    # weight_list = server_classifier.weight_list
    # num_ensembles = server_classifier.num_ensembles

    opts_list = []
    weights_list = []
    for local_step in range(hyperparams.num_local_steps):
        print(local_step)
        key, subkey = jax.random.split(key)
        new_opts, new_weights, client_state = fedalg.vclient_step(sampled_dss, client_states, subkey)

        opts_list.append(new_opts)
        weights_list.append(new_weights/hyperparams.num_sampled_clients)


    params = server_state.classifier.params
    weights = server_state.classifier.weights
    for _opts, _weights in zip(opts_list, weights_list):
        params = jax.tree_multimap(lambda param_1, param_2: jnp.concatenate([param_1, param_2], axis=0), params, _opts.target)
        weights = jax.tree_multimap(lambda weight_1, weight_2: jnp.concatenate([weight_1, weight_2], axis=0), weights, _weights)

    new_classifier = Classifier(params=params, weights=weights, is_empty=False)
    # update server classifier and state.
    print("Start server step of round %3d (distillation)" % server_state.global_round)
    server_state = fedalg.server_step(new_classifier, data_distill, server_state, key)
    print("Finish server step")

    return server_state


def functional_federated_learning(
        model,
        data_train: Batch,
        data_distill: Batch,
        data_test: Batch,
        fedalg: FedAlgorithm,
        hyperparams: ServerHyperParams,
        static_fns: StaticFns,
        key
):
    """Calls run_one_round repeatedly and tracks metrics. Returns final test accuracy and number of rounds run."""
    # prepare the dataset
    assert data_train.x.shape[0] % hyperparams.num_clients == 0

    # split the train dataset
    data_train_tuple = data_split(data_train, hyperparams.num_clients, hyperparams.s)
    # data_distill = jax.tree_map(lambda l: jnp.expand_dims(l, axis=0), data_distill)
    key, subkey = jax.random.split(key)
    server_state = fedalg.server_init(subkey)

    keys = jax.random.split(key, hyperparams.num_rounds)
    global_round = 0
    print("\nStarting training. First round is slower than the rest.")
    while global_round < hyperparams.num_rounds:
        print("\nrunning round {}".format(global_round))
        key = keys[global_round]
        server_state = run_one_round(server_state, fedalg, data_train_tuple,
                                                        data_distill, key, hyperparams, static_fns)


        # compute test metrics.
        print("evaluating testing accuracy")
        correct = test_fn(hyperparams, server_state.classifier, data_test)
        print("Round %2d, testing accuracy %.3f" % (global_round, correct))

        global_round += 1
        # todo: log test metrics
