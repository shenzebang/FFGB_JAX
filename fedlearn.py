from utils.api import ServerHyperParams, Batch, FedAlgorithm, ServerState, Classifier, StaticFns
import jax
from utils.test import test_fn
from typing import List
from utils.data_split import data_split
import jax.numpy as jnp


# one round of functional federated learning.
def run_one_round(
        server_classifier: Classifier,
        fedalg: FedAlgorithm,
        data_train_tuple: List[Batch],
        data_distill: Batch,
        server_state: ServerState,
        key: jax.random.PRNGKey,
        hyperparams: ServerHyperParams,
        static_fns: StaticFns
):
    """Runs one round of functional federated learning."""

    key, subkey = jax.random.split(key)

    client_indxs = fedalg.sampler(hyperparams.num_clients, hyperparams.num_sampled_clients, subkey)
    print("sampled clients: {}".format(client_indxs))

    print("making datasets for sampled clients.")
    sampled_dss = [data_train_tuple[client_indx] for client_indx in list(client_indxs)]

    clients_state = [fedalg.client_init(server_classifier, train_batch, server_state)
                     for train_batch in sampled_dss]


    params_list = server_classifier.params_list
    weight_list = server_classifier.weight_list
    num_ensembles = server_classifier.num_ensembles
    for client_state, batch in zip(clients_state, sampled_dss):
        for local_step in range(hyperparams.num_local_steps):
            print(local_step)
            key, subkey = jax.random.split(key)
            new_params, new_weight, client_state = fedalg.client_step(batch, client_state, subkey)
            params_list.append(new_params)
            weight_list.append(new_weight / hyperparams.num_sampled_clients)
            num_ensembles += 1

    server_ensemble = Classifier(params_list, weight_list, num_ensembles)
    # update server classifier and state.
    server_classifier, server_state = fedalg.server_step(server_ensemble, data_distill, server_state, key)

    return server_classifier, server_state


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

    server_classifier = Classifier(params_list=[], weight_list=[], num_ensembles=0)
    server_state = fedalg.server_init()

    keys = jax.random.split(key, hyperparams.num_rounds)
    global_round = 0
    print("\nStarting training. First round is slower than the rest.")
    while global_round < hyperparams.num_rounds:
        print("\nrunning round {}".format(global_round))
        key = keys[global_round]
        server_classifier, server_state = run_one_round(server_classifier, fedalg, data_train_tuple,
                                                        data_distill, server_state, key, hyperparams, static_fns)
        global_round += 1

        # compute test metrics.
        print("evaluating testing accuracy")
        correct = test_fn(hyperparams, server_classifier, data_test)
        print("Round %2d, testing accuracy %.3f" % (global_round, correct))

        # todo: log test metrics
