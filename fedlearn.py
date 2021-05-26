from utils.api import ServerHyperParams, Batch, FedAlgorithm, Classifier, StaticFns, FFGBDistillServerState
import jax
from utils.test import test_fn
from typing import List
from utils.data_split import data_split
import jax.numpy as jnp
from tqdm import trange, tqdm

# one round of functional federated learning.
def run_one_round(
        fedalg: FedAlgorithm,
        data_train_tuple: List[Batch],
        data_distill: Batch,
        server_state: FFGBDistillServerState,
        key: jax.random.PRNGKey,
        hyperparams: ServerHyperParams,
        static_fns: StaticFns
):
    """Runs one round of functional federated learning."""

    key, subkey = jax.random.split(key)

    client_indxs = fedalg.sampler(hyperparams.num_clients, hyperparams.num_sampled_clients, subkey)
    sampled_x = jnp.stack([data_train_tuple[client_indx].x for client_indx in list(client_indxs)], axis=0)
    sampled_y = jnp.stack([data_train_tuple[client_indx].y for client_indx in list(client_indxs)], axis=0)
    sampled_dss = Batch(x=sampled_x, y=sampled_y)

    # clients_state = [fedalg.client_init(train_batch, server_state)
    #                  for train_batch in sampled_dss]
    client_states = jax.vmap(fedalg.client_init, in_axes=[0, None])(sampled_dss, server_state)


    new_classifiers = []
    for _ in trange(hyperparams.num_local_steps, desc='local steps', leave=False):
        key, subkey = jax.random.split(key)
        new_params, new_weights, client_states = fedalg.client_step(sampled_dss, client_states, subkey)
        new_classifiers.append(
            Classifier(new_params, new_weights/hyperparams.num_sampled_clients)
        )

    # update server classifier and state.
    server_state = fedalg.server_step(new_classifiers, data_distill, server_state, key)

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

    # server_classifier = Classifier(params_list=[], weight_list=[])
    # key, subkey = jax.random.split(key)
    server_state = fedalg.server_init(jax.random.PRNGKey(1))

    keys = jax.random.split(key, hyperparams.num_rounds)

    print("\nStarting training. First round is slower than the rest.")
    for global_round in trange(hyperparams.num_rounds):

        key = keys[global_round]
        server_state = run_one_round(fedalg, data_train_tuple,
                                                        data_distill, server_state, key, hyperparams, static_fns)


        # compute test metrics.
        # print("evaluating testing accuracy")
        correct = test_fn(hyperparams, server_state, data_test)
        print("Round %2d, testing accuracy %.3f" % (global_round, correct))

        # todo: log test metrics
