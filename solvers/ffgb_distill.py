from utils.api import *
import jax
import jax.numpy as jnp
from collections import namedtuple
from utils.oracles import regression_oracle
from utils.dx_loss import vg_ce
from utils.classifier import get_classifier_fn
from typing import List


def _weight(client_state: FFGBDistillClientState):
    return client_state.lr_0 / (
            client_state.num_local_steps * client_state.global_round + client_state.local_step + 1) ** .5


def sampler(num_workers, num_workers_per_round, rng):
    "samples num_sample indices out of num_total."

    return jax.random.permutation(rng, jnp.arange(num_workers))[:num_workers_per_round]


def server_init(model, hyperparams: ServerHyperParams, key):
    x_init = jnp.zeros((1, hyperparams.image_size, hyperparams.image_size, hyperparams.num_channels))
    params = jax.tree_map(lambda p_1: jnp.expand_dims(p_1, axis=0), model.init(key, x_init))
    weight = jnp.zeros((1, 1))
    classifier = Classifier(params, weight)
    return FFGBDistillServerState(classifier=classifier, global_round=0)


def client_init(model, hyperparams: ServerHyperParams, batch: Batch, server_state: FFGBDistillServerState):
    classifier_fn = hyperparams.get_classifier_fn(server_state.classifier)
    f_x = classifier_fn(batch.x)
    residual = jnp.zeros((batch.x.shape[0], hyperparams.num_classes))
    local_step = jnp.zeros(1)
    num_local_steps = jnp.ones(1) * hyperparams.num_local_steps
    global_round = jnp.ones(1) * server_state.global_round
    lr_0 = jnp.ones(1) * hyperparams.lr_0
    return FFGBDistillClientState(f_x, residual, local_step, num_local_steps, global_round, lr_0)

def _get_target(client_state, batch):
    # (negative functional gradient direction) + residual
    return - vg_ce(client_state.f_x, batch.y) + client_state.residual


v_get_target = jax.vmap(_get_target, in_axes=(0, 0))
jv_get_target = jax.jit(v_get_target)

def _update_client_state(model, target, params, new_weight, batch: Batch, client_state, hyperparams:ServerHyperParams):
    predict = model.apply(params, batch.x)
    return FFGBDistillClientState(f_x=client_state.f_x + predict * new_weight,
                                  residual=target - predict, local_step=client_state.local_step + 1,
                                  num_local_steps=client_state.num_local_steps,
                                  global_round=client_state.global_round, lr_0=client_state.lr_0)


v_update_client_state = jax.vmap(_update_client_state, in_axes=(None, 0, 0, 0, 0, 0, None))
vj_update_client_state = jax.jit(v_update_client_state, static_argnums=(0, 6))

def client_step(model, hyperparams: ServerHyperParams,
                batch: Batch, client_states: FFGBDistillClientState, key):
    # (negative functional gradient direction) + residual
    targets = jv_get_target(client_states, batch)
    new_params = regression_oracle(model, batch.x, targets, key, hyperparams)
    new_weights = _weight(client_states)
    new_client_states = vj_update_client_state(model, targets, new_params, new_weights, batch, client_states, hyperparams)
    return new_params, new_weights, new_client_states


# def client_end(self):

def server_step(model, hyperparams: ServerHyperParams,
                classifiers: List[Classifier], batch_distill: Batch, server_state: FFGBDistillServerState, key):
    classifiers = classifiers + [server_state.classifier]
    # print("Start server step of round %3d (distillation)" % server_state.global_round)
    for distill_step in range(hyperparams.num_distill_rounds):
        classifier_fns = [hyperparams.get_classifier_fn(classifier) for classifier in classifiers]
        target = []
        num_batchs = 20
        num_per_batch = batch_distill.x.shape[0] // num_batchs
        for i in range(num_batchs):
            index = jnp.arange(i * num_per_batch, (i + 1) * num_per_batch)
            target.append(
                jnp.sum(jnp.stack(
                    [classifier_fn(batch_distill.x[index]) for classifier_fn in classifier_fns],
                    axis=0), axis=0)
            )
        target = jnp.concatenate(target, axis=0)
        target = jnp.expand_dims(target, axis=0)
        key, subkey = jax.random.split(key)
        params = regression_oracle(model, jnp.expand_dims(batch_distill.x, axis=0), target, key, hyperparams)
        # params = jax.tree_map(lambda p_1: jnp.expand_dims(p_1, axis=0), params)
        classifier = Classifier(params, jnp.ones((1, 1)))
        classifiers = [classifier]
    # finishing the server step, the global round counter increases 1
    # print("Finish server step")
    return FFGBDistillServerState(classifier, server_state.global_round + 1)


def get_optimizer(model, hyperparams: ServerHyperParams):
    ffgb = FedAlgorithm(
        sampler=sampler,
        server_init=jax.partial(server_init, model, hyperparams),
        client_init=jax.jit(jax.partial(client_init, model, hyperparams)),
        client_step=jax.partial(client_step, model, hyperparams),
        client_end=None,
        server_step=jax.partial(server_step, model, hyperparams)
    )

    return ffgb