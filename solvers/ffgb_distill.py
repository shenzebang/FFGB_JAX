from utils.api import *
import jax
import jax.numpy as jnp
from collections import namedtuple
from utils.oracles import regression_oracle
from utils.dx_loss import vg_ce
from utils.classifier import get_classifier_fn


def _weight(client_state: FFGBDistillClientState):
    return client_state.lr_0 / (
            client_state.num_local_steps * client_state.global_round + client_state.local_step + 1) ** .5


def sampler(num_workers, num_workers_per_round, rng):
    "samples num_sample indices out of num_total."

    return jax.random.permutation(rng, jnp.arange(num_workers))[:num_workers_per_round]


def server_init():
    return FFGBDistillServerState(global_round=0)


def client_init(model, hyperparams: ServerHyperParams,
                classifier: Classifier, batch: Batch, server_state: FFGBDistillServerState):
    classifier_fn = hyperparams.get_classifier_fn(classifier)
    f_x = classifier_fn(batch.x)
    residual = jnp.zeros((batch.x.shape[0], hyperparams.num_classes))
    local_step = 0
    num_local_steps = hyperparams.num_local_steps
    global_round = server_state.global_round
    lr_0 = hyperparams.lr_0
    return FFGBDistillClientState(f_x, residual, local_step, num_local_steps, global_round, lr_0)


def client_step(model, hyperparams: ServerHyperParams,
                batch: Batch, client_state: FFGBDistillClientState, key):
    # (negative functional gradient direction) + residual
    target = - vg_ce(client_state.f_x, batch.y) + client_state.residual
    new_params = regression_oracle(model, batch.x, target, key, hyperparams)
    new_weight = _weight(client_state)
    predict = model.apply(new_params, batch.x)
    new_client_state = FFGBDistillClientState(f_x=client_state.f_x + predict * new_weight,
                                              residual=target - predict, local_step=client_state.local_step + 1,
                                              num_local_steps=client_state.num_local_steps,
                                              global_round=client_state.global_round, lr_0=client_state.lr_0)
    return new_params, new_weight, new_client_state


# def client_end(self):

def server_step(model, hyperparams: ServerHyperParams,
                classifier: Classifier, batch_distill: Batch, server_state: FFGBDistillServerState, key):
    print("Start server step of round %3d (distillation)" % server_state.global_round)
    for distill_step in range(hyperparams.num_distill_rounds):
        classifier_fn = hyperparams.get_classifier_fn(classifier)
        target = []
        num_batchs = 100
        num_per_batch = batch_distill.x.shape[0] // num_batchs
        for i in range(num_batchs):
            index = jnp.arange(i * num_per_batch, (i + 1) * num_per_batch)
            target.append(classifier_fn(batch_distill.x[index]))
        target = jnp.concatenate(target, axis=0)
        key, subkey = jax.random.split(key)
        params = regression_oracle(model, batch_distill.x, target, subkey, hyperparams)
        classifier = Classifier([params], [1.], 1)
    # finishing the server step, the global round counter increases 1
    print("Finish server step")
    return classifier, FFGBDistillServerState(server_state.global_round + 1)


def get_optimizer(model, hyperparams: ServerHyperParams):
    ffgb = FedAlgorithm(
        sampler=sampler,
        server_init=server_init,
        client_init=jax.partial(client_init, model, hyperparams),
        client_step=jax.partial(client_step, model, hyperparams),
        client_end=None,
        server_step=jax.partial(server_step, model, hyperparams)
    )

    return ffgb