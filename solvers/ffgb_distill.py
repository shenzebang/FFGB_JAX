from utils.api import *
import jax
import jax.numpy as jnp
from collections import namedtuple
from utils.oracles import regression_oracle, v_regression_oracle
from utils.dx_loss import vg_ce
from utils.classifier import get_classifier_fn


def _weight(client_state: FFGBDistillClientState):
    return client_state.lr_0 / (
            client_state.num_local_steps * client_state.global_round + client_state.local_step + 1) ** .5


def sampler(num_workers, num_workers_per_round, rng):
    "samples num_sample indices out of num_total."

    return jax.random.permutation(rng, jnp.arange(num_workers))[:num_workers_per_round]


def server_init(model, hyperparams: ServerHyperParams, key):
    x_init = jnp.zeros((1, hyperparams.image_size, hyperparams.image_size, hyperparams.num_channels))
    param = jax.tree_map(lambda p_1: jnp.expand_dims(p_1, axis=0), model.init(key, x_init))
    weight = jnp.zeros((1, 1))
    classifier = Classifier(param, weight, True)
    return FFGBDistillServerState(classifier=classifier, global_round=0)


def client_init(model, hyperparams: ServerHyperParams, batch: Batch, server_state: FFGBDistillServerState):
    classifier_fn = hyperparams.get_classifier_fn(server_state.classifier)
    f_x = classifier_fn(batch.x)
    residual = jnp.zeros((batch.x.shape[0], hyperparams.num_classes))
    local_step = jnp.zeros(1)
    num_local_steps = jnp.ones(1)*hyperparams.num_local_steps
    global_round = jnp.ones(1)*server_state.global_round
    lr_0 = jnp.ones(1)*hyperparams.lr_0
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


def _get_target(client_state, batch):
    # (negative functional gradient direction) + residual
    return - vg_ce(client_state.f_x, batch.y) + client_state.residual


v_get_target = jax.vmap(_get_target, in_axes=(0, 0))
jv_get_target = jax.jit(v_get_target)


def _update_client_state(model, target, opt, new_weight, batch: Batch, client_state):
    predict = model.apply(opt.target, batch.x)
    return FFGBDistillClientState(f_x=client_state.f_x + predict * new_weight,
                                  residual=target - predict, local_step=client_state.local_step + 1,
                                  num_local_steps=client_state.num_local_steps,
                                  global_round=client_state.global_round, lr_0=client_state.lr_0)


v_update_client_state = jax.vmap(_update_client_state, in_axes=(None, 0, 0, 0, 0, 0))
vj_update_client_state = jax.jit(v_update_client_state, static_argnums=(0,))


def vclient_step(model, hyperparams: ServerHyperParams, batches: Batch, client_states, keys):
    targets = jv_get_target(client_states, batches)
    opts = v_regression_oracle(model, batches, targets, keys, hyperparams)
    new_weights = jax.vmap(_weight)(client_states)
    new_client_states = vj_update_client_state(model, targets, opts, new_weights, batches, client_states)
    return opts, new_weights, new_client_states


# def client_end(self):

predict_fn = lambda classifier_fn, xs: jnp.concatenate([classifier_fn(_x) for _x in xs])
# predict_fn = jax.jit(predict_fn, static_argnums=(0,))



def server_step(model, hyperparams: ServerHyperParams,
                classifier: Classifier, batch_distill: Batch, server_state: FFGBDistillServerState, key):
    for distill_step in range(hyperparams.num_distill_rounds):
        classifier_fn = hyperparams.get_classifier_fn(classifier)
        num_split = 200
        distill_xs = jnp.split(batch_distill.x, num_split, axis=0)
        # target = [classifier_fn(x) for x in distill_xs]
        # target = jnp.concatenate(target, axis=0)
        target = predict_fn(classifier_fn, distill_xs)
        key, subkey = jax.random.split(key)
        opt = regression_oracle(model, batch_distill, target, subkey, hyperparams)
        params = jax.tree_map(lambda p_1: jnp.expand_dims(p_1, axis=0), opt.target)
        classifier = Classifier(params=params, weights=jnp.ones((1, 1)), is_empty=False)
    # finishing the server step, the global round counter increases 1
    return FFGBDistillServerState(classifier=classifier, global_round=server_state.global_round + 1)


def get_optimizer(model, hyperparams: ServerHyperParams):
    ffgb = FedAlgorithm(
        sampler=sampler,
        server_init=jax.jit(jax.partial(server_init, model, hyperparams)),
        client_init=jax.jit(jax.partial(client_init, model, hyperparams)),
        client_step=jax.partial(client_step, model, hyperparams),
        vclient_step=jax.partial(vclient_step, model, hyperparams),
        client_end=None,
        server_step=jax.partial(server_step, model, hyperparams)
    )

    return ffgb
