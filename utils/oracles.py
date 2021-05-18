from flax.linen import Module
from jax import value_and_grad, jit, vmap
from jax import random
import jax.numpy as jnp
from flax.optim import Adam, Optimizer
from utils.api import ServerHyperParams, Batch
import jax


def regression_loss(net: Module, params, x, y):
    return jnp.mean(jnp.sum((net.apply(params, x) - y) ** 2, axis=(1,)))


vg = value_and_grad(regression_loss, argnums=1)


def train_op(net: Module, opt: Optimizer, x, y, key, hyperparams: ServerHyperParams):
    index = random.randint(
        key,
        shape=(hyperparams.oracle_batch_size,),
        minval=0,
        maxval=x.shape[0]
    )
    v, g = vg(net, opt.target, x[index], y[index])
    return v, opt.apply_gradient(g)


train_op = jit(train_op, static_argnums=(0, 5))


def train_op_n(net: Module, opt: Optimizer, x, y, keys, hyperparams: ServerHyperParams, n: int):
    for i in range(n):
        v, opt = train_op(net, opt, x, y, keys[i], hyperparams)
    return v, opt


train_op_n = jit(train_op_n, static_argnums=(0, 5, 6))


def _regression_oracle(net: Module, x, y, key, hyperparams: ServerHyperParams):
    key, subkey = random.split(key)
    params_init = net.init(subkey, x[0:2])
    opt_def = Adam(learning_rate=hyperparams.oracle_lr)
    opt = opt_def.create(target=params_init)
    for step in range(hyperparams.oracle_num_steps):
        v, opt, key = train_op(net, opt, x, y, key, hyperparams)
        # if step % 500 == 0:
        #     print("step %5d oracle error %.2f" % (step, v))

    return opt.target



def regression_oracle(net: Module, x, target, key, hyperparams: ServerHyperParams):
    x_init = jnp.zeros((1, hyperparams.image_size, hyperparams.image_size, hyperparams.num_channels))
    key, subkey = random.split(key)
    params_init = net.init(subkey, x_init)
    opt_def = Adam(learning_rate=hyperparams.oracle_lr)
    opt = opt_def.create(target=params_init)
    n_loop_unrolling = 5
    for step in range(hyperparams.oracle_num_steps // n_loop_unrolling):
        keys = random.split(key, n_loop_unrolling + 1)
        key = keys[n_loop_unrolling]
        keys = keys[: n_loop_unrolling]
        v, opt = j_train_op_n(net, opt, x, target, keys, hyperparams, n_loop_unrolling)
        # if step % 500 == 0:
        #     print("step %5d oracle error %.2f" % (step, v))

    return opt


def _train_op(net: Module, opt: Optimizer, batch, target, key, hyperparams: ServerHyperParams):
    index = random.randint(
        key,
        shape=(hyperparams.oracle_batch_size,),
        minval=0,
        maxval=batch.x.shape[0]
    )
    v, g = vg(net, opt.target, batch.x[index], target[index])
    return v, opt.apply_gradient(g)


def _train_op_n(net: Module, opt, batch, target, key, hyperparams: ServerHyperParams, n: int):
    for i in range(n):
        v, opt = _train_op(net, opt, batch, target, key[i], hyperparams)
    return v, opt


j_train_op_n = jit(_train_op_n, static_argnums=(0, 5, 6))


v_train_op_n = vmap(_train_op_n, in_axes=[None, 0, 0, 0, 0, None, None])
jv_train_op_n = jit(v_train_op_n, static_argnums=(0, 5, 6))

def v_regression_oracle(net: Module, batches: Batch, targets, key, hyperparams: ServerHyperParams):
    x_init = jnp.zeros((1, hyperparams.image_size, hyperparams.image_size, hyperparams.num_channels))
    num_vmap_batching = batches.x.shape[0]
    keys = random.split(key, num_vmap_batching + 1)
    batch_params_init = vmap(net.init, in_axes=[0, None])(keys[0:num_vmap_batching], x_init)
    key = keys[-1]
    opt_def = Adam(learning_rate=hyperparams.oracle_lr)
    opts = jax.vmap(opt_def.create)(batch_params_init)
    n_loop_unrolling = 5
    for step in range(hyperparams.oracle_num_steps // n_loop_unrolling):
        # generate keys for the following (n_loop_unrolling * num_sampled_clients) loops
        keys = random.split(key, num_vmap_batching * n_loop_unrolling + 1)
        key = keys[num_vmap_batching * n_loop_unrolling]
        keys = jnp.reshape(keys[:num_vmap_batching * n_loop_unrolling],
                           (num_vmap_batching, n_loop_unrolling, 2))
        v, opts = jv_train_op_n(net, opts, batches, targets, keys[:num_vmap_batching * n_loop_unrolling],
                                hyperparams, n_loop_unrolling)
        # if step % 500 == 0:
        #     print("step %5d oracle error %.2f" % (step, v))
    return opts


def distill_oracle(net: Module, x, y, key, hyperparams: ServerHyperParams):
    key, subkey = random.split(key)
    params_init = net.init(subkey, x[0:2])
    opt_def = Adam(learning_rate=hyperparams.distill_oracle_lr)
    opt = opt_def.create(target=params_init)
    n_loop_unrolling = 5
    for step in range(hyperparams.distill_oracle_num_steps // n_loop_unrolling):
        v, opt, key = train_op_n(net, opt, x, y, key, hyperparams, n_loop_unrolling)
        if step % 500 == 0:
            print("step %5d oracle error %.2f" % (step, v))

    return opt.target
