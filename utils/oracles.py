from flax.linen import Module
from jax import value_and_grad, jit, vmap
from jax import random
import jax.numpy as jnp
from flax.optim import Adam, Optimizer, Momentum
from utils.api import ServerHyperParams


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


def train_op_n(net: Module, opt: Optimizer, x, y, key, hyperparams: ServerHyperParams, n: int):
    for i in range(n):
        v, opt = train_op(net, opt, x, y, key[i], hyperparams)
    return v, opt

v_train_op_n = vmap(train_op_n, in_axes=(None, 0, 0, 0, 0, None, None))
jv_train_op_n = jit(v_train_op_n, static_argnums=(0, 5, 6))


# regression_oracle = jit(regression_oracle, static_argnums=(0, 4))

def regression_oracle(net: Module, x, y, key, hyperparams: ServerHyperParams):
    x_init = jnp.zeros((1, hyperparams.image_size, hyperparams.image_size, hyperparams.num_channels))
    num_vmap_batching = x.shape[0]
    assert num_vmap_batching == y.shape[0]
    keys = random.split(key, num_vmap_batching+1)
    batch_params_init = vmap(net.init, in_axes=[0, None])(keys[1:], x_init)
    key = keys[0]
    opt_def = Adam(learning_rate=hyperparams.oracle_lr)
    # opt_def = Momentum(learning_rate=1e-3, weight_decay=1e-4, nesterov=True)
    opts = vmap(opt_def.create)(batch_params_init)
    num_loop_unrolling = 10
    for step in range(hyperparams.oracle_num_steps // num_loop_unrolling):
        keys = random.split(key, num_vmap_batching * num_loop_unrolling + 1)
        key = keys[num_vmap_batching * num_loop_unrolling]
        keys = jnp.reshape(keys[:num_vmap_batching * num_loop_unrolling],
                           (num_vmap_batching, num_loop_unrolling, 2))
        v, opts = jv_train_op_n(net, opts, x, y, keys, hyperparams, num_loop_unrolling)
        if step % 1000 == 0:
            print("step %5d oracle error %.2f" % (step, v))

    return opts.target


# def distill_oracle(net: Module, x, y, key, hyperparams: ServerHyperParams):
#     key, subkey = random.split(key)
#     params_init = net.init(subkey, x[0:2])
#     opt_def = Adam(learning_rate=hyperparams.distill_oracle_lr)
#     opt = opt_def.create(target=params_init)
#     for step in range(hyperparams.distill_oracle_num_steps // 5):
#         v, opt, key = train_op_5(net, opt, x, y, key, hyperparams)
#         if step % 500 == 0:
#             print("step %5d oracle error %.2f" % (step, v))
#
#     return opt.target
