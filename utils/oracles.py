from flax.linen import Module
from jax import value_and_grad, jit
from jax import random
import jax.numpy as jnp
from flax.optim import Adam, Optimizer
from utils.api import ServerHyperParams


def regression_loss(net: Module, params, x, y):
    return jnp.mean(jnp.sum((net.apply(params, x) - y) ** 2, axis=(1,)))


vg = value_and_grad(regression_loss, argnums=1)


def train_op(net: Module, opt: Optimizer, x, y, key, hyperparams: ServerHyperParams):
    key, subkey = random.split(key)
    index = random.randint(
        subkey,
        shape=(hyperparams.oracle_batch_size,),
        minval=0,
        maxval=x.shape[0]
    )
    v, g = vg(net, opt.target, x[index], y[index])
    return v, opt.apply_gradient(g), key


train_op = jit(train_op, static_argnums=(0, 5))


def regression_oracle(net: Module, x, y, key, hyperparams: ServerHyperParams):
    key, subkey = random.split(key)
    params_init = net.init(subkey, x[0:2])
    opt_def = Adam(learning_rate=hyperparams.oracle_lr)
    opt = opt_def.create(target=params_init)
    for step in range(hyperparams.oracle_num_steps):
        v, opt, key = train_op(net, opt, x, y, key, hyperparams)
        # if step % 500 == 0:
        #     print("step %5d oracle error %.2f" % (step, v))

    return opt.target

# regression_oracle = jit(regression_oracle, static_argnums=(0, 4))
