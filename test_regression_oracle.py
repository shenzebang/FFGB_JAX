from flax import linen as nn
from utils.oracles import regression_oracle
import jax.random as random
import jax.numpy as jnp
import jax
from collections import namedtuple
# test the regression oracle with simple modules


State = namedtuple("State", "f_x, residual")





# def f(x):
#     for i in range(10000):
#         x = x + i
#     return State(x, x+1)
#
# vf = jax.vmap(f, in_axes=0)
# x = jnp.ones((11, 10))
# y = vf(x)

# def ff(x: State):
#     print(x.f_x.shape)
#     # for i in range(100):
#     #     x.f_x = x.f_x + i
#     #     x.residual = x.residual + i
#     # return x
#
# vff = jax.vmap(ff, in_axes=0)
#
# state = State(jnp.ones((11, 10)), jnp.ones((11, 10)))
#
# vff(state)

# def fff(x):
#     print(type(x))
#     return x
#
# vfff = jax.vmap(fff, in_axes=0)
#
# y = vfff((jnp.ones(1), jnp.ones(1)))
# print(y)




# net = nn.Dense(features=10)
# n = 1000
# d = 10
# key = random.PRNGKey(1)
# key, subkey = random.split(key)
# x = random.normal(subkey, (n, d))
# w = jnp.ones((d, d))
# y = x@w
# print(y.shape)
# key, subkey = random.split(key)
# lr = 1e-2
# batch_size = 32
# num_steps = 1000
# oracle_state = OracleState(lr, num_steps, batch_size)
# params = regression_oracle(net, x, y, subkey, oracle_state)


v = []
av = jnp.stack(v)
print(av)

