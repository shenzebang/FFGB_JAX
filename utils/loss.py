from jax import grad, vmap
from jax.scipy.special import logsumexp
import jax.numpy as jnp


def cross_entropy(x, label):
    # x: d-dimensional vector
    # label: integer indicating index
    Z = logsumexp(x)
    return -x[label] + Z


v_ce = vmap(cross_entropy, in_axes=(0, 0))

def dense_cross_entropy(x, y):
    # x: d-dimensional vector
    # label: integer indicating index
    Z = logsumexp(x)
    return -jnp.dot(x, y) + Z

v_dce = vmap(dense_cross_entropy, in_axes=(0, 0))


