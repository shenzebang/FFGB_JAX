from jax import grad, vmap
from jax.scipy.special import logsumexp



def cross_entropy(x, label):
    # x: d-dimensional vector
    # label: integer indicating index
    Z = logsumexp(x)
    return -x[label] + Z


v_ce = vmap(cross_entropy, in_axes=(0, 0))


