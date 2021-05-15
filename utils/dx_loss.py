from jax import grad, vmap
from jax.scipy.special import logsumexp



def cross_entropy(x, label):
    # x: d-dimensional vector
    # label: integer indicating index
    Z = logsumexp(x)
    return -x[label] + Z

g_ce =  grad(cross_entropy, argnums=0)

vg_ce = vmap(g_ce, in_axes=(0, 0))


