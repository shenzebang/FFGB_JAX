from flax.linen import Module
from utils.api import Classifier
import jax.numpy as jnp
from jax import jit, vmap


def weighted_net_apply(net, params, x, weight):
    return net.apply(params, x) * weight


v_weighted_net_apply = vmap(weighted_net_apply, in_axes=(None, 0, None, 0))


def get_response(net, params, x, weight):
    return jnp.mean(v_weighted_net_apply(net, params, x, weight), axis=0)


jv_get_response = jit(get_response, static_argnums=0)


def get_classifier_fn(net: Module, classifier: Classifier):
    def classifier_fn(x):
        return jv_get_response(net, classifier.params, x, classifier.weights)

    return classifier_fn
