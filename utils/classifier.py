from flax.linen import Module
from utils.api import Classifier
import jax.numpy as jnp
from jax import jit


def get_response(net, params, x, weight):
    return net.apply(params, x) * weight


get_response = jit(get_response, static_argnums=0)


def get_classifier_fn(net: Module, num_classes: int, classifier: Classifier):
    # todo: make a vectorized version
    def classifier_fn(x):
        response = jnp.zeros((x.shape[0], num_classes))
        for params, weight in zip(classifier.params_list, classifier.weight_list):
            response = response + get_response(net, params, x, weight)
        return response

    return classifier_fn
