from utils.api import Classifier, Batch, ServerHyperParams
from utils.classifier import get_classifier_fn
import jax.numpy as jnp
from jax import jit


def _test_fn(hyperparams: ServerHyperParams, classifier: Classifier, batch: Batch):
    classifier_fn = hyperparams.get_classifier_fn(classifier)
    f_x_test = classifier_fn(batch.x)
    pred = jnp.argmax(f_x_test, axis=1)
    correct = jnp.true_divide(
        jnp.sum(jnp.equal(pred, jnp.reshape(batch.y, pred.shape))),
        batch.y.shape[0])
    return correct


test_fn = jit(_test_fn, static_argnums=0)
