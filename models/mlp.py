import jax.numpy as jnp
from flax import linen as nn


class MLP(nn.Module):
    def setup(self):
        self.dense1 = nn.Dense(features=384)
        self.dense2 = nn.Dense(features=192)
        self.dense3 = nn.Dense(features=10)

        self.activation = nn.leaky_relu

    def __call__(self, x: jnp.DeviceArray):
        x = jnp.reshape(x, (x.shape[0], -1))
        x = self.activation(self.dense1(x))
        x = self.activation(self.dense2(x))
        x = self.dense3(x)
        return x
