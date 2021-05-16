import jax.numpy as jnp
from flax import linen as nn
from jax import jit


class CONVNET(nn.Module):
    def setup(self):
        self.conv1 = nn.Conv(features=32, kernel_size=(3, 3), strides=(1, 1),
                             padding=((0, 0), (0, 0)), use_bias=True)
        self.conv2 = nn.Conv(features=64, kernel_size=(3, 3), strides=(1, 1),
                             padding=((0, 0), (0, 0)), use_bias=True)
        self.conv3 = nn.Conv(features=64, kernel_size=(3, 3), strides=(1, 1),
                             padding=((0, 0), (0, 0)), use_bias=True)
        self.dense1 = nn.Dense(features=64)
        # self.dense2 = nn.Dense(features=192)
        self.dense2 = nn.Dense(features=10)

        self.pool = lambda x: nn.max_pool(x, window_shape=(2, 2), strides=(2, 2),
                                          padding=((0, 0), (0, 0)))

        self.activation = nn.relu

    def __call__(self, x: jnp.DeviceArray):
        x = self.pool(self.activation(self.conv1(x)))
        x = self.pool(self.activation(self.conv2(x)))
        x = self.activation(self.conv3(x))
        x = jnp.reshape(x, (x.shape[0], -1))
        x = self.activation(self.dense1(x))
        # x = self.activation(self.dense2(x))
        x = self.dense2(x)
        return x
