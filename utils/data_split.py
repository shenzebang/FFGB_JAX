import jax.numpy as jnp
from utils.api import Batch


def data_split(batch: Batch, num_workers: int, s: float):
    # s encodes the heterogeneity of the data split
    if num_workers == 1:
        return batch
    n_data = batch.x.shape[0]

    n_homo_data = int(n_data * s)

    assert 0 < n_homo_data < n_data

    data_homo, data_hetero = jnp.split(batch.x, [n_homo_data])
    label_homo, label_hetero = jnp.split(batch.y, [n_homo_data])

    data_homo_list = jnp.split(data_homo, num_workers)
    label_homo_list = jnp.split(label_homo, num_workers)

    index = jnp.argsort(label_hetero)
    label_hetero_sorted = label_hetero[index]
    data_hetero_sorted = data_hetero[index]
    data_hetero_list = jnp.split(data_hetero_sorted, num_workers)
    label_hetero_list = jnp.split(label_hetero_sorted, num_workers)

    data_list = [jnp.concatenate([data_homo, data_hetero], axis=0) for data_homo, data_hetero in
                 zip(data_homo_list, data_hetero_list)]
    label_list = [jnp.concatenate([label_homo, label_hetero], axis=0) for label_homo, label_hetero in
                  zip(label_homo_list, label_hetero_list)]

    return [Batch(x, y) for x, y in zip(data_list, label_list)]
