import jax
import jax.numpy as jnp
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, CIFAR10
import jax.random as random
import os

from models.convnet import CONVNET
from models.mlp import MLP
from solvers.ffgb_distill import get_optimizer
from utils.api import Batch, ServerHyperParams, StaticFns
from fedlearn import functional_federated_learning
from utils.classifier import get_classifier_fn

key = random.PRNGKey(100)

DATA_DIR = os.path.join(os.environ['HOME'], 'DATASET')
model_path = './ckpt'
if not os.path.exists(model_path): os.makedirs(model_path)



# ================= configuration =================
num_rounds = 100
distill_ratio = .1
lr_0 = 1
num_distill_rounds = 1
num_local_steps = 5
num_clients = 100
s = .1
num_classes = 10
num_channels = 3
oracle_num_steps = 10000
oracle_lr = 1e-3
oracle_batch_size = 32
num_sampled_clients = 10
dataset = "mnist"
model = CONVNET()
get_classifier_fn = jax.partial(get_classifier_fn, model, num_classes)
model_apply_fn = jax.jit(model.apply)
hyperparams = ServerHyperParams(num_rounds=num_rounds, distill_ratio=distill_ratio, lr_0=lr_0,
                                num_sampled_clients=num_sampled_clients, num_distill_rounds=num_distill_rounds,
                                num_local_steps=num_local_steps, num_clients=num_clients, s=s,
                                num_classes=num_classes, oracle_num_steps=oracle_num_steps, oracle_lr=oracle_lr,
                                oracle_batch_size=oracle_batch_size, num_channels=num_channels,
                                get_classifier_fn=get_classifier_fn)

static_fns = StaticFns(get_classifier_fn=get_classifier_fn, model_apply_fn=model_apply_fn)

ffgb = get_optimizer(model, hyperparams)
# ================= configuration =================

# ================= load dataset =================
# load dataset with PyTorch
if dataset == "cifar10":
    transform = transforms.Compose(
        [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = {
        "train": CIFAR10(os.path.join(DATA_DIR, "cifar10"), train=True, transform=transform, download=True),
        "test": CIFAR10(os.path.join(DATA_DIR, "cifar10"), train=False, transform=transform, download=True)
    }
    x = jnp.array(dataset["train"].data)
    y = jnp.array(dataset["train"].targets)
    x_test = jnp.array(dataset["test"].data)
    y_test = jnp.array(dataset["test"].targets)
elif dataset == "mnist":
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset = {
        "train": MNIST(os.path.join(DATA_DIR, "mnist"), train=True, transform=transform, download=True),
        "test": MNIST(os.path.join(DATA_DIR, "mnist"), train=False, transform=transform, download=True)
    }
    x = jnp.array(dataset["train"].data.numpy())[:, None, :, :].transpose(0, 2, 3, 1)
    y = jnp.array(dataset["train"].targets.numpy())
    x_test = jnp.array(dataset["test"].data.numpy())[:, None, :, :].transpose(0, 2, 3, 1)
    y_test = jnp.array(dataset["test"].targets.numpy())

# ================= load dataset =================


# ================= prepare dataset =================

key, subkey = random.split(key)
index = random.permutation(subkey, jnp.arange(0, x.shape[0]))
x = x[index]
y = y[index]

x_train, x_distill = jnp.split(x, [int(x.shape[0] * hyperparams.distill_ratio)], axis=0)
y_train = y[0:int(x.shape[0] * hyperparams.distill_ratio)]

data_train = Batch(x_train, y_train)
data_distill = Batch(x_distill, None)


data_test = Batch(x_test, y_test)

del x, y, index
# ================= prepare dataset =================


# ================= run experiment =================
functional_federated_learning(model, data_train, data_distill, data_test, ffgb, hyperparams, static_fns, key)
