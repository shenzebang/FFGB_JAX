import jax.numpy as jnp
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, CIFAR10
import jax.random as random
import os
import jax
from utils.api import ServerHyperParams, Classifier
from flax.optim import Adam
from utils.loss import v_ce
from models.convnet import CONVNET
from models.mlp import MLP
from utils.classifier import get_classifier_fn
from utils.oracles import regression_oracle, distill_oracle

key = random.PRNGKey(100)

DATA_DIR = os.path.join(os.environ['HOME'], 'DATASET')
model_path = './ckpt'
if not os.path.exists(model_path): os.makedirs(model_path)

# load dataset with PyTorch
# transform = transfotransform = transforms.Compose(
#     [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
dataset = {
    "train": CIFAR10(os.path.join(DATA_DIR, "cifar10"), train=True, download=True),
    "test": CIFAR10(os.path.join(DATA_DIR, "cifar10"), train=False, download=True)
}
x = jnp.array(dataset["train"].data)
y = jnp.array(dataset["train"].targets)
x_test = jnp.array(dataset["test"].data)
y_test = jnp.array(dataset["test"].targets)

# normalize the data
# x = (x / 255. - jnp.ones(3) * .5) / (jnp.ones(3) * .5)
# x_test = (x_test / 255. - jnp.ones(3) * .5) / (jnp.ones(3) * .5)

x = x/255.
x_test = x_test/255.
# ================= configuration =================
num_rounds = 100
distill_ratio = .1
lr_0 = 1.
num_distill_rounds = 1
num_local_steps = 3
num_clients = 100
s = .1
num_classes = 10
num_channels = 3
oracle_num_steps = 40000
oracle_lr = 1e-3
oracle_batch_size = 32
distill_oracle_num_steps = 40000
distill_oracle_lr = 1e-3
distill_oracle_batch_size = 32
num_sampled_clients = 10
dataset = "cifar10"
model = CONVNET()
get_classifier_fn = jax.partial(get_classifier_fn, model, num_classes)
model_apply_fn = jax.jit(model.apply)
hyperparams = ServerHyperParams(num_rounds=num_rounds, distill_ratio=distill_ratio, lr_0=lr_0,
                                num_sampled_clients=num_sampled_clients, num_distill_rounds=num_distill_rounds,
                                num_local_steps=num_local_steps, num_clients=num_clients, s=s,
                                num_classes=num_classes, oracle_num_steps=oracle_num_steps, oracle_lr=oracle_lr,
                                oracle_batch_size=oracle_batch_size, num_channels=num_channels,
                                get_classifier_fn=get_classifier_fn,
                                distill_oracle_batch_size=distill_oracle_batch_size,
                                distill_oracle_lr=distill_oracle_lr, distill_oracle_num_steps=distill_oracle_num_steps)

# static_fns = StaticFns(get_classifier_fn=get_classifier_fn, model_apply_fn=model_apply_fn)


# ================= configuration =================

# create a distill dataset

key, subkey = random.split(key)
index = random.permutation(subkey, jnp.arange(0, x.shape[0]))
x = x[index]
y = y[index]

# x_train, x_distill = jnp.split(x, [int(x.shape[0] * hyperparams.distill_ratio)], axis=0)
# y_train = y[0:int(x.shape[0] * hyperparams.distill_ratio)]
x_train = x
y_train = y
del x, y, index

# ================= initialization =================
model = CONVNET()
# model = MLP()
key, subkey = random.split(key)
params = model.init(subkey, x_train[0:2])


def loss(params, x, y):
    f_x = model.apply(params, x)
    return jnp.mean(v_ce(f_x, y))

value_and_grad = jax.value_and_grad(loss)
opt_def = Adam(learning_rate=hyperparams.oracle_lr)
opt = opt_def.create(target=params)

def train_op(opt, x, y):
    v, g = value_and_grad(opt.target, x, y)
    return v, opt.apply_gradient(g)

train_op = jax.jit(train_op)
for step in range(40000):
    key, subkey = random.split(key)
    index = random.randint(
        subkey,
        shape=(hyperparams.oracle_batch_size,),
        minval=0,
        maxval=x_train.shape[0]
    )
    v, opt = train_op(opt, x_train[index], y_train[index])
    if step % 500 == 0:
        print("test sgd result")
        f_x_test = model.apply(opt.target, x_test)
        test_loss = v_ce(f_x_test, y_test)
        pred = jnp.argmax(f_x_test, axis=1)
        corrct = jnp.true_divide(
            jnp.sum(jnp.equal(pred, jnp.reshape(y_test, pred.shape))),
            y_test.shape[0])
        print("step %5d, test accuracy % .4f" % (step, corrct))

