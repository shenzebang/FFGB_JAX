import jax.numpy as jnp
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import jax.random as random
from collections import namedtuple
import os

from utils.dx_loss import vg_ce
from utils.loss import v_ce
from models.convnet import CONVNET
from models.mlp import MLP
from utils.classifier import get_classifier
from utils.oracles import regression_oracle, OracleState

key = random.PRNGKey(100)

DATA_DIR = os.path.join(os.environ['HOME'], 'DATASET')
model_path = './ckpt'
if not os.path.exists(model_path): os.makedirs(model_path)

batch_size = {
    "train": 128,
    "test": 1024
}
num_workers = 8

# load dataset with PyTorch
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
dataset = {
    "train": MNIST(os.path.join(DATA_DIR, "mnist"), train=True, transform=transform, download=True),
    "test": MNIST(os.path.join(DATA_DIR, "mnist"), train=False, transform=transform, download=True)
}


#================= configuration =================


distill_lr = 1e-3
distill_batch_size = 32
distill_num_steps = 10000
distill_options = OracleState(distill_lr, distill_num_steps, distill_batch_size)


weak_learner_lr = 1e-3
weak_learner_batch_size = 32
weak_learner_num_steps = 10000
weak_learner_options = OracleState(weak_learner_lr, weak_learner_num_steps, weak_learner_batch_size)

WorkerOptions = namedtuple("WorkerOptions", "num_steps weak_learner_oracle")
worker_num_steps = 3
weak_learner_oracle = regression_oracle
worker_options = WorkerOptions(worker_num_steps, weak_learner_oracle)

ExperimentOptions = namedtuple("ExperimentOptions", "num_rounds dx_loss distill_oracle distill_ratio lr_0\
                                                    num_distill_rounds")
num_rounds = 100
dx_loss = vg_ce # vectorized gradient of cross entropy
distill_oracle = regression_oracle
distill_ratio = .1
lr_0 = 1
num_distill_rounds = 1
experiment_options = ExperimentOptions(num_rounds, dx_loss, distill_oracle, distill_ratio, lr_0, num_distill_rounds)

#================= configuration =================

# create a distill dataset

key, subkey = random.split(key)
x = jnp.array(dataset["train"].data.numpy())[:, None, :, :].transpose(0, 2, 3, 1)
y = jnp.array(dataset["train"].targets.numpy())
index = random.shuffle(subkey, jnp.arange(0, x.shape[0]))
x = x[index]
y = y[index]

x_train, x_distill = jnp.split(x, [int(x.shape[0] * experiment_options.distill_ratio)], axis=0)
y_train = y[0:int(x.shape[0] * experiment_options.distill_ratio)]

del x, y, index

x_test = jnp.array(dataset["test"].data.numpy())[:, None, :, :].transpose(0, 2, 3, 1)
y_test = jnp.array(dataset["test"].targets.numpy())



# create a distill dataloader

dataloader = {
    "train": DataLoader(dataset["train"], batch_size=batch_size["train"], shuffle=True, num_workers=num_workers),
    "test": DataLoader(dataset["test"], batch_size=batch_size["test"], shuffle=False, num_workers=num_workers)
}




#================= initialization =================
# model = CONVNET()
model = MLP()
key, subkey = random.split(key)
params = model.init(subkey, jnp.ones((1, 28, 28, 1)))

#================= initialization =================




#================= run experiment =================
for round in range(experiment_options.num_rounds):
    if round != 0:
        params_list = [params]
        weight_list = [1.]
        f_data = model.apply(params, x_train)
    else:
        params_list = []
        weight_list = []
        f_data = jnp.zeros((x_train.shape[0], 10))

    # run fgb steps
    residual = jnp.zeros((x_train.shape[0], 10))
    print("start running local updates")
    for local_step in range(worker_options.num_steps):
        key, subkey = random.split(key)
        target = - experiment_options.dx_loss(f_data, y_train) + residual # (negative functional gradient direction)
        new_params = worker_options.weak_learner_oracle(model, x_train, target, subkey, weak_learner_options)
        # new_weight = experiment_options.lr_0/(round * worker_options.num_steps + local_step + 1)
        new_weight = experiment_options.lr_0
        predict = model.apply(new_params, x_train)
        residual = target - predict
        params_list.append(new_params)
        weight_list.append(new_weight)
        f_data += predict * new_weight
    print("finish running local steps")

    print("test fgb result")
    classifier = get_classifier(model, params_list, weight_list)
    f_x_test = classifier(x_test)
    test_loss = v_ce(f_x_test, y_test)
    pred = jnp.argmax(f_x_test, axis=1)
    corrct = jnp.true_divide(
        jnp.sum(jnp.equal(pred, jnp.reshape(y_test, pred.shape))),
        y_test.shape[0])
    print("round %5d, test accuracy % .4f" % (round, corrct))

    # distill
    print("start distillating")
    for distill_step in range(experiment_options.num_distill_rounds):
        classifier = get_classifier(model, params_list, weight_list)
        target = classifier(x_distill)
        key, subkey = random.split(key)
        params = experiment_options.distill_oracle(model, x_distill, target, subkey, distill_options)
        params_list = [params]
        weight_list = [1.]
    print("finish distillating")
    # test

    print("test distill result")
    f_x_test = model.apply(params, x_test)
    test_loss = v_ce(f_x_test, y_test)
    pred = jnp.argmax(f_x_test, axis=1)
    corrct = jnp.true_divide(
        jnp.sum(jnp.equal(pred, jnp.reshape(y_test, pred.shape))),
        y_test.shape[0])
    print("round %5d, test accuracy % .4f" % (round, corrct))












