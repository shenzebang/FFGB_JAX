import jax.numpy as jnp
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, CIFAR10
import jax.random as random
import os
import jax
from utils.api import ServerHyperParams, Classifier, Batch
from utils.dx_loss import vg_ce
from utils.loss import v_ce
from models.convnet import CONVNET
from models.mlp import MLP
from utils.classifier import get_classifier_fn
from utils.oracles import regression_oracle

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
x = (x / 255. - jnp.ones(3) * .5) / (jnp.ones(3) * .5)
x_test = (x_test / 255. - jnp.ones(3) * .5) / (jnp.ones(3) * .5)


# x = (x/255. - jnp.array((0.4914, 0.4822, 0.4465))) / jnp.array((0.2023, 0.1994, 0.2010))
# x_test = (x_test/255. - jnp.array((0.4914, 0.4822, 0.4465))) / jnp.array((0.2023, 0.1994, 0.2010))
# ================= configuration =================
num_rounds = 100
distill_ratio = .5
lr_0 = 1.
num_distill_rounds = 1
num_local_steps = 200
num_clients = 100
s = .1
num_classes = 10
num_channels = 3
image_size = 32
oracle_num_steps = 40000
oracle_lr = 1e-3
oracle_batch_size = 32
distill_oracle_num_steps = 40000
distill_oracle_lr = 1e-4
distill_oracle_batch_size = 32
num_sampled_clients = 10
dataset = "cifar10"
model = CONVNET()
get_classifier_fn = jax.partial(get_classifier_fn, model)
model_apply_fn = jax.jit(model.apply)
hyperparams = ServerHyperParams(num_rounds=num_rounds, distill_ratio=distill_ratio, lr_0=lr_0,
                                num_sampled_clients=num_sampled_clients, num_distill_rounds=num_distill_rounds,
                                num_local_steps=num_local_steps, num_clients=num_clients, s=s,
                                num_classes=num_classes, oracle_num_steps=oracle_num_steps, oracle_lr=oracle_lr,
                                oracle_batch_size=oracle_batch_size, num_channels=num_channels,
                                get_classifier_fn=get_classifier_fn,
                                distill_oracle_batch_size=distill_oracle_batch_size,
                                distill_oracle_lr=distill_oracle_lr, distill_oracle_num_steps=distill_oracle_num_steps,
                                image_size=image_size)

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
x_distill = x[0:int(x.shape[0] * hyperparams.distill_ratio)]
batch = Batch(x_train, y_train)
batch_d = Batch(x_distill, None)
# del x, y, index

# ================= initialization =================
model = CONVNET()
# model = MLP()
key, subkey = random.split(key)

# ================= initialization =================

num_split = 10
xs = jnp.split(x_train, num_split)
xts = jnp.split(x_test, num_split)
xds = jnp.split(x_distill, num_split)

# classifier = Classifier(None, None)
# ================= run experiment =================
for round in range(hyperparams.num_rounds):
    if round == 0:
        f_data = jnp.zeros((x_train.shape[0], 10))
        f_x_test = jnp.zeros((x_test.shape[0], 10))
        f_distill = jnp.zeros((x_distill.shape[0], 10))
    # run fgb steps
    residual = jnp.zeros((x_train.shape[0], 10))
    print("start running local updates")
    for local_step in range(hyperparams.num_local_steps):
        # print(local_step)
        key, subkey = random.split(key)
        target = - vg_ce(f_data, y_train) + residual  # (negative functional gradient direction)
        params = regression_oracle(model, jnp.expand_dims(batch.x, axis=0), jnp.expand_dims(target, axis=0), subkey, hyperparams)
        # new_weight = hyperparams.lr_0 / (round * hyperparams.num_local_steps + local_step + 1) ** .5
        new_weight = hyperparams.lr_0 / (local_step + 1) ** .5 * jnp.ones((1, 1))
        # new_weight = hyperparams.lr_0
        # predict = jnp.concatenate([model.apply(opt.target, _x) for _x in xs])
        classfier = Classifier(params, jnp.ones((1, 1)))
        classfier_fn = get_classifier_fn(classfier)
        predict = jnp.concatenate([classfier_fn(x) for x in xs], axis=0)
        residual = target - predict
        # params_list.append(new_params)
        # weight_list.append(new_weight)
        f_data += predict * new_weight



        # print("test fgb result")
        # predict_test = model.apply(opt.target, x_test)
        predict_test = jnp.concatenate([classfier_fn(x) for x in xts], axis=0)
        f_x_test += predict_test * new_weight
        test_loss = v_ce(f_x_test, y_test)
        pred = jnp.argmax(f_x_test, axis=1)
        corrct = jnp.true_divide(
            jnp.sum(jnp.equal(pred, jnp.reshape(y_test, pred.shape))),
            y_test.shape[0])
        print("step %5d, test accuracy % .4f" % (local_step, corrct))


        # predict_distill = predict_fn(opt.target, xds)
        # f_distill += predict_distill
    print("finish running local steps")

    # print("test fgb result")
    # classifier = Classifier(params_list, weight_list, None)
    # classifier = get_classifier_fn(classifier)
    # f_x_test = classifier(x_test)
    # test_loss = v_ce(f_x_test, y_test)
    # pred = jnp.argmax(f_x_test, axis=1)
    # corrct = jnp.true_divide(
    #     jnp.sum(jnp.equal(pred, jnp.reshape(y_test, pred.shape))),
    #     y_test.shape[0])
    # print("round %5d, test accuracy % .4f" % (round, corrct))

    # distill
    print("start distillating")
    for distill_step in range(hyperparams.num_distill_rounds):
        key, subkey = random.split(key)
        opt = regression_oracle(model, batch_d, f_distill, subkey, hyperparams)
        # opt = distill_oracle(model, batch_d, f_distill, subkey, hyperparams)
        params = opt.target
        f_distill = predict_fn(params, xds)
    print("finish distillating")


    print("test distill result")
    f_x_test = predict_fn(params, xts)
    test_loss = v_ce(f_x_test, y_test)
    pred = jnp.argmax(f_x_test, axis=1)
    corrct = jnp.true_divide(
        jnp.sum(jnp.equal(pred, jnp.reshape(y_test, pred.shape))),
        y_test.shape[0])
    print("round %5d, test accuracy % .4f" % (round, corrct))
