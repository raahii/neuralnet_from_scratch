#speed coding : utf-8
import sys, os
sys.path.append(os.pardir)
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("ggplot")
from PIL import Image
from tqdm import tqdm
import pickle

from dataset.mnist import load_mnist
from lib.common_functions import *
from lib.three_layer_net import ThreeLayerNet

# データセットのロード
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=True)

# 28x28の画像を入力し、0-9の文字のいずれかを知りたい
network = ThreeLayerNet(input_size=28*28, hidden_size1=50, hidden_size2=100, output_size=10)

# 学習
iter_num = 10000
batch_size = 100
train_size = x_train.shape[0]
lr = 0.1

loss_list = []

for i in tqdm(range(iter_num)):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    grads = network.numerical_gradient(x_batch, t_batch)

    for key in {"W1", "W2", "W3", "b1", "b2", "b3"}:
        network.params[key] -= lr * grads[key]

    loss = network.loss(x_batch, t_batch)
    loss_list.append(loss)
    
    # realtime plot
    plt.clf()
    x = np.array(range(1, len(loss_list)+1))
    plt.plot(x, loss_list)
    plt.xlabel("iter")
    plt.ylabel("loss")
    # plt.xlim(1, iter_num+1)
    plt.draw()
    plt.pause(0.05)

x = np.array(range(1, len(loss_list)+1))
plt.plot(x, loss_list)
plt.xlabel("iter")
plt.ylabel("loss")
plt.xlim(1, iter_num+1)
plt.show()
