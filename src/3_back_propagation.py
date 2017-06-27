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
from lib.common_functions import sigmoid, softmax, img_show
from lib.three_layer_net import ThreeLayerNet

# データセットのロード
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=True)

# 28x28の画像を入力し、0-9の文字のいずれかを知りたい
network = ThreeLayerNet(28*28, 50, 100, 10)

# 学習
iter_num = 1000
batch_size = 1000
train_size = x_train.shape[0]
lr = 1e-4

loss_list = []

for i in tqdm(range(iter_num)):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    grads = network.numerical_gradient(x_batch, t_batch)

    for key in network.params.keys():
        network.params[key] -= lr * grads[key]

    loss_list.append(network.loss(x_batch, t_batch))
    
    if i%100 == 0:
        plt.clf()
        x = np.array(range(1, len(loss_list)+1))
        plt.plot(x, loss_list)
        plt.xlabel("iter")
        plt.ylabel("loss")
        plt.xlim(1, iter_num+1)
        plt.draw()
        plt.pause(0.05)

x = np.array(range(1, len(loss_list)+1))
plt.plot(x, loss_list)
plt.xlabel("iter")
plt.ylabel("loss")
plt.xlim(1, iter_num+1)
plt.show()
