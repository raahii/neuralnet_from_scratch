# coding : utf-8
import sys, os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
from lib.common_functions import cross_entropy_error
from lib.my_neural_net import MyNeuralNet
from lib.layers import Affine
from lib.activation_functions import Sigmoid, Softmax, Relu

import numpy as np
import matplotlib.pyplot as plt
plt.style.use("ggplot")

# データセットのロード
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=True)

# 28x28の画像を入力し、0-9の文字のいずれかを知りたい
network = MyNeuralNet(cross_entropy_error)

# レイヤを追加
network.add_layer(Affine(28*28, 50 , Relu()   , "he"))
network.add_layer(Affine(50   , 100, Relu()   , "he"))
network.add_layer(Affine(100  , 20 , Relu()   , "he"))
network.add_layer(Affine(20   , 10 , Softmax(), "he"))

# 学習
train_size = x_train.shape[0]
batch_size = 100

iter_per_epoch = max(int(train_size / batch_size), 1)

iters_num = iter_per_epoch * 300
learning_rate = 0.1

plt.figure(figsize=(12,8))
loss_list = []
train_acc_list = []
test_acc_list = []

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    
    x = x_train[batch_mask]
    t = t_train[batch_mask]

    y = network.forward(x)
    network.backward(y, t)

    for layer in network.layers:
        layer.W -= learning_rate * layer.dW
        layer.b -= learning_rate * layer.db
    
    loss_list.append(network.loss(y, t))

    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(train_acc, test_acc)

        plt.clf()
        x = np.array(range(1, len(loss_list)+1))
        plt.plot(x, loss_list)
        plt.xlabel("iter")
        plt.ylabel("loss")
        plt.draw()
        plt.pause(0.05)
