# coding : utf-8
import sys, os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
from lib.common_functions import cross_entropy_error
from lib.my_neural_net import MyNeuralNet
from lib.layers import Affine
from lib.activate_functions import Sigmoid, Softmax, Relu

import numpy as np
import matplotlib.pyplot as plt
plt.style.use("ggplot")

# データセットのロード
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=True)

# 28x28の画像を入力し、0-9の文字のいずれかを知りたい
network = MyNeuralNet(cross_entropy_error)

# レイヤを追加
network.add_layer(Affine(28*28, 100, Sigmoid()))
network.add_layer(Affine(100,   100, Sigmoid()))
network.add_layer(Affine(100,   100, Sigmoid()))
network.add_layer(Affine(100,   100, Sigmoid()))
network.add_layer(Affine(100,   10 , Softmax()))

# 学習
train_size = x_train.shape[0]
batch_size = 100

iter_per_epoch = max(int(train_size / batch_size), 1)

iters_num = iter_per_epoch * 300
learning_rate = 1.0

loss_list = []
train_acc_list = []
test_acc_list = []
sum_loss = 0.0

plt.figure(figsize=(16,4))
for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    
    x = x_train[batch_mask]
    t = t_train[batch_mask]

    y = network.forward(x)
    network.backward(y, t)

    for layer in network.layers:
        layer.W -= learning_rate * layer.dW
        layer.b -= learning_rate * layer.db
    
    sum_loss += network.loss(y, t)

    if i % iter_per_epoch == 0:
        loss_list.append(sum_loss / iter_per_epoch)
        sum_loss = 0

        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(int(i/iter_per_epoch), ":", train_acc, test_acc)

        plt.clf()
        # x = np.array(range(1, len(loss_list)+1))
        # plt.plot(x, loss_list)
        # plt.xlabel("epoch")
        # plt.ylabel("loss")
        
        plt.suptitle("init param to gaussian random number")
        for i, layer in enumerate(network.layers[0:-1]):
            # import pdb; pdb.set_trace()
            plt.subplot(1, len(network.layers)-1, i+1)
            plt.title("{}-layer".format(i+1))
            plt.hist(layer.activate_function.y.flatten(), bins=30)

        plt.draw()
        plt.pause(0.05)
