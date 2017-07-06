# coding:utf-8

import sys, os 
sys.path.append(os.pardir)
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from dataset.mnist import load_mnist
from lib.my_neural_net import MyNeuralNet
from lib.common_functions import cross_entropy_error
from lib.layers import Affine, Conv
from lib.activation_functions import *


# データセットのロード
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=False, one_hot_label=True)

# ネットワークを定義
network = MyNeuralNet(
            cost_function = cross_entropy_error
        )

# レイヤを追加
C, IH, IW = 1, 28, 28

P, S = 0, 1
FN, FH, FW = 30, 5, 5
OH = int( (IH + 2*P - FH) / S + 1 )
OW = int( (IW + 2*P - FW) / S + 1 )

P, S = 0, 2
PH, PW = 2, 2
OH = int( (OH + 2*P - PH) / S + 1 )
OW = int( (OW + 2*P - PW) / S + 1 )

conv_output_size = FN*OH*OW

conv = Conv( activation_function = [Relu(), Pooling(PH, PW, padding=P, stride=S)],
             filter_shape = (FN, C, FH, FW) )
affine1 = Affine( conv_output_size, 100, Relu(), "he" )
affine2 = Affine( 100, 10, Softmax(), "he" )

network.add_layer(conv)
network.add_layer(affine1)
network.add_layer(affine2)

# 学習
train_size = x_train.shape[0]
batch_size = 100

iter_per_epoch = max(int(train_size / batch_size), 1)

iters_num = iter_per_epoch * 100
learning_rate = 0.1

plt.style.use("ggplot")
plt.figure(figsize=(12,8))
loss_list = []
train_acc_list = []
test_acc_list = []

for i in tqdm(range(iters_num)):
    batch_mask = np.random.choice(train_size, batch_size)
    
    x = x_train[batch_mask]
    t = t_train[batch_mask]

    y = network.forward(x)
    network.backward(y, t)

    for layer in network.layers:
        layer.W -= learning_rate * layer.dW
        layer.b -= learning_rate * layer.db
    
    loss_list.append(network.loss(y, t))
    
    if i!=0 and i % iter_per_epoch == 0:
        plt.clf()

        # plt.subplot(1, 2, 1)
        x = np.array(range(1, len(loss_list)+1))
        plt.plot(x, loss_list)
        plt.xlabel("iter")
        plt.ylabel("loss")

        # train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        print(test_acc)
        # train_acc_list.append(train_acc)
        # test_acc_list.append(test_acc)
        # x = np.array(range(1, len(train_acc_list)+1))
        # plt.subplot(1, 2, 2)
        # plt.plot(x, train_acc_list)
        # plt.plot(x, test_acc_list)
        # plt.xlabel("iter")
        # plt.ylabel("acc")
        # plt.legend()

        plt.draw()
        plt.pause(0.01)

x = np.array(range(1, len(loss_list)+1))
plt.plot(x, loss_list)
plt.xlabel("iter")
plt.ylabel("loss")
plt.savefig("./power_of_cnn.png")

print("--------- last result -----------")
print("test_acc: {}".format(network.accuracy(x_test, t_test)))
# print("train_acc: {}".format(network.accuracy(x_train, t_train)))
print("---------------------------------")
