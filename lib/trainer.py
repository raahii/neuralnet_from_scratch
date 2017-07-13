# coding : utf-8
import sys, os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
from lib.common_functions import cross_entropy_error
from lib.my_neural_net import MyNeuralNet
from lib.layers import *

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
plt.style.use("ggplot")

class Trainer:
    def __init__(self, network, x_train, t_train, x_test, t_test):
        self.network = network
        self.x_train = x_train
        self.t_train = t_train
        self.x_test = x_test
        self.t_test = t_test

        self.train_size = x_train.shape[0]

        self.loss_list = None
        self.train_acc_list = None
        self.test_acc_list = None

    def train(self, lr, epoch_num, batch_size,
              debug=True, plot=True, for_cnn=False):
        train_size = self.train_size
        x_train = self.x_train
        t_train = self.t_train
        x_test = self.x_test
        t_test = self.t_test
        network = self.network

        iter_per_epoch = max(int(train_size / batch_size), 1)
        iters_num = iter_per_epoch * epoch_num
        loss_list = []
        train_acc_list = []
        test_acc_list = []

        for i in tqdm(range(iters_num)):
            batch_mask = np.random.choice(train_size, batch_size)
            
            x = x_train[batch_mask]
            t = t_train[batch_mask]

            y = network.forward(x, train_flg=True)
            network.backward(y, t)

            for layer in network.layers:
                layer.update(lr)

            if i != 0 and i % iter_per_epoch == 0:
                loss_list.append(network.loss(y, t))
                if for_cnn:
                    train_acc = network.accuracy(x_train[:10000], t_train[:10000])
                else:
                    train_acc = network.accuracy(x_train, t_train)
                test_acc = network.accuracy(x_test, t_test)
                train_acc_list.append(train_acc)
                test_acc_list.append(test_acc)

                plt.clf()
                plt.subplot(1, 2, 1)
                x = np.array(range(1, len(loss_list)+1))
                plt.plot(x, loss_list)
                plt.xlabel("epoch")
                plt.ylabel("loss")
                plt.xlim(1, epoch_num+1)

                plt.subplot(1, 2, 2)
                plt.plot(x, train_acc_list, "r", label="train_acc")
                plt.plot(x, test_acc_list, "b", label="test_acc")
                plt.xlabel("epoch")
                plt.ylabel("acc")
                plt.xlim(1, epoch_num+1)
                plt.legend()

                plt.draw()
                plt.pause(0.05)

        self.loss_list = loss_list
        self.train_acc_list = train_acc_list
        self.test_acc_list = test_acc_list
        self.epoch_num = epoch_num

    def savefig(self, title, path):
        loss_list = self.loss_list
        train_acc_list = self.train_acc_list
        test_acc_list = self.test_acc_list
        epoch_num = self.epoch_num
        
        plt.figure(figsize=(20,10))
        plt.suptitle(title)
        plt.subplot(1, 2, 1)
        x = np.array(range(1, len(loss_list)+1))
        plt.plot(x, loss_list)
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.xlim(1, epoch_num+1)

        plt.subplot(1, 2, 2)
        plt.plot(x, train_acc_list, "r", label="train_acc")
        plt.plot(x, test_acc_list, "b", label="test_acc")
        plt.xlabel("epoch")
        plt.ylabel("acc")
        plt.xlim(1, epoch_num+1)
        plt.legend()

        plt.savefig(path)
        print("train acc: {}, test acc: {}".format(train_acc_list[-1], test_acc_list[-1]))
