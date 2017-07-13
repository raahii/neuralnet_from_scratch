# coding : utf-8
import sys, os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
from lib.common_functions import cross_entropy_error
from lib.my_neural_net import MyNeuralNet
from lib.layers import *
from lib.trainer import Trainer

# データセットのロード
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=True)

# 28x28の画像を入力し、0-9の文字のいずれかを知りたい
network = MyNeuralNet(cross_entropy_error)

# レイヤを追加
network.add_layer(Affine(28*28, 50, "he"))
network.add_layer(BatchNormalization())
network.add_layer(Relu())
network.add_layer(Affine(50, 100, "he"))
network.add_layer(BatchNormalization())
network.add_layer(Relu())
network.add_layer(Affine(100, 100, "he"))
network.add_layer(BatchNormalization())
network.add_layer(Relu())
network.add_layer(Affine(100, 100, "he"))
network.add_layer(BatchNormalization())
network.add_layer(Relu())
network.add_layer(Affine(100, 100, "he"))
network.add_layer(BatchNormalization())
network.add_layer(Relu())
network.add_layer(Affine(100, 10, "he"))
network.add_layer(Softmax())

# 学習
trainer = Trainer(network, x_train, t_train, x_test, t_test)
trainer.train(lr=0.1, epoch_num=20, batch_size=100)
trainer.savefig("../data/batch_normalization.png")
