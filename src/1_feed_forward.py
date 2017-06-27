# coding : utf-8
import sys, os
sys.path.append(os.pardir)
import numpy as np
from PIL import Image
import pickle

from dataset.mnist import load_mnist
from lib.common_functions import sigmoid, softmax, img_show
from lib.three_layer_net import ThreeLayerNet

# データセットのロード
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)

# 28x28の画像を入力し、0-9の文字のいずれかを知りたい
network = ThreeLayerNet(28*28, 50, 100, 10)

# パラメータをロード
with open("../dataset/sample_weight.pkl", 'rb') as f:
    params = pickle.load(f)
network.set_params(params)

# 予測
while True:
    randint = np.random.randint(x_test.shape[0])
    x, t = x_test[randint], t_test[randint]
    img_show(x.reshape(28, 28) * 255)

    y_hat = np.argmax(network.forward(x))
    print("predicted: {}".format(y_hat))
    print("answer: {}".format(t))
    sys.stdin.read(1)
