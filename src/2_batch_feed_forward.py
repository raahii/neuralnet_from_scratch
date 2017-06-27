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
batch_size = 100
correct_num = 0
data_size = x_test.shape[0]
for i in range(0, data_size, batch_size):
    x = x_test[i:i+batch_size]
    t = t_test[i:i+batch_size]
    y = np.argmax(network.forward(x), 1)

    correct_num += np.sum((t == y))

print("acc: {0:0.2f}%".format(100 * correct_num / data_size))


# 速度比較
import time

batch_sizes = [1, 10, 100, 1000, data_size]
iter_num = 10000
for batch_size in batch_sizes:
    s = time.time()
    for _ in range(iter_num):
        for i in range(0, data_size, batch_size):
            x = x_test[i:i+batch_size]
            t = t_test[i:i+batch_size]
            y = np.argmax(network.forward(x), 1)
    e = time.time()

    print("{0}: {1:0.2f}s".format(batch_size, e-s))
