# coding:utf-8
import numpy as np

def step_function(x):
    """
    ステップ関数。パーセプトロンの活性化関数。
    線形関数で0, 1しか取らない。
    """

    y = x > 0
    return y.astype(np.int)

def sigmoid(x):
    """
    シグモイド関数。
    非線形関数で[0, 1]の範囲の連続値を取る。
    """

    return 1. / (1 + np.exp(-x))

def relu(x):
    """
    relu関数。
    """
    # y = np.zeros_like(x)
    # y[ x > 0 ] = x[ x > 0 ]

    return np.maximum(0., x)

def softmax(x):
    """
    softmax関数。
    各クラスへの所属確率の大小を保ちつつ、確率として扱えるようにする（総和が1）。
    """
    c = np.max(x)
    exp_x = np.exp(x-c)
    return exp_x / np.sum(exp_x)

def main():
    # y = step_function(np.array([-1, 0, 1]))
    # print(y)
    # y = sigmoid(np.array([-1, 0, 1]))
    # print(y)
    y = relu(np.array([-1, 0, 1, 2]))
    print(y)

if __name__=="__main__":
    main()
