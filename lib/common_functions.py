# coding:utf-8
import numpy as np
from PIL import Image

def img_show(img):
    """
    PILで画像を表示する
    そのまま画素配列を渡せばよい
    """

    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

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

def mean_squared_error(y, t):
    """
    二乗和誤差関数
    """
    return 0.5 * np.sum((y-t)**2)

def cross_entropy_error(y, t):
    """
    交差エントロピー関数
    one-hot表現であるため、正解のクラスのみ値が出る。
    """

    if y.ndim == 1:
        y = y.reshape(1, y.size)
        t = t.reshape(1, t.size)
        
    batch_size = y.shape[0]
    return -np.sum(t * np.log(y)) / batch_size

def numerical_diff(f, x):
    """
    数値微分
    """
    h = 1e-4
    
    if x.ndim == 1:
        x = x.reshape(1, x.size)

    ret = np.zeros_like(x)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x_minus = np.copy(x[i])
            x_minus[j] -= h
            x_plus = np.copy(x[i])
            x_plus[j] += h

            ret[i][j] = (f(x_plus) - f(x_minus)) / (2*h)

    return ret

def main():
    # y = step_function(np.array([-1, 0, 1]))
    # print(y)
    # y = sigmoid(np.array([-1, 0, 1]))
    # print(y)
    y = relu(np.array([-1, 0, 1, 2]))
    print(y)

if __name__=="__main__":
    main()
