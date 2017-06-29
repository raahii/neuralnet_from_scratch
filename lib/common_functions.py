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

    # sigmoid_range = 34.538776394910684
    # np.clip(x, -sigmoid_range, sigmoid_range, out=x)

    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_grad(x):
    """
    シグモイドの勾配関数
    """

    return sigmoid(x) * (1.0-sigmoid(x))

def relu(x):
    """
    relu関数。
    """
    # y = np.zeros_like(x)
    # y[ x > 0 ] = x[ x > 0 ]

    return np.maximum(0, x)

def softmax(x):
    """
    softmax関数。
    各クラスへの所属確率の大小を保ちつつ、確率として扱えるようにする（総和が1）。
    """
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    x = x - np.max(x)
    return np.exp(x) / np.sum(np.exp(x))

def mean_squared_error(y, t):
    """
    二乗和誤差関数
    """
    batch_size = y.shape[0]
    return 0.5/batch_size * np.sum((y-t)**2)

def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
        
    # 教師データがone-hot-vectorの場合、正解ラベルのインデックスに変換
    if t.size == y.size:
        t = t.argmax(axis=1)
             
    batch_size = y.shape[0]

    return -np.sum(np.log(y[np.arange(batch_size), t])) / batch_size

def numerical_diff(f, x):
    """
    数値微分
    """
    h = 1e-4
    grad = np.zeros_like(x)

    for i in range(x.size):
        tmp_val = x[i]
        x[i] = float(tmp_val) + h
        fx_plus= f(x)
        x[i] = float(tmp_val) - h
        fx_minus = f(x)
        
        grad[i] = (fx_plus - fx_minus) / (2*h)
        x[i] = tmp_val

    return grad
    
def numerical_gradient(f, x):
    """
    数値微分を用いて勾配を計算する。
    """
    
    if x.ndim == 1:
        return numerical_diff(f, x)

    ret = np.zeros_like(x)
    for i in range(x.shape[0]):
        ret[i,:] = numerical_diff(f, x[i])

    return ret

def gradient_descent(f, init_params, lr = 0.01, step_num=100):
    """
    最急降下法
    """
    
    params = init_params
    
    for _ in range(step_num):
        grad = numerical_gradient(f, params)
        params -= lr * grad
        # print(params)

    return params

def gaussian_init(input_size, output_size):
    """
    ガウス分布から取ってきたランダム値で初期化した
    パラメータを返す
    """

    return 0.01 * np.random.randn(input_size, output_size)

def xavier_init(input_size, output_size):
    """
    Xavierの初期値で初期化したパラメータを返す
    """

    return np.sqrt(input_size) * np.random.randn(input_size, output_size)

def he_init(input_size, output_size):
    """
    Heの初期値で初期化したパラメータを返す
    """

    return np.sqrt(input_size) * np.random.randn(input_size, output_size)
def main():
    # y = step_function(np.array([-1, 0, 1]))
    # print(y)
    # y = sigmoid(np.array([-1, 0, 1]))
    # print(y)
    y = relu(np.array([-1, 0, 1, 2]))
    print(y)

if __name__=="__main__":
    main()
