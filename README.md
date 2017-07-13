# ゼロから作るニューラルネットワーク

スクラッチでニューラルネットワークを実装する

## 目標

スクラッチでVGG16を実装する

## マイルストーン

- [x] 3 layer neural network

  - [x] feed-forward

    - [x] compare processing time for various batch-size
  
  - [x] gradient descent
  
  - [x] back-propagation

- [x] N layer neural network
  
  - [x] xavier, he initial value

  - [x] batch-normalization

  - [x] dropout

- [x] Convolutional neural network

  - [x] Convolution

  - [x] Pooling

- [ ] AlexNet

  - [ ] Local Response Normalization

- [ ] VGG16


## MNISTベンチマーク

### 実験条件

|条件|値|
|:-:|:-:|
|トレーニングデータ|60,000件|
|テストデータ|10,000件|
|バッチサイズ|100|

### 結果

|名前|対応スクリプト|エポック数|train acc|test acc|
|:--:|:--:|:--:|:--:|:--:|
|3層ニューラルネット|`src/04_back_propagation.py`|20|0.988|0.971|
|5層ニューラルネット|`src/05_deeper_network.py`|20|0.9962|0.9768|
|5層ニューラルネット<br>（BN有り）|`src/07_batch_normalization.py`|20|0.9980|0.9737|
|5層ニューラルネット<br>（BN, Dropout有り）|`src/08_dropout.py`|100|0.9957|0.9767|
|簡単なCNN|`src/10_CNN.py`|20|0.9972|0.9834|

## 参考

[ゼロから作るDeep Learning ――Pythonで学ぶディープラーニングの理論と実装](https://www.oreilly.co.jp/books/9784873117584/)
