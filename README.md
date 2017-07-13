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

- [x] AlexNet

  - [x] Local Response Normalization

- [ ] VGG16


## MNISTベンチマーク

### 実験条件

|条件|値|
:-:|:-:
トレーニングデータ|60,000件
テストデータ|10,000件
バッチサイズ|100
エポック数|20エポック

### 結果

|名前|対応スクリプト|train acc|test acc|
|:--:|:--:|:--:|:--:|
|3層ニューラルネット|`src/04_back_propagation.py`|0.988|0.971|
|5層ニューラルネット|`src/05_deeper_network.py`|0.9962|0.9768|







## 参考

[ゼロから作るDeep Learning ――Pythonで学ぶディープラーニングの理論と実装](https://www.oreilly.co.jp/books/9784873117584/)
