import time
import sys
import os

import numpy as np

import layers
from functions import pad_images

from keras.datasets import mnist, cifar10, fashion_mnist
from keras.utils import to_categorical


if __name__ == '__main__':
    args = sys.argv

    #データセットのロード
    (X_train, Y_train), (X_test,Y_test) = fashion_mnist.load_data()
    
    X_train = pad_images(X_train)
    X_test = pad_images(X_test)

    #データ整形
    X_train = X_train.reshape(-1, 32,32,1)
    X_test = X_test.reshape(-1, 32,32,1)
    Y_train = to_categorical(Y_train,10)
    Y_test = to_categorical(Y_test,10)

    n = int(args[1])  #train
    m = int(args[2])  #test
    emb = args[3]

    X_train = X_train[:n]
    Y_train = Y_train[:n]
    X_test = X_test[:m]
    Y_test = Y_test[:m]

    #モデル定義
    model = layers.Model(display=True)
    model.data_set_name = "Fashion-MNIST"
    model.add_layer(layers.KIMLayer(block_size=5, channels_next = 6, stride = 1, emb=emb, num_blocks=3000))
    model.add_layer(layers.AvgPoolingLayer(pool_size=2))
    model.add_layer(layers.KIMLayer(block_size=5, channels_next = 16, stride = 1, emb=emb, num_blocks=3000))
    model.add_layer(layers.AvgPoolingLayer(pool_size=2))
    #model.add_layer(layers.KIMLayer(block_size=5, channels_next = 120, stride = 1, emb=emb))

    model.fit(X_train, Y_train)
    output = model.predict(X_test, Y_test)
    #print(output[0])
    #print(output[1])