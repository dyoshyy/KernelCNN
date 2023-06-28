import time
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import numpy as np
#import layer_classes_test as layers

import layers
from functions import binarize_images
from functions import display_images

from keras.datasets import mnist
from keras.utils import to_categorical


if __name__ == '__main__':
    args = sys.argv

    #データセットのロード
    (X_train, Y_train), (X_test,Y_test) = mnist.load_data()

    image_rows = 28
    image_cols = 28
    image_color = 1 #グレースケール
    input_shape = (image_rows, image_cols, image_color)
    out_size = 10

    #データ整形
    X_train = X_train.reshape(-1, image_color, image_rows, image_cols) 
    X_test = X_test.reshape(-1, image_color, image_rows, image_cols, ) 
    Y_train = to_categorical(Y_train,out_size)
    Y_test = to_categorical(Y_test,out_size)

    n = int(args[1])  #train
    m = int(args[2])  #test

    X_train = X_train[:n]
    Y_train = Y_train[:n]
    X_test = X_test[:m]
    Y_test = Y_test[:m]

    #X_train = binarize_images(X_train)
    #X_test = binarize_images(X_test)

    #モデル定義
    model = layers.Model(display=True)
    model.add_layer(layers.KIMLayer(block_size=3, channels_next = 6, stride = 1, num_samples=100))
    model.add_layer(layers.MaxPoolingLayer(pool_size=2))
    model.add_layer(layers.KIMLayer(block_size=3, channels_next = 16, stride = 1, num_samples=100))
    model.add_layer(layers.MaxPoolingLayer(pool_size=2))

    start1 = time.time()
    model.fit(X_train, Y_train)
    start2 = time.time()
    output = model.predict(X_test, Y_test)
    print("fitting time:", start2-start1)
    print("predicting time:", time.time()-start2)
    #print(output[0])
    #print(output[1])