import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import layers
#import layers_originalGP as layers

from keras.datasets import mnist
from keras.utils import to_categorical

def main_mnist(num_train , num_test, embedding_method, num_blocks):

    #データセットのロード
    (X_train, Y_train), (X_test,Y_test) = mnist.load_data()

    #データ整形
    X_train = X_train.reshape(-1, 28,28,1)
    X_test = X_test.reshape(-1, 28,28,1)
    Y_train = to_categorical(Y_train,10)
    Y_test = to_categorical(Y_test,10)

    X_train = X_train[:num_train]
    Y_train = Y_train[:num_train]
    X_test = X_test[:num_test]
    Y_test = Y_test[:num_test]

    #モデル定義
    model = layers.Model(display=True)
    model.data_set_name = "MNIST"
    model.add_layer(layers.KIMLayer(block_size=5, channels_next = 6, stride = 1, padding=True, emb=embedding_method, num_blocks=num_blocks))
    model.add_layer(layers.MaxPoolingLayer(pool_size=2))
    model.add_layer(layers.KIMLayer(block_size=5, channels_next = 16, stride = 1, padding=False, emb=embedding_method, num_blocks=num_blocks))
    model.add_layer(layers.MaxPoolingLayer(pool_size=2))
    #model.add_layer(layers.KIMLayer(block_size=5, channels_next = 32, stride = 1, padding=False, emb=embedding_method, num_blocks=num_blocks))
    #model.add_layer(layers.KIMLayer(block_size=5, channels_next = 120, stride = 1, emb=emb))

    model.fit(X_train, Y_train)
    output = model.predict(X_test, Y_test)

if __name__ == '__main__':
    args = sys.argv
    n = int(args[1])  #train
    m = int(args[2])  #test
    emb = args[3]
    
    main_mnist(num_train=n, num_test=m, embedding_method=emb, num_blocks=5000)
    #print(output[0])
    #print(output[1])
    
