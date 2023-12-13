import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import layers
import functions
#import layers_originalGP as layers
from functions import pad_images

from keras.datasets import mnist, cifar10, fashion_mnist
from keras.utils import to_categorical

def main_kernelCNN(num_train , num_test, datasets: str,  B=3000, embedding_method=["LE", "LE"], block_size=[5, 5]):
    
    if (datasets == 'MNIST') or (datasets == 'FMNIST'):
        if datasets == 'MNIST':
            (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
        if datasets == 'FMNIST':
            (X_train, Y_train), (X_test, Y_test) = fashion_mnist.load_data()

        X_train = X_train.reshape(-1, 28, 28, 1) 
        X_test = X_test.reshape(-1, 28, 28, 1)
        X_train = pad_images(X_train, 32)
        X_test = pad_images(X_test, 32)
    elif datasets == 'CIFAR10':
        (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
        X_train = X_train.reshape(-1, 32, 32, 3) 
        X_test = X_test.reshape(-1, 32, 32, 3)

    #データ整形
    Y_train = to_categorical(Y_train,10)
    Y_test = to_categorical(Y_test,10)

    X_train = X_train[:num_train]
    Y_train = Y_train[:num_train]
    X_test = X_test[:num_test]
    Y_test = Y_test[:num_test]

    #モデル定義
    model = layers.Model(display=True)
    model.data_set_name = datasets
    model.add_layer(layers.KIMLayer(block_size=block_size[0], channels_next = 6, stride = 1, padding=False, emb=embedding_method[0], num_blocks=B))
    model.add_layer(layers.MaxPoolingLayer(pool_size=2))
    model.add_layer(layers.KIMLayer(block_size=block_size[1], channels_next = 16, stride = 1, padding=False, emb=embedding_method[1], num_blocks=B))
    model.add_layer(layers.MaxPoolingLayer(pool_size=2))
    #model.add_layer(layers.KIMLayer(block_size=block_size[2], channels_next = 30, stride = 1, padding=True, emb=embedding_method[1], num_blocks=B))
    #model.add_layer(layers.MaxPoolingLayer(pool_size=2))
    #model.add_layer(layers.KIMLayer(block_size=5, channels_next = 32, stride = 1, padding=False, emb=embedding_method, num_blocks=num_blocks))
    #model.add_layer(layers.KIMLayer(block_size=5, channels_next = 120, stride = 1, emb=emb))
    model.add_layer(layers.LabelLearningLayer_GaussianProcess())
    #model.add_layer(layers.LabelLearningLayer_NeuralNetwork())

    model.fit(X_train, Y_train)
    accuracy = model.predict(X_test, Y_test)
    return accuracy

if __name__ == '__main__':
    args = sys.argv
    num_train = int(args[1])  #train
    num_test = int(args[2])  #test
    dataset_name = args[3]
    embedding_method = list(map(str, args[3].split(',')))
    block_size = list(map(int, args[4].split(',')))
    #arguments = [n,m,emb,3000]
    main_kernelCNN(num_train=num_train, num_test=num_test, datasets=dataset_name, B=3000, embedding_method=embedding_method, block_size=block_size)
    
