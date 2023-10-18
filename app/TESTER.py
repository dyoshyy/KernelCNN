
import numpy as np
import GPy
from functions import binarize_images
from functions import pad_images
from scipy import stats
import sys

from keras.datasets import mnist, fashion_mnist, cifar10
from keras.utils import to_categorical

from functions import calculate_similarity

def main_GP(num_train, num_test, datasets:str):
    
    print('Number of training samples:', num_train)
    if (datasets == 'MNIST') or (datasets == 'FMNIST'):
        if datasets == 'MNIST':
            (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
        if datasets == 'FMNIST':
            (X_train, Y_train), (X_test, Y_test) = fashion_mnist.load_data()
        X_train = pad_images(X_train)
        X_test = pad_images(X_test)
        X_train = X_train.reshape(-1, 32, 32, 1) 
        X_test = X_test.reshape(-1, 32, 32, 1)
        input_dim = 32*32
    elif datasets == 'CIFAR10':
        (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
        X_train = X_train.reshape(-1, 32, 32, 3) 
        X_test = X_test.reshape(-1, 32, 32, 3)
        input_dim = 32*32*3
        
    #データ整形
    Y_train = to_categorical(Y_train,10)
    Y_test = to_categorical(Y_test,10)

    X_train = X_train[:num_train]
    Y_train = Y_train[:num_train]
    X_test = X_test[:num_test]
    Y_test = Y_test[:num_test]
    
    n = X_train.shape[0]

    X_train = binarize_images(X_train).reshape(X_train.shape[0], -1)
    X_test = binarize_images(X_test).reshape(X_test.shape[0], -1)
    
    if n > 10000:
        GP = []
        num_GP = int((n-1)/10000 + 1)
        kernel = GPy.kern.RBF(input_dim = input_dim) + GPy.kern.Bias(input_dim = input_dim) + GPy.kern.Linear(input_dim = input_dim)
        for i in range(num_GP):
            print('learning {}'.format(i+1))
            X_sep = X_train[10000*i:10000*(i+1)]
            Y_sep = Y_train[10000*i:10000*(i+1)]
            GP.append(GPy.models.GPRegression(X_sep,Y_sep, kernel=kernel))
            GP[-1].optimize()
    else:
        kernel = GPy.kern.RBF(input_dim = input_dim) + GPy.kern.Bias(input_dim = input_dim) + GPy.kern.Linear(input_dim = input_dim)
        GP = GPy.models.GPRegression(X_train, Y_train, kernel=kernel)
        GP.optimize()
        
    if n > 10000:
        predictions = []
        for i in range(num_GP):
            Y_predicted, _ = GP[i].predict(X_test)
            Y_predicted = np.array(Y_predicted)
            predict = [np.argmax(Y_predicted[n,:]) for n in range(num_test)]
            predictions.append(predict)
        ensemble_predictions = np.vstack(predictions)
        Y_predict = stats.mode(ensemble_predictions, axis=0).mode.ravel()
            
    else:
        Y_predicted, _ = GP.predict(X_test)
        Y_predicted = np.array(Y_predicted)
        Y_predict = [np.argmax(Y_predicted[n,:]) for n in range(num_test)]
    
    Y_test= [np.argmax(Y_test[n,:]) for n in range(Y_test.shape[0])]
    print(calculate_similarity(Y_predict, Y_test))
    
if __name__ == '__main__':
    args = sys.argv
    num_train = int(args[1])  #train
    num_test = int(args[2])  #test
    dataset = args[3]
    main_GP(num_train, num_test, dataset)