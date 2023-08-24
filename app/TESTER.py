import embedding_analysis
import numpy as np
import GPy
from functions import binarize_images
from functions import pad_images

from keras.datasets import mnist
from keras.utils import to_categorical

from functions import calculate_similarity

if __name__ == '__main__':
    
    #データセットのロード
    (X_train, Y_train), (X_test,Y_test) = mnist.load_data()

    X_train = pad_images(X_train)
    X_test = pad_images(X_test)
    
    image_rows = 32
    image_cols = 32
    image_color = 1 #グレースケール
    input_shape = (image_rows, image_cols, image_color)
    out_size = 10

    #データ整形
    X_train = X_train.reshape(-1, image_color, image_rows, image_cols) 
    X_test = X_test.reshape(-1, image_color, image_rows, image_cols, ) 
    Y_train = to_categorical(Y_train,out_size)
    Y_test = to_categorical(Y_test,out_size)

    n = 1000  #train
    #m = int(args[2])  #test
    m = 10000

    X_train = X_train[:n]
    Y_train = Y_train[:n]
    X_test = X_test[:m]
    Y_test = Y_test[:m]

    X_train = binarize_images(X_train).reshape(X_train.shape[0], -1)
    X_test = binarize_images(X_test).reshape(X_test.shape[0], -1)
    
    kernel = GPy.kern.RBF(input_dim = 32*32)
    GP = GPy.models.GPRegression(X_train, Y_train, kernel=kernel)
    GP.optimize()
    
    Y_predict, _ = GP.predict(X_test)
    Y_predict = np.array(Y_predict)
    Y_predict = [np.argmax(Y_predict[n,:]) for n in range(X_test.shape[0])]
    Y_test= [np.argmax(Y_test[n,:]) for n in range(Y_test.shape[0])]
    print(calculate_similarity(Y_predict, Y_test))