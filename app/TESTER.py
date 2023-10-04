import embedding_analysis
import numpy as np
import GPy
from functions import binarize_images
from functions import pad_images
from scipy import stats

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

    n = 60000  #train
    #m = int(args[2])  #test
    m = 10000

    X_train = X_train[:n]
    Y_train = Y_train[:n]
    X_test = X_test[:m]
    Y_test = Y_test[:m]

    X_train = binarize_images(X_train).reshape(X_train.shape[0], -1)
    X_test = binarize_images(X_test).reshape(X_test.shape[0], -1)
    
    if n > 10000:
        GP = []
        num_GP = int((n-1)/10000 + 1)
        kernel = GPy.kern.RBF(input_dim = 32*32)
        for i in range(num_GP):
            print('learning {}'.format(i+1))
            X_sep = X_train[10000*i:10000*(i+1)]
            Y_sep = Y_train[10000*i:10000*(i+1)]
            GP.append(GPy.models.GPRegression(X_sep,Y_sep, kernel=kernel))
            GP[-1].optimize()
    else:
        kernel = GPy.kern.RBF(input_dim = 32*32)
        GP = GPy.models.GPRegression(X_train, Y_train, kernel=kernel)
        GP.optimize()
        
    if n > 10000:
        predictions = []
        for i in range(num_GP):
            Y_predicted, _ = GP[i].predict(X_test)
            Y_predicted = np.array(Y_predicted)
            predict = [np.argmax(Y_predicted[n,:]) for n in range(m)]
            predictions.append(predict)
        ensemble_predictions = np.vstack(predictions)
        Y_predict = stats.mode(ensemble_predictions, axis=0).mode.ravel()
            
    else:
        Y_predicted, _ = GP.predict(X_test)
        Y_predicted = np.array(Y_predicted)
        Y_predict = [np.argmax(Y_predicted[n,:]) for n in range(m)]
    
    Y_test= [np.argmax(Y_test[n,:]) for n in range(Y_test.shape[0])]
    print(calculate_similarity(Y_predict, Y_test))