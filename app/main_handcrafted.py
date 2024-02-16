import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from tqdm import tqdm
from keras.datasets import mnist, cifar10, fashion_mnist
from keras.preprocessing import sequence
from sklearn.svm import SVC
from sklearn import metrics
from functions import pad_images
import numpy as np
import cv2
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from matplotlib import pyplot as plt
from skimage import feature
from skimage.color import rgb2gray

def select_dataset(datasets: str, imagesize: int):
    if (datasets == 'MNIST') or (datasets == 'FMNIST'):
        if datasets == 'MNIST':
            (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
        if datasets == 'FMNIST':
            (X_train, Y_train), (X_test, Y_test) = fashion_mnist.load_data()
        X_train = X_train.reshape(-1, 28, 28, 1) 
        X_test = X_test.reshape(-1, 28, 28, 1)
        X_train = pad_images(X_train, imagesize).astype(np.uint8)  
        X_test = pad_images(X_test, imagesize).astype(np.uint8)  
    elif datasets == 'CIFAR10':
        (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
        X_train = X_train.reshape(-1, 32, 32, 3).astype(np.uint8) 
        X_test = X_test.reshape(-1, 32, 32, 3).astype(np.uint8) 
        if imagesize > 32:
            X_train = pad_images(X_train, imagesize)
            X_test = pad_images(X_test, imagesize)

  
    return X_train, Y_train, X_test, Y_test

def main_HOG(num_train=1000 , num_test=1000, datasets: str = "MNIST"):
    imagesize = 32
    X_train, Y_train, X_test, Y_test = select_dataset(datasets, imagesize)

    X_train = X_train[:num_train]
    Y_train = Y_train[:num_train]
    X_test = X_test[:num_test]
    Y_test = Y_test[:num_test]
    
    # HOG feature extraction
    descriptors_train = [np.concatenate([feature.hog(channel, orientations=8, pixels_per_cell=(4, 4), cells_per_block=(1, 1), transform_sqrt=True, block_norm="L2-Hys", visualize=False) for channel in train_image.transpose((2, 0, 1))]) for train_image in X_train]
    descriptors_test = [np.concatenate([feature.hog(channel, orientations=8, pixels_per_cell=(4, 4), cells_per_block=(1, 1), transform_sqrt=True, block_norm="L2-Hys", visualize=False) for channel in test_image.transpose((2, 0, 1))]) for test_image in X_test]
    # Convert descriptors to numpy arrays
    descriptors_train = np.array(descriptors_train).reshape(len(descriptors_train), -1)
    descriptors_test = np.array(descriptors_test).reshape(len(descriptors_test), -1)
    # Standardize and Normalize the descriptors
    descriptors_train = StandardScaler().fit_transform(descriptors_train)
    descriptors_test = StandardScaler().fit_transform(descriptors_test)
    descriptors_train = MinMaxScaler().fit_transform(descriptors_train)
    descriptors_test = MinMaxScaler().fit_transform(descriptors_test)
    
    # SVM classification
    SVM = SVC(kernel='rbf', C=10.0, gamma='auto', probability=True, decision_function_shape='ovr')
    SVM.fit(descriptors_train, Y_train)
    Y_pred = SVM.predict(descriptors_test)
    accuracy = metrics.accuracy_score(Y_test, Y_pred) * 100
    classification_report = metrics.classification_report(Y_test, Y_pred)
    print(classification_report)
    print("Accuracy:", accuracy)
    
    return accuracy

if __name__ == '__main__':
    args = sys.argv
    if len(args) != 4:
        print("Usage: python main_handcrafted.py <train> <test> <dataset>")
        sys.exit(1)
    else:
        n = int(args[1])  #train
        m = int(args[2])  #test
        dataset_name = args[3]

    main_HOG(num_train=n, num_test=m, datasets=dataset_name)