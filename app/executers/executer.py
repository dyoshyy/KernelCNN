import os
import numpy as np
#import LeNet
datasets_array = ['MNIST', 'FMNIST', 'CIFAR10']
embedding_array = ['LE', 'PCA', 'LLE', 'TSNE']
from components import execute_each_datasets_each_samples
#ベースライン     

if True:
    execute_each_datasets_each_samples(file_dir='./average_accuracy_LeNet.txt', model='LeNet', datasets_array=datasets_array, sample_num_array=[1000, 10000, 30000, 60000])
    #execute_each_datasets_each_samples(file_dir='./average_accuracy_kernel.txt', model='KernelCNN', datasets_array=datasets_array, sample_num_array=[1000, 10000, 30000, 60000])
    #execute_each_datasets_each_samples(file_dir='./average_accuracy_HOG.txt', model='HOG', datasets_array=datasets_array, sample_num_array=[1000, 10000, 30000, 60000])
    