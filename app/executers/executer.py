import os
import numpy as np

# import LeNet
datasets_array = ["MNIST", "CIFAR10", "KTH"]
embedding_array = ["LE", "PCA", "LLE", "TSNE"]
num_samples_array = [10, 100, 300, 648]
from components import execute_each_datasets_each_samples

# ベースライン

if True:
        execute_each_datasets_each_samples(
            file_dir="../results/average_accuracy_LeNet.txt",
            model="LeNet",
            datasets_array=datasets_array,
            sample_num_array=num_samples_array,
        )
        execute_each_datasets_each_samples(
            file_dir="../results/average_accuracy_kernel.txt",
            model="KernelCNN",
            datasets_array=datasets_array,
            sample_num_array=num_samples_array,
        )
        execute_each_datasets_each_samples(
            file_dir="../results/average_accuracy_HOG.txt",
            model="HOG",
            datasets_array=datasets_array,
            sample_num_array=num_samples_array,
        )
