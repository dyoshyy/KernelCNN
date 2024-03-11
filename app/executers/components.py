import dis
import os
import sys

sys.path.append("/workspaces/KernelCNN/app/mains")
# print(sys.path)
from main_KernelCNN import main_kernelCNN
from main_CNN import main_CNN
from main_handcrafted import main_HOG
import numpy as np


def execute_each_datasets_each_samples(
    file_dir: str, model: str, datasets_array: list, sample_num_array: list, model_type: str = "LeNet"
):
    
    for dataset in datasets_array:
        if dataset == "KTH":
            stride = 2
        else:
            stride = 1
        with open(file_dir, "a") as file:
            file.write(f"{dataset}:\n")
            accuracy_each_samples = []        
            for n in sample_num_array:
                N = 1  # iteration number
                accuracy_N_list = []
                for _ in range(N):
                    if model == "KernelCNN":
                        accuracy = main_kernelCNN(
                            n,
                            10000,
                            dataset,
                            B=1000,
                            embedding_method=["SLE", "SLE"],
                            block_size=[5, 5],
                            stride = stride,
                            layers_BOOL=[1, 0, 0, 0],
                        )
                    elif model == "CNN":
                        accuracy = main_CNN(
                            n,
                            10000,
                            dataset,
                            block_size=[5, 5],
                            stride = stride,
                            display=True,
                            layers_BOOL=[1, 0, 0, 0],
                            model_type=model_type,
                        )
                    elif model == "HOG":
                        accuracy = main_HOG(n, 10000, dataset)
                    accuracy_N_list.append(accuracy)
                avg_accuracy = np.mean(accuracy_N_list)
                variance_accuracy = np.var(accuracy_N_list)
                accuracy_each_samples.append(avg_accuracy)
                file.write(f"n={n}\n")
                file.write(f"Average Accuracy: {avg_accuracy}\n")
            file.write(str(accuracy_each_samples) + "\n")
            file.write("----------------------------\n")

def embedding_method_comparison(
    file_dir: str, datasets_array: list, sample_num_array: list, embedding_methods_array: list
):

    for dataset in datasets_array:
        if dataset == "KTH":
            stride = 2
        else:
            stride = 1
        with open(file_dir, "a") as file:
            file.write(f"{dataset}:\n")
            for embedding in embedding_methods_array:
                accuracy_each_samples = []
                file.write(f"{embedding}:\n")
                for n in sample_num_array:
                    accuracy = main_kernelCNN(
                            n,
                            10000,
                            dataset,
                            B=3000,
                            embedding_method=[embedding],
                            block_size=[5, 5],
                            stride = stride,
                            layers_BOOL=[1, 0, 0, 0],
                    )
                    accuracy_each_samples.append(accuracy)
                    file.write(f"n={n}\n")
                    file.write(f"Accuracy: {accuracy}\n")
                file.write(str(accuracy_each_samples) + "\n")
            file.write("----------------------------\n")
            
def features_selection_comparison(
    file_dir: str, datasets_array: list, sample_num_array: list, use_channels_array: list
):

    for dataset in datasets_array:
        if dataset == "KTH":
            stride = 2
        else:
            stride = 1
        with open(file_dir, "a") as file:
            file.write(f"{dataset}:\n")
            for embedding in ["LE", "SLE"]:
                accuracy_each_samples = []
                file.write(f"{embedding}:\n")
                for use_channels in use_channels_array:
                    accuracy = main_kernelCNN(
                            648,
                            10000,
                            dataset,
                            B=1000,
                            embedding_method=[embedding],
                            block_size=[5, 5],
                            stride = stride,
                            layers_BOOL=[1, 0, 0, 0],
                            use_channels = use_channels
                    )
                    accuracy_each_samples.append(accuracy)
                    file.write(f"use channels:{use_channels}\n")
                    file.write(f"Accuracy: {accuracy}\n")
                file.write(str(accuracy_each_samples) + "\n")
            file.write("----------------------------\n")