import dis
import os
import sys

sys.path.append("/workspaces/KernelCNN/app/mains")
# print(sys.path)
from main_KernelCNN import main_kernelCNN
from main_LeNet import main_LeNet
from main_handcrafted import main_HOG
import numpy as np


def execute_each_datasets_each_samples(
    file_dir: str, model: str, datasets_array: list, sample_num_array: list
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
                            embedding_method=["LE", "LE"],
                            block_size=[5, 5],
                            stride = stride,
                            layers_BOOL=[1, 0, 0, 0],
                        )
                    elif model == "LeNet":
                        accuracy = main_LeNet(
                            n,
                            10000,
                            dataset,
                            block_size=[5, 5],
                            stride = stride,
                            display=True,
                            layers_BOOL=[1, 0, 0, 0],
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
