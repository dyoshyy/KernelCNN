import os
import numpy as np
from torch import embedding

# import LeNet
datasets_array = ["MNIST", "CIFAR10", "KTH"]
embedding_array = ["LE", "PCA", "LLE", "TSNE"]
num_samples_array = [10, 100, 300, 648]
from components import execute_each_datasets_each_samples, embedding_method_comparison, features_selection_comparison

# ベースライン
if True:
        # execute_each_datasets_each_samples(
        #     file_dir="../results/average_accuracy_kernel.txt",
        #     model="KernelCNN",
        #     datasets_array=datasets_array,
        #     # sample_num_array=num_samples_array,
        #     sample_num_array = [10, 100, 300, 648, 1000, 5000, 10000],
        # )
        # execute_each_datasets_each_samples(
        #     file_dir="../results/average_accuracy_LeNet.txt",
        #     model="CNN",
        #     datasets_array=datasets_array,
        #     # sample_num_array=num_samples_array,
        #     sample_num_array = [10, 100, 300, 648, 1000, 5000, 10000],
        #     model_type="2LayerCNN"
        # )
        # execute_each_datasets_each_samples(
        #     file_dir="../results/average_accuracy_LeNet.txt",
        #     model="CNN",
        #     datasets_array=datasets_array,
        #     # sample_num_array=num_samples_array,
        #     sample_num_array = [10, 100, 300, 648, 1000, 5000, 10000],
        #     model_type="LeNet"
        # )
        execute_each_datasets_each_samples(
            file_dir="../results/average_accuracy_HOG.txt",
            model="HOG",
            datasets_array=datasets_array,
            # sample_num_array=num_samples_array,
            sample_num_array = [10, 100, 300, 648, 1000, 5000, 10000],
        )

# 埋め込み手法の比較
# embedding_method_comparison(
#     file_dir="../results/average_accuracy_embedding_method_comparison.txt",
#     datasets_array=datasets_array,
#     # datasets_array=["KTH"],
#     sample_num_array=[10, 100, 300, 648, 1000, 5000, 10000],
#     # embedding_methods_array= ["PCA", "LDA", "LE", "SLE"],
#     embedding_methods_array=["SLE"]
# )

# # 特徴選択の比較
# features_selection_comparison(
#     file_dir="../results/feature_selection_comparison.txt",
#     datasets_array=datasets_array,
#     sample_num_array=num_samples_array,
#     use_channels_array=[[1, 9], [10, 18], [1, 6], [5, 13]],
# )

