from main_LeNet import main_LeNet
import os
import numpy as np
#import LeNet
datasets_array = ['MNIST', 'FMNIST', 'CIFAR10']
embedding_array = ['LE', 'PCA', 'LLE', 'TSNE']

#ベースライン     
if True:
    
    for dataset in datasets_array:
        for n in [1000, 10000, 30000, 60000]:
            N = 1 #iteration number
            accuracy_list = []
            for _ in range(N):
                accuracy = main_LeNet(n, 10000, dataset, block_size=[5, 5], display=False, layers_BOOL=[1,1,1,0])
                #accuracy = main_kernelCNN(n, 10000, dataset, B=1000, embedding_method=["LE","LE"], block_size=[5,5])
                accuracy_list.append(accuracy)
            
            avg_accuracy = np.mean(accuracy_list)
            variance_accuracy = np.var(accuracy_list)
            file_dir = './average_accuracy_LeNet.txt'
            with open(file_dir, 'a') as file:
                file.write(f'{dataset}:\n') 
                file.write(f'n={n}\n') 
                file.write(f'average accuracy:{str(avg_accuracy)}\n')
                file.write(f'variance accuracy:{str(variance_accuracy)}\n')
    