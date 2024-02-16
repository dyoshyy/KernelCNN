from functions import calculate_average_accuracy_kernelCNN
from main_KernelCNN import main_kernelCNN
from app.main_LeNet import main_LeNet
import os
#import LeNet
datasets_array = ['MNIST', 'FMNIST', 'CIFAR10']
embedding_array = ['LE', 'PCA', 'LLE', 'TSNE']


#ベースライン     
if True:
    
    #for dataset in datasets_array:
    for dataset in ['FMNIST', 'CIFAR10']:
        for n in [1000, 10000, 30000, 50000, 60000]:
            N=5 #iteration number
            accuracy_sum = 0
            for _ in range(N):
                accuracy_sum += main_kernelCNN(n, 10000, dataset, B=1000, embedding_method=["LE","LE"], block_size=[5,5])
                #accuracy_sum += main_LeNet(n, 10000, dataset, block_size=[5, 5], display=False, layers_BOOL=[1,1,1,0])
            
            avg_accuracy = accuracy_sum / N
            file_dir = './average_accuracy_kernel.txt'
            with open(file_dir, 'a') as file:
                file.write(f'{dataset}:\n') 
                file.write(f'n={n}\n') 
                file.write(str(avg_accuracy)+'\n')
    
   
#ブロックサイズ変化
if False:
    #main_kernelCNN(10000, 10000, 'MNIST', B=3000, embedding_method=['LE','LE'], block_size=[3,3])
    #main_kernelCNN(10000, 10000, 'MNIST', B=3000, embedding_method=['LE','LE'], block_size=[7,7])
    #main_kernelCNN(10000, 10000, 'MNIST', B=3000, embedding_method=['LE','LE'], block_size=[7,3])
    #main_kernelCNN(10000, 10000, 'MNIST', B=3000, embedding_method=['LE','LE'], block_size=[3,7])


    if False:
        main_fmnist(1000, 10000, 'LE', 3000, 3)
        main_fmnist(1000, 10000, 'LE', 3000, 5)
        main_fmnist(1000, 10000, 'LE', 3000, 7)

        main_cifar10(1000, 10000, 'LE', 3000, 3)
        main_cifar10(1000, 10000, 'LE', 3000, 5)
        main_cifar10(1000, 10000, 'LE', 3000, 7)
        
#層数変化
if False:
    #for dataset in datasets_array:
        dataset = "CIFAR10"
        main(60000, 1000, dataset, [5, 5, 5], display=True, layers_BOOL=[1,1,1,1])
        main_kernelCNN(1000, 100, dataset, B=3000, embedding_method=['LE','LE','LE'], block_size=[5,5,5], layers_BOOL=[1,1,1,1])

    
#Bの変化
if False:
    for B in [300, 600, 3000, 6000]:
        #main_kernelCNN(10000, 10000, "MNIST", B=B, embedding_method=["LE","LE"], block_size=[5,5])
        #main_kernelCNN(10000, 10000, "FMNIST", B=B, embedding_method=["LE","LE"], block_size=[5,5])
        main_kernelCNN(10000, 10000, "CIFAR10", B=B, embedding_method=["LE","LE"], block_size=[5,5])

#埋め込み手法比較
if False:
    for emb in ["LLE", "TSNE"]:
        #emb = "TSNE"
        if emb == "TSNE":
            main_kernelCNN(10000, 10000, "MNIST", B=2000, embedding_method=[emb, emb], block_size=[5,5])
        main_kernelCNN(10000, 10000, "FMNIST", B=2000, embedding_method=[emb,emb], block_size=[5,5])
        main_kernelCNN(10000, 10000, "CIFAR10", B=2000, embedding_method=[emb,emb], block_size=[5,5])
        
if False:
    for dataset in datasets_array:
        main_kernelCNN(100, 100, dataset, B=1000, embedding_method=["LE","LE"], block_size=[5,5])

#main_kernelCNN(1000, 100, "CIFAR10", B=120, embedding_method=["LE","LE"], block_size=[5,5])
#main_kernelCNN(1000, 100, "MNIST", B=120, embedding_method=["LE","LE"], block_size=[5,5])
