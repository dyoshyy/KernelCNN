from functions import calculate_average_accuracy_kernelCNN
from main_KernelCNN import main_kernelCNN
from LeNet import main
#import LeNet
datasets_array = ['MNIST', 'FMNIST', 'CIFAR10']
embedding_array = ['LE', 'PCA', 'LLE', 'TSNE']
'''
main_mnist(100, 10000, 'LE', 3000)
main_mnist(1000, 10000, 'LE', 3000)
main_mnist(10000, 10000, 'LE', 3000)
main_mnist(30000, 10000, 'LE', 3000)
main_mnist(60000, 10000, 'LE', 3000)
'''

'''
main_fmnist(100, 10000, 'LE', 3000)
main_fmnist(1000, 10000, 'LE', 3000)
main_fmnist(10000, 10000, 'LE', 3000)
main_fmnist(30000, 10000, 'LE', 3000)
main_fmnist(60000, 10000, 'LE', 3000)

main_cifar10(100, 10000, 'LE', 3000)
main_cifar10(1000, 10000, 'LE', 3000)
main_cifar10(10000, 10000, 'LE', 3000)
main_cifar10(30000, 10000, 'LE', 3000)
main_cifar10(50000, 10000, 'LE', 3000)
'''

#手法ごとの比較
'''
main_mnist(1000, 100, 'PCA', 3000)
main_mnist(1000, 100, 'LLE', 3000)
main_mnist(1000, 100, 'LE', 3000)
main_mnist(1000, 100, 'TSNE', 3000)

main_fmnist(1000, 100, 'PCA', 3000)
main_fmnist(1000, 100, 'LLE', 3000)
main_fmnist(1000, 100, 'LE', 3000)
main_fmnist(1000, 100, 'TSNE', 3000)

main_cifar10(1000, 100, 'PCA', 3000)
main_cifar10(1000, 100, 'LLE', 3000)
main_cifar10(1000, 100, 'LE', 3000)
main_cifar10(1000, 100, 'TSNE', 3000)

'''
'''
main_GP(100, 10000, 'MNIST')
main_GP(1000, 10000, 'MNIST')
main_GP(10000, 10000, 'MNIST')
main_GP(30000, 10000, 'MNIST')
main_GP(50000, 10000, 'MNIST')
main_GP(60000, 10000, 'MNIST')

main_GP(100, 10000, 'FMNIST')
main_GP(1000, 10000, 'FMNIST')
main_GP(10000, 10000, 'FMNIST')
main_GP(30000, 10000, 'FMNIST')
main_GP(50000, 10000, 'FMNIST')
main_GP(60000, 10000, 'FMNIST')

main_GP(100, 10000, 'CIFAR10')
main_GP(1000, 10000, 'CIFAR10')
main_GP(10000, 10000, 'CIFAR10')
main_GP(30000, 10000, 'CIFAR10')
main_GP(50000, 10000, 'CIFAR10')
'''

if True:
    if False:
        calculate_average_accuracy_kernelCNN(main_mnist, [1000, 10000, 'LE', 3000], 'MNIST',10)
        calculate_average_accuracy_kernelCNN(main_mnist, [10000, 10000, 'LE', 3000], 'MNIST',10)
        calculate_average_accuracy_kernelCNN(main_mnist, [30000, 10000, 'LE', 3000], 'MNIST',10)
        calculate_average_accuracy_kernelCNN(main_mnist, [60000, 10000, 'LE', 3000], 'MNIST',10)

    if False:
        calculate_average_accuracy_kernelCNN(main_fmnist, [1000, 10000, 'LE', 3000],'FMNIST', 10)
        calculate_average_accuracy_kernelCNN(main_fmnist, [10000, 10000, 'LE', 3000],'FMNIST', 10)
        calculate_average_accuracy_kernelCNN(main_fmnist, [30000, 10000, 'LE', 3000],'FMNIST', 10)
        calculate_average_accuracy_kernelCNN(main_fmnist, [60000, 10000, 'LE', 3000],'FMNIST', 10)

    if False:
        #calculate_average_accuracy_kernelCNN(main_cifar10, [1000, 10000, 'LE', 3000],'CIFAR10', 10)
        #calculate_average_accuracy_kernelCNN(main_cifar10, [10000, 10000, 'LE', 3000],'CIFAR10', 10)
        calculate_average_accuracy_kernelCNN(main_cifar10, [30000, 10000, 'LE', 3000],'CIFAR10', 10)
        calculate_average_accuracy_kernelCNN(main_cifar10, [50000, 10000, 'LE', 3000],'CIFAR10', 10)



#ベースライン     
if False:
    main_kernelCNN(1000, 100, "MNIST", B=3000, embedding_method=["LE","LE"], block_size=[5,5])
    main_kernelCNN(1000, 100, "FMNIST", B=3000, embedding_method=["LE","LE"], block_size=[5,5])
    main_kernelCNN(1000, 100, "CIFAR10", B=3000, embedding_method=["LE","LE"], block_size=[5,5])
    
#ブロックサイズ変化
if True:
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
if True:
    for emb in ["LLE", "TSNE"]:
        #emb = "TSNE"
        main_kernelCNN(10000, 10000, "MNIST", B=3000, embedding_method=[emb, emb], block_size=[5,5])
        main_kernelCNN(10000, 10000, "FMNIST", B=3000, embedding_method=[emb,emb], block_size=[5,5])
        main_kernelCNN(10000, 10000, "CIFAR10", B=3000, embedding_method=[emb,emb], block_size=[5,5])
    