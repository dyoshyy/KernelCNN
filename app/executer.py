from functions import calculate_average_accuracy_kernelCNN
from main_KernelCNN import main_kernelCNN
#import LeNet

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

#ブロックサイズ変化
if False:
    
    main_mnist(1000, 10000, 'LE', 3000, [7,5])
    main_mnist(1000, 10000, 'LE', 3000, [7,3])
    main_mnist(1000, 10000, 'LE', 3000, [5,3])

    if False:
        main_fmnist(1000, 10000, 'LE', 3000, 3)
        main_fmnist(1000, 10000, 'LE', 3000, 5)
        main_fmnist(1000, 10000, 'LE', 3000, 7)

        main_cifar10(1000, 10000, 'LE', 3000, 3)
        main_cifar10(1000, 10000, 'LE', 3000, 5)
        main_cifar10(1000, 10000, 'LE', 3000, 7)
        
if False:
    main_kernelCNN(1000, 100, "MNIST", B=3000, embedding_method=["LE","LE"], block_size=[5,5])
    main_kernelCNN(1000, 100, "FMNIST", B=3000, embedding_method=["LE","LE"], block_size=[5,5])
    main_kernelCNN(1000, 100, "CIFAR10", B=3000, embedding_method=["LE","LE"], block_size=[5,5])
    
#main_kernelCNN(1000, 100, "CIFAR10", B=50, embedding_method=["LE","LE"], block_size=[5,5])
main_kernelCNN(1000, 100, "MNIST", B=300, embedding_method=["LE","LE"], block_size=[5,5])

    