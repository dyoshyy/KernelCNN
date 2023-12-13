from LeNet import main
from functions import calculate_average_accuracy_CNN
#import LeNet

if False:
    print('Dataset: MNIST')
    main(100, 10000, 'MNIST')
    main(1000, 10000, 'MNIST')
    main(10000, 10000, 'MNIST')
    main(30000, 10000, 'MNIST')
    main(50000, 10000, 'MNIST')
    main(60000, 10000, 'MNIST')

    print('Dataset: FMNIST')
    main(100, 10000, 'FMNIST')
    main(1000, 10000, 'FMNIST')
    main(10000, 10000, 'FMNIST')
    main(30000, 10000, 'FMNIST')
    main(50000, 10000, 'FMNIST')
    main(60000, 10000, 'FMNIST')

    print('Dataset: CIFAR10')
    main(100, 10000, 'CIFAR10')
    main(1000, 10000, 'CIFAR10')
    main(10000, 10000, 'CIFAR10')
    main(30000, 10000, 'CIFAR10')
    main(50000, 10000, 'CIFAR10')

'''
main(10000, 10000, 'MNIST')
#main(10000, 10000, 'FMNIST')
#main(10000, 10000, 'CIFAR10')
'''

if False:
    calculate_average_accuracy(main, [100, 10000, 'MNIST', None], 10)
    calculate_average_accuracy(main, [1000, 10000, 'MNIST', None], 10)
    calculate_average_accuracy(main, [10000, 10000, 'MNIST', None], 10)
    calculate_average_accuracy(main, [30000, 10000, 'MNIST', None], 10)
    calculate_average_accuracy(main, [50000, 10000, 'MNIST', None], 10)
    calculate_average_accuracy(main, [60000, 10000, 'MNIST', None], 10)

    calculate_average_accuracy(main, [100, 10000, 'FMNIST', None], 10)
    calculate_average_accuracy(main, [1000, 10000, 'FMNIST', None], 10)
    calculate_average_accuracy(main, [10000, 10000, 'FMNIST', None], 10)
    calculate_average_accuracy(main, [30000, 10000, 'FMNIST', None], 10)
    calculate_average_accuracy(main, [50000, 10000, 'FMNIST', None], 10)
    calculate_average_accuracy(main, [60000, 10000, 'FMNIST', None], 10)

    calculate_average_accuracy(main, [100, 10000, 'CIFAR10', None], 10)
    calculate_average_accuracy(main, [1000, 10000, 'CIFAR10', None], 10)
    calculate_average_accuracy(main, [10000, 10000, 'CIFAR10', None], 10)
    calculate_average_accuracy(main, [30000, 10000, 'CIFAR10', None], 10)
    calculate_average_accuracy(main, [50000, 10000, 'CIFAR10', None], 10)

#ブロックサイズ変化
if False:
    main(60000, 10000, 'MNIST', [3,3], display=True)
    main(60000, 10000, 'MNIST', [3,5], display=True)
    main(60000, 10000, 'MNIST', [3,7], display=True)
    
    main(60000, 10000, 'MNIST', [5,3], display=True)
    main(60000, 10000, 'MNIST', [5,5], display=True)
    main(60000, 10000, 'MNIST', [5,7], display=True)
    
    main(60000, 10000, 'MNIST', [7,3], display=True)
    main(60000, 10000, 'MNIST', [7,5], display=True)
    main(60000, 10000, 'MNIST', [7,7], display=True)

if False:
    calculate_average_accuracy_CNN(main, [60000, 10000, 'MNIST', [3,3], False], 10)
    calculate_average_accuracy_CNN(main, [60000, 10000, 'MNIST', [3,5], False], 10)
    calculate_average_accuracy_CNN(main, [60000, 10000, 'MNIST', [3,7], False], 10)
    calculate_average_accuracy_CNN(main, [60000, 10000, 'MNIST', [5,3], False], 10)
    calculate_average_accuracy_CNN(main, [60000, 10000, 'MNIST', [5,5], False], 10)
    calculate_average_accuracy_CNN(main, [60000, 10000, 'MNIST', [5,7], False], 10)
    calculate_average_accuracy_CNN(main, [60000, 10000, 'MNIST', [7,3], False], 10)
    calculate_average_accuracy_CNN(main, [60000, 10000, 'MNIST', [7,5], False], 10)
    calculate_average_accuracy_CNN(main, [60000, 10000, 'MNIST', [7,7], False], 10)
    
    calculate_average_accuracy_CNN(main, [60000, 10000, 'FMNIST', [3,3], False], 10)
    calculate_average_accuracy_CNN(main, [60000, 10000, 'FMNIST', [3,5], False], 10)
    calculate_average_accuracy_CNN(main, [60000, 10000, 'FMNIST', [3,7], False], 10)
    calculate_average_accuracy_CNN(main, [60000, 10000, 'FMNIST', [5,3], False], 10)
    calculate_average_accuracy_CNN(main, [60000, 10000, 'FMNIST', [5,5], False], 10)
    calculate_average_accuracy_CNN(main, [60000, 10000, 'FMNIST', [3,7], False], 10)
    calculate_average_accuracy_CNN(main, [60000, 10000, 'FMNIST', [7,3], False], 10)
    calculate_average_accuracy_CNN(main, [60000, 10000, 'FMNIST', [7,5], False], 10)
    calculate_average_accuracy_CNN(main, [60000, 10000, 'FMNIST', [7,7], False], 10)
    
    calculate_average_accuracy_CNN(main, [60000, 10000, 'CIFAR10', [3,3], False], 10)
    calculate_average_accuracy_CNN(main, [60000, 10000, 'CIFAR10', [3,5], False], 10)
    calculate_average_accuracy_CNN(main, [60000, 10000, 'CIFAR10', [3,7], False], 10)
    calculate_average_accuracy_CNN(main, [60000, 10000, 'CIFAR10', [5,3], False], 10)
    calculate_average_accuracy_CNN(main, [60000, 10000, 'CIFAR10', [5,5], False], 10)
    calculate_average_accuracy_CNN(main, [60000, 10000, 'CIFAR10', [5,7], False], 10)    
    calculate_average_accuracy_CNN(main, [60000, 10000, 'CIFAR10', [7,3], False], 10)
    calculate_average_accuracy_CNN(main, [60000, 10000, 'CIFAR10', [7,5], False], 10)
    calculate_average_accuracy_CNN(main, [60000, 10000, 'CIFAR10', [7,7], False], 10)
    
    
#層数変化
if False:
    main(60000, 10000, 'MNIST', layers_BOOL=[0,0,0,0,0])
    main(60000, 10000, 'MNIST', layers_BOOL=[1,0,0,0,0])
    main(60000, 10000, 'MNIST', layers_BOOL=[1,1,0,0,0])
    main(60000, 10000, 'MNIST', layers_BOOL=[1,1,1,0,0])
    main(60000, 10000, 'MNIST', layers_BOOL=[1,1,1,1,0])
    main(60000, 10000, 'MNIST', layers_BOOL=[1,1,1,1,1])
    
    main(60000, 10000, 'FMNIST', layers_BOOL=[0,0,0,0,0])
    main(60000, 10000, 'FMNIST', layers_BOOL=[1,0,0,0,0])
    main(60000, 10000, 'FMNIST', layers_BOOL=[1,1,0,0,0])
    main(60000, 10000, 'FMNIST', layers_BOOL=[1,1,1,0,0])
    main(60000, 10000, 'FMNIST', layers_BOOL=[1,1,1,1,0])
    main(60000, 10000, 'FMNIST', layers_BOOL=[1,1,1,1,1])
    
    main(50000, 10000, 'CIFAR10', layers_BOOL=[0,0,0,0,0])
    main(50000, 10000, 'CIFAR10', layers_BOOL=[1,0,0,0,0])
    main(50000, 10000, 'CIFAR10', layers_BOOL=[1,1,0,0,0])
    main(50000, 10000, 'CIFAR10', layers_BOOL=[1,1,1,0,0])
    main(50000, 10000, 'CIFAR10', layers_BOOL=[1,1,1,1,0])
    main(50000, 10000, 'CIFAR10', layers_BOOL=[1,1,1,1,1])

#中間層出力
if True:
    #main(60000, 100, 'MNIST', block_size=[5,5], display=True)
    #main(60000, 100, 'FMNIST', block_size=[5,5], display=True)
    main(60000, 100, 'CIFAR10', block_size=[5,5], display=True)

    