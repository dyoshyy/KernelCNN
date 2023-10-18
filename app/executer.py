from main_MNIST import main_mnist
from main_FMNIST import main_fmnist
from main_cifar10 import main_cifar10
#import LeNet

'''

'''
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
'''

'''
#main_cifar10(100, 10000, 'LE', 3000)
#main_cifar10(1000, 10000, 'LE', 3000)
#main_cifar10(10000, 10000, 'LE', 3000)
main_cifar10(30000, 10000, 'LE', 3000)
main_cifar10(50000, 10000, 'LE', 3000)
'''

#手法ごとの比較
'''
main_mnist(1000, 10000, 'PCA', 3000)
main_mnist(1000, 10000, 'LLE', 3000)
main_mnist(1000, 10000, 'LE', 3000)
main_mnist(1000, 10000, 'TSNE', 3000)

main_fmnist(1000, 10000, 'PCA', 3000)
main_fmnist(1000, 10000, 'LLE', 3000)
main_fmnist(1000, 10000, 'LE', 3000)
main_fmnist(1000, 10000, 'TSNE', 3000)

main_cifar10(1000, 10000, 'PCA', 3000)
main_cifar10(1000, 10000, 'LLE', 3000)
#main_cifar10(1000, 10000, 'LE', 3000)
main_cifar10(1000, 10000, 'TSNE', 3000)
'''

main_cifar10(1000, 100, 'LE', 3000)
main_mnist(1000, 100, 'LE', 3000)

main_mnist(1000, 100, 'PCA', 3000)
main_mnist(1000, 100, 'LLE', 3000)
main_mnist(1000, 100, 'LE', 3000)
main_mnist(1000, 100, 'TSNE', 3000)