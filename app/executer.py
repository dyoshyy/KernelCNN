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

if False:
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
    for dataset in datasets_array:
        for block_size in [[3,3], [5,5], [7,7]]:
            accuracy_sum = 0
            n=1
            for _ in range(n):
                accuracy_sum += main_kernelCNN(10000, 10000, dataset, B=2000, embedding_method=['LE','LE'], block_size=block_size)
            
            average_accuracy = accuracy_sum/n
            print(f'accuarcy:{average_accuracy}')
            with open('./average_accuracy.txt', 'a') as file:
                file.write(f'{dataset}:\n') 
                file.write(f'block size:{block_size}\n') 
                file.write(str(average_accuracy)+'\n')
            
        
#層数変化
if False:
    for dataset in datasets_array:
        #dataset = "CIFAR10"
        #if dataset == "MNIST":
            #continue
        #else:
            for layers_status in [[0,0,0,0], [1,0,0,0], [1,1,0,0], [1,1,1,0], [1,1,1,1]]:
                #main(60000, 10000, dataset, [5, 5, 5], display=True, layers_BOOL=layers_status)
                main_kernelCNN(10000, 10000, dataset, B=2000, embedding_method=['LE','LE','LE'], block_size=[5,5,5], layers_BOOL=layers_status)

    
#Bの変化
if False:
    for B in [300, 600, 3000, 6000]:
        #main_kernelCNN(10000, 10000, "MNIST", B=B, embedding_method=["LE","LE"], block_size=[5,5])
        #main_kernelCNN(10000, 10000, "FMNIST", B=B, embedding_method=["LE","LE"], block_size=[5,5])
        main_kernelCNN(10000, 10000, "CIFAR10", B=B, embedding_method=["LE","LE"], block_size=[5,5])

#埋め込み手法比較
if False:
    #for emb in ["LLE", "TSNE"]:
    for emb in embedding_array:
        #emb = "TSNE"
        main_kernelCNN(10000, 10000, "MNIST", embedding_method=[emb, emb], block_size=[5,5])
        main_kernelCNN(10000, 10000, "FMNIST", embedding_method=[emb,emb], block_size=[5,5])
        main_kernelCNN(10000, 10000, "CIFAR10", embedding_method=[emb,emb], block_size=[5,5])
        
if False:
    for dataset in datasets_array:
        main_kernelCNN(100, 100, dataset, B=1000, embedding_method=["LE","LE"], block_size=[5,5])

#main_kernelCNN(1000, 100, "CIFAR10", B=120, embedding_method=["LE","LE"], block_size=[5,5])
#main_kernelCNN(1000, 100, "MNIST", B=120, embedding_method=["LE","LE"], block_size=[5,5])
