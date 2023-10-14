from prml.app.main_MNIST import main
#import LeNet
'''
main(1000, 10000, 'PCA', 100)
main(1000, 10000, 'LLE', 100)
main(1000, 10000, 'LE', 100)
main(1000, 10000, 'TSNE', 100)
'''
main(1000, 10000, 'LE', 30)

'''
LeNet.main(100, 10000, 3000)
LeNet.main(1000, 10000, 3000)
LeNet.main(10000, 10000, 3000)
LeNet.main(60000, 10000, 3000)
'''