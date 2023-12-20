import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# CIFAR-10データセットのロード


for datasets in ['MNIST', 'FMNIST', 'CIFAR10']:

    # クラス名
    if datasets == 'MNIST':
        (train_images, train_labels), _= tf.keras.datasets.mnist.load_data()
        class_names = ["0","1","2","3","4","5","6","7","8","9"]
    elif datasets == "FMNIST": 
        (train_images, train_labels), _= tf.keras.datasets.fashion_mnist.load_data()
        class_names = ["T-shirt/Top","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot"]
    elif datasets == 'CIFAR10':
        (train_images, train_labels), _= tf.keras.datasets.cifar10.load_data()
        class_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

    # 画像を1x10のグリッドに配置し、各画像の下にキャプションを追加する
    arr = [0,1,7]
    fig, axes = plt.subplots(1, 3, figsize=(6, 2.5))
    for i, ax in enumerate(axes.ravel()):
        ax.imshow(train_images[arr[i]],cmap="gray")
        ax.set_title(class_names[int(train_labels[arr[i]])], fontsize=20)  # クラス名をキャプションとして追加
        ax.axis('off')  # 軸を非表示にする

    # 画像をファイルに保存
    plt.tight_layout()
    plt.close(fig)  # メモリを節約するために閉じる
    fig.savefig(f'{datasets}_class_images.png')
    
    
