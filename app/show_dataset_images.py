import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# CIFAR-10データセットのロード
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

# CIFAR-10のクラス名
class_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

# 画像を1x10のグリッドに配置し、各画像の下にキャプションを追加する
fig, axes = plt.subplots(1, 10, figsize=(20, 2))
for i, ax in enumerate(axes.ravel()):
    ax.imshow(test_images[i])
    ax.set_title(class_names[int(test_labels[i])])  # クラス名をキャプションとして追加
    ax.axis('off')  # 軸を非表示にする

# 画像をファイルに保存
fig.savefig('cifar10_first_10_images_with_labels.png')
plt.close(fig)  # メモリを節約するために閉じる
