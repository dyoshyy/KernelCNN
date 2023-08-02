import cv2
import numpy as np
import os
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.pyplot as plt

def display_images(data, layer):
    data = data[:10]
    N = data.shape[0]
    C = data.shape[1]
    fig, axes = plt.subplots(N, C, figsize=(C, N))
    for n in range(N):
        for c in range(C):
            if C == 1:
                ax = axes[n]
            else:
                ax = axes[n,c]
            image = data[n, c,:,:]
            ax.imshow(image, cmap='gray')
            ax.set_title('Channel {}'.format(c+1))
            ax.axis('off')
    plt.tight_layout()
    plt.savefig("./results/layer_{}_result.png".format(layer))
    plt.show()

def visualize_emb(compressed_data, sampled_blocks, sampled_blocks_label, emb, block_size):
    '''
    埋め込み後の点を可視化
        compressed_data : 次元削減後のデータの第1,2次元
        sampled_blocks : サンプルしたブロック
        sampled_blocks_label : ブロックのラベル
        emb : 埋め込み手法
        block_size : ブロックのサイズ
    '''
    filename = 'emb_' + emb
    file_exists = os.path.exists("./emb_results/" + filename + ".png")
    counter = 1
    changed = False
    while file_exists:
        new_filename = filename + f"({counter})"
        file_exists = os.path.exists("./emb_results/"+new_filename+".png")
        counter += 1
        changed = True
    if changed:
        filename = new_filename

    fig = plt.figure(figsize=(36, 30)) #12 10

    # 圧縮後の散布図
    ax = fig.add_subplot(111)
    sc = ax.scatter(compressed_data[:, 0], compressed_data[:, 1], cmap='tab10', c=sampled_blocks_label, marker='o', s=60, edgecolors='black')
    plt.colorbar(sc, label='label')
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.set_title('Embedded data '+"("+emb+")")

    # ランダムに一部の点にのみブロックの画像を表示
    #num_samples = len(compressed_data)
    #num_blocks_to_display = min(200, num_samples)
    #random_indices = random.sample(range(num_samples), num_blocks_to_display)
    #print(np.shape(sampled_blocks))
    if sampled_blocks.shape[1] == block_size * block_size:
        for i in range(0, 300, 1):
            x, y = compressed_data[i]
            img = sampled_blocks[i].reshape(block_size, block_size)# ブロック画像を5x5に変形
            img_rgb = np.zeros((block_size, block_size, 3))
            
            if 0<=i & i<=30:
                img_rgb[:, :, 0] = img
            else:
                img_rgb = img
                
            #imgbox = OffsetImage(img_rgb, zoom=17-block_size, cmap='gray')  # 解像度を上げるためにzoomパラメータを調整
            imgbox = OffsetImage(img_rgb, zoom=8, cmap='gray')
            ab = AnnotationBbox(imgbox, (x, y), frameon=True, xycoords='data', boxcoords="offset points", pad=0.0)
            ax.add_artist(ab)

    plt.tight_layout()

    # 画像として保存
    plt.savefig("./emb_results/"+filename+'.png')
    plt.show()

def calculate_similarity(array1, array2):
    count = 0
    total = len(array1)

    for i in range(total):
        if array1[i] == array2[i]:
            count += 1

    similarity = count / total
    return similarity

def binarize_images(images):

    min = np.min(images)
    max = np.max(images)
    images = ((images - min) / (max - min)) * 255
    images = np.uint8(images)

    binarized_images = np.zeros_like(images)  # 二値化された画像を格納する配列を作成

    for i in range(images.shape[0]):
        image = images[i, 0]  # 画像を取得（shape: (1, 28, 28)）
        _, binary_image = cv2.threshold(image, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # 二値化

        binarized_images[i, 0] = binary_image  # 二値化した画像を保存

    return binarized_images

def binarize_mnist_data(data):
    """
    MNISTデータを最大値と最小値から閾値を求めて二値化します。

    Parameters:
        data (ndarray): MNISTデータの配列 (サンプル数, 高さ, 幅)

    Returns:
        ndarray: 二値化されたMNISTデータの配列 (サンプル数, 高さ, 幅)
    """
    # データの最大値と最小値を取得
    max_value = np.max(data)
    min_value = np.min(data)

    # 閾値を求める（最大値と最小値の中間値）
    threshold = (max_value + min_value) / 2

    # 閾値を使ってデータを二値化する
    binarized_data = (data > threshold).astype(np.uint8)

    return binarized_data

def pad_images(images):
    # 元の画像サイズ (MNISTは28x28)
    original_size = images.shape[1]
    
    # パディング後のサイズ (32x32)
    padded_size = 32
    
    # パディング量を計算
    pad_width = (padded_size - original_size) // 2
    
    # パディング後の画像データを格納する配列を作成
    padded_images = np.zeros((images.shape[0], padded_size, padded_size))
    
    # 元の画像を中央に配置してパディングする
    padded_images[:, pad_width:pad_width + original_size, pad_width:pad_width + original_size] = images
    
    return padded_images