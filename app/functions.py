import cv2
import numpy as np
import os
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.pyplot as plt
import math
import random
from sklearn.manifold import SpectralEmbedding, TSNE, LocallyLinearEmbedding
from sklearn.decomposition import PCA, KernelPCA

random.seed(0)

def display_images(data, layer_number, embedding_method : str, dataset_name : str):
    for n in range(5):
        data_to_display = data[n]
        num_in_a_row = 4 #default 4
        Channels = data_to_display.shape[2]
        Rows = math.ceil(Channels/5)

        fig, axes = plt.subplots(Rows, num_in_a_row, figsize=(15, 3*Rows))
        fig.subplots_adjust(hspace=0.4)

        for r in range(Rows):
            for c in range(num_in_a_row):
                if Channels == 1:
                    ax = axes[r]
                else:
                    ax = axes[r, c]
                ax.axis('off')
                index = r * num_in_a_row + c
                if(index < Channels):
                    image = data_to_display[:,:, index]
                    ax.imshow(image, cmap='gray')
                    ax.set_title('Channel {}'.format(index+1))

        plt.tight_layout()
        plt.savefig(f"./results/{dataset_name}_{embedding_method}_{layer_number}_{n+1}_.png")
        #plt.show()

def visualize_emb(compressed_data, data_to_embed, data_to_embed_label, emb, block_size, channel1, channel2, dataset_name: str):
    '''
    埋め込み後の点を可視化
        compressed_data : 次元削減後のデータの第1,2次元
        data_to_embed : サンプルしたブロック
        data_to_embed_label : ブロックのラベル
        emb : 埋め込み手法
        block_size : ブロックのサイズ
    '''
    
    #ファイル名の重複を防ぐ処理
    filename = f'{dataset_name}_emb_{emb}' 
    file_exists = os.path.exists("./emb_results/" + filename + ".png")
    counter = 1
    changed = False
    while file_exists:
        new_filename = filename + f"({counter})"
        file_exists = os.path.exists("./emb_results/" + new_filename + ".png")
        counter += 1
        changed = True
    if changed:
        filename = new_filename

    fig = plt.figure(figsize=(10, 10)) #12 10

    # 圧縮後の散布図
    ax = fig.add_subplot(111)
    sc = ax.scatter(compressed_data[:, 0], compressed_data[:, 1], cmap='tab10', c=data_to_embed_label, marker='o', s=60, edgecolors='black')
    #plt.colorbar(sc, label='label')
    ax.set_title('Embedded data ' + 'Channel{}'.format(str(channel1)) + '&' + str(channel2) +" ("+emb+")")

    # ランダムに一部の点にのみブロックの画像を表示
    num_samples = len(compressed_data)
    num_blocks_to_display = min(30, num_samples)
    random_indices = random.sample(range(num_samples), num_blocks_to_display)
    #print(np.shape(data_to_embed))
    if data_to_embed.shape[1] == block_size * block_size:
        for i in random_indices:
            x, y = compressed_data[i]
            img = data_to_embed[i].reshape(block_size, block_size)# ブロック画像を5x5に変形
            imgbox = OffsetImage(img, zoom=15-block_size, cmap='gray')  # 解像度を上げるためにzoomパラメータを調整
            #imgbox = OffsetImage(img, zoom=8, cmap='gray')
            ab = AnnotationBbox(imgbox, (x, y), frameon=True, xycoords='data', boxcoords="offset points", pad=0.0)
            ax.add_artist(ab) 

    plt.tight_layout()

    # 画像として保存
    plt.savefig("./emb_results/"+filename+'.png')
    #plt.show()

def calculate_similarity(array1, array2):
    count = 0
    total = len(array1)

    for i in range(total):
        if array1[i] == array2[i]:
            count += 1

    similarity = count / total
    return similarity

def binarize_images(images):
    min_val = np.min(images)
    max_val = np.max(images)
    images = ((images - min_val) / (max_val - min_val)) * 255
    images = np.uint8(images)

    binarized_images = np.zeros_like(images)  # 二値化された画像を格納する配列を作成

    for i in range(images.shape[0]):
        for c in range(images.shape[-1]):  # Loop through each channel
            image = images[i, :, :, c]
            
            _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # 二値化
            binarized_images[i, :, :, c] = binary_image  # 二値化した画像を保存

    return binarized_images

def binarize_2d_array(array_2d):
    """
    Binarize a 2D array (n_samples, n_features) using Otsu's method for each feature.
    
    Parameters:
        array_2d: np.ndarray
            The 2D array to binarize, shape (n_samples, n_features).
            
    Returns:
        binarized_array: np.ndarray
            The binarized array.
    """
    # 結果を格納する配列を初期化
    binarized_array = np.zeros_like(array_2d, dtype=np.uint8)
    
    # 各特徴量に対してOtsuの二値化を適用
    for i in range(array_2d.shape[0]):
        _, tmp = cv2.threshold(array_2d[i].astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binarized_array[i] = tmp.reshape(tmp.shape[0])
    return binarized_array

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

def select_embedding_method(embedding_method : str, Channels_next : int, data_to_embed):
        if embedding_method == "PCA":
            pca = PCA(n_components=Channels_next, svd_solver='auto')
            embedded_blocks = pca.fit_transform(data_to_embed)
            
        elif embedding_method == "KPCA":
            kpca = KernelPCA(n_components=Channels_next, kernel="rbf")
            embedded_blocks = kpca.fit_transform(data_to_embed)
                
        elif embedding_method == "LE":
            n = int(data_to_embed.shape[0]/Channels_next)
            
            print(f"k for knn:{n}")
            LE = SpectralEmbedding(n_components=Channels_next, n_neighbors=n)
            embedded_blocks = LE.fit_transform(data_to_embed)
            
        elif embedding_method == "TSNE":
            tsne = TSNE(n_components=Channels_next, random_state = 0, method='exact', perplexity = 30, n_iter = 1000, init='pca', learning_rate='auto')
            embedded_blocks = tsne.fit_transform(data_to_embed)
                
        elif embedding_method == 'LLE':
            lle = LocallyLinearEmbedding(n_components=Channels_next, n_neighbors= int(data_to_embed.shape[0]/Channels_next))
            embedded_blocks = lle.fit_transform(data_to_embed)
        else:
            print('Error: No embedding selected.')
        
        return embedded_blocks.astype(np.float32)