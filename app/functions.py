import cv2
import numpy as np
import os
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import pathpatch_2d_to_3d
from matplotlib.patches import PathPatch
from matplotlib.path import Path
import math
import random
from skimage import util
from sklearn.manifold import SpectralEmbedding, TSNE, LocallyLinearEmbedding
from sklearn.decomposition import PCA, KernelPCA

random.seed(0)

def normalize_output(img):
    img_min, img_max = np.min(img, axis=(0, 1)), np.max(img, axis=(0, 1))
    return (img - img_min) / (img_max - img_min)


def display_images(data, layer_number, embedding_method: str, dataset_name: str, suptitle : str):
    for n in range(3):
        data_to_display = data[n]
        num_in_a_row = 4  # default 4
        Channels = data_to_display.shape[2]
        Rows = math.ceil(Channels / 5)

        fig, axes = plt.subplots(Rows, num_in_a_row, figsize=(15, 3 * Rows))
        fig.subplots_adjust(hspace=0.4)

        for r in range(Rows):
            for c in range(num_in_a_row):
                if Channels == 1:
                    ax = axes[r]
                else:
                    ax = axes[r, c]
                ax.axis("off")
                index = r * num_in_a_row + c
                if index < Channels:
                    image = data_to_display[:, :, index]
                    ax.imshow(image, cmap="gray")
                    ax.set_title("Channel {}".format(index + 1))

        fig.suptitle(suptitle)
        plt.tight_layout()
        plt.savefig(
            f"./results/{dataset_name}_{embedding_method}_{layer_number}_{n+1}_.png"
        )
        plt.close()
        # plt.show()

def display_weights(weights, dataset_name, layer_idx):
    num_input_channels = weights.shape[2]
    num_output_channels = weights.shape[3]
    
    fig, axs = plt.subplots(num_output_channels,num_input_channels, figsize=(num_input_channels+1,num_output_channels+10))
    fig.suptitle(f'Layer{layer_idx} weights')
    
    for i in range(num_output_channels):
        for j in range(num_input_channels):
            filter_weights = weights[:,:,j, i]
            if num_input_channels == 1:
                axs[i].imshow(filter_weights, cmap='gray')
                axs[i].axis('off')
            else:
                axs[i, j].imshow(filter_weights, cmap='gray')
                axs[i, j].axis('off')
    
    # 横の余白を小さくし、縦の余白を大きくする
    plt.subplots_adjust(wspace=0.6, hspace=1.5)
    plt.tight_layout()
    plt.savefig(f"./emb_results/{dataset_name}_LeNet_weights_layer{layer_idx}.png")
    plt.close()

def visualize_emb(input_data,input_data_label,convolved_data,block_size: int,stride: int,B: int,embedding_method: str,dataset_name: str, output_dots=True):
    """
    埋め込み後の点を可視化
        input_data : 層の入力画像 (データ数, 高さ, 幅, 入力チャンネル数)
        input_data_label : ブロックのラベル (データ数, 高さ, 幅, 出力チャンネル数)
        convolved_data : 次元削減後のデータの第1,2次元
        block_size : ブロックのサイズ
        stride : ストライド
        embedding_method : 埋め込み手法
        dataset_name : データセットの名前
    """
    # ファイル名の重複を防ぐ処理
    filename = f"{dataset_name}_emb_{embedding_method}"
    file_exists = os.path.exists("./emb_results/" + filename + '.png')
    counter = 1
    changed = False
    while file_exists:
        new_filename = filename + f"({counter})"
        file_exists = os.path.exists("./emb_results/" + new_filename + '.png')
        counter += 1
        changed = True
    if changed:
        filename = new_filename

    # 画像データからブロックに変換
    print(input_data.shape)
    input_data = util.view_as_windows(input_data, (1, block_size, block_size, input_data.shape[3]), stride)
    input_data = input_data.reshape(-1, block_size, block_size, input_data.shape[-1])
    input_data_label = np.repeat(np.argmax(input_data_label, axis=1),convolved_data.shape[1] * convolved_data.shape[2],) #１枚の画像のラベルをブロックの個数分繰り返す　（例：３のラベルを２８ｘ２８繰り返す）
    convolved_data = convolved_data.reshape(-1, convolved_data.shape[3])
    
    #重複を削除
    print(convolved_data.shape, input_data.shape)
    convolved_data, unique_indices = np.unique(convolved_data, axis=0, return_index=True)
    input_data_label = input_data_label[unique_indices]
    input_data = input_data[unique_indices]

    #２チャネルごとに列方向に描画
    num_images = int(convolved_data.shape[1] / 2)
    num_input_channels = int(input_data.shape[3])
    fig, axs = plt.subplots(num_images, 1, figsize=(10, num_images*10))
    
    # ランダムに一部の点にのみブロックの画像を表示
    num_samples = len(convolved_data)
    num_blocks_to_display = min(10, num_samples)
    random_indices = np.random.choice(num_samples, num_blocks_to_display, replace=False)
    convolved_data = convolved_data[random_indices]
    input_data_label = input_data_label[random_indices]
    
    for img_idx in range(num_images):
        convolved_data_sep = convolved_data[:, (2 * img_idx) : (2 * (img_idx + 1))]
        ax = axs[img_idx]
        
        #軸の範囲を設定
        x_min = np.min(convolved_data_sep[:,0])
        x_max = np.max(convolved_data_sep[:,0])
        y_min = np.min(convolved_data_sep[:,1])
        y_max = np.max(convolved_data_sep[:,1])
            
        k = 0.2*(x_max-x_min)/2
        l = 0.2*(y_max-y_min)/2
        
        #散布図のプロット
        sc = ax.scatter(convolved_data_sep[:,0], convolved_data_sep[:,1], cmap="tab10", c=input_data_label, marker="o",s=80,edgecolors="black",)
        
        #Annotationのプロット
        for dot_idx in range(len(convolved_data)):
            x, y = convolved_data_sep[dot_idx]
            ax.annotate(chr(dot_idx+65), (x, y), (x+k/2, y+l/2),arrowprops=dict(arrowstyle="->"), size=30, color="black")
        

        ax.set_box_aspect (1)
        ax.set_xlim(x_min-k,x_max+k)
        ax.set_ylim(y_min-l,y_max+l)
        ax.set_title(f"Channel {2*img_idx+1}-{2*(img_idx+1)} Dataset:{dataset_name}, Embedding:{embedding_method}\n(B={B}, b={block_size})")
            
    # 画像として保存
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)
    plt.savefig(f"./emb_results/{filename}.png")
    plt.close()
    
    output_dots=False
    if output_dots:
        visualize_emb_dots(input_data_label, convolved_data, block_size, B, embedding_method, dataset_name)
        

def visualize_emb_dots(input_data_label, convolved_data, b: int, B: int,embedding_method: str,dataset_name: str, ):
    """
    埋め込み後の点を可視化
        input_data : 層の入力画像 (データ数, 高さ, 幅, 入力チャンネル数)
        input_data_label : ブロックのラベル (データ数, 高さ, 幅, 出力チャンネル数)
        convolved_data : 次元削減後のデータの第1,2次元
        block_size : ブロックのサイズ
        stride : ストライド
        embedding_method : 埋め込み手法
        dataset_name : データセットの名前
    """
    # ファイル名の重複を防ぐ処理
    filename = f"{dataset_name}_emb_{embedding_method}_dots"
    file_exists = os.path.exists("./emb_results/" + filename + '.png')
    counter = 1
    changed = False
    while file_exists:
        new_filename = filename + f"({counter})"
        file_exists = os.path.exists("./emb_results/" + new_filename + '.png')
        counter += 1
        changed = True
    if changed:
        filename = new_filename

    #２チャネルごとに列方向に描画
    num_images = int(convolved_data.shape[1] / 2)
    fig, axs = plt.subplots(num_images, 1, figsize=(10,num_images*10))
    
    #一部の点を選ぶ
    num_to_select = min(convolved_data.shape[0], 3000)
    select_indices = np.random.choice(convolved_data.shape[0], num_to_select, replace=False)
    convolved_data = convolved_data[select_indices]
    input_data_label = input_data_label[select_indices]
    
    for img_idx in range(num_images):
        convolved_data_sep = convolved_data[:, (2 * img_idx) : (2 * (img_idx + 1))]
        ax = axs[img_idx]
        sc = ax.scatter(convolved_data_sep[:,0], convolved_data_sep[:,1], cmap="tab10", c=input_data_label, marker="o",s=50,edgecolors="black",)
        #plt.colorbar(sc, label='label')
        
        #軸の範囲を設定
        x_min = np.min(convolved_data_sep[:,0])
        x_max = np.max(convolved_data_sep[:,0])
        y_min = np.min(convolved_data_sep[:,1])
        y_max = np.max(convolved_data_sep[:,1])
            
        k = 0.2*(x_max-x_min)/2
        l = 0.2*(y_max-y_min)/2
        ax.set_box_aspect (1)
        ax.set_xlim(x_min-k,x_max+k)
        ax.set_ylim(y_min-l,y_max+l)
        ax.set_title(f"Channel {2*img_idx+1}-{2*(img_idx+1)} Dataset:{dataset_name}, Embedding:{embedding_method}\n(B={B}, b={b})")
            
    # 画像として保存
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)
    plt.savefig(f"./emb_results/{filename}.png")
    plt.close()

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

            _, binary_image = cv2.threshold(
                image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )  # 二値化
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
        _, tmp = cv2.threshold(
            array_2d[i].astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        binarized_array[i] = tmp.reshape(tmp.shape[0])
    return binarized_array


def pad_images(images, out_size):
    # 元の画像サイズ (MNISTは28x28)
    original_size = images.shape[1]

    # パディング量を計算
    pad_width = (out_size - original_size) // 2

    # パディング後の画像データを格納する配列を作成
    padded_images = np.zeros((images.shape[0], out_size, out_size, images.shape[3]))

    # 元の画像を中央に配置してパディングする
    padded_images[:, pad_width : pad_width + original_size, pad_width : pad_width + original_size, :] = images

    return padded_images


def select_embedding_method(embedding_method: str, Channels_next: int, data_to_embed):
    if embedding_method == "PCA":
        pca = PCA(n_components=Channels_next, svd_solver="auto")
        embedded_blocks = pca.fit_transform(data_to_embed)

    elif embedding_method == "KPCA":
        kpca = KernelPCA(n_components=Channels_next, kernel="rbf")
        embedded_blocks = kpca.fit_transform(data_to_embed)

    elif embedding_method == "LE":
        n = int(data_to_embed.shape[0] / Channels_next)
        # n = k_for_knn
        print(f"k for knn:{n}")
        LE = SpectralEmbedding(n_components=Channels_next, n_neighbors=n)
        embedded_blocks = LE.fit_transform(data_to_embed)

    elif embedding_method == "TSNE":
        tsne = TSNE(
            n_components=Channels_next,
            random_state=0,
            method="exact",
            perplexity=30,
            n_iter=1000,
            init="pca",
            learning_rate="auto",
        )
        embedded_blocks = tsne.fit_transform(data_to_embed)

    elif embedding_method == "LLE":
        lle = LocallyLinearEmbedding(
            n_components=Channels_next,
            n_neighbors=int(data_to_embed.shape[0] / Channels_next),
        )
        embedded_blocks = lle.fit_transform(data_to_embed)
    else:
        print("Error: No embedding selected.")

    return embedded_blocks.astype(np.float32)
