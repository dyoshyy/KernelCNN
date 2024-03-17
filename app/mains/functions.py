import json
from pathlib import Path
import random
from turtle import pd
import cv2
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
import numpy as np
import os
import sys

from sklearn.naive_bayes import LabelBinarizer
from sympy import N

sys.path.append("/workspaces/KernelCNN/app/data")
import matplotlib.pyplot as plt
import math
from skimage import util
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, StandardScaler
from sklearn.manifold import SpectralEmbedding, TSNE, LocallyLinearEmbedding
from sklearn.decomposition import PCA, KernelPCA
from laplacianEigenmap import LaplacianEigenmap
from KSLE import SLE
import pickle
import os
from sklearn.model_selection import train_test_split
from keras.datasets import mnist, fashion_mnist, cifar10
from keras.utils import to_categorical
import pandas as pd
import networkx as nx


def make_unique_filename(preliminary_name: str, file_path: str):
    file_exists = os.path.exists(file_path + "/" + preliminary_name + "_1.png")
    counter = 2
    changed = False
    # preliminary_nameのファイルが存在する限りカウンターをインクリメント
    while file_exists:
        new_filename = preliminary_name + f"_{counter}"
        file_exists = os.path.exists(file_path + "/" + new_filename + ".png")
        counter += 1
        changed = True

    # 一度でも名前が変更されたらnew_filenameを返し，一度も変更されなかったらpreliminary_nameを返す
    if changed:
        unique_name = new_filename
    else:
        unique_name = preliminary_name + "_1"
    return unique_name


def normalize_output(img):
    img_min, img_max = np.min(img, axis=(0, 1)), np.max(img, axis=(0, 1))
    return (img - img_min) / (img_max - img_min)


def scale_to_0_255(data):
    """
    配列データを0から255の範囲にスケーリングする関数
    :param data: スケーリング対象の配列
    :return: スケーリングされた配列
    """
    min_val = np.min(data)
    max_val = np.max(data)

    if min_val == max_val:
        # 配列のすべての値が同じ場合はスケーリング不要
        return data
    else:
        scaled_data = 255 * (data - min_val) / (max_val - min_val)
        return scaled_data.astype(np.uint8)


def display_images(
    data, label, layer_number, embedding_method: str, dataset_name: str, suptitle: str
):
    label = np.argmax(label, axis=1)
    indices = [
        np.where(label == i)[0][0] for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    ]  # 1,2,3,4,5のラベルを持つデータの最初のインデックスを取得

    img_idx = 1
    for n in indices:
        data_to_display = data[n]
        # data_to_display = scale_to_0_255(data_to_display)
        if data_to_display.shape[2] == 6:
            num_in_a_row = 9
        else:
            # data_to_display = data_to_display[:, :, :16]
            num_in_a_row = 9  # default 4
        Channels = data_to_display.shape[2]
        Rows = math.ceil(Channels / num_in_a_row)

        fig, axes = plt.subplots(
            Rows, num_in_a_row, figsize=(num_in_a_row * 1.5, 2 * Rows)
        )
        fig.subplots_adjust(hspace=0.4)

        for r in range(Rows):
            for c in range(num_in_a_row):
                if Rows == 1:
                    ax = axes[c]
                else:
                    ax = axes[r, c]
                ax.axis("off")
                index = r * num_in_a_row + c
                if index < Channels:
                    image = data_to_display[:, :, index]
                    ax.imshow(image, cmap="gray")
                    ax.set_title("Channel {}".format(index + 1))

        # fig.suptitle(suptitle)
        filename = (
            f"{embedding_method}_layer{layer_number}_{dataset_name}_class{img_idx}"
        )
        file_dir = "../results/results_output"
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)
        filename = make_unique_filename(filename, file_dir)
        plt.tight_layout()
        plt.savefig(file_dir + f"/{filename}.png")
        plt.close()
        # plt.show()
        img_idx += 1


def display_weights(weights, dataset_name, layer_idx):
    num_input_channels = weights.shape[2]
    num_output_channels = weights.shape[3]

    fig, axs = plt.subplots(
        # num_input_channels,
        nrows=1,
        ncols=num_output_channels,
        figsize=(num_output_channels + 10,  2),
        dpi=50,
    )
    # fig.suptitle(f"Layer{layer_idx} weights")

    # for i in range(num_input_channels):
    #     for j in range(num_output_channels):
    #         filter_weights = weights[:, :, i, j]
    #         if num_input_channels == 1:
    #             axs[j].imshow(filter_weights, cmap="gray")
    #             axs[j].axis("off")
    #         else:
    #             axs[i, j].imshow(filter_weights, cmap="gray")
    #             axs[i, j].axis("off")
    for i in range(num_output_channels):
        filter_weights = weights[:, :, :, i]
        if num_input_channels <= 3:
            # filter_weights = (filter_weights - filter_weights.min()) / (filter_weights.max() - filter_weights.min())
            # RGBチャンネルごとに最小値と最大値を取得
            min_val = np.amin(filter_weights, axis=(0, 1))
            max_val = np.amax(filter_weights, axis=(0, 1))
            # RGBチャンネルごとに正規化
            filter_weights = (filter_weights - min_val) / (max_val - min_val)
            if filter_weights.shape[2] == 1:
                axs[i].imshow(filter_weights, cmap="gray")
            else:
                axs[i].imshow(filter_weights)
            axs[i].axis("off")
            
    # 横の余白を小さくし、縦の余白を大きくする
    plt.subplots_adjust(wspace=0.6, hspace=1.5)
    plt.tight_layout()
    file_dir = "../results/results_weights"
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    filename = f"{dataset_name}_LeNet_weights_layer_{layer_idx}"
    filename = make_unique_filename(filename, file_dir)
    plt.savefig(file_dir + f"/{filename}.png")
    plt.close()


def visualize_emb(
    input_data,
    input_data_label,
    convolved_data,
    block_size: int,
    stride: int,
    B: int,
    embedding_method: str,
    dataset_name: str,
    output_dots=True,
):
    """
    埋め込み後の点を可視化
        input_data : 層の入力画像 (データ数, 高さ, 幅, 入力チャンネル数)
        input_data_label : ブロックのラベル (データ数, 高さ, 幅, 出力チャンネル数)
        convolved_data : 次元削減後のデータ
        block_size : ブロックのサイズ
        stride : ストライド
        embedding_method : 埋め込み手法
        dataset_name : データセットの名前
    """
    plt.rcParams["xtick.labelsize"] = 10  # 軸だけ変更されます。
    plt.rcParams["ytick.labelsize"] = 10  # 軸だけ変更されます
    plt.rcParams["font.size"] = 20  # 全体のフォントサイズが変更されます。

    # ファイル名の重複を防ぐ処理

    file_dir = "../results/results_emb"
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    filename = f"{embedding_method}_{dataset_name}"
    filename = make_unique_filename(filename, file_dir)

    input_data = input_data[:100]
    input_data_label = input_data_label[:100]
    convolved_data = convolved_data[:100]

    # 画像データからブロックに変換
    blocks = np.empty((0, block_size, block_size, input_data.shape[3]))
    for img_idx in range(input_data.shape[0]):
        img = input_data[img_idx]
        block = util.view_as_windows(
            img, (block_size, block_size, input_data.shape[3]), stride
        ).reshape(-1, block_size, block_size, input_data.shape[3])
        blocks = np.concatenate((blocks, block), axis=0)

    input_data = blocks
    input_data_label = np.repeat(
        np.argmax(input_data_label, axis=1),
        convolved_data.shape[1] * convolved_data.shape[2],
    )  # １枚の画像のラベルをブロックの個数分繰り返す　（例：３のラベルを２８ｘ２８繰り返す）
    convolved_data = convolved_data.reshape(-1, convolved_data.shape[3])

    # # convolved_dataの重複を削除
    # convolved_data, unique_indices = np.unique(
    #     convolved_data, axis=0, return_index=True
    # )
    # input_data_label = input_data_label[unique_indices]
    # input_data = input_data[unique_indices]

    # ランダムに一部の点にのみブロックの画像を表示
    num_samples = len(convolved_data)
    num_blocks_to_display = min(15, num_samples)
    np.random.seed(0)  # Fix the seed for reproducibility
    random_indices = np.random.choice(num_samples, num_blocks_to_display, replace=False)
    convolved_data = convolved_data[random_indices]
    input_data = input_data[random_indices]
    input_data_label = input_data_label[random_indices]

    # 0-255の範囲にスケール
    input_data = scale_to_0_255(input_data)

    # ２チャネルごとに列方向に描画
    num_images = 1
    
    # figureの定義
    fig1, axs1 = plt.subplots(
        ncols=1, 
        nrows=1, 
        figsize=(8, 8), 
        dpi=300
        )
    fig2, axs2 = plt.subplots(
        ncols=num_blocks_to_display,
        nrows=2,
        figsize=(15, 2),
        dpi = 300
    )
    
    # 散布図の描画
    for img_idx in range(num_images):
        convolved_data_sep = convolved_data[:, (2 * img_idx) : (2 * (img_idx + 1))]
        if num_images != 1:
            ax = axs1[img_idx]
        else:
            ax = axs1
            
        # 軸の範囲を設定
        x_min = np.min(convolved_data_sep[:, 0])
        x_max = np.max(convolved_data_sep[:, 0])
        y_min = np.min(convolved_data_sep[:, 1])
        y_max = np.max(convolved_data_sep[:, 1])
        k = 0.2 * (x_max - x_min) / 2
        l = 0.2 * (y_max - y_min) / 2

        # 散布図のプロット
        sc = ax.scatter(
            convolved_data_sep[:, 0],
            convolved_data_sep[:, 1],
            cmap="tab10",
            c=input_data_label,
            marker="o",
            s=600,
            edgecolors="black",
        )
        # plt.colorbar(sc, label="label") #凡例のプロット
        # Annotationのプロット
        for dot_idx in range(len(convolved_data)):
            x, y = convolved_data_sep[dot_idx]
            ax.annotate(
                chr(dot_idx + 65),
                (x, y),
                weight="bold",
                ha="center",
                va="center",
                size=20,
                color="black",
            )

        ax.set_box_aspect(1)
        ax.set_xlim(x_min - k, x_max + k)
        ax.set_ylim(y_min - l, y_max + l)
        ax.set_xlabel(r"Channel 1")
        ax.set_ylabel(r"Channel 2")
        # ax.set_title(f"Channel {2*img_idx+1}-{2*(img_idx+1)} Dataset:{dataset_name}, Embedding:{embedding_method}\n(B={B}, b={block_size})")
    # ブロックの描画
    for dot_idx in range(len(convolved_data)):
        for channel_idx in range(2):
            ax2 = axs2[channel_idx, dot_idx]
            ax2.axis("off")
            if channel_idx == 0:
                # ax2に文字を書く
                ax2.text(
                    0.5,
                    0.5,
                    chr(dot_idx + 65),
                    horizontalalignment="center",
                    verticalalignment="center",
                    fontsize=15,
                )
            else:
                img = input_data[dot_idx, :, :, :3]
                imgbox = OffsetImage(
                    img, zoom=12, cmap="gray"
                )  # 解像度を上げるためにzoomパラメータを調整
                # print(input_data[dot_idx])
                # print(img)
                ab = AnnotationBbox(imgbox, (0.5, 0.5), frameon=True, pad=0.0)
                ax2.add_artist(ab)
                # ax2.imshow(imgbox, cmap="gray")  # vmin, vmaxの指定
                # if dot_idx == 0:
                    # ax2.set_title(f"Channel{channel_idx}")

    # 画像として保存
    # plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)
    fig2.subplots_adjust(wspace=0.1)
    fig1.tight_layout()
    fig2.tight_layout()
    
    fig1.savefig(file_dir + f"/{filename}.png")
    fig2.savefig(file_dir + f"/{filename}_blocks.png")
    plt.close()

    return None


# def visualize_emb(
#     input_data,
#     input_data_label,
#     convolved_data,
#     block_size: int,
#     stride: int,
#     B: int,
#     embedding_method: str,
#     dataset_name: str,
#     output_dots=True,
# ):
#     """
#     埋め込み後の点を可視化
#         input_data : 層の入力画像 (データ数, 高さ, 幅, 入力チャンネル数)
#         input_data_label : ブロックのラベル (データ数, 高さ, 幅, 出力チャンネル数)
#         convolved_data : 次元削減後のデータ
#         block_size : ブロックのサイズ
#         stride : ストライド
#         embedding_method : 埋め込み手法
#         dataset_name : データセットの名前
#     """
#     # ファイル名の重複を防ぐ処理

#     file_dir = "../results/results_emb"
#     if not os.path.exists(file_dir):
#         os.makedirs(file_dir)
#     filename = f"{embedding_method}_{dataset_name}"
#     filename = make_unique_filename(filename, file_dir)

#     # ランダムに１００データに制限
#     indices = np.random.choice(len(input_data), size=min(len(input_data), 100), replace=False)
#     input_data = input_data[indices]
#     input_data_label = input_data_label[indices]
#     convolved_data = convolved_data[indices]

#     # 画像データからブロックに変換
#     blocks = np.empty((0, block_size, block_size, input_data.shape[3]))
#     for img_idx in range(input_data.shape[0]):
#         img = input_data[img_idx]
#         block = util.view_as_windows(
#             img, (block_size, block_size, input_data.shape[3]), stride
#         ).reshape(-1, block_size, block_size, input_data.shape[3])
#         blocks = np.concatenate((blocks, block), axis=0)

#     input_data = blocks
#     input_data_label = np.repeat(
#         np.argmax(input_data_label, axis=1),
#         convolved_data.shape[1] * convolved_data.shape[2],
#     )  # １枚の画像のラベルをブロックの個数分繰り返す　（例：３のラベルを２８ｘ２８繰り返す）
#     convolved_data = convolved_data.reshape(-1, convolved_data.shape[3])

#     # convolved_dataの重複を削除
#     convolved_data, unique_indices = np.unique(
#         convolved_data, axis=0, return_index=True
#     )
#     input_data_label = input_data_label[unique_indices]
#     input_data = input_data[unique_indices]

#     # ランダムに一部の点にのみブロックの画像を表示
#     num_samples = len(convolved_data)
#     num_blocks_to_display = min(10, num_samples)  # 15個のデータを表示
#     np.random.seed(0)
#     random_indices = np.random.choice(num_samples, num_blocks_to_display, replace=False)
#     # random_indices = [157222, 771083, 203848, 231814, 517608, 630900, 174863, 861036, 749684, 262324, 8638,  77385, 283762, 592353, 752354]
#     print(random_indices)
#     convolved_data = convolved_data[random_indices]
#     input_data = input_data[random_indices]
#     input_data_label = input_data_label[random_indices]

#     # 0-255の範囲にスケール
#     input_data = scale_to_0_255(input_data)

#     # ２チャネルごとに描画するfigを作成
#     # num_images = int(convolved_data.shape[1] / 2)
#     num_images = 3
#     if int(input_data.shape[3]) == 1 or int(input_data.shape[3]) == 3:
#         num_input_channels = int(input_data.shape[3])
#     else:
#         num_input_channels = 6
#     # num_input_channels = int(input_data.shape[3])
#     fig, axs = plt.subplots(num_images, 1, figsize=(8, num_images * 6 + 1))

#     # ブロックの描画
#     for img_idx in range(num_images):
#         convolved_data_sep = convolved_data[
#             :, (2 * img_idx) : (2 * (img_idx + 1))
#         ]  # ２チャネルごとに切り取った写像先の値
#         ax = axs[img_idx]

#         # 軸の範囲を設定
#         x_min = np.min(convolved_data_sep[:, 0])
#         x_max = np.max(convolved_data_sep[:, 0])
#         y_min = np.min(convolved_data_sep[:, 1])
#         y_max = np.max(convolved_data_sep[:, 1])
#         k = 0.2 * (x_max - x_min) / 2
#         l = 0.2 * (y_max - y_min) / 2

#         # 散布図のプロット
#         sc = ax.scatter(
#             convolved_data_sep[:, 0],
#             convolved_data_sep[:, 1],
#             cmap="tab10",
#             c=input_data_label,
#             marker="o",
#             s=600,
#             edgecolors="black",
#         )
#         # plt.colorbar(sc, label="label") #凡例のプロット
        
#         # Annotationのプロット
#         if input_data.shape[3] <= 3:
#             for dot_idx in range(len(convolved_data)):
#                 x, y = convolved_data_sep[dot_idx]
#                 img = input_data[dot_idx]
#                 imgbox = OffsetImage(
#                     img, zoom=15 - block_size, cmap="gray"
#                 )  # 解像度を上げるためにzoomパラメータを調整
#                 ab = AnnotationBbox(
#                     imgbox,
#                     xy=(x, y),
#                     frameon=True,
#                     xycoords="data",
#                     boxcoords="data",
#                     pad=0.0,
#                 )
#                 ax.add_artist(ab)

#         ax.set_box_aspect(1)
#         ax.set_xlim(x_min - k, x_max + k)
#         ax.set_ylim(y_min - l, y_max + l)

#         # ax.set_title(f"Channel {2*img_idx+1}-{2*(img_idx+1)} Dataset:{dataset_name}, Embedding:{embedding_method}\n(B={B}, b={block_size})")

#     # 画像として保存
#     # plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)
#     fig.tight_layout()
#     fig.savefig(file_dir + f"/{filename}.png")
#     plt.close()

#     return None


def visualize_emb_dots(
    input_data_label,
    convolved_data,
    b: int,
    B: int,
    embedding_method: str,
    dataset_name: str,
):
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

    # ２チャネルごとに列方向に描画
    num_images = int(convolved_data.shape[1] / 2)
    fig, axs = plt.subplots(num_images, 1, figsize=(10, num_images * 10))

    # 一部の点を選ぶ
    num_to_select = min(convolved_data.shape[0], 3000)
    select_indices = np.random.choice(
        convolved_data.shape[0], num_to_select, replace=False
    )
    convolved_data = convolved_data[select_indices]
    input_data_label = input_data_label[select_indices]

    for img_idx in range(num_images):
        convolved_data_sep = convolved_data[:, (2 * img_idx) : (2 * (img_idx + 1))]
        ax = axs[img_idx]
        sc = ax.scatter(
            convolved_data_sep[:, 0],
            convolved_data_sep[:, 1],
            cmap="tab10",
            c=input_data_label,
            marker="o",
            s=50,
            edgecolors="black",
        )
        # plt.colorbar(sc, label='label')

        # 軸の範囲を設定
        x_min = np.min(convolved_data_sep[:, 0])
        x_max = np.max(convolved_data_sep[:, 0])
        y_min = np.min(convolved_data_sep[:, 1])
        y_max = np.max(convolved_data_sep[:, 1])

        k = 0.2 * (x_max - x_min) / 2
        l = 0.2 * (y_max - y_min) / 2
        ax.set_box_aspect(1)
        ax.set_xlim(x_min - k, x_max + k)
        ax.set_ylim(y_min - l, y_max + l)
        ax.set_title(
            f"Channel {2*img_idx+1}-{2*(img_idx+1)} Dataset:{dataset_name}, Embedding:{embedding_method}\n(B={B}, b={b})"
        )

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
    padded_images[
        :,
        pad_width : pad_width + original_size,
        pad_width : pad_width + original_size,
        :,
    ] = images

    return padded_images


def select_embedding_method(
    embedding_method: str, Channels_next: int, data_to_embed, data_to_embed_label
):
    if embedding_method == "PCA":
        pca = PCA(n_components=Channels_next, svd_solver="auto")
        return pca.fit(data_to_embed)
    elif embedding_method == "LDA":
        lda = LinearDiscriminantAnalysis(n_components=Channels_next)
        return lda.fit(data_to_embed, data_to_embed_label)

    elif embedding_method == "KPCA":
        kpca = KernelPCA(n_components=Channels_next, kernel="rbf")
        embedded_blocks = kpca.fit_transform(data_to_embed)

    elif embedding_method == "LE":
        # k = int(data_to_embed.shape[0] / Channels_next)
        k = None
        # n = k_for_knn
        print(f"k for knn:{k}")
        # LE = SpectralEmbedding(n_components=Channels_next, n_neighbors=n, random_state=0, n_jobs=-1)
        # embedded_blocks = LE.fit_transform(data_to_embed)
        # print("LE params:", LE.affinity_matrix_)
        embedded_blocks, _ = SLE(
            X=data_to_embed,
            Y=data_to_embed_label,
            la=1.0,
            map_d=Channels_next,
            n_neighbors=k,
        )

    elif embedding_method == "SLE":
        la = 0.2 # 値が高いほど近傍点の重みが大きくなる(小さいほどラベルの情報が使われる)
        # k = int(data_to_embed.shape[0] / Channels_next)
        k = None
        embedded_blocks, _ = SLE(
            X=data_to_embed,
            Y=data_to_embed_label,
            la=la,
            map_d=Channels_next,
            n_neighbors=k,
        )
        print(f"SLE parameter: {la}")
    elif embedding_method == "TSNE":
        tsne = TSNE(
            n_components=Channels_next,
            random_state=0,
            method="exact",
            perplexity=int(data_to_embed.shape[0] / Channels_next),
            n_iter=1500,
            init="pca",
            learning_rate="auto",
        )
        embedded_blocks = tsne.fit_transform(data_to_embed)

    elif embedding_method == "LLE":

        lle = LocallyLinearEmbedding(n_components=Channels_next, n_neighbors=100)
        embedded_blocks = lle.fit_transform(data_to_embed)
    else:
        print("Error: No embedding selected.")
        exit()

    # normalized_blocks = []
    # for channel in range(embedded_blocks.shape[1]):
    #     channel_data = embedded_blocks[:, channel]
    #     # normalized_channel = MinMaxScaler().fit_transform(channel_data.reshape(-1, 1))
    #     channel_data = StandardScaler().fit_transform(channel_data.reshape(-1, 1))
    #     normalized_blocks.append(channel_data)
    # normalized_blocks = np.stack(normalized_blocks, axis=1).reshape(-1, Channels_next)
    
    return embedded_blocks


def get_KTH_data():
    w = 200
    h = 200
    folder = "/workspaces/KernelCNN/app/data/KTH_TIPS"
    image_raw = []
    label_raw = []

    for i in os.listdir(folder):
        for j in os.listdir(os.path.join(folder, i)):
            path = os.path.join(folder, i, j)
            image = cv2.imread(path)
            image = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
            image = np.array(image)
            image = image.astype("float32")
            image /= 255
            image_raw.append(image)
            label_raw.append(i)
    image_raw = np.array(image_raw)
    label_names = np.unique(label_raw)
    y_label = pd.get_dummies(label_raw)
    one_hot = LabelBinarizer().fit(label_names)
    y_label = one_hot.transform(label_raw)
    y_label = np.argmax(y_label, axis=1)

    train_X, test_X, train_Y, test_Y = train_test_split(
        image_raw, y_label, test_size=0.2, random_state=0
    )

    return train_X, train_Y, test_X, test_Y


def select_datasets(num_train: int, num_test: int, datasets: str):
    if (datasets == "MNIST") or (datasets == "FMNIST"):
        image_size = 32
        if datasets == "MNIST":
            (train_X, train_Y), (test_X, test_Y) = mnist.load_data()
        if datasets == "FMNIST":
            (train_X, train_Y), (test_X, test_Y) = fashion_mnist.load_data()

        train_X = train_X.reshape(-1, 28, 28, 1)
        test_X = test_X.reshape(-1, 28, 28, 1)
        train_X = pad_images(train_X, image_size)
        test_X = pad_images(test_X, image_size)
        channel = 1
    elif datasets == "CIFAR10":
        image_size = 32
        (train_X, train_Y), (test_X, test_Y) = cifar10.load_data()
        train_X = train_X.reshape(-1, 32, 32, 3)
        test_X = test_X.reshape(-1, 32, 32, 3)
        train_X = pad_images(train_X, image_size)
        test_X = pad_images(test_X, image_size)
        channel = 3
    elif datasets == "KTH":
        image_size = 200
        train_X, train_Y, test_X, test_Y = get_KTH_data()
        # train_X, train_Y, test_X, test_Y = load_KTH_TIPS_dataset()
        channel = train_X.shape[3]

    train_Y = to_categorical(train_Y, 10)
    test_Y = to_categorical(test_Y, 10)
    train_X = train_X / 255
    test_X = test_X[:num_test]/ 255
    test_Y = test_Y[:num_test]

    # Select equal number of images from each class
    train_X_selected = []
    train_Y_selected = []
    # test_X_selected = []
    # test_Y_selected = []

    num_classes = train_Y.shape[1]
    num_train = min(num_train, train_X.shape[0])
    num_images_per_class = num_train // num_classes

    for class_label in range(num_classes):
        class_indices = np.where(train_Y[:, class_label] == 1)[0]
        selected_indices = np.random.choice(class_indices, num_images_per_class, replace=True)
        train_X_selected.extend(train_X[selected_indices])
        train_Y_selected.extend(train_Y[selected_indices])

        # class_indices = np.where(test_Y[:, class_label] == 1)[0]
        # selected_indices = np.random.choice(class_indices, num_images_per_class, replace=False)
        # test_X_selected.extend(test_X[selected_indices])
        # test_Y_selected.extend(test_Y[selected_indices])

    train_X_selected = np.array(train_X_selected)
    train_Y_selected = np.array(train_Y_selected)
    # test_X_selected = np.array(test_X_selected)
    # test_Y_selected = np.array(test_Y_selected)
    
    return train_X_selected, train_Y_selected, test_X, test_Y, channel, image_size

    # return train_X, train_Y, test_X, test_Y, channel, image_size
