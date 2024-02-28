from pathlib import Path
from turtle import pd
import cv2
import numpy as np
import os
import sys

from sklearn.naive_bayes import LabelBinarizer

sys.path.append("/workspaces/KernelCNN/app/data")
import matplotlib.pyplot as plt
import math
from skimage import util
from sklearn.discriminant_analysis import StandardScaler
from sklearn.manifold import SpectralEmbedding, TSNE, LocallyLinearEmbedding
from sklearn.decomposition import PCA, KernelPCA
from KSLE import SLE
import pickle
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10
from tensorflow.keras.utils import to_categorical
import pandas as pd


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
    data, layer_number, embedding_method: str, dataset_name: str, suptitle: str
):
    num_data = 3
    img_idx = 1
    for n in range(3):
        data_to_display = data[n]
        # data_to_display = scale_to_0_255(data_to_display)
        if data_to_display.shape[2] == 6:
            num_in_a_row = 6
        else:
            data_to_display = data_to_display[:, :, :16]
            num_in_a_row = 8  # default 4
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
        filename = f"{layer_number}_{embedding_method}_{dataset_name}_{img_idx}"
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
        num_input_channels,
        num_output_channels,
        figsize=(num_output_channels + 10, num_input_channels + 1),
        dpi=72,
    )
    # fig.suptitle(f"Layer{layer_idx} weights")

    for i in range(num_input_channels):
        for j in range(num_output_channels):
            filter_weights = weights[:, :, i, j]
            if num_input_channels == 1:
                axs[j].imshow(filter_weights, cmap="gray")
                axs[j].axis("off")
            else:
                axs[i, j].imshow(filter_weights, cmap="gray")
                axs[i, j].axis("off")

    # 横の余白を小さくし、縦の余白を大きくする
    plt.subplots_adjust(wspace=0.6, hspace=1.5)
    plt.tight_layout()
    filename = f"{dataset_name}_LeNet_weights_layer{layer_idx}"
    filename = make_unique_filename(filename, "./weights_results")
    plt.savefig(f"../results/results_weights/{filename}.png")
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

    # convolved_dataの重複を削除
    convolved_data, unique_indices = np.unique(
        convolved_data, axis=0, return_index=True
    )
    input_data_label = input_data_label[unique_indices]
    input_data = input_data[unique_indices]

    # ランダムに一部の点にのみブロックの画像を表示
    num_samples = len(convolved_data)
    num_blocks_to_display = min(15, num_samples)
    random_indices = np.random.choice(num_samples, num_blocks_to_display, replace=False)
    print(random_indices)
    convolved_data = convolved_data[random_indices]
    input_data = input_data[random_indices]
    input_data_label = input_data_label[random_indices]

    # 0-255の範囲にスケール
    input_data = scale_to_0_255(input_data)

    # ２チャネルごとに列方向に描画
    # num_images = int(convolved_data.shape[1] / 2)
    num_images = 3
    if int(input_data.shape[3]) == 1 or int(input_data.shape[3]) == 3:
        num_input_channels = int(input_data.shape[3])
    else:
        num_input_channels = 6
    # num_input_channels = int(input_data.shape[3])
    fig, axs = plt.subplots(num_images, 1, figsize=(8, num_images * 6 + 1))
    fig2, axs2 = plt.subplots(
        num_blocks_to_display,
        num_input_channels + 1,
        figsize=(3 + 2 * num_input_channels / 3, num_blocks_to_display),
    )

    for img_idx in range(num_images):
        convolved_data_sep = convolved_data[:, (2 * img_idx) : (2 * (img_idx + 1))]
        ax = axs[img_idx]

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
        # ax.set_title(f"Channel {2*img_idx+1}-{2*(img_idx+1)} Dataset:{dataset_name}, Embedding:{embedding_method}\n(B={B}, b={block_size})")
    for dot_idx in range(len(convolved_data)):
        for channel_idx in range(num_input_channels + 1):
            ax2 = axs2[dot_idx, channel_idx]
            ax2.axis("off")
            if channel_idx == 0:
                # ax2に文字を書く
                ax2.text(
                    0.5,
                    0.5,
                    chr(dot_idx + 65),
                    horizontalalignment="center",
                    verticalalignment="center",
                    fontsize=20,
                )
            else:
                img = input_data[dot_idx, :, :, channel_idx - 1]
                # print(input_data[dot_idx])
                # print(img)
                ax2.imshow(img, cmap="gray")  # vmin, vmaxの指定
                if dot_idx == 0:
                    ax2.set_title(f"Channel{channel_idx}")

    # 画像として保存
    # plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)
    fig.tight_layout()
    fig2.tight_layout()
    fig.savefig(file_dir + f"/{filename}.png")
    fig2.savefig(file_dir + f"/{filename}_blocks.png")
    plt.close()

    output_dots = False
    if output_dots:
        visualize_emb_dots(
            input_data_label,
            convolved_data,
            block_size,
            B,
            embedding_method,
            dataset_name,
        )


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

    elif embedding_method == "SLE":
        embedded_blocks, _ = SLE(
            X=data_to_embed, Y=data_to_embed_label, la=0.5, map_d=Channels_next
        )

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

    normalized_blocks = []
    for channel in range(embedded_blocks.shape[1]):
        channel_data = embedded_blocks[:, channel]
        # normalized_channel = MinMaxScaler().fit_transform(channel_data.reshape(-1, 1))
        channel_data = StandardScaler().fit_transform(channel_data.reshape(-1, 1))
        normalized_blocks.append(channel_data)
    normalized_blocks = np.stack(normalized_blocks, axis=1).reshape(-1, Channels_next)
    return normalized_blocks

def get_KTH_data():
    w=200
    h=200
    folder = "/workspaces/KernelCNN/app/data/KTH_TIPS"
    image_raw = []
    label_raw = []

    for i in os.listdir(folder):
        for j in os.listdir(os.path.join(folder,i)):
            path = os.path.join(folder, i, j)
            image = cv2.imread(path)
            image = cv2.resize(image, (w, h), interpolation = cv2.INTER_AREA)
            image = np.array(image)
            image = image.astype('float32')
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

def load_KTH_TIPS_dataset():
    cache_file = "dataset_cache.pkl"

    # Check if cache file exists
    if os.path.exists(cache_file):
        # Load data from cache
        with open(cache_file, "rb") as file:
            return pickle.load(file)

    file_dir = "/workspaces/KernelCNN/app/data/KTH_TIPS"
    images = np.ndarray((0, 200, 200, 3))
    labels = np.ndarray(0)
    label_to_number = {}
    number = 0
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if file.endswith(".png"):
                image_path = os.path.join(root, file)
                label = os.path.basename(root)
                if label not in label_to_number:
                    label_to_number[label] = number
                    number += 1

                image = cv2.imread(image_path)
                image = cv2.resize(image, (200, 200)).reshape(1, 200, 200, 3)
                # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).reshape(1, 200, 200, 1)
                images = np.append(images, image, axis=0)
                labels = np.append(labels, label_to_number[label])

    images = np.array(images)
    labels = np.array(labels)

    # Shuffle the data
    indices = np.arange(len(images))
    np.random.shuffle(indices)
    images = images[indices]
    labels = labels[indices]

    # Split the data into training and test sets while maintaining class balance
    train_X, test_X, train_Y, test_Y = train_test_split(
        images, labels, test_size=0.2, stratify=labels, random_state=0
    )

    # Save data to cache
    with open(cache_file, "wb") as file:
        pickle.dump((train_X, train_Y, test_X, test_Y), file)

    return train_X, train_Y, test_X, test_Y


def check_KTH_loading():
    train_X, train_Y, test_X, test_Y = load_KTH_TIPS_dataset()

    # Print the shape of the loaded data
    print("Train X shape:", train_X.shape)
    print("Train Y shape:", train_Y.shape)
    print("Test X shape:", test_X.shape)
    print("Test Y shape:", test_Y.shape)

    # Print a sample image and its corresponding label
    sample_index = 0
    sample_image = train_X[sample_index]
    sample_label = train_Y[sample_index]
    print("Sample Image:")
    plt.imshow(sample_image)
    plt.title("Label: " + sample_label)
    plt.savefig("sample_image.png")


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
    train_X = train_X[:num_train] / 255
    train_Y = train_Y[:num_train]
    test_X = test_X[:num_test] / 255
    test_Y = test_Y[:num_test]

    return train_X, train_Y, test_X, test_Y, channel, image_size
