import cv2
import matplotlib.pyplot as plt
import numpy as np

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