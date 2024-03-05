from email.mime import image
import sys
import os
from matplotlib.pylab import multi_dot

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from tqdm import tqdm
from sklearn import metrics
from functions import *
import numpy as np
from skimage import feature
import layers as my_layers
from functions import *


def main_HOG(num_train=1000, num_test=1000, datasets: str = "MNIST"):
    X_train, Y_train, X_test, Y_test, channel, image_size = select_datasets(
        num_train, num_test, datasets
    )

    orientations = 9
    pixels_per_cell = (8, 8)
    cells_per_block = (3, 3)

    descriptors_train = []
    descriptors_test = []
    images_train = []
    images_test = []

    for train_image in X_train:
        hog_feature, hog_image = feature.hog(
            train_image,
            orientations=orientations,
            pixels_per_cell=pixels_per_cell,
            cells_per_block=cells_per_block,
            transform_sqrt=True,
            block_norm="L2-Hys",
            visualize=True,
            channel_axis=-1,
        )
        descriptors_train.append(hog_feature)
        images_train.append(hog_image)

    for test_image in X_test:
        hog_feature, hog_image = feature.hog(
            test_image,
            orientations=orientations,
            pixels_per_cell=pixels_per_cell,
            cells_per_block=cells_per_block,
            transform_sqrt=True,
            block_norm="L2-Hys",
            visualize=True,
            channel_axis=-1,
        )
        descriptors_test.append(hog_feature)
        images_test.append(hog_image)

    # Convert descriptors to numpy arrays
    descriptors_train = np.array(descriptors_train).reshape(len(descriptors_train), -1)
    descriptors_test = np.array(descriptors_test).reshape(len(descriptors_test), -1)
    images_train = np.array(images_train)
    images_test = np.array(images_test)
    images_train = images_train.reshape(
        images_train.shape[0], images_train.shape[1], images_train.shape[2], 1
    )
    images_test = images_test.reshape(
        images_test.shape[0], images_test.shape[1], images_test.shape[2], 1
    )

    # Show HOG images
    display_images(images_test, Y_test, 1, "HOG", datasets, "")

    # SVM classification
    # classifier = my_layers.SupportVectorsMachine()
    # classifier = my_layers.RandomForest()
    # classifier = my_layers.GaussianProcess()
    classifier = my_layers.kNearestNeighbors(n_neighbors=1)
    classifier.fit(descriptors_train, Y_train)
    Y_pred = classifier.predict(descriptors_test)
    accuracy = metrics.accuracy_score(Y_test, Y_pred) * 100
    classification_report = metrics.classification_report(Y_test, Y_pred)
    print(classification_report)
    print("Accuracy:", accuracy)

    return accuracy


if __name__ == "__main__":
    args = sys.argv
    if len(args) != 4:
        print("Usage: python main_handcrafted.py <train> <test> <dataset>")
        sys.exit(1)
    else:
        n = int(args[1])  # train
        m = int(args[2])  # test
        dataset_name = args[3]

    main_HOG(num_train=n, num_test=m, datasets=dataset_name)
