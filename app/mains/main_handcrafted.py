import sys
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from tqdm import tqdm
from sklearn import metrics
from functions import *
import numpy as np
from skimage import feature
import layers as my_layers


def main_HOG(num_train=1000, num_test=1000, datasets: str = "MNIST"):
    X_train, Y_train, X_test, Y_test, channel, image_size = select_datasets(
        num_train, num_test, datasets
    )

    # HOG feature extraction
    descriptors_train = [
        np.concatenate(
            [
                feature.hog(
                    channel,
                    orientations=8,
                    pixels_per_cell=(4, 4),
                    cells_per_block=(1, 1),
                    transform_sqrt=True,
                    block_norm="L2-Hys",
                    visualize=False,
                )
                for channel in train_image.transpose((2, 0, 1))
            ]
        )
        for train_image in X_train
    ]
    descriptors_test = [
        np.concatenate(
            [
                feature.hog(
                    channel,
                    orientations=8,
                    pixels_per_cell=(4, 4),
                    cells_per_block=(1, 1),
                    transform_sqrt=True,
                    block_norm="L2-Hys",
                    visualize=False,
                )
                for channel in test_image.transpose((2, 0, 1))
            ]
        )
        for test_image in X_test
    ]
    # Convert descriptors to numpy arrays
    descriptors_train = np.array(descriptors_train).reshape(len(descriptors_train), -1)
    descriptors_test = np.array(descriptors_test).reshape(len(descriptors_test), -1)

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
