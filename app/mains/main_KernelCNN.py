import sys
import os

# import layers
import layers as layers
import functions
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
np.random.seed(0)


def main_kernelCNN(
    num_train,
    num_test,
    datasets: str,
    B=3000,
    embedding_method=["LE", "LE"],
    block_size=[5, 5],
    layers_BOOL=[1, 0, 0, 0],
):

    train_X, train_Y, test_X, test_Y, channel, image_size = functions.select_datasets(
        num_train, num_test, datasets
    )
    stride = 2
    # モデル定義
    model = layers.Model(display=True)
    model.data_set_name = datasets
    model.add_layer(
        layers.KIMLayer(
            block_size=block_size[0],
            channels_next=40,
            stride=stride,
            padding=False,
            emb=embedding_method[0],
            num_blocks=B,
        )
    )
    if layers_BOOL[0]:
        model.add_layer(layers.MaxPoolingLayer(pool_size=2))
    if layers_BOOL[1]:
        model.add_layer(
            layers.KIMLayer(
                block_size=block_size[1],
                channels_next=60,
                stride=stride,
                padding=False,
                emb=embedding_method[1],
                num_blocks=B,
            )
        )
    if layers_BOOL[2]:
        model.add_layer(layers.MaxPoolingLayer(pool_size=2))
    if layers_BOOL[3]:
        model.add_layer(
            layers.KIMLayer(
                block_size=block_size[2],
                channels_next=32,
                stride=stride,
                padding=False,
                emb=embedding_method[2],
                num_blocks=B,
            )
        )
    # model.add_layer(layers.MaxPoolingLayer(pool_size=2))
    # model.add_layer(layers.KIMLayer(block_size=5, channels_next = 32, stride = 1, padding=False, emb=embedding_method, num_blocks=num_blocks))
    # model.add_layer(layers.KIMLayer(block_size=5, channels_next = 120, stride = 1, emb=emb))

    # model.add_layer(layers.GaussianProcess())
    # model.add_layer(layers.SupportVectorsMachine())
    # model.add_layer(layers.RandomForest())
    model.add_layer(layers.kNearestNeighbors(n_neighbors=1))
    # model.add_layer(layers.QuadraticDiscriminantAnalysis())

    print("========================================")
    print("Summary of the training:")
    print("Dataset:", datasets)
    print("Number of training samples:", num_train)
    print("Number of testing samples:", num_test)
    print("Embedding method:", embedding_method)
    print("Block size:", block_size)
    print("KIM blocks:", B)
    print("Layers activation status:", layers_BOOL)
    print("========================================")

    model.fit(train_X, train_Y)
    accuracy = model.predict(test_X, test_Y)
    return accuracy


if __name__ == "__main__":
    args = sys.argv
    if len(args) < 6:
        print(
            "Usage: python main_KernelCNN.py [num_train] [num_test] [dataset] [embedding_method] [block_size] [layers_BOOL]"
        )
        sys.exit(1)
    num_train = int(args[1])  # train
    num_test = int(args[2])  # test
    dataset_name = args[3]
    embedding_method = list(map(str, args[4].split(",")))
    block_size = list(map(int, args[5].split(",")))
    if len(args) > 6:
        layers_BOOL = list(map(int, args[6].split(",")))
    else:
        layers_BOOL = [1, 0, 0, 0]
    # arguments = [n,m,emb,3000]
    main_kernelCNN(
        num_train=num_train,
        num_test=num_test,
        datasets=dataset_name,
        B=100,
        embedding_method=embedding_method,
        block_size=block_size,
        layers_BOOL=layers_BOOL,
    )
