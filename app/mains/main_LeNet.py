import os
from matplotlib import pyplot as plt

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import tensorflow as tf
from keras import backend
from keras import layers, models, losses
from keras import optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn import metrics
from app.mains.pkg.functions import visualize_emb
from app.mains.pkg.functions import display_images, display_weights
from app.mains.pkg.functions import select_datasets
import layers as my_layers
import sys

np.random.seed(0)
tf.random.set_seed(0)


def get_intermediate_output(model, layer_index, data):
    intermediate_layer_model = models.Model(
        inputs=model.input, outputs=model.layers[layer_index].output
    )
    return intermediate_layer_model.predict(data)


def main_LeNet(
    num_train: int,
    num_test: int,
    datasets: str,
    block_size=[5, 5],
    display=True,
    layers_BOOL=[1, 1, 1, 0],
):
    backend.clear_session()
    print("Number of training samples:", num_train)
    stride = 2

    train_X, train_Y, test_X, test_Y, channel, image_size = select_datasets(
        num_train, num_test, datasets
    )
    # 訓練データの表示
    # display_images(
    #     train_X, train_Y, 2, "LeNettrain", datasets, f"LeNet Input n={num_train}"
    # )
    # print(train_Y[:10])
    
    # model definition
    activation = "relu"
    # activation = None
    # activation = 'tanh'
    model = models.Sequential()
    model.add(
        layers.Conv2D(
            40,
            kernel_size=(block_size[0], block_size[0]),
            activation=activation,
            strides=stride,
            input_shape=(image_size, image_size, channel),
        )
    )

    if layers_BOOL[0]:
        model.add(layers.MaxPooling2D((2, 2)))
        # model.add(layers.Activation('sigmoid'))
        if layers_BOOL[1]:
            model.add(
                layers.Conv2D(
                    8,
                    kernel_size=(block_size[1], block_size[1]),
                    activation=activation,
                    strides=stride,
                    padding="valid",
                )
            )
            if layers_BOOL[2]:
                model.add(layers.MaxPooling2D((2, 2)))
                # model.add(layers.Activation('sigmoid'))
                if layers_BOOL[3]:
                    model.add(
                        layers.Conv2D(
                            32,
                            kernel_size=(block_size[1], block_size[1]),
                            activation=activation,
                            strides=stride,
                            padding="valid",
                        )
                    )
    model.add(layers.Flatten())
    # model.add(layers.Dense(120, activation=activation))
    # model.add(layers.Dense(84, activation=activation))
    model.add(layers.Dense(10, activation="softmax"))
    # model.summary()

    # Model parameters
    batch_size = 64
    epochs = 50

    # Callbacks
    lrr = ReduceLROnPlateau(
        monitor="val_accuracy", patience=2, verbose=1, factor=0.5, min_lr=0.0001
    )
    es = EarlyStopping(monitor="val_loss", mode="auto", patience=3, verbose=0)
    cp = ModelCheckpoint(
        "./weights/model_weights_epoch_{epoch:02d}.h5",
        save_weights_only=True,
        save_freq="epoch",
        period=1,
    )

    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    result = model.fit(
        train_X,
        train_Y,
        batch_size=batch_size,
        verbose=1,
        epochs=epochs,
        callbacks=[es, lrr],
        validation_data=(test_X, test_Y),
    )

    # SVMによる識別
    train_features = get_intermediate_output(model, 1, train_X)

    # Train a classifier
    # classifier = my_layers.SupportVectorsMachine()
    # classifier = my_layers.RandomForest()
    # classifier = my_layers.GaussianProcess()
    classifier = my_layers.kNearestNeighbors(n_neighbors=1)
    classifier.fit(train_features, train_Y)

    test_features = get_intermediate_output(model, 1, test_X)
    predictions = classifier.predict(test_features)
    accuracy = metrics.accuracy_score(test_Y, predictions) * 100
    classification_report = metrics.classification_report(test_Y, predictions)
    confusion_matrix = metrics.confusion_matrix(
        np.argmax(test_Y, axis=1), np.argmax(predictions, axis=1)
    )
    print(classification_report)
    print(confusion_matrix)
    print(f"Accuracy: {accuracy:.4f}")

    if True:  # 学習の過程を可視化
        plt.figure(figsize=[20, 8])
        plt.plot(result.history["accuracy"])
        plt.plot(result.history["val_accuracy"])
        plt.title("Epoch VS Model Accuracy", size=25, pad=20)
        plt.ylabel("Accuracy", size=15)
        plt.xlabel("Epoch", size=15)
        plt.legend(["train", "test"], loc="upper left")
        plt.savefig("./learning_historyAccuracy.png")

        plt.figure(figsize=[20, 8])
        plt.plot(result.history["loss"])
        plt.plot(result.history["val_loss"])
        plt.title("Epoch VS Model Loss", size=25, pad=20)
        plt.ylabel("Loss", size=15)
        plt.xlabel("Epoch", size=15)
        plt.legend(["train", "test"], loc="upper right")
        plt.savefig("./learning_historyLoss.png")

    if display:
        # 学習後のモデルの出力
        block_outputs = []
        for layer_idx in range(len(model.layers) - 2):
            block_outputs.append(get_intermediate_output(model, layer_idx, train_X))
            print(f"Block {layer_idx+1} output shape:", block_outputs[layer_idx].shape)

        weights1 = model.layers[0].get_weights()[0]
        visualize_emb(
            train_X,
            train_Y,
            block_outputs[0],
            block_size=block_size[0],
            stride=stride,
            B=0,
            embedding_method="LeNet",
            dataset_name=datasets,
        )
        display_images(
            block_outputs[0],
            train_Y,
            2,
            "LeNet",
            datasets,
            f"LeNet Output layer 2 n={num_train}",
        )
        display_weights(weights1, datasets, layer_idx=2)

    epochs_to_check = [1, 5, 10, 20, 30, 40, 50]

    # 学習途中のモデルの出力
    if False:
        for epoch in epochs_to_check:
            block_outputs = []
            model.load_weights(f"./weights/model_weights_epoch_{epoch:02d}.h5")
            block_outputs.append(get_intermediate_output(model, "conv2d", train_X))
            block_outputs.append(
                get_intermediate_output(model, "max_pooling2d", train_X)
            )
            block_outputs.append(get_intermediate_output(model, "conv2d_1", train_X))
            weights1 = model.get_layer("conv2d").get_weights()[0]
            weights2 = model.get_layer("conv2d_1").get_weights()[0]

            display_images(
                block_outputs[0],
                99,
                "LeNet",
                datasets,
                suptitle=f"LeNet Output layer 2 epoch={epoch} n={num_train}",
            )
            display_images(
                block_outputs[2],
                100,
                "LeNet",
                datasets,
                suptitle=f"LeNet Output layer 4 epoch={epoch} n={num_train}",
            )
            # display_weights(weights1, datasets)

            # Print the shapes of the intermediate outputs
            for i, output in enumerate(block_outputs):
                print(f"Block {i+1} output shape:", output.shape)

            # 畳み込み第1層を可視化
            maxPool_1_out = np.array(block_outputs[1])
            visualize_emb(
                train_X,
                train_Y,
                block_outputs[0],
                block_size=block_size,
                stride=stride,
                B=None,
                embedding_method="LeNet",
                dataset_name=datasets,
            )
            visualize_emb(
                maxPool_1_out,
                train_Y,
                block_outputs[2],
                block_size=block_size,
                stride=stride,
                B=None,
                embedding_method="LeNet",
                dataset_name=datasets,
            )
    return accuracy


if __name__ == "__main__":
    args = sys.argv
    num_train = int(args[1])
    num_test = int(args[2])
    datasets = args[3]
    block_size = list(map(int, args[4].split(",")))
    layers_BOOL = list(map(int, args[5].split(",")))
    arguments = [num_train, num_test, datasets, None]
    main_LeNet(
        num_train, num_test, datasets, block_size=block_size, layers_BOOL=layers_BOOL
    )
