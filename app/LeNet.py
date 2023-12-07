import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import backend
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10
from tensorflow.keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint

from functions import visualize_emb
from functions import display_images, display_weights, make_unique_filename
from functions import pad_images

import sys

import functions

# Define a function to get intermediate outputs for the convolutional layers

def get_intermediate_output(model, layer_name, data):
    intermediate_layer_model = models.Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    return intermediate_layer_model.predict(data)


def main(num_train: int, test_num : int, datasets : str, block_size=[5,5], display=True):
    backend.clear_session()
    print('Number of training samples:', num_train)
    #block_size = [7,3]
    stride = 1
    
    if (datasets == 'MNIST') or (datasets == 'FMNIST'):
        if datasets == 'MNIST':
            (train_X, train_Y), (test_X, test_Y) = mnist.load_data()
        if datasets == 'FMNIST':
            (train_X, train_Y), (test_X, test_Y) = fashion_mnist.load_data()

        train_X = train_X.reshape(-1, 28, 28, 1) 
        test_X = test_X.reshape(-1, 28, 28, 1)
        train_X = pad_images(train_X, 32)
        test_X = pad_images(test_X, 32)
        channel = 1
    elif datasets == 'CIFAR10':
        (train_X, train_Y), (test_X, test_Y) = cifar10.load_data()
        train_X = train_X.reshape(-1, 32, 32, 3) 
        test_X = test_X.reshape(-1, 32, 32, 3)
        channel = 3
    
    train_Y = to_categorical(train_Y, 10)
    test_Y = to_categorical(test_Y, 10)

    train_X = train_X[:num_train]/255
    train_Y = train_Y[:num_train]
    test_X = test_X[:test_num]/255
    test_Y = test_Y[:test_num]

    # LeNet-5 model definition
    model = models.Sequential([
        layers.Conv2D(6, kernel_size=(block_size[0], block_size[0]), activation='relu', strides= stride, input_shape=(32, 32, channel)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(16, kernel_size=(block_size[1], block_size[1]),activation='relu', strides= stride),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(120, activation='relu'),
        layers.Dense(84, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    #model.summary()
    # Compile the model
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    batch_size = 64
    epochs = 1000
    es = EarlyStopping(monitor='val_loss', mode='auto', patience=5, verbose=0)
    #cp = ModelCheckpoint("./weights/model_weights_epoch_{epoch:02d}.h5", save_weights_only=True, save_freq='epoch', period = 10)
    
    history = model.fit(train_X, train_Y, batch_size=batch_size, verbose=0, epochs=epochs, callbacks=[es], validation_split=0.2)

    # predict test samples
    test_loss, test_acc = model.evaluate(test_X, test_Y)
    test_acc*=100
    print('Accuracy:',test_acc)

    if display:
        #学習後のモデルの出力
        block_outputs = []
        block_outputs.append(get_intermediate_output(model, 'conv2d', train_X))
        block_outputs.append(get_intermediate_output(model, 'max_pooling2d', train_X))
        block_outputs.append(get_intermediate_output(model, 'conv2d_1', train_X))
        #for i, output in enumerate(block_outputs):
        #    print(f"Block {i+1} output shape:", output.shape)
        weights1 = model.get_layer("conv2d").get_weights()[0]
        weights2 = model.get_layer("conv2d_1").get_weights()[0]
        visualize_emb(train_X, train_Y, block_outputs[0], block_size=block_size[0], stride=stride, B=None, embedding_method='LeNet', dataset_name=datasets)
        visualize_emb(block_outputs[1], train_Y, block_outputs[2], block_size=block_size[1], stride=stride, B=None, embedding_method='LeNet', dataset_name=datasets)
        display_images(block_outputs[0], 2, 'LeNet', datasets, f'LeNet Output layer 2 n={num_train}')
        display_images(block_outputs[2], 4, 'LeNet', datasets, f'LeNet Output layer 4 n={num_train}')
        display_weights(weights1, datasets, layer_idx=2)
        display_weights(weights2, datasets, layer_idx=4)
    
    '''
    # 学習の履歴から損失と精度を取得
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']

    # 損失のプロット
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # 精度のプロット
    plt.subplot(1, 2, 2)
    plt.plot(accuracy, label='Training Accuracy')
    plt.plot(val_accuracy, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    filename = make_unique_filename("LeNet_Learning_history", "./histories")
    plt.savefig(f"./histories/{filename}")
    '''
    epochs_to_check = [10, 50, 100]
    # 学習途中のモデルの出力
    if False:
        for epoch in epochs_to_check:
            block_outputs=[]
            model.load_weights(f"./weights/model_weights_epoch_{epoch:02d}.h5")
            block_outputs.append(get_intermediate_output(model, 'conv2d', train_X))
            block_outputs.append(get_intermediate_output(model, 'conv2d_1', train_X))
            weights1 = model.get_layer("conv2d").get_weights()[0]
            weights2 = model.get_layer("conv2d_1").get_weights()[0]

            display_images(block_outputs[0], 99, 'LeNet', datasets)
            display_images(block_outputs[1], 100, 'LeNet', datasets)
            display_weights(weights1, datasets)
            
            #for i in range(weights2.shape[0]):
            #    weight = weights2[i:]
            #    display_images(weight, 102+i)

            # Print the shapes of the intermediate outputs
            for i, output in enumerate(block_outputs):
                print(f"Block {i+1} output shape:", output.shape)

            # 畳み込み第1層を可視化

            visualize_emb(train_X, train_Y, block_outputs[0], block_size=block_size, stride=stride, B=None, embedding_method='LeNet', dataset_name=datasets)
    return test_acc

if __name__ == '__main__':

    args = sys.argv
    num_train = int(args[1])
    num_test = int(args[2])
    datasets = args[3]
    arguments = [num_train, num_test, datasets, None]
    #functions.calculate_average_accuracy(main, arguments, datasets, num_train, num_test, 10)
    main(num_train, num_test, datasets)
