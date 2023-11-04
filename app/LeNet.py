import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10
from tensorflow.keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint

from functions import visualize_emb
from functions import display_images, display_weights
from functions import pad_images

import sys

# Define a function to get intermediate outputs for the convolutional layers

def get_intermediate_output(model, layer_name, data):
    intermediate_layer_model = models.Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    return intermediate_layer_model.predict(data)


def main(train_num: int, test_num : int, datasets : str):
    print('Number of training samples:', train_num)
    block_size = 5
    stride = 1
    
    if (datasets == 'MNIST') or (datasets == 'FMNIST'):
        if datasets == 'MNIST':
            (train_X, train_Y), (test_X, test_Y) = mnist.load_data()
        if datasets == 'FMNIST':
            (train_X, train_Y), (test_X, test_Y) = fashion_mnist.load_data()
        train_X = pad_images(train_X)
        test_X = pad_images(test_X)
        train_X = train_X.reshape(-1, 32, 32, 1) 
        test_X = test_X.reshape(-1, 32, 32, 1)
        channel = 1
    elif datasets == 'CIFAR10':
        (train_X, train_Y), (test_X, test_Y) = cifar10.load_data()
        train_X = train_X.reshape(-1, 32, 32, 3) 
        test_X = test_X.reshape(-1, 32, 32, 3)
        channel = 3
    
    train_Y = to_categorical(train_Y, 10)
    test_Y = to_categorical(test_Y, 10)

    train_X = train_X[:train_num]
    train_Y = train_Y[:train_num]
    test_X = test_X[:test_num]
    test_Y = test_Y[:test_num]

    # LeNet-5 model definition
    model = models.Sequential([
        layers.Conv2D(6, kernel_size=(block_size, block_size), activation='relu', strides= stride, input_shape=(32, 32, channel)),
        layers.AveragePooling2D(pool_size=(2, 2)),
        layers.Conv2D(16, kernel_size=(block_size, block_size),activation='relu', strides= stride),
        layers.AveragePooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(120, activation='relu'),
        layers.Dense(84, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    model.summary()
    # Compile the model
    model.compile(optimizer='adagrad', loss='categorical_crossentropy',
                metrics=['accuracy'])

    # Train the model
    batch_size = 256
    epochs = 1000
    es = EarlyStopping(monitor='val_loss', mode='auto', patience=5, verbose=0)
    cp = ModelCheckpoint("./weights/model_weights_epoch_{epoch:02d}.h5", save_weights_only=True, save_freq='epoch', period = 10)
    
    history = model.fit(train_X, train_Y, batch_size=batch_size, verbose=1, epochs=epochs, callbacks=[es, cp], validation_split=0.1)

    # predict test samples
    test_loss, test_acc = model.evaluate(test_X, test_Y)

    print('Accuracy:',test_acc)

    #学習後のモデルの出力
    block_outputs = []
    block_outputs.append(get_intermediate_output(model, 'conv2d', train_X))
    block_outputs.append(get_intermediate_output(model, 'average_pooling2d', train_X))
    block_outputs.append(get_intermediate_output(model, 'conv2d_1', train_X))
    for i, output in enumerate(block_outputs):
        print(f"Block {i+1} output shape:", output.shape)
    weights1 = model.get_layer("conv2d").get_weights()[0]
    weights2 = model.get_layer("conv2d_1").get_weights()[0]
    visualize_emb(train_X, train_Y, block_outputs[0], block_size=block_size, stride=stride, B=None, embedding_method='LeNet', dataset_name=datasets)
    visualize_emb(block_outputs[1], train_Y, block_outputs[2], block_size=block_size, stride=stride, B=None, embedding_method='LeNet', dataset_name=datasets)
    display_images(block_outputs[0], 2, 'LeNet', datasets)
    display_images(block_outputs[1], 4, 'LeNet', datasets)
    display_weights(weights1, datasets, layer_idx=2)
    display_weights(weights2, datasets, layer_idx=4)
    
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

if __name__ == '__main__':

    args = sys.argv
    num_train = int(args[1])
    num_test = int(args[2])
    datasets = args[3]
    
    main(num_train, num_test, datasets)
