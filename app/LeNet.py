import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10
from tensorflow.keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint

from functions import visualize_emb
from functions import display_images
from functions import pad_images

import sys

# Define a function to get intermediate outputs for the convolutional layers

def get_intermediate_output(model, layer_name, data):
    intermediate_layer_model = models.Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    return intermediate_layer_model.predict(data)


def main(train_num: int, test_num : int, datasets : str):
    print('Number of training samples:', train_num)
    
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
        layers.Conv2D(6, kernel_size=(5, 5), activation='relu', input_shape=(32, 32, channel)),
        layers.AveragePooling2D(pool_size=(2, 2)),
        layers.Conv2D(16, kernel_size=(5, 5), activation='relu'),
        layers.AveragePooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(120, activation='relu'),
        layers.Dense(84, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    #model.summary()
    # Compile the model
    model.compile(optimizer='adagrad', loss='categorical_crossentropy',
                metrics=['accuracy'])

    # Train the model
    batch_size = 256
    epochs = 1000
    es = EarlyStopping(monitor='val_loss', mode='auto', patience=5, verbose=0)
    checkpoint = ModelCheckpoint("./weights/model_weights_epoch_{epoch:02d}.h5", save_weights_only=True, save_freq=10)
    
    history = model.fit(train_X, train_Y, batch_size=batch_size, verbose=0, epochs=epochs, callbacks=[es], validation_split=0.1)

    # predict test samples
    test_loss, test_acc = model.evaluate(test_X, test_Y)

    print('Accuracy:',test_acc)

    #epochs_to_check = [10, 50, 100, 500, 1000]
    epochs_to_check = [1000]

    # Get intermediate outputs for the convolutional layers and save them in the list
    if False:
        for epoch in epochs_to_check:
            block_outputs=[]
            #model.load_weights(f"./weights/model_weights_epoch_{epoch:02d}.h5")
            block_outputs.append(get_intermediate_output(model, 'conv2d', train_X))
            block_outputs.append(get_intermediate_output(model, 'conv2d_1', train_X))
            weights1 = model.get_layer("conv2d").get_weights()[0]
            weights2 = model.get_layer("conv2d_1").get_weights()[0]
            weights1 = weights1.transpose(2, 3, 0, 1)
            weights2 = weights2.transpose(2, 3, 0, 1)

            display_images(block_outputs[0].transpose(0, 3, 1, 2), 99)
            display_images(block_outputs[1].transpose(0, 3, 1, 2), 100)
            display_images(weights1, 101)
            for i in range(weights2.shape[0]):
                weight = weights2[i:]
                display_images(weight, 102+i)

            # Print the shapes of the intermediate outputs
            for i, output in enumerate(block_outputs):
                print(f"Block {i+1} output shape:", output.shape)

            # 畳み込み第1層を可視化
            sampled_blocks = []
            block_labels = []
            kernel_size = 5

            for n in range(train_X.shape[0]):
                for i in range(train_X.shape[1] - (kernel_size - 1)):
                    for j in range(train_X.shape[2] - (kernel_size - 1)):
                        block = train_X[n, i:i+kernel_size, j:j+kernel_size].reshape(kernel_size, kernel_size)
                        sampled_blocks.append(block)
                        block_labels.append(train_labels[n])

            sampled_blocks = np.array(sampled_blocks)
            print("sampled_blocks shape:", np.shape(sampled_blocks))

            compressed_blocks = []
            conv_outputs = block_outputs[0]

            for n in range(conv_outputs.shape[0]):   
                for i in range(conv_outputs.shape[1]):
                    for j in range(conv_outputs.shape[2]):
                        compressed_blocks.append(conv_outputs[n, i, j])

            #selected_indices = np.random.choice(sampled_blocks.shape[0], 5000, replace=False)

            sampled_blocks, selected_indices = np.unique(np.array(sampled_blocks), axis=0, return_index=True)
            print("unique blocks shape:", np.shape(sampled_blocks))
            compressed_blocks = np.array(compressed_blocks)[selected_indices]
            block_labels = np.array(block_labels)[selected_indices]

            sampled_blocks = sampled_blocks.reshape(sampled_blocks.shape[0], 25)

            visualize_emb(compressed_blocks[:, 0:2], sampled_blocks, block_labels, f"LeNet_epoch{epoch}", 5, 1, 2)
            visualize_emb(compressed_blocks[:, 2:4], sampled_blocks, block_labels, f"LeNet_epoch{epoch}", 5, 3, 4)
            visualize_emb(compressed_blocks[:, 4:6], sampled_blocks, block_labels, f"LeNet_epoch{epoch}", 5, 1, 2)

if __name__ == '__main__':

    args = sys.argv
    num_train = int(args[1])
    num_test = int(args[2])
    datasets = args[3]
    
    main(num_train, num_test, datasets)
