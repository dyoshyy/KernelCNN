import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import backend
from tensorflow.keras import layers, models, losses
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10
from tensorflow.keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.svm import SVC
from sklearn import metrics
from functions import visualize_emb
from functions import display_images, display_weights, make_unique_filename
from functions import pad_images
import layers as my_layers

import sys

import functions
np.random.seed(0)
# Define a function to get intermediate outputs for the convolutional layers

def get_intermediate_output(model, layer_name, data):
    intermediate_layer_model = models.Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    return intermediate_layer_model.predict(data)


def main_LeNet(num_train: int, test_num : int, datasets : str, block_size=[5,5], display=True, layers_BOOL=[1,1,1,0]):
    backend.clear_session()
    print('Number of training samples:', num_train)
    #block_size = [7,3]
    stride = 1
    image_size = 32
    
    if (datasets == 'MNIST') or (datasets == 'FMNIST'):
        if datasets == 'MNIST':
            (train_X, train_Y), (test_X, test_Y) = mnist.load_data()
        if datasets == 'FMNIST':
            (train_X, train_Y), (test_X, test_Y) = fashion_mnist.load_data()

        train_X = train_X.reshape(-1, 28, 28, 1) 
        test_X = test_X.reshape(-1, 28, 28, 1)
        train_X = pad_images(train_X, image_size)
        test_X = pad_images(test_X, image_size)
        channel = 1
    elif datasets == 'CIFAR10':
        (train_X, train_Y), (test_X, test_Y) = cifar10.load_data()
        train_X = train_X.reshape(-1, 32, 32, 3) 
        test_X = test_X.reshape(-1, 32, 32, 3)
        train_X = pad_images(train_X, image_size)
        test_X = pad_images(test_X, image_size)
        channel = 3
    
    train_Y = to_categorical(train_Y, 10)
    test_Y = to_categorical(test_Y, 10)
    train_X = train_X[:num_train]/255
    train_Y = train_Y[:num_train]
    test_X = test_X[:test_num]/255
    test_Y = test_Y[:test_num]

    # LeNet-5 model definition
    activation = 'relu'
    #activation = 'tanh'
    model = models.Sequential()
    model.add(layers.Conv2D(30, kernel_size=(block_size[0], block_size[0]), activation=activation, strides= stride, input_shape=(image_size, image_size, channel)))
    if layers_BOOL[0]:
        model.add(layers.MaxPooling2D((2, 2)))
        #model.add(layers.Activation('sigmoid'))
    if layers_BOOL[1]:
        model.add(layers.Conv2D(60, kernel_size=(block_size[1], block_size[1]),activation=activation, strides= stride, padding='valid'))
    if layers_BOOL[2]:
        model.add(layers.MaxPooling2D((2, 2)))
        #model.add(layers.Activation('sigmoid'))
    if layers_BOOL[3]:
        model.add(layers.Conv2D(32, kernel_size=(block_size[1], block_size[1]),activation=activation, strides= stride, padding='valid'))
    model.add(layers.Flatten())
    model.add(layers.Dense(120, activation=activation))
    model.add(layers.Dense(84, activation=activation))
    model.add(layers.Dense(10, activation='softmax'))
    #model.summary()
    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    batch_size = 64
    epochs = 50
    es = EarlyStopping(monitor='val_loss', mode='auto', patience=5, verbose=0)
    cp = ModelCheckpoint("./weights/model_weights_epoch_{epoch:02d}.h5", save_weights_only=True, save_freq='epoch', period = 1)
    
    history = model.fit(train_X, train_Y, batch_size=batch_size, verbose=0, epochs=epochs, callbacks=[cp], validation_split=0.1)

    #SVMによる識別
    train_features = get_intermediate_output(model, 'max_pooling2d_1', train_X)

    # Train an SVM classifier
    '''
    svm = SVC(kernel='rbf', C=10.0, gamma='auto', probability=True, decision_function_shape='ovr')
    train_Y_1d = np.argmax(train_Y, axis=1)
    svm.fit(flattened_output, train_Y_1d)
    '''
    train_Y = np.argmax(train_Y, axis=1)
    #classifier = my_layers.SupportVectorsMachine()
    classifier = my_layers.GaussianProcess()
    classifier.fit(train_features, train_Y)

    # Use the trained SVM classifier for prediction
    test_features = get_intermediate_output(model, 'max_pooling2d_1', test_X)
    test_features = test_features.reshape(test_features.shape[0], -1)
    
    predictions = classifier.predict(test_features)
    accuracy = metrics.accuracy_score(np.argmax(test_Y, axis=1), predictions) * 100
    classification_report = metrics.classification_report(np.argmax(test_Y, axis=1), predictions)
    print(classification_report)
    print(f"Accuracy: {accuracy:.4f}")

    if display:
        #学習後のモデルの出力
        block_outputs = []
        block_outputs.append(get_intermediate_output(model, 'conv2d', train_X))
        if layers_BOOL[1]:
            block_outputs.append(get_intermediate_output(model, 'max_pooling2d', train_X))
            block_outputs.append(get_intermediate_output(model, 'conv2d_1', train_X))
            if layers_BOOL[3]:
                block_outputs.append(get_intermediate_output(model, 'max_pooling2d_1', train_X)) 
                block_outputs.append(get_intermediate_output(model, 'conv2d_2', train_X))

        weights1 = model.get_layer("conv2d").get_weights()[0]
        visualize_emb(train_X, train_Y, block_outputs[0], block_size=block_size[0], stride=stride, B=None, embedding_method='LeNet', dataset_name=datasets)
        display_images(block_outputs[0], 2, 'LeNet', datasets, f'LeNet Output layer 2 n={num_train}')
        display_weights(weights1, datasets, layer_idx=2)
        
        if layers_BOOL[1]:
            if block_outputs[1].shape[1] == block_outputs[2].shape[1]: #paddingしてるとき
                block_outputs[1]=(pad_images(block_outputs[1],18))

            weights2 = model.get_layer("conv2d_1").get_weights()[0]
            visualize_emb(block_outputs[1], train_Y, block_outputs[2], block_size=block_size[1], stride=stride, B=None, embedding_method='LeNet', dataset_name=datasets)
            display_images(block_outputs[2], 4, 'LeNet', datasets, f'LeNet Output layer 4 n={num_train}')
            display_weights(weights2, datasets, layer_idx=4)
            
            if layers_BOOL[3]:
                if block_outputs[3].shape[1] == block_outputs[4].shape[1]:
                    block_outputs[3]=(pad_images(block_outputs[3],11))
                weights3 = model.get_layer("conv2d_2").get_weights()[0]
                visualize_emb(block_outputs[3], train_Y, block_outputs[4], block_size=block_size[1], stride=stride, B=None, embedding_method='LeNet', dataset_name=datasets)
                display_images(block_outputs[4], 6, 'LeNet', datasets, f'LeNet Output layer 6 n={num_train}')
                display_weights(weights3, datasets, layer_idx=6)
    

    epochs_to_check = [1, 5, 10, 20, 30, 40, 50]
    
    # 学習途中のモデルの出力
    if False:
        for epoch in epochs_to_check:
            block_outputs=[]
            model.load_weights(f"./weights/model_weights_epoch_{epoch:02d}.h5")
            block_outputs.append(get_intermediate_output(model, 'conv2d', train_X))
            block_outputs.append(get_intermediate_output(model, 'max_pooling2d', train_X))
            block_outputs.append(get_intermediate_output(model, 'conv2d_1', train_X))
            weights1 = model.get_layer("conv2d").get_weights()[0]
            weights2 = model.get_layer("conv2d_1").get_weights()[0]

            display_images(block_outputs[0], 99, 'LeNet', datasets, suptitle=f'LeNet Output layer 2 epoch={epoch} n={num_train}')
            display_images(block_outputs[2], 100, 'LeNet', datasets, suptitle=f'LeNet Output layer 4 epoch={epoch} n={num_train}')
            #display_weights(weights1, datasets)

            # Print the shapes of the intermediate outputs
            for i, output in enumerate(block_outputs):
                print(f"Block {i+1} output shape:", output.shape)

            # 畳み込み第1層を可視化
            maxPool_1_out = np.array(block_outputs[1])
            visualize_emb(train_X, train_Y, block_outputs[0], block_size=block_size, stride=stride, B=None, embedding_method='LeNet', dataset_name=datasets)
            visualize_emb(maxPool_1_out, train_Y, block_outputs[2], block_size=block_size, stride=stride, B=None, embedding_method='LeNet', dataset_name=datasets)
    return accuracy

if __name__ == '__main__':

    args = sys.argv
    num_train = int(args[1])
    num_test = int(args[2])
    datasets = args[3]
    arguments = [num_train, num_test, datasets, None]
    #functions.calculate_average_accuracy(main, arguments, datasets, num_train, num_test, 10)
    main_LeNet(num_train, num_test, datasets)
