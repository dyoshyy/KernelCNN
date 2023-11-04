import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, datasets, optimizers
from functions import display_weights, display_images

def get_intermediate_output(model, layer_name, data):
    intermediate_layer_model = models.Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    return intermediate_layer_model.predict(data)

# MNISTデータセットのロード
dataset_name = 'CIFAR10'
(train_X, train_Y), (_, _) = datasets.cifar10.load_data()
train_X = train_X.reshape((50000, 32, 32, 3)).astype('float32') / 255

# LeNetモデルの定義
model = models.Sequential()
model.add(layers.Conv2D(6, (5, 5), activation='relu', padding='valid', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(16, (5, 5), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(120, activation='relu'))
model.add(layers.Dense(84, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

# モデルのコンパイル
model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 学習
model.fit(train_X, train_Y, epochs=50, batch_size=256)

# 重みの可視化 (最初の畳み込み層の重みを可視化)
weights = model.layers[0].get_weights()[0]
display_weights(weights, dataset_name)

block_outputs = []
block_outputs.append(get_intermediate_output(model, 'conv2d', train_X))
block_outputs.append(get_intermediate_output(model, 'conv2d_1', train_X))
display_images(block_outputs[0], 99, 'LeNet', dataset_name)
display_images(block_outputs[1], 100, 'LeNet', dataset_name)