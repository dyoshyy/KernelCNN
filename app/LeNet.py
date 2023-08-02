import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import numpy as np

from embedding_analysis import visualize_data
from functions import binarize_mnist_data
from functions import display_images

# Load and preprocess the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_X = train_images.reshape(-1, 28, 28, 1) 
test_X = test_images.reshape(-1, 28, 28, 1)
train_Y = to_categorical(train_labels, 10)
test_Y = to_categorical(test_labels, 10)

train_num = 5000
test_num = 1000

train_X = binarize_mnist_data(train_X[:train_num])
train_Y = binarize_mnist_data(train_Y[:train_num])
test_X = test_X[:test_num]
test_Y = test_Y[:test_num]


# LeNet-5 model definition
model = models.Sequential([
    layers.Conv2D(6, kernel_size=(5, 5), activation='relu', input_shape=(28, 28, 1)),
    layers.AveragePooling2D(pool_size=(2, 2)),
    layers.Conv2D(16, kernel_size=(5, 5), activation='relu'),
    layers.AveragePooling2D(pool_size=(2, 2)),
    layers.Conv2D(120, kernel_size=(5, 5), activation='relu'),
    layers.Flatten(),
    layers.Dense(84, activation='relu'),
    layers.Dense(10, activation='softmax')
])

#model.summary()

# Create a list to store intermediate outputs
block_outputs = []

# Define a function to get intermediate outputs for the convolutional layers

def get_intermediate_output(model, layer_name, data):
    intermediate_layer_model = models.Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    return intermediate_layer_model.predict(data)

# Compile the model
model.compile(optimizer='sgd', loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
batch_size = 64
epochs = 30
model.fit(train_X, train_Y, batch_size=batch_size,epochs=epochs, validation_split=0.1)

# Get intermediate outputs for the convolutional layers and save them in the list
block_outputs.append(get_intermediate_output(model, 'conv2d', train_X))

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

principal_data = compressed_blocks[:, 0:2]

visualize_data(principal_data, sampled_blocks, block_labels, "LeNet", "LeNet", kernel_size)




        
