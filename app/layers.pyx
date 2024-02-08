# distutils: language=c++
# cython: language_level=3
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
# distutils: extra_compile_args = ["-O3"]
# cython: language_level=3, boundscheck=False, wraparound=False
# cython: cdivision=True

import GPy
import cython
import numpy as np
cimport numpy as cnp
from cython import boundscheck, wraparound 
import math
import time
from keras import layers, models    
from keras.callbacks import EarlyStopping
from functions import calculate_similarity, display_images, binarize_images, visualize_emb, visualize_emb_dots, select_embedding_method, pad_images

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy import stats

from skimage import util
from tqdm import tqdm

np.set_printoptions(precision=3, threshold=10000, linewidth=200, edgeitems=10)
ctypedef cnp.float64_t DTYPE_t 

class KIMLayer:
    def __init__(self, block_size : int, channels_next : int, stride : int, padding : bool, emb : str, num_blocks : int):
        cdef int b = block_size
        cdef int b_radius = int((b - 1) / 2)
        cdef int stride = stride
        cdef int C_next = channels_next
        cdef int C_prev = None
        cdef int H = None
        cdef int W = None
        cdef object output_data = None
        cdef object input_data = None
        cdef str embedding = emb
        cdef object GP = None
        cdef int B = num_blocks
        cdef str dataset_name = None
        cdef bint padding = padding

    def sample_and_embed_blocks(self, n_train: int, train_X: np.ndarray, train_Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        '''
        Args:
            n_train (int): 画像の枚数
            train_X (ndarray): 学習する画像データ
            train_Y (ndarray): 画像のラベル(NOT One-hot vector)
        
        Returns:
            ndarray: サンプリングされたブロックの配列
            ndarray: サンプリングされたブロックの埋め込み
        '''
        cdef np.ndarray[np.float64_t, ndim=4] sampled_blocks = np.empty((n_train*(self.H-self.b+1)**2, self.b, self.b, self.C_prev))
        print('sampling...')
        cdef int n
        cdef cnp.ndarray[DTYPE_t, ndim=3] data
        cdef cnp.ndarray[DTYPE_t, ndim=4] blocks
        for n in range(n_train):
            # 一枚持ってくる
            data = train_X[n,:,:,:]
            # すべてのブロックをサンプリング
            blocks = util.view_as_windows(data, (self.b, self.b, self.C_prev), self.stride).reshape((self.H-self.b+1)**2, self.b, self.b, self.C_prev)
            sampled_blocks[(n)*(self.H-self.b+1)**2 : (n+1)*(self.H-self.b+1)**2 ] = blocks
        
        train_Y = train_Y[:n_train]
        sampled_blocks = sampled_blocks.reshape(-1, self.b, self.b, self.C_prev)
        sampled_blocks_label = np.repeat(np.argmax(train_Y, axis=1), int(sampled_blocks.shape[0]/train_Y.shape[0]))
        print('sampling completed')
        print('All samples shape:',np.shape(sampled_blocks))
        
        #画像を二値化
        binarized_sampled_blocks = binarize_images(sampled_blocks)
        binarized_sampled_blocks = binarized_sampled_blocks.reshape(sampled_blocks.shape[0], self.b * self.b * self.C_prev)
        
        #重複を削除し，そのインデックスのブロックのみを使う
        _, unique_index= np.unique(binarized_sampled_blocks, axis=0, return_index=True)
        sampled_blocks = sampled_blocks.reshape(sampled_blocks.shape[0], self.b * self.b * self.C_prev)[unique_index]
        sampled_blocks_label = sampled_blocks_label[unique_index]
        print('unique samples shape:', np.shape(sampled_blocks))
        
        #サンプル数が5000を超える場合は5000にする
        embedding_samples_threshold = 5000
        if sampled_blocks.shape[0] > embedding_samples_threshold:
            selected_indices = np.random.choice(sampled_blocks.shape[0], embedding_samples_threshold, replace=False)
            sampled_blocks = sampled_blocks[selected_indices]
            sampled_blocks_label = sampled_blocks_label[selected_indices]
            
        #埋め込み
        print('embedding...')
        embedded_blocks = select_embedding_method(self.embedding, self.C_next, sampled_blocks, sampled_blocks_label)
        print('embedding completed')
        
        #B個だけランダムに取り出す
        self.B = min(sampled_blocks.shape[0], self.B)  #Bより少ないサンプル数の場合はそのまま
        selected_indices = np.random.choice(embedded_blocks.shape[0], self.B, replace=False)
        sampled_blocks = sampled_blocks[selected_indices]
        embedded_blocks = embedded_blocks[selected_indices]
        
        return sampled_blocks, embedded_blocks
            
    def learn_embedding(self, train_X, train_Y): 
        '''
        埋め込みをKIMで学習
            train_X: 学習に使うX
            train_Y: Xのラベルデータ
        '''
        n_train = 300 #埋め込みを学習するサンプル数

        if self.GP is None:
            sampled_blocks, embedded_blocks = self.sample_and_embed_blocks(n_train, train_X, train_Y)
            kernel = GPy.kern.RBF(input_dim = self.b * self.b * self.C_prev) + GPy.kern.Bias(input_dim = self.b * self.b * self.C_prev) + GPy.kern.Linear(input_dim = self.b * self.b * self.C_prev)
            self.GP = GPy.models.GPRegression(sampled_blocks, embedded_blocks, kernel=kernel)
            print('optimizing KIM parameters...')
            
            self.GP.optimize()
            print('model summary:', self.GP)
            print('optimizing completed')
        else:
            print('[KIM] GPmodel found')
        
    def convert_image_batch(self, batch_size: int = 10):
        """
        Converts the input image batch into a batch of predictions.

        Args:
            batch_size (int, optional): The size of each batch. Defaults to 10.
        """
        cdef int num_batches = math.ceil(self.input_data.shape[0] / batch_size)
        cdef int batch_index
        cdef tuple batch_images
        cdef tuple blocks_to_convert
        cdef tuple predictions

        for batch_index in tqdm(range(num_batches)):
            batch_images = self.input_data[batch_size * batch_index: batch_size * (batch_index + 1)]
            blocks_to_convert = util.view_as_windows(batch_images, (1, self.b, self.b, self.C_prev), self.stride)
            blocks_to_convert = blocks_to_convert.reshape(batch_size * (self.H - self.b + 1) ** 2, self.b * self.b * self.C_prev)  # ex) (10*784, 5*5*1)
            predictions, _ = self.GP.predict(blocks_to_convert)  # shape: (10*784, 6)
            predictions = predictions.reshape(batch_size, self.H - self.b + 1, self.H - self.b + 1, self.C_next)
            self.output_data[batch_size * batch_index: batch_size * (batch_index + 1)] = predictions

    def calculate(self, input_X, input_Y):
        """
        このメソッドは、入力データに対してKIM (Kernelized Input Mapping) を適用し、
        結果を出力データとして保存します。

        Parameters:
        input_X (numpy.ndarray): 入力データ。形状は (num_inputs, H, W, C_prev) で、
                                num_inputs は入力の数、H と W はそれぞれ高さと幅、
                                C_prev は前の層のチャネル数を表します。
        input_Y (numpy.ndarray): 入力データに対応するラベル。

        Returns:
        numpy.ndarray: KIM を適用した後の出力データ。形状は (num_inputs, (H-b+1)/stride, (W-b+1)/stride, C_next) です。
        """
        
        #インスタンス変数に格納
        num_inputs = input_X.shape[0]
        self.H = input_X.shape[1]
        self.W = input_X.shape[2]
        self.C_prev = input_X.shape[3]
        self.input_data = input_X
        self.output_data = np.zeros((num_inputs, int((self.H-self.b+1)/self.stride), int((self.W-self.b+1)/self.stride), self.C_next))
        
        #KIMで埋め込みを学習
        self.learn_embedding(input_X, input_Y) 
        
        #学習したKIMで変換
        print('[KIM] Converting the image...')
        self.convert_image_batch(batch_size=100)
        print('completed')
        #ReLU
        #self.output_data = np.maximum(0, self.output_data)
        #Tanh
        #self.output_data = np.tanh(self.output_data)
        return self.output_data
    
class TanhLayer:
    def calculate(self, input_data):
        """
        Applies the hyperbolic tangent (tanh) activation function to the input data.

        Args:
            input_data (numpy.ndarray): The input data.

        Returns:
            numpy.ndarray: The output data after applying the tanh activation function.
        """
        return np.tanh(input_data)
    
class SigmoidLayer:
    def calculate(self, input_data):
        """
        Applies the sigmoid activation function to the input data.

        Args:
            input_data (numpy.ndarray): The input data.

        Returns:
            numpy.ndarray: The output data after applying the sigmoid activation function.
        """
        return 1 / (1 + np.exp(-input_data))


class AvgPoolingLayer:
    def __init__(self, pool_size):
        self.pool_size = pool_size

    def calculate(self, input_data):
        print('[AVG] Converting')
        N, H, W, C = input_data.shape
        p = self.pool_size
        H_out = H // p
        W_out = W // p

        # Initialize output data
        output_data = np.zeros((N, H_out, W_out, C))

        # Perform average pooling
        for i in range(H_out):
            for j in range(W_out):
                # Extract pooling window
                window = input_data[:, i*p:(i+1)*p, j*p:(j+1)*p, :]
                # Calculate mean value
                output_data[:, i, j, :] = np.mean(window, axis=(1, 2))
        
        #sigmoid
        #output_data = 1 / (1 + np.exp(-output_data))
        print('[AVG] Completed')
        return output_data

class MaxPoolingLayer:
    def __init__(self, pool_size):
        self.pool_size = pool_size

    def calculate(self, input_data):
        print('[MAX] Converting')
        N, H, W, C = input_data.shape
        p = self.pool_size
        H_out = H // p
        W_out = W // p

        # Initialize output data
        output_data = np.zeros((N, H_out, W_out, C))

        # Perform max pooling
        for i in range(H_out):
            for j in range(W_out):
                # Extract pooling window
                window = input_data[:, i*p:(i+1)*p, j*p:(j+1)*p, :]
                # Calculate max value
                output_data[:, i, j, :] = np.max(window, axis=(1, 2))
        
        print('[MAX] Completed')
        return output_data
    
class LabelLearningLayer_NeuralNetwork:
    def __init__(self):
        self.model = None

    def fit(self, X, Y):
        input_dim = X.shape[1] * X.shape[2] * X.shape[3]
        #ベクトル化し学習
        X = X.reshape(X.shape[0], input_dim)
        X = StandardScaler().fit_transform(X)
        if self.model is None:
            print('Learning labels')
            self.model = models.Sequential([
                layers.Dense(120, activation='tanh'),
                layers.Dense(84, activation='tanh'),
                layers.Dense(10, activation='softmax')
            ])
            self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            batch_size = 64
            epochs = 100
            es = EarlyStopping(monitor='val_loss', mode='auto', patience=30, verbose=0)
            self.model.fit(X, Y, batch_size=batch_size, verbose=0, epochs=epochs, callbacks=[es], validation_split=0.2)
            print('Completed')
        else:
            print('GPmodel found')

    def predict(self, X):
        #ベクトル化し予測
        X = X.reshape(X.shape[0], X.shape[1] * X.shape[2] * X.shape[3])
        X = StandardScaler().fit_transform(X)
        Y_predicted = self.model.predict(X)
        output = [np.argmax(Y_predicted[n,:]) for n in range(X.shape[0])]
        return output

class LabelLearningLayer_GaussianProcess:
    def __init__(self):
        self.GP = None
        self.num_GP = None
        self.OVER_threshold = False
        self.threshold = 10000

    def fit(self, X, Y):
        input_dim = X.shape[1] * X.shape[2] * X.shape[3]
        #ベクトル化し学習
        X = X.reshape(X.shape[0], input_dim)
        X = StandardScaler().fit_transform(X)
        if self.GP is None:
            print('Learning labels')

            # 訓練サンプルが10000超える場合はthresholdずつに分けて学習
            
            if X.shape[0] > self.threshold:
                self.OVER_threshold = True
                self.GP = []
                #必要なGPの数
                self.num_GP = int((X.shape[0]-1)/self.threshold + 1)
                kernel = GPy.kern.RBF(input_dim = input_dim) + GPy.kern.Bias(input_dim = input_dim) + GPy.kern.Linear(input_dim = input_dim)
                for i in range(self.num_GP):
                    print('learning {}'.format(i+1))
                    X_sep = X[self.threshold*i:self.threshold*(i+1)]
                    Y_sep = Y[self.threshold*i:self.threshold*(i+1)]
                    self.GP.append(GPy.models.GPRegression(X_sep,Y_sep, kernel=kernel))
                    self.GP[-1].optimize()
            else:
                kernel = GPy.kern.RBF(input_dim = input_dim) + GPy.kern.Bias(input_dim = input_dim) + GPy.kern.Linear(input_dim = input_dim)
                self.GP = GPy.models.GPRegression(X, Y, kernel=kernel)
                self.GP.optimize()
                print('Completed')
        else:
            print('GPmodel found')

    def predict(self, X):
        #ベクトル化し予測
        X = X.reshape(X.shape[0], X.shape[1] * X.shape[2] * X.shape[3])
        X = StandardScaler().fit_transform(X)
        if self.OVER_threshold:
            predictions = []
            for i in range(self.num_GP):
                Y_predicted, _ = self.GP[i].predict(X)
                Y_predicted = np.array(Y_predicted)
                predict = [np.argmax(Y_predicted[n,:]) for n in range(X.shape[0])]
                predictions.append(predict)
            ensemble_predictions = np.vstack(predictions)
            output = stats.mode(ensemble_predictions, axis=0).mode.ravel()
            
        else:
            Y_predicted, _ = self.GP.predict(X)
            Y_predicted = np.array(Y_predicted)
            output = [np.argmax(Y_predicted[n,:]) for n in range(X.shape[0])]
        return output

class Model:
    def __init__(self, display):
        self.layers = []
        self.display = display
        self.shapes = []
        self.time_fitting = 0
        self.time_predicting = 0
        self.data_set_name = ""
    
    def add_layer(self, layer):
        self.layers.append(layer)

    def fit(self, X, Y):
        start_time = time.time()
        self.num_train = X.shape[0]
        
        for n, layer in enumerate(self.layers):
            self.shapes.append(np.shape(X)[1:])
            layer.dataset_name = self.data_set_name
            if isinstance(layer, KIMLayer):
                if layer.padding:
                    out_size = X.shape[1] + layer.b - 1 # 28 + 5 - 1
                    X = pad_images(X, out_size)
                X_temp = X
                X = layer.calculate(X, Y)
                #中間層の出力を可視化
                if self.display:
                    visualize_emb(X_temp, Y, X, layer.b, layer.stride, layer.B, layer.embedding, self.data_set_name)
                    display_images(X, n+2, layer.embedding, self.data_set_name, f'KernelCNN train output Layer{n+2} (b={layer.b}, B={layer.B}, Embedding:{layer.embedding})')
            elif isinstance(layer, LabelLearningLayer_GaussianProcess) or isinstance(layer, LabelLearningLayer_NeuralNetwork): #最後の層のとき
                layer.fit(X, Y)
            else: #プーリング層
                X = layer.calculate(X)
        
        self.time_fitting = time.time() - start_time

    def predict(self, test_X, test_Y):
        start_time = time.time()
        self.num_test = test_X.shape[0]
        for n,layer in enumerate(self.layers):
            if isinstance(layer, KIMLayer):
                if layer.padding:
                    out_size = test_X.shape[1] + layer.b - 1 # 28 + 5 - 1
                    test_X = pad_images(test_X, out_size)
                test_X = layer.calculate(test_X, None)
                if self.display:
                    continue
                    #display_images(test_X, n+7, layer.embedding, self.data_set_name, f'KernelCNN test output Layer{n+2} (b={layer.b}, B={layer.B}, Embedding:{layer.embedding})')
            elif isinstance(layer, LabelLearningLayer_GaussianProcess) or isinstance(layer, LabelLearningLayer_NeuralNetwork):
                Y_predicted = self.layers[-1].predict(test_X)
                Y_answer= [np.argmax(test_Y[n,:]) for n in range(test_Y.shape[0])]
            else:
                test_X = layer.calculate(test_X)

        self.time_predicting = time.time() - start_time
        accuracy = calculate_similarity(Y_predicted, Y_answer)*100 #%単位
        
        print('Layers shape:',self.shapes)
        print('Fitting time:', self.time_fitting)
        print('Predicting time:', self.time_predicting)
        print('Accuracy:', accuracy)
        
        # パラメータをテキストファイルに保存
        with open('model_parameters.txt', 'a') as param_file:
            param_file.write(f'Datasets: {self.data_set_name}\n')
            param_file.write('================================================================================\n')
            for i, layer in enumerate(self.layers):
                if isinstance(layer, LabelLearningLayer_GaussianProcess) or isinstance(layer, LabelLearningLayer_NeuralNetwork):
                    continue  
                if isinstance(layer, KIMLayer):
                    param_file.write(f'Layer {i+2}\n')
                    param_file.write(f'Embedding method: {layer.embedding}\n')
                    param_file.write(f'block size: {layer.b}\n')
                    param_file.write(f'stride: {layer.stride}\n')
                    param_file.write(f'B: {layer.B}\n')
                    param_file.write('-------------------------------\n')

            # 正解率を保存
            param_file.write(f'Train samples: {self.num_train}\n')
            param_file.write(f'Test samples: {self.num_test}\n')
            param_file.write(f'Layer shape: {self.shapes}\n')
            param_file.write(f'Fitting time: {self.time_fitting} seconds\n')
            param_file.write(f'Predicting time: {self.time_predicting} seconds\n')
            param_file.write(f'Accuracy: {accuracy} %\n')
            param_file.write('================================================================================\n')
            
        return accuracy

