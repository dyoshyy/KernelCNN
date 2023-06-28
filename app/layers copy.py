import numpy as np
import GPy
from sklearn.manifold import SpectralEmbedding
from functions import calculate_similarity
from functions import display_images
from functions import binarize_images

class KIMLayer:
    def __init__(self, block_size, channels_next, stride, num_samples=100):
        self.b = block_size
        self.stride = stride
        self.C_next = channels_next
        self.C_prev = None
        self.H = None
        self.W = None
        self.num_samples = num_samples
        self.GP = None

    def learn_embedding(self, train_data): #埋め込みをGPで学習

        input_dim = self.b * self.b * self.C_prev
        n_train = train_data.shape[0]

        train_data = binarize_images(train_data)

        if self.GP is None:
            print('[KIM] Sampling blocks...')
            sampled_blocks = []
            for n in range(n_train):
                data = train_data[n,:,:,:] #画像を一枚持ってくる
                for i in range(self.num_samples): # 画像からランダムな位置でブロックをサンプリング
                    start_i = np.random.randint(0, self.H - self.b + 1)
                    start_j = np.random.randint(0, self.W - self.b + 1)
                    sampled_blocks.append(data[:, start_i : start_i + self.b, start_j : start_j + self.b])
            print('[KIM] Completed')
            sampled_blocks = np.array(sampled_blocks).reshape(self.num_samples * n_train, input_dim)
            print('samples shape:',np.shape(sampled_blocks))
            sampled_blocks = np.unique(sampled_blocks, axis=0) #重複を削除
            print('unique samples shape:',np.shape(sampled_blocks))

            # ラプラシアン固有マップによる次元削減
            LE = SpectralEmbedding(n_components=self.C_next)
            embedded_blocks = LE.fit_transform(sampled_blocks)

            #ガウス過程回帰で学習
            print('[KIM] Fitting samples...')
            kernel = GPy.kern.RBF(input_dim = input_dim)
            self.GP = GPy.models.GPRegression(sampled_blocks, embedded_blocks, kernel=kernel)
            self.GP.optimize()
            print('[KIM] Completed')
        else:
            print('[KIM] GPmodel found')

    def convert_image(self, input_data):
        num_inputs = input_data.shape[0]
        b_radius = int((self.b-1)/2)
        output_data = np.zeros((num_inputs, self.C_next, int((self.H-self.b+1)/self.stride), int((self.W-self.b+1)/self.stride)))

        for n in range(num_inputs):
            print('[KIM] Converting the image %d' % (n+1))
            for i in range(b_radius, self.H-b_radius, self.stride):
                i_output = int((i - b_radius)/self.stride)
                for j in range(b_radius, self.W-b_radius, self.stride):
                    j_output = int((j - b_radius)/self.stride)
                    input_cropped = input_data[n, :, (i-b_radius):(i+b_radius+1), (j-b_radius):(j+b_radius+1)].reshape(1, self.b * self.b * self.C_prev)
                    output_data[n, :, i_output,j_output], _ = self.GP.predict(input_cropped)

        return output_data

    def calculate(self, input_data):

        self.C_prev = input_data.shape[1]
        self.H = input_data.shape[2]
        self.W = input_data.shape[3]

        train_data = input_data[:100] #先頭から１００枚の画像を埋め込みの学習に使う
        self.learn_embedding(train_data) 

        output_data = self.convert_image(input_data) 

        return output_data

class MaxPoolingLayer:
    def __init__(self, pool_size):
        self.pool_size = pool_size

    def calculate(self, input_data):
        N, C, H, W = input_data.shape
        p = self.pool_size
        H_out = H // p
        W_out = W // p
        output_data = np.zeros((N, C, H_out, W_out))
        for n in range(N):
            print('[MaxPooling] Converting the image %d' % (n+1))
            for c in range(C):
                for i in range(H_out):
                    for j in range(W_out):
                        start_i = i * p
                        start_j = j * p
                        output_data[n, c, i, j] = np.max(np.abs(input_data[n, c, start_i:start_i + p, start_j:start_j + p]))

        return output_data

class AvgPoolingLayer:
    def __init__(self, pool_size):
        self.pool_size = pool_size

    def calculate(self, input_data):
        N, C, H, W = input_data.shape
        p = self.pool_size
        H_out = H // p
        W_out = W // p
        output_data = np.zeros((N, C, H_out, W_out))
        for n in range(N):
            print('[AvgPooling] Converting the image %d' % (n+1))
            for c in range(C):
                for i in range(H_out):
                    for j in range(W_out):
                        start_i = i * p
                        start_j = j * p
                        output_data[n, c, i, j] = np.mean(input_data[n, c, start_i:start_i + p, start_j:start_j + p])

        return output_data

class KernelLearningLayer:
    def __init__(self):
        self.GP = None

    def fit(self, X, Y):
        input_dim = X.shape[1] * X.shape[2] * X.shape[3]
        train_X = X.reshape(X.shape[0], input_dim) #全部のデータを学習に使う
        train_Y = Y

        if self.GP is None:
            print('Learning labels')
            kernel = GPy.kern.RBF(input_dim = input_dim)
            self.GP = GPy.models.GPRegression(train_X, train_Y, kernel=kernel)
            self.GP.optimize()
            print('Completed')
        else:
            print('GPmodel found')

    def predict(self, X):
        input_dim = X.shape[1] * X.shape[2] * X.shape[3]
        #ベクトル化し予測
        X = X.reshape(X.shape[0], input_dim)
        Y_predicted, _ = self.GP.predict(X)
        Y_predicted = np.array(Y_predicted)

        output = [np.argmax(Y_predicted[n,:]) for n in range(X.shape[0])]
        return output

class Model:
    
    def __init__(self, display):
        self.layers = []
        self.display = display
        self.shapes = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def fit(self, X, Y):
        for layer in self.layers:
            self.shapes.append(np.shape(X)[1:])
            X = layer.calculate(X)
            if self.display:
                display_images(X)

        self.shapes.append(np.shape(X)[1:])
        if not isinstance(self.layers[-1], KernelLearningLayer):
            self.layers.append(KernelLearningLayer())
        self.layers[-1].fit(X, Y)

    def predict(self, test_X, test_Y):
        for layer in self.layers:
            if isinstance(layer, KernelLearningLayer):
                break
            test_X = layer.calculate(test_X)

        Y_predicted = self.layers[-1].predict(test_X)
        Y_answer= [np.argmax(test_Y[n,:]) for n in range(test_Y.shape[0])]

        print('Accuracy:', calculate_similarity(Y_predicted, Y_answer))
        print('Layers shape:',self.shapes)
        return Y_predicted, Y_answer

