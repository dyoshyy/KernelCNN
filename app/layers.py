import GPy
import numpy as np
import math
import os, sys, time

from functions import calculate_similarity, display_images, binarize_images, binarize_2d_array, visualize_emb, select_embedding_method

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from scipy import stats

from skimage import util
from tqdm import tqdm

np.random.seed(1)
np.set_printoptions(precision=3, threshold=10000, linewidth=200, edgeitems=10)

class KIMLayer:
    def __init__(self, block_size : int, channels_next : int, stride : int, emb : str, num_blocks : int):
        self.b = block_size
        self.b_radius = int((self.b - 1) / 2)
        self.stride = stride
        self.C_next = channels_next
        self.C_prev = None
        self.H = None
        self.W = None
        self.output_data = None
        self.input_data = None
        self.embedding = emb
        self.GP = None
        self.B = num_blocks

    def sample_block(self, n_train, train_X, train_Y):
        '''
        画像データからブロックをサンプリング
            n_train : 画像の枚数
            train_X : 学習する画像データ
            train_Y : 画像のラベル(NOT One-hot vector)
        '''
        sampled_blocks = np.empty((n_train*(self.H-self.b+1)**2, self.b, self.b, self.C_prev))
        sampled_blocks_label = []
        train_Y = np.argmax(train_Y, axis=1)
        for n in range(n_train):
            # 一枚持ってくる
            data = train_X[n,:,:,:]
            # すべてのブロックをサンプリング
            blocks = util.view_as_windows(data, (self.b, self.b, self.C_prev), self.stride).reshape((self.H-self.b+1)**2, self.b, self.b, self.C_prev)
            sampled_blocks[(n)*(self.H-self.b+1)**2 : (n+1)*(self.H-self.b+1)**2 ] = blocks
            sampled_blocks_label.extend([train_Y[n]] * blocks.shape[0])
        sampled_blocks = sampled_blocks.reshape(-1, self.b, self.b, self.C_prev)
            
        #画像を二値化
        sampled_blocks = binarize_images(sampled_blocks)
        sampled_blocks = sampled_blocks.reshape(sampled_blocks.shape[0], self.b * self.b * self.C_prev)
        print('samples shape:',np.shape(sampled_blocks))
        
        #重複を削除
        sampled_blocks, indices= np.unique(sampled_blocks, axis=0, return_index=True) 
        sampled_blocks_label = np.array(sampled_blocks_label)[indices]

        print('unique samples shape:',np.shape(sampled_blocks))
        
        return sampled_blocks, sampled_blocks_label

    def learn_embedding(self, train_X, train_Y): 
        '''
        埋め込みをKIMで学習
            train_X: 学習に使うX
            train_Y: Xのラベルデータ
        '''
        n_train = train_X.shape[0]

        if self.GP is None:
            sampled_blocks, sampled_blocks_label = self.sample_block(n_train, train_X, train_Y)
            embedded_blocks = select_embedding_method(self.embedding, self.C_next, sampled_blocks)

            #B個のブロックだけランダムに取り出す
            selected_indices = np.random.choice(sampled_blocks.shape[0], self.B, replace=False)
            sampled_blocks = sampled_blocks[selected_indices]
            embedded_blocks = embedded_blocks[selected_indices]

            #ガウス過程回帰で学習
            print('[KIM] Fitting samples...')
            #埋め込みデータを正規化,標準化
            ms = MinMaxScaler()
            ss = StandardScaler()
            rs = RobustScaler()
            ms.fit(embedded_blocks)
            embedded_blocks = ms.transform(embedded_blocks)
            embedded_blocks = ss.fit_transform(embedded_blocks)
            
            # 埋め込みデータを可視化
            principal_data_12 = embedded_blocks[:,0:2]
            visualize_emb(principal_data_12, sampled_blocks, sampled_blocks_label[selected_indices], self.embedding, self.b, 1, 2)

            print("Training sample shape:", np.shape(embedded_blocks))
            
            print('[KIM] Training KIM')
            kernel = GPy.kern.RBF(input_dim = self.b * self.b * self.C_prev) + GPy.kern.Bias(input_dim = self.b * self.b * self.C_prev) + GPy.kern.Linear(input_dim = self.b * self.b * self.C_prev)
            self.GP = GPy.models.GPRegression(sampled_blocks, embedded_blocks, kernel=kernel)
            self.GP.optimize()
            print('[KIM] Completed')
            
        else:
            print('[KIM] GPmodel found')
        
    def convert_image_batch(self, batch_size=10):
        '''
        学習済みのKIMで元の画像を変換
        '''
        num_batches = math.ceil(self.input_data.shape[0]/batch_size)

        for batch_index in tqdm(range(num_batches)):
            batch_images = self.input_data[batch_size * batch_index : batch_size * (batch_index + 1)]
            blocks_to_convert = util.view_as_windows(batch_images, (1, self.b, self.b, self.C_prev), self.stride)
            blocks_to_convert = blocks_to_convert.reshape(batch_size * (self.H-self.b+1)**2, self.b * self.b * self.C_prev) # ex) (10*784, 5*5*1)        
            predictions, _ = self.GP.predict(blocks_to_convert) # shape: (10*784, 6)
            predictions = predictions.reshape(batch_size, self.H-self.b+1, self.H-self.b+1, self.C_next)
            self.output_data[batch_size * batch_index : batch_size * (batch_index + 1)] = predictions

    def calculate(self, input_X, input_Y):
        
        num_inputs = input_X.shape[0]
        self.H = input_X.shape[1]
        self.W = input_X.shape[2]
        self.C_prev = input_X.shape[3]
        self.input_data = input_X
        self.output_data = np.zeros((num_inputs, int((self.H-self.b+1)/self.stride), int((self.W-self.b+1)/self.stride), self.C_next))
        
        #先頭からtrain_numの画像を埋め込みの学習に使う
        train_num = 50
        train_X = input_X[:train_num] 
        train_Y = input_Y[:train_num]
        
        self.learn_embedding(train_X, train_Y) 
        
        print('[KIM] Converting the image...')
        self.convert_image_batch(batch_size=100)
        print('completed')

        return self.output_data

class AvgPoolingLayer:
    def __init__(self, pool_size):
        self.pool_size = pool_size

    def calculate(self, input_data, Y):
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
        
        print('[AVG] Completed')
        return output_data

class LabelLearningLayer:
    def __init__(self):
        self.GP = None
        self.num_GP = None
        self.OVER_10000 = False

    def fit(self, X, Y):
        input_dim = X.shape[1] * X.shape[2] * X.shape[3]
        #ベクトル化し学習
        X = X.reshape(X.shape[0], input_dim)
        if self.GP is None:
            print('Learning labels')

            # 訓練サンプルが10000超える場合は10000ずつに分けて学習
            if X.shape[0] > 10000:
                self.OVER_10000 = True
                self.GP = []
                #必要なGPの数
                self.num_GP = int((X.shape[0]-1)/10000 + 1)
                kernel = GPy.kern.RBF(input_dim = input_dim) + GPy.kern.Bias(input_dim = input_dim) + GPy.kern.Linear(input_dim = input_dim)
                for i in range(self.num_GP):
                    print('learning {}'.format(i+1))
                    X_sep = X[10000*i:10000*(i+1)]
                    Y_sep = Y[10000*i:10000*(i+1)]
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
        if self.OVER_10000:
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
            X = layer.calculate(X, Y)
            if self.display:
                display_images(X, n+1)

        self.shapes.append(np.shape(X)[1:])
        if not isinstance(self.layers[-1], LabelLearningLayer):
            self.layers.append(LabelLearningLayer())
            
        self.layers[-1].fit(X, Y)
        self.time_fitting = time.time() - start_time

    def predict(self, test_X, test_Y):
        start_time = time.time()
        self.num_test = test_X.shape[0]
        for n,layer in enumerate(self.layers):
            if isinstance(layer, LabelLearningLayer):
                break
            test_X = layer.calculate(test_X, test_Y)
            if self.display:
                display_images(test_X, n+5)

        Y_predicted = self.layers[-1].predict(test_X)
        Y_answer= [np.argmax(test_Y[n,:]) for n in range(test_Y.shape[0])]

        self.time_predicting = time.time() - start_time
        
        accuracy = calculate_similarity(Y_predicted, Y_answer)
        
        print('Layers shape:',self.shapes)
        print('Fitting time:', self.time_fitting)
        print('Predicting time:', self.time_predicting)
        print('Accuracy:', accuracy)
        
        # パラメータをテキストファイルに保存
        with open('model_parameters.txt', 'a') as param_file:
            param_file.write(f'Datasets: {self.data_set_name}\n')
            param_file.write('================================================================================\n')
            for i, layer in enumerate(self.layers):
                if isinstance(layer, LabelLearningLayer):
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
            param_file.write(f'Fitting time: {self.time_fitting}\n')
            param_file.write(f'Predicting time: {self.time_predicting}\n')
            param_file.write(f'Accuracy: {accuracy}\n')
            param_file.write('================================================================================\n')
            
        return Y_predicted, Y_answer

