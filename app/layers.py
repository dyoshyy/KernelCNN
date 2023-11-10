import gpytorch_models
import gpytorch, torch
import numpy as np
import math
import time

from functions import calculate_similarity, display_images, binarize_images, visualize_emb, select_embedding_method, pad_images

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy import stats

from skimage import util
from tqdm import tqdm

np.random.seed(1)
np.set_printoptions(precision=3, threshold=10000, linewidth=200, edgeitems=10)

class KIMLayer:
    def __init__(self, block_size : int, channels_next : int, stride : int, padding : bool, emb : str, num_blocks : int):
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
        self.likelihood = None
        self.B = num_blocks
        self.dataset_name = None
        self.padding = padding

    def sample_block(self, n_train, train_X):
        '''
        画像データからブロックをサンプリング
            n_train : 画像の枚数
            train_X : 学習する画像データ
            train_Y : 画像のラベル(NOT One-hot vector)
        '''
        sampled_blocks = np.empty((n_train*(self.H-self.b+1)**2, self.b, self.b, self.C_prev))
    
        for n in range(n_train):
            # 一枚持ってくる
            data = train_X[n,:,:,:]
            # すべてのブロックをサンプリング
            blocks = util.view_as_windows(data, (self.b, self.b, self.C_prev), self.stride).reshape((self.H-self.b+1)**2, self.b, self.b, self.C_prev)
            sampled_blocks[(n)*(self.H-self.b+1)**2 : (n+1)*(self.H-self.b+1)**2 ] = blocks
        
        sampled_blocks = sampled_blocks.reshape(-1, self.b, self.b, self.C_prev)
            
        #画像を二値化
        sampled_blocks = binarize_images(sampled_blocks)
        sampled_blocks = sampled_blocks.reshape(sampled_blocks.shape[0], self.b * self.b * self.C_prev)
        print('samples shape:',np.shape(sampled_blocks))
        
        #重複を削除
        sampled_blocks= np.unique(sampled_blocks, axis=0) 

        print('unique samples shape:', np.shape(sampled_blocks))
        
        #B個だけランダムに取り出す
        self.B = min(sampled_blocks.shape[0], self.B)  #Bより少ないサンプル数の場合はそのまま
        selected_indices = np.random.choice(sampled_blocks.shape[0], self.B, replace=False)
        sampled_blocks = sampled_blocks[selected_indices]
        
        return torch.from_numpy(sampled_blocks.astype(np.float32)).clone()

    def learn_embedding(self, train_X): 
        '''
        埋め込みをKIMで学習
            train_X: 学習に使うX
            train_Y: Xのラベルデータ
        '''
        n_train = train_X.shape[0]

        if self.GP is None:
            sampled_blocks= self.sample_block(n_train, train_X)
            embedded_blocks = select_embedding_method(self.embedding, self.C_next, sampled_blocks)

            #ガウス過程回帰で学習
            print('[KIM] Fitting samples...')
            
            #埋め込みデータを正規化,標準化
            ms = MinMaxScaler()
            ss = StandardScaler()
            embedded_blocks = ms.fit_transform(embedded_blocks)
            embedded_blocks = ss.fit_transform(embedded_blocks)

            print("Training sample shape:", np.shape(embedded_blocks))
            print('[KIM] Training KIM')
            embedded_blocks = torch.from_numpy(embedded_blocks.astype(np.float32)).clone()
            self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
            self.GP = gpytorch_models.ExactGPModel(sampled_blocks, embedded_blocks, self.likelihood)
            self.GP.train()
            self.likelihood.train()
            optimizer= torch.optim.Adam(self.GP.parameters(), lr=0.1)
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.GP)
            for i in range(50):
                optimizer.zero_grad()
                output = self.GP(sampled_blocks)
                print(embedded_blocks.shape)
                loss = -mll(output, embedded_blocks)
                loss.backward()
                print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
                    i + 1, 50, loss.item(),
                    self.GP.covar_module.base_kernel.lengthscale.item(),
                    self.GP.likelihood.noise.item()
                ))
                optimizer.step()
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
            predictions = self.likelihood(self.GP(blocks_to_convert))
            predictions = predictions.numpy().reshape(batch_size, self.H-self.b+1, self.H-self.b+1, self.C_next)
            self.output_data[batch_size * batch_index : batch_size * (batch_index + 1)] = predictions

    def calculate(self, input_X):
        '''
        KIM層の全体の計算
        '''
        #パディング
        if self.padding:
            out_size = input_X.shape[1] + self.b - 1 # 28 + 5 - 1
            input_X = pad_images(input_X, out_size)
        
        #インスタンス変数に格納
        num_inputs = input_X.shape[0]
        self.H = input_X.shape[1]
        self.W = input_X.shape[2]
        self.C_prev = input_X.shape[3]
        self.input_data = input_X
        self.output_data = np.zeros((num_inputs, int((self.H-self.b+1)/self.stride), int((self.W-self.b+1)/self.stride), self.C_next))
        
        #KIMで埋め込みを学習
        self.learn_embedding(input_X) 
        
        #学習したKIMで変換
        print('[KIM] Converting the image...')
        self.convert_image_batch(batch_size=100)
        print('completed')

        return self.output_data

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
                for i in range(self.num_GP):
                    print('learning {}'.format(i+1))
                    X_sep = X[10000*i:10000*(i+1)]
                    Y_sep = Y[10000*i:10000*(i+1)]
                    self.GP.append(gpflow.models.GPR((X_sep,Y_sep), kernel=kernel))
                    opt.minimize(self.GP[-1].training_loss, self.GP[-1].trainable_variables)
            else:
                self.GP = gpflow.models.GPR((X, Y), kernel=kernel)
                opt.minimize(self.GP.training_loss, self.GP.trainable_variables)
                print('Completed')
        else:
            print('GPmodel found')

    def predict(self, X):
        #ベクトル化し予測
        X = X.reshape(X.shape[0], X.shape[1] * X.shape[2] * X.shape[3])
        if self.OVER_10000:
            predictions = []
            for i in range(self.num_GP):
                Y_predicted, _ = self.GP[i].predict_y(X)
                Y_predicted = np.array(Y_predicted)
                predict = [np.argmax(Y_predicted[n,:]) for n in range(X.shape[0])]
                predictions.append(predict)
            ensemble_predictions = np.vstack(predictions)
            output = stats.mode(ensemble_predictions, axis=0).mode.ravel()
            
        else:
            Y_predicted, _ = self.GP.predict_y(X)
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
            X_temp = X
            X = layer.calculate(X)
            if self.display:
                if isinstance(layer, KIMLayer):
                    visualize_emb(X_temp, Y, X, layer.b, layer.stride, layer.B, layer.embedding, self.data_set_name)
                    display_images(X, n+2, layer.embedding, self.data_set_name, f'KernelCNN train output Layer{n+2} (B={layer.B}, Embedding:{layer.embedding})')

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
            test_X = layer.calculate(test_X)
            if self.display:
                if isinstance(layer, KIMLayer):
                    display_images(test_X, n+7, layer.embedding, self.data_set_name, f'KernelCNN test output Layer{n+2} (B={layer.B}, Embedding:{layer.embedding})')

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

