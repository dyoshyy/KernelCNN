import numpy as np
import random
import GPy
from functions import calculate_similarity
from functions import display_images
from functions import binarize_images
from functions import visualize_emb

import embedding
from sklearn.manifold import SpectralEmbedding
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, KernelPCA

from skimage import util
from tqdm import tqdm


np.random.seed(1)

class KIMLayer:
    def __init__(self, block_size, channels_next, stride, emb="LE"):
        self.b = block_size
        self.stride = stride
        self.C_next = channels_next
        self.C_prev = None
        self.H = None
        self.W = None
        self.output_data = None
        self.input_data = None
        self.embedding = emb
        self.GP = None

    def sample_block(self, n_train, train_X, train_Y):
        '''
        画像データからブロックをサンプリング
            n_train : 画像の枚数
            train_X : 学習する画像データ
            train_Y : 画像のラベル(NOT One-hot vector)
        '''
        sampled_blocks = []
        sampled_blocks_label = []
        train_Y = np.argmax(train_Y, axis=1)
        for n in range(n_train):
            # 一枚持ってくる
            data = train_X[n,:,:,:]
            # すべてのブロックをサンプリング
            for i in range(self.H - self.b + 1):
                for j in range(self.W - self.b + 1):
                    block = data[:, i:i+self.b, j:j+self.b]
                    sampled_blocks.append(block)
                    sampled_blocks_label.append(train_Y[n])      

        #画像を二値化
        sampled_blocks = binarize_images(sampled_blocks)
        sampled_blocks = np.array(sampled_blocks).reshape(sampled_blocks.shape[0], self.b * self.b * self.C_prev)
        print('samples shape:',np.shape(sampled_blocks))
        
        #重複を削除
        sampled_blocks, indices, counts = np.unique(sampled_blocks, axis=0, return_index=True, return_counts=True) 
        sampled_blocks_label = np.array(sampled_blocks_label)[indices]
        
        #重複回数の多い順にソート
        sorted_indices = np.argsort(counts)[::-1]
        sampled_blocks = sampled_blocks[sorted_indices]
        
        sampled_blocks_label = sampled_blocks_label[sorted_indices]
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
            # 埋め込み
            if self.embedding == "PCA":
                pca = PCA(n_components=self.C_next, svd_solver='auto')
                embedded_blocks = pca.fit_transform(sampled_blocks)
            
            elif self.embedding == "KPCA":
                kpca = KernelPCA(n_components=self.C_next, kernel="rbf")
                embedded_blocks = kpca.fit_transform(sampled_blocks)
                
            elif self.embedding == "LE":
                n_neighbors = int(sampled_blocks.shape[0]/10)
                #n_neighbors=10
                LE = embedding.LaplacianEigenmap(self.C_next, n_neighbors)
                embedded_blocks = LE.transform(sampled_blocks)
            
            elif self.embedding == "LPP":
                LPP_model = embedding.LPP(n_components=self.C_next)
                embedded_blocks = LPP_model.transform(sampled_blocks)
                
            elif self.embedding == "GPLVM":
                sigma=3
                alpha=0.05
                beta=0.08
                model = embedding.GPLVM(sampled_blocks,self.C_next, np.array([sigma**2,alpha/beta]))
                embedded_blocks = model.fit(epoch=100,epsilonX=0.05,epsilonSigma=0.0005,epsilonAlpha=0.00001)
            
            elif self.embedding == "TSNE":
                tsne = TSNE(n_components=self.C_next, random_state = 0, method='exact', perplexity = 30, n_iter = 500)
                embedded_blocks = tsne.fit_transform(sampled_blocks)
            
            else:
                print('Error: No embedding selected.')

            print("embedded shape:", np.shape(embedded_blocks))
            
            # 埋め込みデータを可視化
            principal_data = embedded_blocks[:,0:2]
            visualize_emb(principal_data, sampled_blocks, sampled_blocks_label, self.embedding, self.b)

            #ガウス過程回帰で学習
            print('[KIM] Fitting samples...')
            
            #３分の１だけ無作為に取り出す
            selected_indices = random.sample(range(sampled_blocks.shape[0]), int(sampled_blocks.shape[0]/3))
            sampled_blocks = sampled_blocks[selected_indices]
            embedded_blocks = embedded_blocks[selected_indices]
            
            kernel = GPy.kern.RBF(input_dim = self.b * self.b * self.C_prev)
            self.GP = GPy.models.GPRegression(sampled_blocks, embedded_blocks, kernel=kernel)
            self.GP.optimize()
            print('[KIM] Completed')
            
        else:
            print('[KIM] GPmodel found')
    
    def convert_image(self, n):
        '''
        学習済みのKIMで元の画像を変換
            n : 何枚目の画像を変換するか
        '''
        b_radius = int((self.b-1)/2)
        output_tmp = np.zeros((self.C_next, int((self.H-self.b+1)/self.stride), int((self.W-self.b+1)/self.stride)))

        blocks = []
        for i in range(b_radius, self.H-b_radius, self.stride):
            i_output = int((i - b_radius)/self.stride)
            for j in range(b_radius, self.W-b_radius, self.stride):
                j_output = int((j - b_radius)/self.stride)
                input_cropped = self.input_data[n, :, (i-b_radius):(i+b_radius+1), (j-b_radius):(j+b_radius+1)].reshape(1, self.b * self.b * self.C_prev)
                blocks.append(input_cropped)
        
        blocks = np.concatenate(blocks, axis=0)
        predictions, _ = self.GP.predict(blocks)
        
        #再配置
        idx = 0
        for i in range(b_radius, self.H-b_radius, self.stride):
            i_output = int((i - b_radius)/self.stride)
            for j in range(b_radius, self.W-b_radius, self.stride):
                j_output = int((j - b_radius)/self.stride)
                output_tmp[:, i_output, j_output] = predictions[idx]
                idx += 1
        
        self.output_data[n] = output_tmp


    def calculate(self, input_X, input_Y):
        
        num_inputs = input_X.shape[0]
        self.C_prev = input_X.shape[1]
        self.H = input_X.shape[2]
        self.W = input_X.shape[3]
        
        self.input_data = input_X
        self.output_data = np.zeros((num_inputs, self.C_next, int((self.H-self.b+1)/self.stride), int((self.W-self.b+1)/self.stride)))

        #先頭からtrain_numの画像を埋め込みの学習に使う
        train_num = 300
        train_X = input_X[:train_num] 
        train_Y = input_Y[:train_num]
        
        self.learn_embedding(train_X, train_Y) 
        
        print('[KIM] Converting the image...')
        for n in tqdm(range(num_inputs)):
            self.convert_image(n)
            
        return self.output_data

class MaxPoolingLayer:
    def __init__(self, pool_size):
        self.pool_size = pool_size
        self.output_data = None

    def pooling(self, n, input_data, H_out, W_out, C):
        print('[MaxPooling] Converting the image %d' % (n+1))
        for c in range(C):
            for i in range(H_out):
                    for j in range(W_out):
                        start_i = i * self.pool_size
                        start_j = j * self.pool_size
                        self.output_data[n, c, i, j] = np.max(np.abs(input_data[n, c, start_i:start_i + self.pool_size, start_j:start_j + self.pool_size]))

    def calculate(self, input_data, Y):
        N, C, H, W = input_data.shape
        p = self.pool_size
        H_out = H // p
        W_out = W // p
        self.output_data = np.zeros((N, C, H_out, W_out))
        
        for n in range(N):
            self.pooling(n, input_data, H_out, W_out, C)

        return self.output_data

class AvgPoolingLayer:
    def __init__(self, pool_size):
        self.pool_size = pool_size

    def calculate(self, input_data, Y):
        N, C, H, W = input_data.shape
        p = self.pool_size
        H_out = H // p
        W_out = W // p
        output_data = np.zeros((N, C, H_out, W_out))
        print('[AvgPooling] Converting the image...')
        for n in tqdm(range(N)):
            for c in range(C):
                for i in range(H_out):
                    for j in range(W_out):
                        start_i = i * p
                        start_j = j * p
                        output_data[n, c, i, j] = np.mean(input_data[n, c, start_i:start_i + p, start_j:start_j + p])

        return output_data

class LabelLearningLayer:
    def __init__(self):
        self.GP = None

    def fit(self, X, Y):
        input_dim = X.shape[1] * X.shape[2] * X.shape[3]
        X = X.reshape(X.shape[0], input_dim)
        if self.GP is None:
            print('Learning labels')
            kernel = GPy.kern.RBF(input_dim = input_dim)
            self.GP = GPy.models.GPRegression(X, Y, kernel=kernel)
            self.GP.optimize()
            print('Completed')
        else:
            print('GPmodel found')

    def predict(self, X):
        #ベクトル化し予測
        X = X.reshape(X.shape[0], X.shape[1] * X.shape[2] * X.shape[3])
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
        for n, layer in enumerate(self.layers):
            self.shapes.append(np.shape(X)[1:])
            X = layer.calculate(X, Y)
            if self.display:
                display_images(X, n+1)

        self.shapes.append(np.shape(X)[1:])
        if not isinstance(self.layers[-1], LabelLearningLayer):
            self.layers.append(LabelLearningLayer())
        self.layers[-1].fit(X, Y)

    def predict(self, test_X, test_Y):
        for n,layer in enumerate(self.layers):
            if isinstance(layer, LabelLearningLayer):
                break
            test_X = layer.calculate(test_X, test_Y)
            if self.display:
                display_images(test_X, n+5)

        Y_predicted = self.layers[-1].predict(test_X)
        Y_answer= [np.argmax(test_Y[n,:]) for n in range(test_Y.shape[0])]

        print('Accuracy:', calculate_similarity(Y_predicted, Y_answer))
        print('Layers shape:',self.shapes)
        return Y_predicted, Y_answer

