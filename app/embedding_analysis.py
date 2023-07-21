import time
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import numpy as np
import layers
#import layers_originalGP as layers
import embedding
from functions import display_images
from functions import binarize_images
import GPy
import random
from sklearn.manifold import SpectralEmbedding
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from skimage import util
from tqdm import tqdm

from keras.datasets import mnist
from keras.utils import to_categorical

random.seed(1)

def visualize_data(compressed_data, sampled_blocks, filename, emb):

    fig = plt.figure(figsize=(10, 10))

    # 圧縮後の散布図
    ax2 = fig.add_subplot(111)
    ax2.scatter(compressed_data[:, 0], compressed_data[:, 1], c='blue', marker='o', s=30)
    ax2.set_xlabel('Component 1')
    ax2.set_ylabel('Component 2')
    ax2.set_title('Embedded data '+"("+emb+")")

    # ランダムに一部の点にのみブロックの画像を表示
    num_samples = len(compressed_data)
    num_blocks_to_display = min(300, num_samples) 
    random_indices = random.sample(range(num_samples), num_blocks_to_display)

    for i in random_indices:
        x, y = compressed_data[i]
        img = sampled_blocks[i].reshape(5, 5)  # ブロック画像を5x5に変形
        imgbox = OffsetImage(img, zoom=4, cmap='gray')  # 解像度を上げるためにzoomパラメータを調整
        ab = AnnotationBbox(imgbox, (x, y), frameon=True, xycoords='data', boxcoords="offset points", pad=0.0)
        ax2.add_artist(ab)

    plt.tight_layout()

    # 画像として保存
    plt.savefig(filename)
    plt.show()


class KIMLayer:
    def __init__(self, block_size, channels_next, stride, num_samples=100, emb="LE"):
        self.b = block_size
        self.stride = stride
        self.C_next = channels_next
        self.C_prev = None
        self.H = None
        self.W = None
        self.num_samples = num_samples
        self.GP = None
        self.output_data = None
        self.input_data = None
        self.embedding = emb

    def sample_block(self, n_train, train_data, input_dim):
        print('[KIM] Sampling blocks...')
        sampled_blocks = []
        for n in range(n_train):
            data = train_data[n,:,:,:] #画像を一枚持ってくる
            for i in range(self.num_samples): # 画像からランダムな位置でブロックをサンプリング
                start_i = np.random.randint(0, self.H - self.b + 1)
                start_j = np.random.randint(0, self.W - self.b + 1)
                sampled_blocks.append(data[:, start_i : start_i + self.b, start_j : start_j + self.b])
                
        print('[KIM] Completed')
        #train_data = binarize_images(train_data) #画像を二値化
        sampled_blocks = binarize_images(sampled_blocks)
        sampled_blocks = np.array(sampled_blocks).reshape(self.num_samples * n_train, input_dim)
        print('samples shape:',np.shape(sampled_blocks))
        sampled_blocks = np.unique(sampled_blocks, axis=0) #重複を削除
        print('unique samples shape:',np.shape(sampled_blocks))
        
        return sampled_blocks

    def learn_embedding(self, train_data): 
        
        '''
        埋め込みをGPで学習
        '''
        input_dim = self.b * self.b * self.C_prev
        n_train = train_data.shape[0]

        if self.GP is None:
            sampled_blocks = self.sample_block(n_train, train_data, input_dim)
            #print("sampled_blocks shape:",sampled_blocks.shape())
            
            # 埋め込み
            if self.embedding == "PCA":
                embedded_blocks = embedding.pca(sampled_blocks, self.C_next)
                
            elif self.embedding == "LE":
                n_neighbors = int(sampled_blocks.shape[0]/10)
                LE = embedding.LaplacianEigenmap(self.C_next, n_neighbors)
                embedded_blocks = LE.transform(sampled_blocks)
                
            elif self.embedding == "GPLVM":
                GPLVM_model = embedding.GPLVM(θ1=1.0, θ2=0.03, θ3=0.05)
                embedded_blocks = GPLVM_model.fit(sampled_blocks,latent_dim=self.C_next, epoch=50, eta=0.001)
                
            elif self.embedding == "GPLVM2":
                sigma=3
                alpha=0.05
                beta=0.08
                model = embedding.GPLVM2(sampled_blocks,self.C_next, np.array([sigma**2,alpha/beta]))
                embedded_blocks = model.fit(epoch=100,epsilonX=0.05,epsilonSigma=0.0005,epsilonAlpha=0.00001)
                
            elif self.embedding == "LPP":
                LPP_model = embedding.LPP(n_components=self.C_next)
                embedded_blocks = LPP_model.transform(sampled_blocks)
            
            else:
                print('Error: No embedding selected.')

            print("embedded shape:", np.shape(embedded_blocks))
            # 埋め込みデータを可視化
            principal_data = embedded_blocks[:,0:2]
            visualize_data(principal_data, sampled_blocks, "emb_"+self.embedding+".png", self.embedding)

            #ガウス過程回帰で学習
            print('[KIM] Fitting samples...')
            kernel = GPy.kern.RBF(input_dim = input_dim)
            self.GP = GPy.models.GPRegression(sampled_blocks, embedded_blocks, kernel=kernel)
            self.GP.optimize()
            print('[KIM] Completed')
        else:
            print('[KIM] GPmodel found')
    
    def convert_image(self, n):
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


    def calculate(self, input_data):

        self.C_prev = input_data.shape[1]
        self.H = input_data.shape[2]
        self.W = input_data.shape[3]
        num_inputs = input_data.shape[0]

        self.input_data = input_data
        self.output_data = np.zeros((num_inputs, self.C_next, int((self.H-self.b+1)/self.stride), int((self.W-self.b+1)/self.stride)))

        train_data = input_data[:100] #先頭から１００枚の画像を埋め込みの学習に使う
        self.learn_embedding(train_data) 
        
        print('[KIM] Converting the image...')
        for n in tqdm(range(num_inputs)):
            self.convert_image(n)
        
        return self.output_data


if __name__ == '__main__':
    args = sys.argv

    #データセットのロード
    (X_train, Y_train), (X_test,Y_test) = mnist.load_data()

    image_rows = 28
    image_cols = 28
    image_color = 1 #グレースケール
    input_shape = (image_rows, image_cols, image_color)
    out_size = 10

    #データ整形
    X_train = X_train.reshape(-1, image_color, image_rows, image_cols) 
    X_test = X_test.reshape(-1, image_color, image_rows, image_cols, ) 
    Y_train = to_categorical(Y_train,out_size)
    Y_test = to_categorical(Y_test,out_size)

    n = int(args[1])  #train
    #m = int(args[2])  #test
    emb = args[2]

    X_train = X_train[:n]
    Y_train = Y_train[:n]
    #X_test = X_test[:m]
    #Y_test = Y_test[:m]

    #X_train = binarize_images(X_train)
    #X_test = binarize_images(X_test)
    
    KIM = KIMLayer(block_size=5, channels_next = 20, stride = 2, num_samples=100, emb=emb)
    output = KIM.calculate(X_train)
    display_images(output,1)