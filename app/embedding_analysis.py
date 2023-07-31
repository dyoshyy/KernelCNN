import time
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import numpy as np
#import layers_originalGP as layers
import embedding
from functions import display_images
from functions import binarize_images

import GPy
import random

from sklearn.manifold import SpectralEmbedding
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, KernelPCA

import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from skimage import util
from tqdm import tqdm

from keras.datasets import mnist
from keras.utils import to_categorical

random.seed(1)

def visualize_data(compressed_data, sampled_blocks, sampled_blocks_label, filename, emb, block_size):

    file_exists = os.path.exists("./emb_results/"+filename+".png")
    counter = 1
    changed = False
    while file_exists:
        new_filename = filename + f"({counter})"
        file_exists = os.path.exists("./emb_results/"+new_filename+".png")
        counter += 1
        changed = True
    if changed:
        filename = new_filename

    fig = plt.figure(figsize=(36, 30)) #12 10

    # 圧縮後の散布図
    ax2 = fig.add_subplot(111)
    sc = ax2.scatter(compressed_data[:, 0], compressed_data[:, 1], cmap='tab10', c=sampled_blocks_label, marker='o', s=60, edgecolors='black')
    plt.colorbar(sc, label='label')
    ax2.set_xlabel('Component 1')
    ax2.set_ylabel('Component 2')
    ax2.set_title('Embedded data '+"("+emb+")")

    # ランダムに一部の点にのみブロックの画像を表示
    num_samples = len(compressed_data)
    num_blocks_to_display = min(300, num_samples) 
    random_indices = random.sample(range(num_samples), num_blocks_to_display)

    for i in random_indices:
        x, y = compressed_data[i]
        img = sampled_blocks[i].reshape(block_size, block_size)  # ブロック画像を5x5に変形
        imgbox = OffsetImage(img, zoom=17-block_size, cmap='gray')  # 解像度を上げるためにzoomパラメータを調整
        ab = AnnotationBbox(imgbox, (x, y), frameon=True, xycoords='data', boxcoords="offset points", pad=0.0)
        ax2.add_artist(ab)

    plt.tight_layout()

    # 画像として保存
    plt.savefig("./emb_results/"+filename+'.png')
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
        self.GP_learn = False

    def sample_block(self, n_train, train_X, train_Y, input_dim):
        print('[KIM] Sampling blocks...')
        sampled_blocks = []
        sampled_blocks_label = []
        for n in range(n_train):
            data = train_X[n,:,:,:] #画像を一枚持ってくる
            for i in range(self.num_samples): # 画像からランダムな位置でブロックをサンプリング
                start_i = np.random.randint(0, self.H - self.b + 1)
                start_j = np.random.randint(0, self.W - self.b + 1)
                sampled_blocks.append(data[:, start_i : start_i + self.b, start_j : start_j + self.b])
                sampled_blocks_label.append(train_Y[n])
                
        print('[KIM] Completed')
        #train_data = binarize_images(train_data) #画像を二値化
        sampled_blocks = binarize_images(sampled_blocks)
        sampled_blocks = np.array(sampled_blocks).reshape(self.num_samples * n_train, input_dim)
        print('samples shape:',np.shape(sampled_blocks))
        sampled_blocks, indices = np.unique(sampled_blocks, axis=0, return_index=True) #重複を削除
        sampled_blocks_label = [sampled_blocks_label[index] for index in indices]
        print('unique samples shape:',np.shape(sampled_blocks))
        
        return sampled_blocks, sampled_blocks_label

    def learn_embedding(self, train_X, train_Y): 
        
        '''
        埋め込みをGPで学習
        '''
        input_dim = self.b * self.b * self.C_prev
        n_train = train_X.shape[0]

        if self.GP is None:
            sampled_blocks, sampled_blocks_label = self.sample_block(n_train, train_X, train_Y, input_dim)
            #print("sampled_blocks shape:",sampled_blocks.shape())
            
            # 埋め込み
            if self.embedding == "PCA":
                pca = PCA(n_components=self.C_next)
                embedded_blocks = pca.fit_transform(sampled_blocks)
            
            elif self.embedding == "KPCA":
                kpca = KernelPCA(n_components=self.C_next, kernel="rbf")
                embedded_blocks = kpca.fit_transform(sampled_blocks)
                
            elif self.embedding == "LE":
                n_neighbors = int(sampled_blocks.shape[0]/10)
                #n_neighbors=10
                LE = embedding.LaplacianEigenmap(self.C_next, n_neighbors)
                embedded_blocks = LE.transform(sampled_blocks)
                
            elif self.embedding == "GPLVM":
                sigma=3
                alpha=0.05
                beta=0.08
                model = embedding.GPLVM(sampled_blocks,self.C_next, np.array([sigma**2,alpha/beta]))
                embedded_blocks = model.fit(epoch=100,epsilonX=0.05,epsilonSigma=0.0005,epsilonAlpha=0.00001)
                
            elif self.embedding == "LPP":
                LPP_model = embedding.LPP(n_components=self.C_next)
                embedded_blocks = LPP_model.transform(sampled_blocks)
            
            elif self.embedding == "TSNE":
                tsne = TSNE(n_components=self.C_next, random_state = 0, method='exact', perplexity = 30, n_iter = 500)
                embedded_blocks = tsne.fit_transform(sampled_blocks)
            
            
            else:
                print('Error: No embedding selected.')

            print("embedded shape:", np.shape(embedded_blocks))
            # 埋め込みデータを可視化
            principal_data = embedded_blocks[:,0:2]
            visualize_data(principal_data, sampled_blocks, sampled_blocks_label, "emb_"+self.embedding, self.embedding, self.b)

            if self.GP_learn:
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


    def calculate(self, input_X, input_Y):

        self.C_prev = input_X.shape[1]
        self.H = input_X.shape[2]
        self.W = input_X.shape[3]
        num_inputs = input_X.shape[0]

        self.input_data = input_X
        self.output_data = np.zeros((num_inputs, self.C_next, int((self.H-self.b+1)/self.stride), int((self.W-self.b+1)/self.stride)))

        train_X = input_X[:100] #先頭から１００枚の画像を埋め込みの学習に使う
        train_Y = input_Y[:100]
        #train_Y = train_Y[np.newaxis, np.newaxis, np.newaxis, :]
        
        self.learn_embedding(train_X, train_Y) 
        
        if self.GP_learn:
            print('[KIM] Converting the image...')
            for n in tqdm(range(num_inputs)):
                self.convert_image(n)
            
        return self.output_data


def main(channels_next, emb):

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

    n = 100 #train
    #m = int(args[2])  #test

    X_train = X_train[:n]
    Y_train = Y_train[:n]
    #X_test = X_test[:m]
    #Y_test = Y_test[:m]

    #X_train = binarize_images(X_train)
    #X_test = binarize_images(X_test)
    
    KIM = KIMLayer(block_size=5, channels_next = channels_next, stride = 2, num_samples=100, emb=emb)
    output = KIM.calculate(X_train, Y_train)
    display_images(output,1)

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
    #Y_train = to_categorical(Y_train,out_size)
    #Y_test = to_categorical(Y_test,out_size)

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
    output = KIM.calculate(X_train, Y_train)
    display_images(output,1)