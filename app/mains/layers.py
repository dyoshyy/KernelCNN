import GPy
import numpy as np
import math
import time
from keras import layers, models    
from keras.callbacks import EarlyStopping
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from functions import calculate_similarity, display_images, binarize_images, visualize_emb, visualize_emb_dots, select_embedding_method, pad_images

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC
from sklearn import metrics
from scipy import stats

from skimage import util
from tqdm import tqdm

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
        self.B = num_blocks
        self.dataset_name = None
        self.padding = padding
        self.X_for_KIM = None
        self.Y_for_KIM = None

    def sample_and_embed_blocks(self):
            '''
            Args:
                n_images (int): 画像の枚数
                train_X (ndarray): 学習する画像データ
                train_Y (ndarray): 画像のラベル(NOT One-hot vector)
            
            Returns:
                ndarray: サンプリングされたブロックの配列
                ndarray: サンプリングされたブロックの埋め込み
            '''
            n_images = self.X_for_KIM.shape[0]
            sampled_blocks = np.empty((n_images*(self.H-self.b+1)**2, self.b, self.b, self.C_prev))
            print('sampling...')
            sampled_blocks = [util.view_as_windows(self.X_for_KIM[n,:,:,:], (self.b, self.b, self.C_prev), self.stride).reshape((self.H-self.b+1)**2, self.b, self.b, self.C_prev) for n in range(n_images)]
            sampled_blocks = np.concatenate(sampled_blocks, axis=0)

            sampled_blocks = sampled_blocks.reshape(-1, self.b, self.b, self.C_prev)
            sampled_blocks_label = np.repeat(np.argmax(self.Y_for_KIM, axis=1), int(sampled_blocks.shape[0]/self.Y_for_KIM.shape[0]))
            print('sampling completed')

            #画像を二値化
            sampled_blocks = binarize_images(sampled_blocks)
            sampled_blocks = sampled_blocks.reshape(sampled_blocks.shape[0], self.b * self.b * self.C_prev)
            print('All samples shape:',np.shape(sampled_blocks))
            
            #重複を削除
            sampled_blocks, unique_index= np.unique(sampled_blocks, axis=0, return_index=True) 
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
            
    def learn_embedding(self): 
        '''
        埋め込みをKIMで学習
            train_X: 学習に使うX
            train_Y: Xのラベルデータ
        '''
        '''
        #クラスの偏りがないようにサンプルを選ぶ
        n_images = 100 #埋め込みの学習に用いる画像枚数
        n_classes = 10
        selected_X = []
        selected_Y = []
        for label in range(n_classes):
            indices = np.where(train_Y == label)[0][:int(n_images/n_classes)]
            selected_X.extend(train_X[indices])
            selected_Y.extend(train_Y[indices])
        
        selected_X = np.array(selected_X)
        selected_Y = np.array(selected_Y) 
        '''
            
        if self.GP is None:
            sampled_blocks, embedded_blocks = self.sample_and_embed_blocks()
            #kernel = GPy.kern.RBF(input_dim = self.b * self.b * self.C_prev, variance=0.001, lengthscale=1.0) + GPy.kern.Bias(input_dim = self.b * self.b * self.C_prev, variance=60000) + GPy.kern.Linear(input_dim = self.b * self.b * self.C_prev, variances=0.05)
            kernel = GPy.kern.RBF(input_dim = self.b * self.b * self.C_prev, variance=0.001) + GPy.kern.Bias(input_dim = self.b * self.b * self.C_prev) + GPy.kern.Linear(input_dim = self.b * self.b * self.C_prev, variances=0.05)
            self.GP = GPy.models.SparseGPRegression(sampled_blocks, embedded_blocks, num_inducing=100, kernel=kernel)
            self.GP.Gaussian_noise.variance = 0.001
            print('optimizing the GP model')
            self.GP.optimize(optimizer='lbfgs')
            
            #print('model parameters:', self.GP)
            print('completed')
        else:
            print('[KIM] GPmodel found')
        
    def convert_image_batch(self, batch_size=10):
            """
            Converts the input image batch into a batch of predictions.

            Args:
                batch_size (int, optional): The size of each batch. Defaults to 10.
            """
            num_batches = math.ceil(self.input_data.shape[0]/batch_size)

            for batch_index in tqdm(range(num_batches)):
                batch_images = self.input_data[batch_size * batch_index : batch_size * (batch_index + 1)]
                blocks_to_convert = util.view_as_windows(batch_images, (1, self.b, self.b, self.C_prev), self.stride)
                blocks_to_convert = blocks_to_convert.reshape(batch_size * (self.H-self.b+1)**2, self.b * self.b * self.C_prev) # ex) (10*784, 5*5*1)        
                predictions, _ = self.GP.predict(blocks_to_convert) # shape: (10*784, 6)
                predictions = predictions.reshape(batch_size, self.H-self.b+1, self.H-self.b+1, self.C_next)
                self.output_data[batch_size * batch_index : batch_size * (batch_index + 1)] = predictions

    def calculate(self, input_X, input_Y):
        """
        このメソッドは、入力データに対してKIM (Kernelized Input Mapping) を適用し、
        結果を出力データとして保存します。KIMモデルが存在しない場合は入力データから学習します。

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
        self.learn_embedding() 
        
        #学習したKIMで変換
        print('[KIM] Converting the image...')
        self.convert_image_batch(batch_size=100)
        print('completed')
        #ReLU
        #self.output_data = np.maximum(0, self.output_data)
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
    def __init__(self) -> None:
        self.classifier = None

    def vectorize_standarize(self, X):
        #if X.ndim == 4:
        #    input_dim = X.shape[1] * X.shape[2] * X.shape[3]
            
        X = X.reshape(X.shape[0], -1)
        X = StandardScaler().fit_transform(X)
        #X = MinMaxScaler().fit_transform(X)
        return X

class SupportVectorsMachine(LabelLearningLayer):
    def __init__(self):
        super().__init__()

    def fit(self, X, Y):
        X = self.vectorize_standarize(X)
        Y = np.argmax(Y, axis=1)
        if self.classifier is None:
            print('Learning labels')
            self.classifier = SVC(kernel='rbf', C=10.0, gamma='auto', probability=True, decision_function_shape='ovr')
            #np.savetxt('train_X.csv', X, delimiter=',')
            #np.savetxt('train_Y.csv', Y, delimiter=',')
            self.classifier.fit(X, Y)
            print('Completed')
        else:
            print('SVM model found')

    def predict(self, X):
        #ベクトル化し予測
        X = self.vectorize_standarize(X)
        #np.savetxt('test_X.csv', X, delimiter=',')
        output = self.classifier.predict(X)
        return output

class RandomForest(LabelLearningLayer):
    def __init__(self):
        super().__init__()

    def fit(self, X, Y):
        X = self.vectorize_standarize(X)
        if len(Y.shape) > 1:
            Y = np.argmax(Y, axis=1)
        if self.classifier is None:
            print('Learning labels')
            self.classifier = RandomForestClassifier(n_estimators=300, max_depth=30, random_state=0)
            self.classifier.fit(X, Y)
            print('Completed')
        else:
            print('RandomForest model found')

    def predict(self, X):
        #ベクトル化し予測
        X = self.vectorize_standarize(X)
        output = self.classifier.predict(X)
        return output
    
class GaussianProcess(LabelLearningLayer):
    def __init__(self):
        super().__init__()
        self.num_GP = None
        self.OVER_threshold = False
        self.threshold = 10000

    def fit(self, X, Y):
        input_dim = X.shape[1] * X.shape[2] * X.shape[3]
        X = self.vectorize_standarize(X)
        
        if self.classifier is None:
            print('Learning labels')

            # 訓練サンプルが10000超える場合はthresholdずつに分けて学習
            if X.shape[0] > self.threshold:
                self.OVER_threshold = True
                self.classifier = []
                #必要なGPの数
                self.num_GP = int((X.shape[0]-1)/self.threshold + 1)
                kernel = GPy.kern.RBF(input_dim = input_dim) + GPy.kern.Bias(input_dim = input_dim) + GPy.kern.Linear(input_dim = input_dim)
                for i in range(self.num_GP):
                    print('learning {}'.format(i+1))
                    X_sep = X[self.threshold*i:self.threshold*(i+1)]
                    Y_sep = Y[self.threshold*i:self.threshold*(i+1)]
                    self.classifier.append(GPy.models.GPRegression(X_sep,Y_sep, kernel=kernel))
                    self.classifier[-1].optimize()
            else:
                kernel = GPy.kern.RBF(input_dim = input_dim) + GPy.kern.Bias(input_dim = input_dim) + GPy.kern.Linear(input_dim = input_dim)
                self.classifier = GPy.models.GPRegression(X, Y, kernel=kernel)
                self.classifier.optimize()
                print('Completed')
        else:
            print('GPmodel found')
    
    def predict(self, X):
        X = self.vectorize_standarize(X)
        if self.OVER_threshold:
            predictions = []
            for i in range(self.num_GP):
                Y_predicted, _ = self.classifier[i].predict(X)
                Y_predicted = np.array(Y_predicted)
                predict = [np.argmax(Y_predicted[n,:]) for n in range(X.shape[0])]
                predictions.append(predict)
            ensemble_predictions = np.vstack(predictions)
            output = stats.mode(ensemble_predictions, axis=0).mode.ravel()
            
        else:
            Y_predicted, _ = self.classifier.predict(X)
            Y_predicted = np.array(Y_predicted)
            output = [np.argmax(Y_predicted[n,:]) for n in range(X.shape[0])]
        return output

class kNearestNeighbors(LabelLearningLayer):
    def __init__(self, num_neighbors=1):
        super().__init__()
        self.num_neighbors = num_neighbors
        self.num_classes = 10

    def fit(self, X, Y):
        X = self.vectorize_standarize(X)
        Y = np.argmax(Y, axis=1)
        if self.classifier is None:
            print('Learning labels')
            self.classifier = KNeighborsClassifier(n_neighbors=self.num_neighbors)
            self.classifier.fit(X, Y)
            print('Completed')
        else:
            print('k-NN model found')
    
    def predict(self, X):
        X = self.vectorize_standarize(X)
        output = self.classifier.predict(X)
        return output
    
class QuadraticDiscriminantAnalysis(LabelLearningLayer):
    def __init__(self):
        super().__init__()

    def fit(self, X, Y):
        X = self.vectorize_standarize(X)
        Y = np.argmax(Y, axis=1)
        if self.classifier is None:
            print('Learning labels')
            self.classifier = QDA()
            self.classifier.fit(X, Y)
            print('Completed')
        else:
            print('QDA model found')
    
    def predict(self, X):
        X = self.vectorize_standarize(X)
        output = self.classifier.predict(X)
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
                #クラスの偏りがないようにサンプルを選ぶ
                n_train = 300 #埋め込みの学習に用いる画像枚数
                n_classes = 10
                selected_X = []
                selected_Y = []
                for label in range(n_classes):
                    indices = np.where(np.argmax(Y, axis=1) == label)[0][:int(n_train/n_classes)]
                    selected_X.extend(X[indices])
                    selected_Y.extend(Y[indices]) 
                layer.X_for_KIM = np.array(selected_X)
                layer.Y_for_KIM = np.array(selected_Y)
                
                #paddingが有効の場合は画像をパディング
                if layer.padding:
                    out_size = X.shape[1] + layer.b - 1 # 28 + 5 - 1
                    X = pad_images(X, out_size)
                X_temp = X
                X = layer.calculate(X, Y)
                #displayが有効の場合は中間層の出力と埋め込みを可視化
                if self.display:
                    visualize_emb(X_temp, Y, X, layer.b, layer.stride, layer.B, layer.embedding, self.data_set_name)
                    display_images(X, n+2, layer.embedding, self.data_set_name, f'KernelCNN train output Layer{n+2} (b={layer.b}, B={layer.B}, Embedding:{layer.embedding})')
            elif isinstance(layer, LabelLearningLayer): #識別層のとき
                layer.fit(X, Y)
            else: #プーリング層
                X = layer.calculate(X)
        
        self.time_fitting = time.time() - start_time

    def predict(self, test_X, test_Y):
        start_time = time.time()
        self.num_test = test_X.shape[0]
        for n, layer in enumerate(self.layers):
            if isinstance(layer, KIMLayer): #畳み込みのとき
                if layer.padding:
                    out_size = test_X.shape[1] + layer.b - 1 # 28 + 5 - 1
                    test_X = pad_images(test_X, out_size)
                test_X = layer.calculate(test_X, None)
                if self.display:
                    continue
                    #display_images(test_X, n+7, layer.embedding, self.data_set_name, f'KernelCNN test output Layer{n+2} (b={layer.b}, B={layer.B}, Embedding:{layer.embedding})')
            elif isinstance(layer, LabelLearningLayer):
                Y_predicted = self.layers[-1].predict(test_X)
            else: #プーリング層のとき
                test_X = layer.calculate(test_X)
                
        Y_answer= [np.argmax(test_Y[n,:]) for n in range(test_Y.shape[0])]
        #np.savetxt('test_Y.csv', Y_answer, delimiter=',')
        self.time_predicting = time.time() - start_time
        accuracy = metrics.accuracy_score(Y_answer, Y_predicted) * 100
        classification_report = metrics.classification_report(Y_answer, Y_predicted)
        print(classification_report)
        
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
            param_file.write(f'Fitting time: {self.time_fitting} seconds\n')
            param_file.write(f'Predicting time: {self.time_predicting} seconds\n')
            param_file.write(f'Accuracy: {accuracy} %\n')
            param_file.write('================================================================================\n')
            
        return accuracy

