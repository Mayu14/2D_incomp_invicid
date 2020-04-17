# -- coding: utf-8 --

import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Reshape, merge, Input, Concatenate
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D, Conv2D

from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Flatten, Dropout

import math
import numpy as np

import os
from keras.datasets import mnist
from keras.optimizers import Adam
from PIL import Image

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from msdf_shader_02_copy import test_3d_rendering

import tensorflow as tf
import keras.backend as KTF

BATCH_SIZE = 2
NUM_EPOCH = 5000
CLASS_NUM = 1  # 10    # lift only
IMG_TOTAL_NUM = 100

class CGAN():
    def __init__(self):
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.52
        KTF.set_session(tf.Session(config = config))
        # self.path = '/volumes/data/dataset/gan/MNIST/cgan/cgan_generated_images/'
        self.path = 'images/'
        # mnistデータ用の入力データサイズ
        self.img_rows = 16
        self.img_cols = 16
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        
        # 潜在変数の次元数
        self.z_dim = 1000
        
        # 画像保存の際の列、行数
        self.row = 5
        self.col = 5
        self.row2 = 1  # 連続潜在変数用
        self.col2 = 10  # 連続潜在変数用
        
        # 画像生成用の固定された入力潜在変数
        self.noise_fix1 = np.random.normal(0, 1, (self.row * self.col, self.z_dim))
        # 連続的に潜在変数を変化させる際の開始、終了変数
        self.noise_fix2 = np.random.normal(0, 1, (1, self.z_dim))
        self.noise_fix3 = np.random.normal(0, 1, (1, self.z_dim))
        
        # 横軸がiteration数のプロット保存用np.ndarray
        self.g_loss_array = np.array([])
        self.d_loss_array = np.array([])
        self.d_accuracy_array = np.array([])
        self.d_predict_true_num_array = np.array([])
        self.c_predict_class_list = []
        
        self.discriminator_optimizer = Adam(lr = 1e-5, beta_1 = 0.1)
        self.combined_optimizer = Adam(lr = .8e-4, beta_1 = 0.5)
        
        # discriminatorモデル
        self.discriminator = self.build_discriminator()
        if os.path.exists("discriminator.h5"):
            self.discriminator.load_weights("discriminator.h5")
        self.discriminator.compile(loss = 'binary_crossentropy',
                                   optimizer = self.discriminator_optimizer,
                                   metrics = ['accuracy'])
        
        # for layer in self.discriminator.layers:
        #    layer.trainable = False
        self.discriminator.trainable = False
        
        # Generatorモデル
        self.generator = self.build_generator()
        if os.path.exists("generator.h5"):
            self.generator.load_weights("generator.h5")
        # generatorは単体で学習しないのでコンパイルは必要ない
        # self.generator.compile(loss='binary_crossentropy', optimizer=optimizer)
        
        self.combined = self.build_combined()
        # self.combined = self.build_combined2()
        self.combined.compile(loss = 'binary_crossentropy',
                              optimizer = self.combined_optimizer,
                              metrics = ['accuracy'])
    
    def build_generator(self):
        inputs = Input(shape = (self.z_dim + CLASS_NUM,))
        x = Dense(units = 1024)(inputs)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Dense(units = 128 * 4 * 4)(inputs)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Reshape((4, 4, 128), input_shape = (128 * 4 * 4,))(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(64, (5, 5), padding = 'same')(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(3, (5, 5), padding = 'same')(x)
        generated = Activation("tanh")(x)
        model = Model(inputs = [inputs], outputs = [generated])
        return model
    
    def build_discriminator(self):
        inputs = Input(shape = (self.img_rows, self.img_cols, (self.channels + CLASS_NUM)))
        x = Conv2D(64, (5, 5), strides = (2, 2), padding = 'same')(inputs)
        x = LeakyReLU(alpha = 0.2)(x)
        x = Conv2D(128, (5, 5), strides = (2, 2))(x)
        x = LeakyReLU(alpha = 0.2)(x)
        x = Flatten()(x)
        x = Dense(units = 256)(x)
        x = LeakyReLU(alpha = 0.2)(x)
        x = Dropout(0.5)(x)
        x = Dense(units = 1)(x)
        discriminated = Activation('tanh')(x)
        model = Model(inputs = [inputs], outputs = [discriminated])
        return model
    
    def build_combined(self):
        z = Input(shape = (self.z_dim,))
        y = Input(shape = (CLASS_NUM,))
        img_10 = Input(shape = (self.img_rows, self.img_cols, CLASS_NUM,))
        z_y = Concatenate()([z, y])
        
        img = self.generator(z_y)
        img_11 = Concatenate(axis = 3)([img, img_10])
        self.discriminator.trainable = False
        valid = self.discriminator(img_11)
        model = Model(inputs = [z, y, img_10], outputs = valid)
        return model
    
    def combine_images(self, generated_images):
        total = generated_images.shape[0]
        cols = int(math.sqrt(total))
        rows = int(math.ceil(float(total) / cols))
        WIDTH, HEIGHT = generated_images.shape[1:3]
        combined_image = np.zeros((HEIGHT * rows, WIDTH * cols, 3),
                                  dtype = generated_images.dtype)
        
        for index, image in enumerate(generated_images):
            i = int(index / cols)
            j = index % cols
            pilImg = Image.fromarray(np.uint8(image * 127.5 + 127.5)[:, :, ::-1]).transpose(Image.FLIP_TOP_BOTTOM)   # RBG -> RGB
            image = test_3d_rendering(pilImg, render_size=1000, fromObj=True, imgshow=True)
            exit()
            combined_image[WIDTH * i:WIDTH * (i + 1), HEIGHT * j:HEIGHT * (j + 1), :] = image[:, :, :]
        return combined_image
    
    def label2images(self, label):
        images = np.zeros((self.img_rows, self.img_cols, CLASS_NUM))
        images[:, :, 0] += label
        return images
    
    def label2onehot(self, label):
        onehot = np.zeros(CLASS_NUM)
        # onehot[label] = 1
        onehot[0] = label
        return onehot
    
    def label2output(self, label):
        onehot = np.zeros(CLASS_NUM)
        # onehot[label] = 1
        onehot[0] = label
        return onehot
    
    def train(self):

        with np.load("16_i_i_4.npz", allow_pickle = True) as f:
            X_train, y_train = f["x_train"], f["y_train"]
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        scalar = MinMaxScaler()
        scalar.fit(X = y_train.reshape(-1, 1))
        X_train = X_train[::8,:,:,:]
        y_train = y_train[::8]

        discriminator = self.build_generator()
        d_opt = Adam(lr = 1e-5, beta_1 = 0.1)
        discriminator.compile(loss = 'binary_crossentropy', optimizer = d_opt, metrics = ['accuracy'])
        
        g_opt = Adam(lr = 8e-4, beta_1 = 0.5)
        self.combined.compile(loss = 'binary_crossentropy', optimizer = g_opt)
        
        self.g_loss_array = np.zeros(NUM_EPOCH)
        self.d_loss_array = np.zeros(NUM_EPOCH)
        self.d_accuracy_array = np.zeros(NUM_EPOCH)
        self.d_predict_true_num_array = np.zeros(NUM_EPOCH)
        
        num_batches = int(X_train.shape[0] / BATCH_SIZE)
        print("number of batches:", num_batches)
        
        for epoch in range(NUM_EPOCH):
            
            for index in range(num_batches):
                print(index)
                noise_z = np.array([np.random.uniform(-1, 1, self.z_dim) for _ in range(BATCH_SIZE)])
                # noise_y_int = np.random.randint(0, CLASS_NUM, BATCH_SIZE)
                noise_y_int = np.random.uniform(-1, 1, BATCH_SIZE)
                noise_y = np.array(np.array([self.label2onehot(i) for i in noise_y_int]))
                noise_z_y = np.concatenate((noise_z, noise_y), axis = 1)
                f_img = self.generator.predict(noise_z_y, verbose = 0)
                f_img_10 = np.array([self.label2images(i) for i in noise_y_int])
                f_img_11 = np.concatenate((f_img, f_img_10), axis = 3)
                
                r_img = X_train[index * BATCH_SIZE:(index + 1) * BATCH_SIZE]
                label_batch = y_train[index * BATCH_SIZE:(index + 1) * BATCH_SIZE]
                r_img_10 = np.array([self.label2images(i) for i in label_batch])
                r_img_11 = np.concatenate((r_img, r_img_10), axis = 3)
                
                if index % 500 == 0:
                    noise = np.array([np.random.uniform(-1, 1, self.z_dim) for _ in range(IMG_TOTAL_NUM)])
                    randomLabel_batch = np.random.uniform(-1, 1, IMG_TOTAL_NUM)  # arrange(IMG_TOTAL_NUM)%10
                    randomLabel_batch_onehot = np.array([self.label2onehot(i) for i in randomLabel_batch])
                    noise_with_randomLabel = np.concatenate((noise, randomLabel_batch_onehot), axis = 1)
                    generated_images = self.generator.predict(noise_with_randomLabel, verbose = 0)
                    image = self.combine_images(generated_images)
                    image = image * 127.5 + 127.5
                    if not os.path.exists(self.path):
                        os.mkdir(self.path)
                    Image.fromarray(image.astype(np.uint8)).save(self.path + "_msdf_%04d_%04d.png" % (epoch, index))
                
                X = np.concatenate((r_img_11, f_img_11))
                y = [1] * BATCH_SIZE + [0] * BATCH_SIZE
                y = np.array(y)
                d_loss = self.discriminator.train_on_batch(X, y)
                
                noise = np.array([np.random.uniform(-1, 1, self.z_dim) for _ in range(BATCH_SIZE)])
                randomLabel_batch = np.random.uniform(-1, 1, BATCH_SIZE)  # randint(0, CLASS_NUM, BATCH_SIZE)
                randomLabel_batch_onehot = np.array([self.label2onehot(i) for i in randomLabel_batch])
                randomLabel_batch_image = np.array([self.label2images(i) for i in randomLabel_batch])
                g_loss = self.combined.train_on_batch([noise, randomLabel_batch_onehot, randomLabel_batch_image],
                                                      np.array([1] * BATCH_SIZE))
                print("epoch: %d, batch: %d, g_loss: %f, d_loss: %f" % (epoch, index, g_loss, d_loss[0]))
            
            self.g_loss_array[epoch] = g_loss
            self.d_loss_array[epoch] = d_loss[0]
            self.d_accuracy_array[epoch] = 100. * d_loss[1]
            
            self.generator.save_weights("generator.h5")
            self.discriminator.save_weights("discriminator.h5")


if __name__ == '__main__':
    gan = CGAN()
    gan.train()
