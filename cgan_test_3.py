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

from matplotlib import pyplot as plt

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from msdf_shader_02_copy import test_3d_rendering
import tensorflow as tf
import keras.backend as KTF
from arc_length_transformer import get_decode_img_from_bnxy, decode_into_cplx
from dvm_beta import zu2circ

BATCH_SIZE = 6250
NUM_EPOCH = 5000
CLASS_NUM = 1  # 1  # 10    # lift only
IMG_TOTAL_NUM = 100
SHAPE_REPRESENTATION = "modified_fourier"
CASE_HEADER = "mdfr_CL20_"


def eval(y_obs, y_pred):
    r2 = r2_score(y_obs, y_pred)
    rmse = mean_squared_error(y_obs, y_pred)
    mae = mean_absolute_error(y_obs, y_pred)
    return r2, rmse, mae

def yyplot(y_obs, y_pred, imshow=True):
    r2, rmse, mae = eval(y_obs, y_pred)
    label = "R2:" + str(r2) + " RMSE:" + str(rmse) + " MAE:" + str(mae) + "\nRMSE/MAE = " + str(rmse / mae)
    yvalues = np.concatenate([y_obs.flatten(), y_pred.flatten()])
    ymin, ymax, yrange = np.amin(yvalues), np.amax(yvalues), np.ptp(yvalues)
    fig = plt.figure(figsize=(8, 8))
    plt.scatter(y_obs, y_pred)

    plt.plot([ymin - yrange * 0.01, ymax + yrange * 0.01], [ymin - yrange * 0.01, ymax + yrange * 0.01], label=label)
    plt.legend(fontsize=18, loc="upper left")
    plt.xlim(ymin - yrange * 0.01, ymax + yrange * 0.01)
    plt.ylim(ymin - yrange * 0.01, ymax + yrange * 0.01)
    plt.xlabel('y_observed', fontsize=24)
    plt.ylabel('y_predicted', fontsize=24)
    plt.title('Observed-Predicted Plot', fontsize=24)
    plt.tick_params(labelsize=16)
    if imshow:
        plt.show()
    return r2, rmse, mae, fig


def lift2label(lift, yMin=-2.3378143865772794, yMax=2.4768136879522054, eps=0.001):
    liftRatio = lift - yMin / (yMax + eps - yMin)
    return min(max(int(liftRatio * CLASS_NUM), 0), CLASS_NUM - 1)


class CGAN():
    def __init__(self):
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.8
        KTF.set_session(tf.Session(config=config))
        # self.path = '/volumes/data/dataset/gan/MNIST/cgan/cgan_generated_images/'
        self.path = 'images/'
        # mnistデータ用の入力データサイズ
        self.img_rows = 201
        self.img_cols = 1
        self.channels = 1
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

        self.discriminator_optimizer = Adam(lr=1e-5, beta_1=0.1)
        self.combined_optimizer = Adam(lr=.8e-4, beta_1=0.5)

        # discriminatorモデル
        self.discriminator = self.build_discriminator()
        if os.path.exists(CASE_HEADER + SHAPE_REPRESENTATION + "discriminator.h5"):
            self.discriminator.load_weights(CASE_HEADER + SHAPE_REPRESENTATION + "discriminator.h5")
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=self.discriminator_optimizer,
                                   metrics=['accuracy'])

        # for layer in self.discriminator.layers:
        #    layer.trainable = False
        self.discriminator.trainable = False

        # Generatorモデル
        self.generator = self.build_generator()
        if os.path.exists(CASE_HEADER + SHAPE_REPRESENTATION + "generator.h5"):
            self.generator.load_weights(CASE_HEADER + SHAPE_REPRESENTATION + "generator.h5")
        # generatorは単体で学習しないのでコンパイルは必要ない
        # self.generator.compile(loss='binary_crossentropy', optimizer=optimizer)

        self.combined = self.build_combined()
        # self.combined = self.build_combined2()
        self.combined.compile(loss='binary_crossentropy',
                              optimizer=self.combined_optimizer,
                              metrics=['accuracy'])

    def build_generator(self):
        inputs = Input(shape=(self.z_dim + CLASS_NUM,))
        x = Dense(units=128 * 4 * 4)(inputs)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        """
        x = Reshape((4, 4, 128), input_shape = (128 * 4 * 4,))(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(64, (5, 5), padding = 'same')(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(3, (5, 5), padding = 'same')(x)
        """
        x = Dense(units=128 * 6)(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Dense(units=self.img_rows)(x)
        x = BatchNormalization()(x)
        x = Reshape(self.img_shape, input_shape=(self.img_rows,))(x)
        generated = Activation("tanh")(x)
        model = Model(inputs=[inputs], outputs=[generated])
        return model

    def build_discriminator(self):
        inputs = Input(shape=(self.img_rows, self.img_cols, (self.channels + CLASS_NUM)))
        x = Flatten()(inputs)
        """
        x = Conv2D(64, (5, 5), strides = (2, 2), padding = 'same')(inputs)
        x = LeakyReLU(alpha = 0.2)(x)
        x = Conv2D(128, (5, 5), strides = (2, 2))(x)
        x = LeakyReLU(alpha = 0.2)(x)
        x = Flatten()(x)
        """
        x = Dense(units=512)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dense(units=1024)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dense(units=256)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(0.5)(x)
        x = Dense(units=1)(x)
        discriminated = Activation('tanh')(x)
        model = Model(inputs=[inputs], outputs=[discriminated])
        return model

    def build_combined(self):
        z = Input(shape=(self.z_dim,))
        y = Input(shape=(CLASS_NUM,))
        img_10 = Input(shape=(self.img_rows, self.img_cols, CLASS_NUM,))
        z_y = Concatenate()([z, y])

        img = self.generator(z_y)
        img_11 = Concatenate(axis=3)([img, img_10])
        self.discriminator.trainable = False
        valid = self.discriminator(img_11)
        model = Model(inputs=[z, y, img_10], outputs=valid)
        return model

    def combine_images(self, generated_images):
        total = generated_images.shape[0]
        cols = int(math.sqrt(total))
        rows = int(math.ceil(float(total) / cols))
        # WIDTH, HEIGHT = generated_images.shape[1:3]
        WIDTH, HEIGHT = 100, 100
        combined_image = np.zeros((HEIGHT * rows, WIDTH * cols, 3),
                                  dtype=generated_images.dtype)

        if SHAPE_REPRESENTATION == "modified_fourier":
            for index, image in enumerate(generated_images):
                i = int(index / cols)
                j = index % cols

                image = get_decode_img_from_bnxy(image[:, 0, 0][1:], from_concat_bn_xy=True, angle=image[:, 0, 0][0],
                                                 imgWidth=WIDTH, imgHeight=HEIGHT, deg=True)
                # pilImg = Image.fromarray(np.uint8(image * 127.5 + 127.5)[:, :, ::-1]).transpose(Image.FLIP_TOP_BOTTOM)  # RBG -> RGB
                # image = test_3d_rendering(pilImg, render_size = 1000, fromObj = True, imgshow = True)
                combined_image[WIDTH * i:WIDTH * (i + 1), HEIGHT * j:HEIGHT * (j + 1), :] = image[:, :, :]
        return combined_image

    def evalGenShape(self, x_output, y_pred):
        samples = x_output.shape[0]
        y_exact = np.zeros(samples)
        if SHAPE_REPRESENTATION == "modified_fourier":
            for index, shape_vector in enumerate(x_output):
                image = get_decode_img_from_bnxy(shape_vector[1:], from_concat_bn_xy=True,
                                                 angle=shape_vector[0], imgWidth=100, imgHeight=100,
                                                 deg=True)
                z_aoa0 = decode_into_cplx(shape_vector[1:], from_concat_bn_xy=True, angle=0.0, deg=True)
                lift, y_exact[index] = zu2circ(z_aoa0, v_in=1.0, aoa=shape_vector[0], deg=True)
                print(lift, y_exact[index])
                exit()
                print(shape_vector[0], y_pred[index], y_exact[index])

            r2, rmse, mae, fig = yyplot(y_exact, y_pred)
            print(r2, rmse, mae)
            return r2, rmse, mae

    def label2images(self, label):
        images = np.zeros((self.img_rows, self.img_cols, CLASS_NUM))
        # label = lift2label(label)  # add
        images[:, :, 0] += label
        return images

    def label2onehot(self, label):
        onehot = np.zeros(CLASS_NUM)
        # label = lift2label(label)
        # onehot[label] = 1
        onehot[0] = label
        return onehot

    def label2output(self, label):
        onehot = np.zeros(CLASS_NUM)
        # label = lift2label(label)
        # onehot[label] = 1
        onehot[0] = label
        return onehot

    def train(self):
        source = "G:\\Toyota\\Data\\Incompressible_Invicid\\training_data\\"
        fname_shape_train = "NACA4\\shape_modified_fourier_5000_odd.csv"
        fname_shape_test = "NACA5\\shape_modified_fourier_all.csv"
        fname_lift_train = "NACA4\\s0000_e5000_a040_odd.csv"
        # fname_lift_test = "NACA5\\s11001_e65199_a040.csv"
        fname_lift_test = "NACA5\\s21001_e25199_a040.csv"
        rr = 1
        sr = 1
        s_odd = 0
        s_skiptype = True
        unballanced = False
        from read_training_data import read_csv_type3
        X_train, y_train = read_csv_type3(source, fname_lift_train, fname_shape_train, shape_odd=s_odd,
                                          read_rate=rr, skip_rate=sr, skip_angle=s_skiptype,
                                          unbalance=unballanced)

        scalar_y = StandardScaler()
        y_train = scalar_y.fit_transform(y_train)
        scalar = StandardScaler()
        X_train = scalar.fit_transform(X_train)
        X_train = X_train.reshape(-1, self.img_rows, self.img_cols, self.channels)[::8, :, :, :]

        discriminator = self.build_generator()
        d_opt = Adam(lr=1e-5, beta_1=0.1)
        discriminator.compile(loss='binary_crossentropy', optimizer=d_opt, metrics=['accuracy'])

        g_opt = Adam(lr=8e-4, beta_1=0.5)
        self.combined.compile(loss='binary_crossentropy', optimizer=g_opt)

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
                # noise_y_int = np.random.uniform(-1, 1, BATCH_SIZE)
                noise_y_int = np.random.normal(0, 1, BATCH_SIZE)
                # noise_y_int = np.random.uniform(-2.3378143865772794, 2.4768136879522054, BATCH_SIZE)
                noise_y = np.array(np.array([self.label2onehot(i) for i in noise_y_int]))
                noise_z_y = np.concatenate((noise_z, noise_y), axis=1)
                f_img = self.generator.predict(noise_z_y, verbose=0)
                f_img_10 = np.array([self.label2images(i) for i in noise_y_int])
                f_img_11 = np.concatenate((f_img, f_img_10), axis=3)

                r_img = X_train[index * BATCH_SIZE:(index + 1) * BATCH_SIZE]
                label_batch = y_train[index * BATCH_SIZE:(index + 1) * BATCH_SIZE]
                r_img_10 = np.array([self.label2images(i) for i in label_batch])

                r_img_11 = np.concatenate((r_img, r_img_10), axis=3)

                if epoch % 100 == 0:
                    noise = np.array([np.random.uniform(-1, 1, self.z_dim) for _ in range(IMG_TOTAL_NUM)])
                    randomLabel_batch = np.random.uniform(-1, 1, IMG_TOTAL_NUM)  # arrange(IMG_TOTAL_NUM)%10
                    randomLabel_batch_onehot = np.array([self.label2onehot(i) for i in randomLabel_batch])

                    noise_with_randomLabel = np.concatenate((noise, randomLabel_batch_onehot), axis=1)
                    generated_images = self.generator.predict(noise_with_randomLabel, verbose=0)
                    generated_images = scalar.inverse_transform(generated_images.reshape(IMG_TOTAL_NUM, -1))
                    y_pred = scalar_y.inverse_transform(randomLabel_batch_onehot)
                    if epoch > 1000:
                        self.evalGenShape(generated_images, y_pred)
                    image = self.combine_images(generated_images.reshape(IMG_TOTAL_NUM, -1, 1, 1))

                    # image = image * 127.5 + 127.5
                    if not os.path.exists(self.path):
                        os.mkdir(self.path)
                    Image.fromarray(image.astype(np.uint8)).save(
                        self.path + CASE_HEADER + "%04d_%04d.png" % (epoch, index))

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

            self.generator.save_weights(CASE_HEADER + SHAPE_REPRESENTATION + "generator.h5")
            self.discriminator.save_weights(CASE_HEADER + SHAPE_REPRESENTATION + "discriminator.h5")


if __name__ == '__main__':
    gan = CGAN()
    gan.train()

