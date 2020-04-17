# coding: utf-8
import math, json, sys

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from keras.backend import tensorflow_backend as KTF
from keras.models import Model, model_from_json
from keras.layers import Input, Dense, Conv2D, Conv2DTranspose, Flatten, Reshape, Dropout
from keras.layers import BatchNormalization, PReLU, Activation
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy, mean_squared_error
from keras.metrics import categorical_accuracy
from keras.utils import plot_model
from keras.utils.np_utils import to_categorical
from keras.initializers import RandomNormal
from keras import regularizers
from sklearn.metrics import r2_score

from read_training_data import load_mixed_and_shuffle

#path_json = 'cnn_modeldefs.json'
path_json = 'cnn_modeldefs_200000.json'
#path_weight = 'cnn_modelweights.hdf5'
path_weight = 'cnn_modelweights_200000.hdf5'

batch_size = 2500
num_classes = 1 # only lift
epochs = 100

img_size = 16
img_rows, img_cols, img_channels = img_size, img_size, 3
input_shape = (img_rows, img_cols, img_channels)

def regressor_model():
    __s = Input(shape = input_shape, name="input")
    reg = regularizers.l2(0.00001)
    rand_init = RandomNormal(stddev = 0.02)
    # 1st conv
    __h = Conv2D(filters=64,  kernel_size=3, strides=2, padding='same',
                 activation=None, kernel_initializer=rand_init,
                 kernel_regularizer=reg, bias_regularizer=reg)(__s)
    __h = PReLU(shared_axes = [1,2,3])(__h)
    # 2nd conv
    __h = Conv2D(filters = 128, kernel_size = 3, strides = 2, padding = 'same',
                 activation = None, kernel_initializer = rand_init,
                 kernel_regularizer = reg, bias_regularizer = reg)(__h)
    __h = BatchNormalization(axis = -1)(__h)
    __h = PReLU(shared_axes = [1, 2, 3])(__h)
    # 3nd conv
    __h = Conv2D(filters = 256, kernel_size = 4, strides = 1, padding = 'valid',
                 activation = None, kernel_initializer = rand_init,
                 kernel_regularizer = reg, bias_regularizer = reg)(__h)
    __h = BatchNormalization(axis = -1)(__h)
    __h = PReLU(shared_axes = [1, 2, 3])(__h)
    # 4th conv
    __h = Conv2D(filters = 512, kernel_size = 3, strides = 2, padding = 'same',
                 activation = None, kernel_initializer = rand_init,
                 kernel_regularizer = reg, bias_regularizer = reg)(__h)
    __h = BatchNormalization(axis = -1)(__h)
    __h = PReLU(shared_axes = [1, 2, 3])(__h)
    __h = Dropout(0.25)(__h)
    # 5th fc
    __h = Flatten()(__h)
    __h = Dense(units = 256, activation = None, kernel_initializer = RandomNormal(stddev = 0.02),
                kernel_regularizer = reg, bias_regularizer = reg)(__h)
    __h = Activation("relu")(__h)
    __h = Dropout(0.5)(__h)
    __y = Dense(units = num_classes, activation = None, kernel_initializer = RandomNormal(stddev = 0.02),
                kernel_regularizer = reg, bias_regularizer = reg)(__h)
    
    return Model(__s, __y, name="regressor")

def compile(model):
    reg_optimizer = Adam(lr=0.0002, beta_1=0.5)
    model.compile(optimizer=reg_optimizer,
                  loss=mean_squared_error,
                  metrics=["mae"])
    return model

def compiled_regressor():
    reg = regressor_model()
    return compile(reg)

if __name__ == '__main__':
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    session = tf.Session(config=config)
    KTF.set_session(session)
    np.random.seed(1)
    tf.set_random_seed(1)

    load_weight = False

    # x_train, x_test, y_train, y_test, rescale_x, rescale_y = load_mixed_and_shuffle(mode = "msdf", msdf_res = img_size, shuffle=False, __step=8)
    # x_train, x_test, y_train, y_test, rescale_x, rescale_y = load_mixed_and_shuffle(mode="msdf", msdf_res=img_size, n_samples=400000, useN5w2=True)
    x_train, x_test, y_train, y_test, rescale_x, rescale_y =load_mixed_and_shuffle(mode = "msdf",n_samples = 200000,msdf_res = img_size,
                                  useN5w2 = False,
                                  shuffle = False,
                                  __step = 1,
                                  __dimReduce = None,
                                  __test_size = 0.01,
                                  __dataReductMethod = None,
                                  __mix = False)

    if not load_weight:
        model = compiled_regressor()
        history = model.fit(x=x_train, y=y_train,
                            batch_size = batch_size,
                            epochs = epochs,
                            validation_split = 0.05)
        with open(path_json, 'w') as fout:
            model_json_str = model.to_json(indent=4)
            fout.write(model_json_str)
            ## model weights
        model.save_weights(path_weight)
        ## training history
        # json.dump(history, open('cnn_training_history.json', 'w'))
    else:
        model = model_from_json(open(path_json, "r").read())
        model.load_weights(path_weight)
        model = compile(model)


    from simple_regressions import error_print

    y_pred = rescale_y(model.predict(x_test))
    y_test_r = rescale_y(y_test)
    r2_train = r2_score(y_train, model.predict(x_train))
    r2_test = r2_score(y_test_r, y_pred)
    log = error_print(y_pred, y_test_r, csv = True).replace("\n", "")
    csvline = "{0},{1},{2},{3}\n".format(r2_train, r2_test, log, None)
    with open("cnn200000.csv") as f:
        f.write(csvline)
        print(csvline)
        
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    r2_train = r2_score(y_train, model.predict(x_train))
    r2_test = r2_score(y_test, model.predict(x_test))
    print(r2_train, r2_test)

    if not load_weight:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(history.history['loss'], marker = ".", label = 'loss')
        ax.plot(history.history['val_loss'], marker = '.', label = 'val_loss')
        ax.set_title('model loss')
        ax.grid(True)
        ax.set_xlabel('epoch')
        ax.set_ylabel('loss')
        ax.legend(loc = 'best')
        plt.show()


