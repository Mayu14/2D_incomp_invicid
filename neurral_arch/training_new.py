# -*- coding: UTF-8 -*-
from keras.models import Sequential, model_from_json, Model
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import LeakyReLU, PReLU, Lambda, Input, Layer
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from keras.layers import Flatten, Concatenate, Reshape, BatchNormalization, Conv2D
from keras import regularizers
from keras.initializers import RandomNormal
from read_training_data import load_mixed_and_shuffle
from keras.optimizers import Adadelta
import keras.backend.tensorflow_backend as KTF
from keras.utils import plot_model
from keras.losses import mean_squared_error
from keras import backend as K
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from read_training_data import old_skip_angle, old_skip_wing, old_skip_unballanced, old_skip_clustering
from scatter_plot import make_scatter_plot
from sklearn.metrics import r2_score

from itertools import product
from pathlib import Path
from simple_regressions import error_print

def getNewestModel(model, dirname):
    from glob import glob
    target = os.path.join(dirname, '*')
    files = [(f, os.path.getmtime(f)) for f in glob(target)]
    if len(files) == 0:
        return model
    else:
        newestModel = sorted(files, key=lambda files: files[1])[-1]
        model.load_weights(newestModel[0])
        return model

class Swish(Layer):
    def __init__(self, beta=0.3, **kwargs):
        super(Swish, self).__init__(**kwargs)
        self.beta = K.cast_to_floatx(beta)
        
    def call(self, inputs):
        return K.sigmoid(self.beta * inputs) * inputs
    
    def get_config(self):
        config = {'beta': float(self.beta)}
        base_config = super(Swish, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    def compute_output_shape(self, input_shape):
        return input_shape

def huber_loss_mean(y_true, y_pred, clip_delta=1.0):
    def huber_loss(y_true, y_pred, clip_delta = 1.0):
        error = y_true - y_pred
        cond = tf.keras.backend.abs(error) < clip_delta
        
        squared_loss = 0.5 * tf.keras.backend.square(error)
        linear_loss = clip_delta * (tf.keras.backend.abs(error) - 0.5 * clip_delta)
        
        return tf.where(cond, squared_loss, linear_loss)
    
    return tf.keras.backend.mean(huber_loss(y_true, y_pred, clip_delta))

def mean_sqrt_error(y_true, y_pred):
    return tf.keras.backend.mean(tf.keras.backend.sqrt(tf.keras.backend.abs(y_true - y_pred)))

def mean_lp_error(y_true, y_pred, p=3):
    return tf.keras.backend.mean(tf.keras.backend.pow(tf.keras.backend.abs(y_true - y_pred), p))

def mean_log_error(y_true, y_pred):
    # return K.mean(K.log(tf.add(K.abs(y_true - y_pred), K.ones_like(y_true))))
    return K.mean(K.log(K.abs(y_true - y_pred) + K.ones_like(y_true)))
    # return tf.keras.backend.mean(tf.keras.backend.log())

class NNPotential(object):
    totalData = 200000
    input_dim = 201
    used_layer = 0
    debug = False#True # for CNN
    def warning_message(self):
        if self.debug:
            print("A T T E N T I O N ! \n D E B U G   M O D E \n")
    
    def __init__(self, n_samples, shapeExp, dataReductMethod="kmeans_norm_pca", dimReductMethod="pca", deepset=False, noAngle=False):
        self.warning_message()
        self.n_samples = n_samples
        self.skip_rate = int(self.totalData / self.n_samples)
        self.mode = shapeExp.lower()
        self.dataReductMethod = dataReductMethod
        self.dimReductMethod = dimReductMethod
        self.deepset = deepset
        self.noAngle = noAngle  # deepset without AoA
        if not deepset and noAngle:
            raise ValueError
        self.x_train, self.x_test, self.y_train, self.y_test, self.rescale_x, self.rescale_y = self.loadData()
        

    def loadData(self):
        def transformShapeData(data):
            def reduceAngle(x_data, rescale_x, updateScaler=False):
                from sklearn.preprocessing import StandardScaler
                x_data = rescale_x(x_data)
                n_sample = x_data.shape[0]
                s_data = x_data[:, 1:].reshape(n_sample, 2, -1)
                a_data = x_data[:, 0] * np.pi / 180.0
                rot = np.array([[np.cos(a_data), -np.sin(a_data)], [np.sin(a_data), np.cos(a_data)]])
                s_data = np.einsum("ijk,jli->ilk",s_data, rot).reshape(n_sample, -1)
                scalar = StandardScaler()
                s_data = scalar.fit_transform(s_data)
                if updateScaler:
                    return s_data, scalar.inverse_transform
                else:
                    return s_data, rescale_x

            x_train, x_test, y_train, y_test, rescale_x, rescale_y = data[0], data[1], data[2], data[3], data[4], data[5]
            if self.noAngle:
                x_test, rescale_x = reduceAngle(x_test, rescale_x, updateScaler = False)
                x_train, rescale_x = reduceAngle(x_train, rescale_x, updateScaler = True)
            return x_train, x_test, y_train, y_test, rescale_x, rescale_y
            
        if not self.debug:
            drm = self.dataReductMethod
            args = [self.mode, self.skip_rate]
            if self.skip_rate != 1:
                if ("wing" in drm):
                    args.append(40)
                    loadFunc = old_skip_wing
                elif ("angle" in drm):
                    args.append(40)
                    loadFunc = old_skip_angle
                elif("unbalance" in drm):
                    args.append(40)
                    loadFunc = old_skip_unballanced
                elif "kmeans" in drm:
                    args.append(self.dataReductMethod)
                    loadFunc = old_skip_clustering
                else:
                    raise ValueError
            else:
                return transformShapeData(load_mixed_and_shuffle(self.mode, self.n_samples))
            return transformShapeData(loadFunc(args[0], args[1], args[2]))
        else:
            return load_mixed_and_shuffle(mode = "msdf",n_samples = 200000,msdf_res = 16,
                                  useN5w2 = False, shuffle = False, __step = 1, __dimReduce = None, __test_size = 0.01,
                                  __dataReductMethod = None, __mix = False)

    def regressorMLP(self, layers, activator, bn=True, dr=0.25):
        inputs = Input(shape=(self.input_dim,), name="input_layer")
        reg = regularizers.l2(0.00001)
        rand_init = RandomNormal(stddev=0.02)

        if activator == PReLU:
            x = PReLU(shared_axes=[1])(inputs)
        else:
            x = activator()(inputs)

        for layer in layers:
            x = Dense(units=layer, activation=None, kernel_initializer=rand_init,
                      kernel_regularizer=reg, bias_regularizer=reg)(x)
            if bn:
                x = BatchNormalization(axis=-1)(x)
            if activator == PReLU:
                x = PReLU(shared_axes=[1])(x)
            else:
                x = activator()(x)

        if dr != 0.0:
            x = Dropout(dr)(x)
        prediction = Dense(units=1, activation=None,
                           kernel_initializer=RandomNormal(stddev=0.02),
                           kernel_regularizer=reg, bias_regularizer=reg, name="output_layer")(x)
        return Model(inputs=[inputs], outputs=[prediction])

    def regressorDS(self, layers, layers2, activator, bn=False, bn2=True, vars=3):
        max_points = 100
        pts_dim=2
        aoa_dim = 1
        if self.noAngle:
            aoa_dim = 0
        reg = regularizers.l2(0.00001)
        rand_init = RandomNormal(stddev=0.02)

        def dense_block(num_points, layers, bn, lastShape=None):
            inputs = Input(shape=(num_points,), name="input_layer")
            if activator == PReLU:
                x = PReLU(shared_axes=[1])(inputs)
            else:
                x = activator()(inputs)

            for layer in layers:
                x = Dense(units=layer, activation=None, kernel_initializer=rand_init,
                          kernel_regularizer=reg, bias_regularizer=reg)(x)
                if bn:
                    x = BatchNormalization(axis=-1)(x)

                if activator == PReLU:
                    x = PReLU(shared_axes=[1])(x)
                else:
                    x = activator()(x)

            if lastShape is None:
                outputs = Activation("linear")(x)
            else:
                outputs = Dense(units=lastShape, activation=None, kernel_initializer=rand_init,
                          kernel_regularizer=reg, bias_regularizer=reg)(x)
            return Model([inputs], [outputs])

        inputs = Input(shape=(aoa_dim + max_points * pts_dim,), name="input_layer")
        input_pts = Lambda(lambda x: x[:, aoa_dim:], name="points", output_shape=(max_points * pts_dim,))(inputs)

        # x = Embedding(max_points, pts_dim, mask_zero = True, trainable = False)(input_pts)
        all_pts = dense_block(max_points * pts_dim, layers, bn)(input_pts)
        Adder_xy = Lambda(lambda x: K.sum(x, axis = 1), name = "adderXY", output_shape = (1,))
        all_pts = Adder_xy(all_pts)
        x = Reshape((1,))(all_pts)
        
        if vars == 3:
            x_pts = Lambda(lambda x: x[:, :max_points], name = "x_pts", output_shape = (max_points,))(input_pts)
            y_pts = Lambda(lambda x: x[:, max_points:], name = "y_pts", output_shape = (max_points,))(input_pts)
            x_pts = dense_block(max_points, layers, bn)(x_pts)
            y_pts = dense_block(max_points, layers, bn)(y_pts)
    
            Adder_x = Lambda(lambda x: K.sum(x, axis=1), name="adderX", output_shape=(1,))
            Adder_y = Lambda(lambda x: K.sum(x, axis=1), name="adderY", output_shape=(1,))
            
            x_pts = Adder_x(x_pts)
            y_pts = Adder_y(y_pts)
            x_pts = Reshape((1,))(x_pts)
            y_pts = Reshape((1,))(y_pts)
        
        if not self.noAngle:
            input_aoas = Lambda(lambda x: x[:, 0], name = "aoas", output_shape = (1,))(inputs)
            input_aoas = Reshape((1,))(input_aoas)
            if vars == 3:
                x = Concatenate()([x_pts, y_pts, x, input_aoas])
            else:
                x = Concatenate()([x, input_aoas])
        else:
            if vars == 3:
                x = Concatenate()([x_pts, y_pts, x])

        lift = dense_block(num_points=vars+aoa_dim, layers=layers2, bn=bn2, lastShape=1)(x)
        return Model(inputs=[inputs], outputs=[lift])

    # def regressorCNN(self, layers, activator, bn, layers2, bn2, input_shape=16**2):
    def regressorCNN(self, img_size=16):
        num_classes = 1
        input_shape = (img_size, img_size, 3)
        __s = Input(shape = input_shape, name = "input")
        reg = regularizers.l2(0.00001)
        rand_init = RandomNormal(stddev = 0.02)
        # 1st conv
        __h = Conv2D(filters = 64, kernel_size = 3, strides = 2, padding = 'same',
                     activation = None, kernel_initializer = rand_init,
                     kernel_regularizer = reg, bias_regularizer = reg)(__s)
        __h = PReLU(shared_axes = [1, 2, 3])(__h)
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
    
        return Model(__s, __y)

    def gridSearch(self, single):
        def replaceALL(name):
            return name.replace("<", "").replace(">", "").replace("(", "").replace(")", "").replace("'", "").replace(
                ", ", "_").replace("class keras.layers.advanced_activations.", "")

        def path_generator(name):
            json_name ="nnModel\\{0}_model.json".format(name)
            weight_name = "nnModel\\{0}_weight.h5".format(name)
            return json_name, weight_name, name

        if self.debug:
            names = path_generator("CNNii200000")
            regressor = self.regressorCNN()
            yield regressor, names
        else:
            """
            model = Sequential()
            model.add(Dense(units = 2, input_dim = 201))
            model.add(LeakyReLU())
            model.add(Dense(units = 16))
            model.add(LeakyReLU())
            model.add(Dense(units = 1))
            yield model, path_generator("{0}_2_16".format(self.mode))
            exit()
            """
            activators = [PReLU]#[PReLU]#[PReLU, LeakyReLU]
            bns = [True, False]
            drs = [0.0, 0.125, 0.25, 0.375]
            layer_num = [3]#[2, 3, 4]
            units_num = [25, 51, 101, 151]
            layers = []
            for layer in layer_num:
                layers.extend(list(product(units_num, repeat=layer)))

            if not self.deepset:
                #"""
                if single:
                    layers = [(151, 101, 51)]
                    bns = [True]
                    drs = [0.25]
                    # """
                for layer, activator, bn, dr in product(layers, activators, bns, drs):
                    if False:#self.n_samples == 5000 and "kmeans" in self.dataReductMethod and self.mode == "circum":
                        name = replaceALL("{0}_{1}_{2}_{3}_{4}".format(self.mode, layer, activator, bn, dr))
                    else:
                        name = replaceALL("{0}_{1}_{2}_{3}_{4}_{5}_{6}_{7}_{8}".format(self.mode, layer, activator, bn, dr, self.n_samples, self.dimReductMethod, self.dataReductMethod, loss))
                    names = path_generator(name)
                    regressor = self.regressorMLP(layer, activator, bn, dr)
                    yield regressor, names
            else:
                layers = [(25, 151)]
                layers2 = [(51, 151)]
                bns1 = [False]
                bns2 = [True]
                vars = 3
                tail = ""
                if vars != 3:
                    tail += "_" + str(vars)
                if self.noAngle:
                    tail += "_NoA"
                for layer1, layer2, activator, bn1, bn2  in product(layers, layers2, activators, bns1, bns2):
                    name = replaceALL("DS_{0}_{1}_{2}_{3}_{4}_{5}{6}".format(self.mode, layer1, activator, bn1, layer2, bn2, tail))
                    names = path_generator(name)
                    regressor = self.regressorDS(layers = layer1, layers2 = layer2, activator = activator, bn = bn1, bn2=bn2, vars=vars)
                    yield regressor, names
        
            

    def train(self, single=False):
        self.warning_message()
        config = tf.ConfigProto(gpu_options = tf.GPUOptions(allow_growth = True))
        KTF.set_session(tf.Session(config=config))

        for model, names in self.gridSearch(single):
            print(names[2])
            if False:#Path(names[0]).exists() and Path(names[1]).exists():
                print("model exists")
                model = model_from_json(open(names[0]).read())
                model.load_weights(names[1])
                model.summary()

            else:
                plot_model(model, to_file="arch\\{0}.png".format(names[2]), show_shapes=True, show_layer_names=True)
                model.summary()
                epoch = 5000

                es_cb = EarlyStopping(monitor='val_loss', patience=500, verbose=0, mode='auto')
                if not self.deepset:
                    baseSaveDir = "checkpoint{0}\\".format(self.mode)
                else:
                    baseSaveDir = "checkpointDS{0}\\".format(self.mode)

                chkpt = baseSaveDir + 'iiMLPGS' + '_.{epoch:02d}-{val_loss:.2f}.hdf5'
                cp_cb = ModelCheckpoint(filepath=chkpt, monitor="val_loss", verbose=0, save_best_only=True,
                                        mode='auto')
                # model.compile(loss="mean_squared_error", optimizer=Adadelta(), metrics=["mae"])
                losses = [huber_loss_mean]#[mean_squared_error, mean_sqrt_error, mean_log_error]
                step = 1
                epoch = int(epoch / step)
                for i in range(step):
                    for loss in losses:
                        model.compile(loss=loss, optimizer=Adadelta(), metrics=["mae"])
                        
                    
                        if not self.debug:
                            batch_size = max(25000, min(int(self.n_samples / 2), 65000))
                        else:
                            batch_size = 2500
        
                        history = model.fit(x=self.x_train, y=self.y_train,
                                            batch_size=batch_size, epochs=epoch,
                                            validation_split=0.3, callbacks=[cp_cb, es_cb])
                        model = getNewestModel(model, baseSaveDir)
                    
                print(names[0])
                open(names[0], 'w').write(model.to_json())
                model.save_weights(names[1])

                fig = plt.figure()
                ax = fig.add_subplot(1, 1, 1)
                ax.plot(history.history['loss'], marker=".", label='loss')
                ax.plot(history.history['val_loss'], marker='.', label='val_loss')
                ax.set_title('model loss')
                ax.grid(True)
                ax.set_xlabel('epoch')
                ax.set_ylabel('loss')
                ax.legend(loc='best')
                plt.savefig("history\\{0[2]}.png".format(names))
                plt.close()

            y_pred = self.rescale_y(model.predict(self.x_test))
            y_test = self.rescale_y(self.y_test)
            r2_train = r2_score(self.y_train, model.predict(self.x_train))
            r2_test = r2_score(y_test, y_pred)
            log = error_print(y_pred, y_test, csv=True).replace("\n", "")
            csvline = "{0},{1},{2},{3}\n".format(r2_train, r2_test, log, names[2].replace("_",","))
            print(r2_train, r2_test)
            if self.debug:
                fname = names[2] + ".csv"
            else:
                if self.deepset:
                    fname = "newModelDSlog.csv"
                    # fname = "newModelMLPlog5000shape.csv"
                else:
                    fname = "newModelMLPlog6433.csv"
                    
            with open("nnModel\\" + fname, "a") as f:
                f.write(csvline)

            make_scatter_plot(y_test, y_pred, "CL(Exact)", "CL(Predict)",
                              path="D:\\Toyota\\github\\2D_incomp_invicid\\neurral_arch\\nnModel\\", fname=names[2])
            K.clear_session()

def main():
    nnp = NNPotential(n_samples=200000, shapeExp="circum", dimReductMethod=None, deepset=True, noAngle = True)
    from read_training_data import load_noaoa
    nnp.x_train, nnp.x_test, nnp.y_train, nnp.y_test, nnp.rescale_x, nnp.rescale_y = load_noaoa()
    nnp.train()
    exit()
    shapes = ["circum", "equidistant", "crowd", "fourier"]
    for shape in shapes:
        nnp = NNPotential(n_samples=200000, shapeExp=shape, dimReductMethod=None, deepset=False, noAngle = False)
        nnp.train()

def singleTrain(n_samples=5000):
    shapes = ["circum"]#["circum", "equidistant", "crowd", "fourier"]
    dataReductMethods = ["angle", "wing", "unbalanced", "kmeans_norm_pca"]
    drMethod = None#"kmeans_norm_pca"#None#"kmeans_norm_pca"
    # for drMethod in dataReductMethods:
    # for i in range(10):
    for shape in shapes:
        dimReduce = None#""pca" + str(201 - 20 * (i+1))
        nnp = NNPotential(n_samples = n_samples, shapeExp = shape, dataReductMethod = drMethod, deepset = False,
                          noAngle = False)
        nnp.train(single = True)
        
def simpleTrain(n_samples=9000):
    shapes = ["circum"]
    drMethod = "kmeans_norm_pca"
    for shape in shapes:
        dimReduce = None#""pca" + str(201 - 20 * (i+1))
        nnp = NNPotential(n_samples = n_samples, shapeExp = shape, dataReductMethod = drMethod, deepset = False,
                          noAngle = False)

        nnp.train(single = True)


if __name__ == '__main__':
    loss = mean_squared_error
    np.random.seed(1)
    tf.set_random_seed(1)
    #"""
    singleTrain(200000)
    exit()
    singleTrain(25000)
    exit()
    # main()
    simpleTrain(25000)
    """
    # for i in range(10):
        # singleTrain(100*(i+1))
    for i in range(235):
        singleTrain(30+2*(i+1))
    #"""