# -*- coding: UTF-8 -*-
# 単純な関数のデータを与えて元のデータを予測させてみる
# 学習用データが(TesraK80の)メモリに乗らないため,Generatorを使ってバッチごとにデータをロードさせる感じで
# 20万件のデータを200件ずつ取り出す
from keras.models import Sequential, model_from_json, Model
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import LeakyReLU, PReLU, Lambda, Input
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from keras.layers import Flatten, Concatenate, Reshape, BatchNormalization
from keras import regularizers
from keras.initializers import RandomNormal
from read_training_data import load_mixed_and_shuffle
from keras.losses import mean_squared_error
from keras.optimizers import Adadelta
import keras.backend.tensorflow_backend as KTF
from keras import backend as K
from keras.utils import plot_model

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from read_training_data import read_csv_type3
from scatter_plot import make_scatter_plot
from sklearn.metrics import r2_score
import csv
from sklearn.preprocessing import StandardScaler
from dataset_reduction_inviscid import data_reduction


def getNewestModel(model, dirname):
    from glob import glob
    target = os.path.join(dirname, '*')
    files = [(f, os.path.getmtime(f)) for f in glob(target)]
    if len(files) == 0:
        return model
    else:
        newestModel = sorted(files, key = lambda files: files[1])[-1]
        model.load_weights(newestModel[0])
        return model


def batch_iter(data, labels, batch_size, shuffle = True):
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    
    def data_generator():
        data_size = len(data)
        while True:
            # Shuffle the data at each epoch
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(data_size))
                shuffled_data = data[shuffle_indices]
                shuffled_labels = labels[shuffle_indices]
            else:
                shuffled_data = data
                shuffled_labels = labels
            
            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)
                X = shuffled_data[start_index: end_index]
                y = shuffled_labels[start_index: end_index]
                yield X, y
    
    return num_batches_per_epoch, data_generator()


def save_my_log(source, case_number, fname_lift_train, fname_shape_train, model_sum):
    with open(source + str(case_number).zfill(4) + "_log.txt", "w") as f:
        f.write("case number :" + str(case_number).zfill((3)) + "\n")
        f.write("training_data of Lift :" + fname_lift_train + "\n")
        f.write("training_data of Shape :" + fname_shape_train + "\n")
        f.write("model summary" + "\n")
        f.write(str(model_sum) + "\n")


def get_case_number(source, env, case_number):
    flag = 0
    source = source + "learned\\"
    if env == "Colab":
        source = source.replace("\\", "/")
        case_number += 10000
    while flag == 0:
        if os.path.exists(source + str(case_number).zfill(5) + "_mlp_model_.json"):
            case_number += 1
        else:
            flag = 1
    return str(case_number).zfill(5)


# case_numberから何のデータだったか思い出せない問題が起きたのでファイル名の命名規則を変更する
# (形状)_(データ数)とする
def get_case_number_beta(case_number, rr, sr, skiptype, shape_data = 200, total_data = 200000, reductioning = False,
                         inputNormalize = False,
                         unballanced = False, outputNormalize = False):
    if int(case_number) / 1000 == 0:
        head = "fourierSr"
    elif int(case_number) / 1000 == 1:
        head = "equidistant"
    elif int(case_number) / 1000 == 2:
        head = "concertrate"
    elif int(case_number) / 1000 == 3:
        head = "modFrSr"
    elif int(case_number) / 1000 == 4:
        head = "circum"
    else:
        print("case number error")
        exit()
    mid1 = str(int(total_data / sr))
    if skiptype:
        mid2 = "less_angle"
    else:
        mid2 = "less_shape"
    tail = str(int(shape_data / rr))
    
    if reductioning:
        tail += "_reduct"
    if inputNormalize:
        tail += "_inNorm"
    if unballanced:
        tail += "_unb"
    if outputNormalize:
        tail += "_outNorm"
    
    return head + "_" + mid1 + "_" + mid2 + "_" + tail


def main(fname_lift_train, fname_shape_train, fname_lift_test, fname_shape_test, case_number, case_type = 3,
         env = "Lab"):
    # r_rate = [1, 2, 4, 8]
    r_rate = [1]
    # s_rate = [2, 4, 8]
    s_rate = [1]
    # s_skiptype = [True, False]
    s_skiptype = False
    unballanced = False
    # r_rate = [1, 2]
    # r_rate = [4, 8]
    # r_rate = [16, 32]
    # r_rate = [64, 160]
    reductioning = False
    inputNormalization = True
    outputNormalization = True
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.7
    KTF.set_session(tf.Session(config = config))
    for sr in s_rate:
        for rr in r_rate:
            if rr == 1:
                s_odd = 0  # 全部読みだす
            elif fname_shape_train.find("fourier") != -1:
                s_odd = 3  # 前方から読み出す(fourier用)
            else:
                s_odd = 4  # 全体にわたって等間隔に読み出す(equidistant, dense用)
            
            old_session = KTF.get_session()
            graph = tf.get_default_graph()
            with tf.Graph().as_default():
                source = "Incompressible_Invicid\\training_data\\"
                if env == "Lab":
                    source = "G:\\Toyota\\Data\\" + source
                    # case_num = get_case_number(source, env, case_number)
                    case_num = get_case_number_beta(case_number, rr, sr, s_skiptype, reductioning = reductioning,
                                                    inputNormalize = inputNormalization, unballanced = unballanced,
                                                    outputNormalize = outputNormalization)
                    
                    log_name = "learned\\" + case_num + "_tb_log.hdf5"
                    json_name = "learned\\" + case_num + "_mlp_model_.json"
                    weight_name = "learned\\" + case_num + "_mlp_weight.h5"
                elif env == "Colab":
                    source = "/content/drive/Colab Notebooks/" + source.replace("\\", "/")
                    case_num = get_case_number(source, env, case_number)
                    log_name = "learned/" + case_num + "_log.hdf5"
                    json_name = "learned/" + case_num + "_mlp_model_.json"
                    weight_name = "learned/" + case_num + "_mlp_weight.h5"
                
                if case_type == 3:
                    X_train, y_train = read_csv_type3(source, fname_lift_train, fname_shape_train, shape_odd = s_odd,
                                                      read_rate = rr, skip_rate = sr, skip_angle = s_skiptype,
                                                      unbalance = unballanced)
                    x_test, y_test = read_csv_type3(source, fname_lift_test, fname_shape_test, shape_odd = s_odd,
                                                    read_rate = rr)
                
                if (sr == 1) and (reductioning):
                    X_train, y_train = data_reduction(X_train, y_train, reduction_target = 25000, preprocess = "None")
                
                if inputNormalization:
                    scalar = StandardScaler()
                    scalar.fit(X_train)
                    X_train = scalar.transform(X_train)
                    x_test = scalar.transform(x_test)
                
                if outputNormalization:
                    scalar2 = StandardScaler()
                    scalar2.fit(y_train)
                    y_train = scalar2.transform(y_train)
                    y_test = scalar2.transform(y_test)
                
                session = tf.Session('')
                KTF.set_session(session)
                print(source + json_name)
                if True:  # not os.path.exists(source + json_name):
                    KTF.set_learning_phase(1)
                    useSequential = False
                    input_vector_dim = X_train.shape[1]
                    if useSequential:
                        model = Sequential()
                        with tf.name_scope("inference") as scope:
                            model.add(Dense(units = 2, input_dim = input_vector_dim))
                            model.add(LeakyReLU())
                            model.add(Dense(units = 16))
                            model.add(LeakyReLU())
                            model.add(Dense(units = 1))
                        x = Dense(units = 101)(x)
                        x = LeakyReLU()(x)
                        x = Dense(units = 17)(x)
                        x = LeakyReLU()(x)
                        prediction = Dense(units = 1, name = "output_layer")(x)
                        model = Model(inputs = [inputs], outputs = [prediction])
                    
                    plot_model(model, to_file = "test.png", show_shapes = True, show_layer_names = True)
                    model.summary()
                    
                    save_my_log(source, case_number, fname_lift_train, fname_shape_train, model.summary())
                    epoch = 2000
                else:
                    model = model_from_json(open(source + json_name).read())
                    model.load_weights(source + weight_name)
                    model.summary()
                    epoch = 1
                es_cb = EarlyStopping(monitor = 'val_loss', patience = 500, verbose = 0, mode = 'auto')
                tb_cb = TensorBoard(log_dir = source + log_name, histogram_freq = 0, write_grads = True)
                baseSaveDir = source + "checkpoints\\"
                
                chkpt = baseSaveDir + 'incompMLP_.{epoch:02d}-{val_loss:.2f}.hdf5'
                cp_cb = ModelCheckpoint(filepath = chkpt, monitor = "val_loss", verbose = 0, save_best_only = True,
                                        mode = 'auto')
                model.compile(loss = "mean_squared_error",
                              optimizer = Adadelta(),
                              metrics = ["mae"])
                
                # batch_size = 500
                # train_steps, train_batches = batch_iter(X_train, y_train, batch_size)
                # valid_steps, valid_batches = batch_iter(x_test, y_test, batch_size)
                # """
                history = model.fit(x = X_train, y = y_train,
                                    batch_size = 16384, epochs = epoch,
                                    validation_split = 0.1, callbacks = [tb_cb, cp_cb, es_cb])
                model = getNewestModel(model, baseSaveDir)
                # """
                """
                model.fit_generator(train_batches, train_steps,
                                    epochs=1000,
                                    validation_data=valid_batches,
                                    validation_steps=valid_steps,
                                    callbacks=[tb_cb])
                """
                # X_train: [number, angle, shape001, shape002, ..., shapeMAX]
                # y_train: [number, lift]
                # 適当に中央付近の翼を抜き出しての-40-38degreeをプロットさせてみる
                tekito = 1306 * 40  # NACA2613 or NACA2615
                plt.figure()
                plt.plot(X_train[tekito:tekito + 40, 0], y_train[tekito:tekito + 40])
                plt.plot(X_train[tekito:tekito + 40, 0], model.predict(X_train)[tekito:tekito + 40])
                plt.savefig(source + case_num + "_train.png")
                
                y_predict = model.predict(x_test)
                tekito = (99 + 13) * 40  # 22012
                plt.figure()
                plt.plot(x_test[tekito:tekito + 40, 0], y_test[tekito:tekito + 40])
                plt.plot(x_test[tekito:tekito + 40, 0], y_predict[tekito:tekito + 40])
                plt.savefig(source + case_num + "_test.png")
                
                make_scatter_plot(y_test, y_predict, "CL(Exact)", "CL(Predict)",
                                  path = "G:\\Toyota\\Data\\Incompressible_Invicid\\fig\\", fname = case_num)
                
                layer_name = "dense_1"
                hidden_layer_model = Model(inputs = model.input, outputs = model.get_layer(layer_name).output)
                hidden_output = hidden_layer_model.predict(x_test)
            
            json_string = model.to_json()
            open(source + json_name, 'w').write(json_string)
            model.save_weights(source + weight_name)
            
            fig = plt.figure()
            ax = fig.add_subplot(111, projection = "3d")
            np.save("hidden_output_2", hidden_output)
            ax.scatter3D(hidden_output[:, 0], hidden_output[:, 1], y_test)
            ax.set_xlabel("1st component")
            ax.set_ylabel("2nd component")
            ax.set_title("Scatter Plot of y_test")
            plt.show()
            
            if (not os.path.exists(baseSaveDir + "fig_post\\" + case_num + ".png")) and (epoch > 10):
                fig = plt.figure()
                ax = fig.add_subplot(1, 1, 1)
                ax.plot(history.history['loss'], marker = ".", label = 'loss')
                ax.plot(history.history['val_loss'], marker = '.', label = 'val_loss')
                ax.set_title('model loss')
                ax.grid(True)
                ax.set_xlabel('epoch')
                ax.set_ylabel('loss')
                ax.legend(loc = 'best')
                plt.savefig(baseSaveDir + "fig_post\\" + case_num + ".png")
                plt.close()
            
            score_test = model.evaluate(x_test, y_test, verbose = 0)
            y_test_pred = model.predict(x_test)
            r2_test = r2_score(y_test, y_test_pred)
            print(r2_test, score_test[0], score_test[1])
            log = [r2_test, score_test[0], score_test[1]]
            log.extend(case_num.split("_"))
            with open(baseSaveDir + "incomplog.csv", "a") as f:
                writer = csv.writer(f, lineterminator = '\n')
                writer.writerow(log)
            KTF.set_session(old_session)


def pclowd_deepset(mode = "circum", n_samples = 200000):
    max_points = 100
    pts_dim = 2
    reg = regularizers.l2(0.00001)
    rand_init = RandomNormal(stddev = 0.02)
    
    def regressor():
        def dense_block(num_points):
            inputs = Input(shape = (num_points,), name = "input_layer")
            x = PReLU(shared_axes = [1])(inputs)
            x = Dense(units = num_points, activation = None, kernel_initializer = rand_init,
                      kernel_regularizer = reg, bias_regularizer = reg, name = "dense_01")(x)
            x = BatchNormalization(axis = -1)(x)
            outputs = PReLU(shared_axes = [1])(x)
            return Model([inputs], [outputs])
        
        inputs = Input(shape = (1 + max_points * pts_dim,), name = "input_layer")
        input_pts = Lambda(lambda x: x[:, 1:], name = "points", output_shape = (max_points * pts_dim,))(inputs)
        input_aoas = Lambda(lambda x: x[:, 0], name = "aoas", output_shape = (1,))(inputs)
        
        # x = Embedding(max_points, pts_dim, mask_zero = True, trainable = False)(input_pts)
        x_pts = Lambda(lambda x: x[:, :max_points], name = "x_pts", output_shape = (max_points,))(input_pts)
        y_pts = Lambda(lambda x: x[:, max_points:], name = "y_pts", output_shape = (max_points,))(input_pts)
        all_pts = dense_block(max_points * pts_dim)(input_pts)
        x_pts = dense_block(max_points)(x_pts)
        y_pts = dense_block(max_points)(y_pts)
        
        Adder_x = Lambda(lambda x: K.sum(x, axis = 1), name = "adderX", output_shape = (1,))
        Adder_y = Lambda(lambda x: K.sum(x, axis = 1), name = "adderY", output_shape = (1,))
        Adder_xy = Lambda(lambda x: K.sum(x, axis = 1), name = "adderXY", output_shape = (1,))
        
        x_pts = Adder_x(x_pts)
        y_pts = Adder_y(y_pts)
        all_pts = Adder_xy(all_pts)
        x_pts = Reshape((1,))(x_pts)
        y_pts = Reshape((1,))(y_pts)
        all_pts = Reshape((1,))(all_pts)
        input_aoas = Reshape((1,))(input_aoas)
        x = Concatenate()([x_pts, y_pts, all_pts, input_aoas])
        x = BatchNormalization(axis = -1)(x)
        x = Dense(units = 16, activation = None, kernel_initializer = rand_init,
                  kernel_regularizer = reg, bias_regularizer = reg, name = "dense_final")(x)
        x = PReLU(shared_axes = [1])(x)
        lift = Dense(1, activation = None, kernel_initializer = rand_init,
                     kernel_regularizer = reg, bias_regularizer = reg, name = "output")(x)
        return Model(inputs = [inputs], outputs = [lift])
    
    x_train, x_test, y_train, y_test, rescale_x, rescale_y = load_mixed_and_shuffle(mode = mode,
                                                                                    n_samples = n_samples,
                                                                                    useN5w2 = True,
                                                                                    shuffle = False,
                                                                                    __step = 1,
                                                                                    __dimReduce = None,
                                                                                    __test_size = 0.01,
                                                                                    __dataReductMethod = None,
                                                                                    __mix = False)
    
    model = regressor()
    model.compile(loss = mean_squared_error,
                  optimizer = Adadelta(),
                  metrics = ["mae"])
    
    plot_model(model, to_file = "test.png", show_shapes = True, show_layer_names = True)
    model.summary()
    
    source = "G:\\Toyota\\Data\\Incompressible_Invicid\\training_data\\"
    epoch = 10000
    es_cb = EarlyStopping(monitor = 'val_loss', patience = 500, verbose = 0, mode = 'auto')
    tb_cb = TensorBoard(log_dir = source + "tboardPClowd", histogram_freq = 0, write_grads = True)
    baseSaveDir = source + "checkpoints\\"
    
    chkpt = baseSaveDir + 'pClowdMLP_.{epoch:02d}-{val_loss:.2f}.hdf5'
    cp_cb = ModelCheckpoint(filepath = chkpt, monitor = "val_loss", verbose = 0, save_best_only = True,
                            mode = 'auto')
    
    history = model.fit(x = x_train, y = y_train,
                        batch_size = 16384, epochs = epoch,
                        validation_split = 0.1, callbacks = [tb_cb, cp_cb, es_cb])
    model = getNewestModel(model, baseSaveDir)
    
    y_pred_t = rescale_y(model.predict(x_train))
    y_pred = rescale_y(model.predict(x_test))
    y_train = rescale_y(y_train)
    y_test = rescale_y(y_test)
    
    r2_train = r2_score(y_train, y_pred_t)
    r2_pred = r2_score(y_test, y_pred)
    log = "{0},{1},".format(r2_train, r2_pred)
    from simple_regressions import error_print
    log += error_print(y_pred_t, y_train).replace("\n", ",")
    log += error_print(y_pred, y_test)
    print(log)
    with open("pClowdLog.csv", "a") as f:
        f.write(log)


if __name__ == '__main__':
    np.random.seed(1)
    # pclowd_deepset()
    # exit()
    # env_in = input("Please set envirionment: 0:Lab, 1:Colab")
    env_in = str(0)
    if env_in == str(0):
        env = "Lab"
    elif env_in == str(1):
        env = "Colab"
    else:
        print("env_error")
        exit()
    
    # shape_type = input("please set shape_type: 0:fourier, 1:equidistant, 2:dense")
    # for i in range(3):
    shape_type = str(4)
    fname_lift_train = "NACA4\\s0000_e5000_a040_odd.csv"
    # fname_lift_test = "NACA5\\s11001_e65199_a040.csv"
    fname_lift_test = "NACA5\\s21001_e25199_a040.csv"
    
    if shape_type == str(0):
        fname_shape_train = "NACA4\\shape_fourier_5000_odd_closed.csv"
        fname_shape_test = "NACA5\\shape_fourier_all_closed.csv"
        case_number = 0
    
    elif shape_type == str(1):
        fname_shape_train = "NACA4\\shape_equidistant_5000_odd_closed.csv"
        fname_shape_test = "NACA5\\shape_equidistant_all_closed.csv"
        case_number = 1000
    
    elif shape_type == str(2):
        fname_shape_train = "NACA4\\shape_crowd_0.1_0.15_30_50_20_5000_odd_xy_closed.csv"
        fname_shape_test = "NACA5\\shape_crowd_all_mk2_xy_closed.csv"
        case_number = 2000
    elif shape_type == str(3):
        fname_shape_train = "NACA4\\shape_modified_fourier_5000_odd.csv"
        fname_shape_test = "NACA5\\shape_modified_fourier_all.csv"
        case_number = 3000
    elif shape_type == str(4):
        fname_shape_train = "NACA4\\shape_circumferential_5000_odd.csv"
        fname_shape_test = "NACA5\\shape_circumferential_all_mk2.csv"
        case_number = 4000
    else:
        print("shape_type error")
        exit()
    
    if env == "Colab":
        fname_lift_train = fname_lift_train.replace("\\", "/")
        fname_shape_train = fname_shape_train.replace("\\", "/")
        fname_lift_test = fname_lift_test.replace("\\", "/")
        fname_shape_test = fname_shape_test.replace("\\", "/")
    
    main(fname_lift_train, fname_shape_train, fname_lift_test, fname_shape_test, case_number, case_type = 3, env = env)