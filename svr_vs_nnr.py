# coding: utf-8
import numpy as np
from sklearn.metrics import r2_score
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

from keras.models import Model
from keras.layers.core import Dense
from keras.layers import LeakyReLU, Input
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adadelta
from keras import backend as K
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf

from scatter_plot import make_scatter_plot

def svr(x_train, y_train, kernel='linear', params_cnt=20, figname="sample.png"):
    # generator for closs validation
    def gen_cv():
        m_train = np.floor(y_train.shape[0] * 0.75).astype(int)
        train_indices = np.arange(m_train)
        test_indices = np.arange(m_train, y_train.shape[0])
        yield (train_indices, test_indices)

    params = {'C': np.logspace(-10, 10, params_cnt), 'epsilon': np.logspace(-10, 10, params_cnt)}
    gridsearch = GridSearchCV(SVR(kernel=kernel, gamma="scale"), params, cv=gen_cv(), scoring='r2',
                              return_train_score=True)
    gridsearch.fit(x_train, y_train)
    # 検証曲線
    plt_x, plt_y = np.meshgrid(params["C"], params["epsilon"])
    fig = plt.figure(figsize=(8, 8))
    fig.subplots_adjust(hspace=0.3)
    for i in range(2):
        if i == 0:
            plt_z = np.array(gridsearch.cv_results_["mean_train_score"]).reshape(params_cnt, params_cnt, order="F")
            title = "Train"
        else:
            plt_z = np.array(gridsearch.cv_results_["mean_test_score"]).reshape(params_cnt, params_cnt, order="F")
            title = "Cross Validation"
        ax = fig.add_subplot(2, 1, i + 1)
        CS = ax.contour(plt_x, plt_y, plt_z, levels=[0.6, 0.65, 0.7, 0.75, 0.8, 0.85])
        ax.clabel(CS, CS.levels, inline=True)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("C")
        ax.set_ylabel("epsilon")
        ax.set_title(title)
    plt.suptitle("Validation curve / Gaussian SVR")
    plt.savefig(figname)
    return SVR(kernel=kernel, C=gridsearch.best_params_["C"], epsilon=gridsearch.best_params_['epsilon'], gamma="scale")


def sample_x_y(kind=1, plot=False):
    def test_func1(x, m=3, b=1):
        return m * x + b * np.random.normal(loc=0, scale=1, size=x.shape[0])

    x = 0.001 * np.random.randint(low=-1000, high=1000, size=100)
    if kind == 1:
        y = test_func1(x, b=0.5)

    if plot:
        plt.plot(x, y, "x")
        plt.show()
    return x.reshape(-1, 1), y

def svr_main(x_train, y_train, x_test, y_test, case_name):
    regresser = svr(x_train, y_train, kernel='rbf', figname=case_name + ".png")
    regresser.fit(x_train, y_train)
    y_pred_t = regresser.predict(x_train)
    y_pred = regresser.predict(x_test)
    make_scatter_plot(data_a=y_train, data_b=y_pred_t, label_a="observed (train data)", label_b="predicted",
                      fname=case_name + "train_svr")
    make_scatter_plot(data_a=y_test, data_b=y_pred, label_a="observed (test data)", label_b="predicted",
                      fname=case_name + "test_svr")
    log = str(regresser.score(x_train, y_train)) + " " + str(regresser.score(x_test, y_test)) + "\n"
    print("SVR:" + log)
    log += str(regresser) + "\n\n"
    return log

def nnr_main(x_train, y_train, x_test, y_test, case_name, standardized=True):
    if not standardized:
        Xtrain = x_train
        Xtest = x_test
        ytrain = y_train
        x_train = (Xtrain - Xtrain.mean(axis=0)) / Xtrain.std(axis=0, ddof=1)
        y_train = (ytrain - ytrain.mean()) / ytrain.std(ddof=1)
        x_test = (Xtest - Xtrain.mean(axis=0)) / Xtrain.std(axis=0, ddof=1)

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.1
    KTF.set_session(tf.Session(config=config))
    batch_size = min(int(x_train.shape[0] / 2), 25000)
    with tf.Session(config=config) as sess:
        K.set_session(sess)
        input_vector_dim = x_train.shape[1]
        inputs = Input(shape=(input_vector_dim,), name="input_layer")
        x = LeakyReLU()(inputs)
        x = Dense(units=2, name="dense_01")(x)
        x = LeakyReLU()(x)
        x = Dense(units=16, name="dense_02")(x)
        x = LeakyReLU()(x)
        prediction = Dense(units=1, name="output_layer")(x)
        model = Model(inputs=[inputs], outputs=[prediction])
        model.compile(loss="mean_squared_error",
                      optimizer=Adadelta(),
                      metrics=["mae"])

        epoch = 100000
        chkpt = "checkpoint\\" + 'MLP_.{epoch:02d}-{val_loss:.2f}.hdf5'

        es_cb = EarlyStopping(monitor='val_loss', patience=500, verbose=0, mode='auto')
        cp_cb = ModelCheckpoint(filepath=chkpt, monitor="val_loss", verbose=0, save_best_only=True, mode='auto')

        history = model.fit(x=x_train,
                            y=y_train,
                            verbose=0,
                            batch_size=batch_size,
                            epochs=epoch,
                            validation_split=0.1,
                            callbacks=[es_cb, cp_cb]
                            )
        y_pred_t = model.predict(x_train)
        y_pred = model.predict(x_test)
        if not standardized:
            y_pred_t = y_pred_t * ytrain.std(ddof=1) + ytrain.mean()
            y_pred = y_pred * ytrain.std(ddof=1) + ytrain.mean()
        r2_train = r2_score(ytrain, y_pred_t)
        r2_pred = r2_score(y_test, y_pred)
        # log0 = str(history.history)
    K.clear_session()
    make_scatter_plot(data_a=ytrain, data_b=y_pred_t, label_a="observed (train data)", label_b="predicted",
                      fname=case_name + "_train_nnr")
    make_scatter_plot(data_a=y_test, data_b=y_pred, label_a="observed (test data)", label_b="predicted", fname=case_name + "_test_nnr")
    log = str(r2_train) + " " + str(r2_pred) + "\n"
    print("NNR:" + log)
    # log += str(log0) + "\n\n"
    return log

def main():
    for shape_type in range(4):
        step = 6251
        for i in range(8):
            step = int(step / 2) + 1
            print(step)

            x_train, y_train, x_test, y_test = load_x_y(str(shape_type), step)
            y_train = y_train.flatten()
            y_test = y_test.flatten()

            case_name = "log\\" + str(x_train.shape) + "_" + str(shape_type)

            log = ""
            log += svr_main(x_train, y_train, x_test, y_test, case_name)
            log += nnr_main(x_train, y_train, x_test, y_test, case_name)

            with open(case_name + "_c.txt", "w") as f:
                f.write(str(x_train.shape) + str(shape_type) + "\n")
                f.write(log)


def load_x_y(shape_type=str(0), step=2500):
    fname_lift_train = "NACA4\\s0000_e5000_a040_odd.csv"
    fname_lift_test = "NACA5\\s21001_e25199_a040.csv"
    if shape_type == str(0):
        fname_shape_train = "NACA4\\shape_fourier_5000_odd_closed.csv"
        fname_shape_test = "NACA5\\shape_fourier_all_closed.csv"
    elif shape_type == str(1):
        fname_shape_train = "NACA4\\shape_equidistant_5000_odd_closed.csv"
        fname_shape_test = "NACA5\\shape_equidistant_all_closed.csv"
    elif shape_type == str(2):
        fname_shape_train = "NACA4\\shape_crowd_0.1_0.15_30_50_20_5000_odd_xy_closed.csv"
        fname_shape_test = "NACA5\\shape_crowd_all_mk2_xy_closed.csv"
    elif shape_type == str(3):
        fname_shape_train = "NACA4\\shape_modified_fourier_5000_odd.csv"
        fname_shape_test = "NACA5\\shape_modified_fourier_all.csv"
    else:
        print(shape_type)
        exit()
    from read_training_data import read_csv_type3
    source = "G:\\Toyota\\Data\\Incompressible_Invicid\\training_data\\"
    rr = 1
    sr = 1
    s_odd = 0
    s_skiptype = False
    unballanced = False
    reductioning = False
    X_train, y_train = read_csv_type3(source, fname_lift_train, fname_shape_train, shape_odd=s_odd, read_rate=rr,
                                      skip_rate=sr, skip_angle=s_skiptype, unbalance=unballanced)

    X_train, y_train = X_train[::step], y_train[::step]
    print(step, X_train.shape)
    x_test, y_test = read_csv_type3(source, fname_lift_test, fname_shape_test, shape_odd=s_odd, read_rate=rr)
    return X_train, y_train, x_test, y_test


if __name__ == '__main__':
    main()
