# -- coding: utf-8 --
import pandas as pd
import numpy as np

import re
from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors
import os
import math

# source:データの置いてあるディレクトリのパス(絶対or相対)
# fpath_lift:揚力係数の入ったcsvデータのsourceからの相対パス
# fpath_shape:形状データの入ったcsvデータのsourceからの相対パス
# shape_odd:物体形状ベクトルの次元の奇偶等（読み飛ばしに関する変数）
# read_rate:物体形状ベクトルの次元をread_rateで割った値に変更する
# skip_rate:揚力係数データ(教師データ)数をskip_rateで割った数に減らす
def read_csv_type3(source, fpath_lift, fpath_shape, shape_odd=0, read_rate=1, skip_rate=4, total_data=200000,
                   angle_data=40, skip_angle=True, unbalance=False, debug_mode=False, param="none"):
    name = ("naca4", "angle", "lift_coef")
    data_type = {"naca4": int, "angle": float, "lift_coef": float}

    def make_skip_rows_for_lift(total_data, angle_data, skip_rate, skip_angle):
        skip_row = []
        if skip_angle:  # 角度データを間引く
            for j in range(total_data):
                if j % skip_rate != 0:
                    skip_row.append(j)
        else:
            for j in range(total_data):
                if j % (skip_rate * angle_data) > angle_data - 1:
                    skip_row.append(j)
        return skip_row

    if unbalance:
        # skip_row = np.arange(int(total_data/skip_rate), total_data, 1).tolist()
        skip_row = np.arange(0, total_data - int(total_data / skip_rate), 1).tolist()
    else:
        skip_row = make_skip_rows_for_lift(total_data, angle_data, skip_rate, skip_angle)
    df_l = pd.read_csv(source + fpath_lift, header=None, usecols=[1, 3, 4],
                       names=name, dtype=data_type, skiprows=skip_row)

    def make_use_cols_for_shape(data, shape_odd, rate):
        # dataは，shapeを何項まで計算したのかによって変更(今回は200固定でよさげ)
        # shape_oddによって全部読む・奇数のみ読む・偶数のみ読む，前半だけ読む、みたいな順番変更
        # dataのうち1/rateだけ読む
        odd_num = lambda index: (2 * index) + 1
        even_num = lambda index: (index + 1) * 2
        original_num = lambda index: index + 1
        skip_n_num = lambda index: (rate * index) + 1

        if ((shape_odd != 0) and (rate == 1) or (rate == 0)):
            print("rate error!")
            exit()
        num = int(data / rate)
        if shape_odd == 1:  # 奇数番のみを抽出
            next = odd_num
        elif shape_odd == 2:  # 偶数番のみを抽出
            next = even_num
        elif shape_odd == 3:  # 並び順は同じだけど，数は半分
            next = original_num
        elif shape_odd == 4:
            next = skip_n_num  # 最後まで読むけどrateごとにスキップ
        else:  # 元々のまま読み出す
            next = original_num
            num = data

        col = [0]
        name = ["naca4"]
        data_type = {"naca4": int}
        for index in range(num):
            col.append(next(index))
            name.append("t" + str(index).zfill(3))
            data_type["t" + str(index).zfill(3)] = float
        return col, name, data_type

    col, name, data_type = make_use_cols_for_shape(data=200, shape_odd=shape_odd, rate=read_rate)

    df_s = pd.read_csv(source + fpath_shape, header=None,
                       usecols=col, names=name, dtype=data_type
                       )  # .set_index("naca4")

    """このままだとクッソ重いので名前を被らせてメモリ節約する
    本当に書きたい処理はこれ
    df_xy = pd.merge(df_l, df_s, on="naca4")
    y_train = df_xy["lift_coef"].values
    X_train = df_xy.drop("lift_coef", axis=1).drop("naca4", axis=1).values
    """
    X_train = pd.merge(df_l, df_s, on="naca4", how="inner")
    if param == "none":
        if debug_mode:
            return X_train, None

        y_train = X_train["lift_coef"].values.reshape(-1, 1)
        X_train = X_train.drop("lift_coef", axis=1).drop("naca4", axis=1).values
        return X_train, y_train
    else:
        def returnNdigit(N):
            return (X_train["naca4"].values.reshape(-1) % 10**N) // 10**(N-1)
        
        digit = len(str(X_train["naca4"].values[0]))
        if param == "NACA":
            return np.array((X_train["naca4"].values.reshape(-1)), dtype = str)
        elif param == "thickness":
            return (X_train["naca4"].values.reshape(-1) % 100) / 100.0
        elif (param == "maxCamber"):
            return returnNdigit(4)
        elif (param == "maxCamberPosition" and digit == 4) or (param == "reflect" and digit == 5):
            return returnNdigit(3)
        elif (param == "designedLift" and digit == 5):
            return returnNdigit(5)
        else:
            print(param + " can't use!")
            exit()

def reductData(source, reductTarget, __x, __mix, mode):
    k_cluster = int(re.sub(r'\D', '', reductTarget))
    header = ""
    if not "fourier" in mode.lower():
        header = mode
    if __mix:
        midFile = source + "mid\\" + header + reductTarget + ".npz"
    else:
        midFile = source + "mid\\" + header + reductTarget + "NoMix.npz"
        
    if os.path.exists(midFile):
        loaded_array = np.load(midFile)
        #  __x = loaded_array["__x"]
        # centroids = loaded_array["centroids"]
        nearest_indices = loaded_array["nearest_indices"]
        x_mixed = loaded_array["__x"]
    else:
        print("calculated training data is not found, making now...")
        if "norm" in reductTarget.lower():
            scalar = StandardScaler()
            __x = scalar.fit_transform(__x)
        
        if "pca" in reductTarget.lower():
            if "kpca" in reductTarget.lower():
                pca = KernelPCA(n_components = 20, kernel = "rbf", n_jobs = -1, random_state = 1)
            else:
                pca = PCA(n_components = 20, random_state = 1)
            __x = pca.fit_transform(__x)
        
        threshold = 2500
        if k_cluster > threshold:
            split = math.floor(float(k_cluster) / threshold)
            batch_size = math.floor(float(k_cluster) / split)
            kmeans = MiniBatchKMeans(n_clusters = k_cluster,  # クラスタ数
                                     batch_size = batch_size,
                                     init = "k-means++",  # セントロイド初期化アルゴリズム
                                     n_init = 10,  #
                                     init_size = k_cluster + batch_size,
                                     max_iter = 300,  # 最大反復回数
                                     tol = 1e-04,  # 収束判定条件
                                     random_state = 1)
        else:
            kmeans = KMeans(n_clusters = k_cluster,  # クラスタ数
                            init = "k-means++",  # セントロイド初期化アルゴリズム
                            n_init = 10,  #
                            max_iter = 300,  # 最大反復回数
                            tol = 1e-04,  # 収束判定条件
                            n_jobs = -1,  # 並列実行数(-1 = 環境限界まで利用)
                            random_state = 1)
        kmeans.fit(__x)
        centroids = kmeans.cluster_centers_
        nn = NearestNeighbors(metric = "minkowski")
        nn.fit(__x)
        distances, nearest_indices = nn.kneighbors(centroids, n_neighbors = 1)
        np.savez_compressed(str(midFile), __x = __x, centroids = centroids, nearest_indices = nearest_indices)
    return nearest_indices.reshape(-1)

def scaleAndRescale(data):
    def __scaleAndRescaleMain(data):
        scalar = StandardScaler()
        scalar.fit(data)
        return scalar.transform, scalar.inverse_transform

    std = data.std(axis = 0, ddof = 1)
    if not np.any(std == 0.0):
        mean = data.mean(axis = 0)
        scaler = lambda data: np.where(std > 0, (data - mean) / std, 0.0)
        rescaler = lambda data: np.where(std > 0, data * std + mean, data)
    else:
        scaler, rescaler = __scaleAndRescaleMain(data)
    return scaler, rescaler

def load_mixed_and_shuffle(mode="fourier", n_samples=50000, shuffle=True, useN5w2=False, msdf_res=16, __step=1, __dimReduce=None, __test_size=0.33, __dataReductMethod=None, __mix=True):
    """

    :param mode:    shape expression
    :param n_samples:   return X_train[n_samples, n_features]
    :param shuffle:     NACA4 and 5 mixed
    :param useN5w2:     use NACA1xxxx ~ NACA6xxxx
    :param msdf_res: ONLY MSDF  msdf resolution
    :param __step: ONLY SHUFFLE=FALSE   X_train = X_train[::__step]
    :param __dimReduce: Specify  Dimension Reduction Method
    :param __test_size: parameter of train_test_split
    :param __dataReductMethod: Specify Data Reduction Method (use with shuffle == True)
    :return:
    """
    from sklearn.model_selection import train_test_split
    source = "G:\\Toyota\\Data\\Incompressible_Invicid\\training_data\\"
    fname_lift_train = "NACA4\\s0000_e5000_a040_odd.csv"
    if useN5w2:
        fname_lift_test = "NACA5\\s11001_e65199_a040.csv"
    else:
        fname_lift_test = "NACA5\\s21001_e25199_a040.csv"
    
    if "fourier" in mode.lower():
        fname_shape_train = "NACA4\\shape_modified_fourier_5000_odd.csv"
        if useN5w2:
            fname_shape_test = "NACA5\\shape_modified_fourier_all_mk2.csv"
        else:
            fname_shape_test = "NACA5\\shape_modified_fourier_all.csv"
    
    elif "equidistant" in mode.lower():
        fname_shape_train = "NACA4\\shape_equidistant_5000_odd_closed.csv"
        if useN5w2:
            fname_shape_test = "NACA5\\shape_equidistant_all_mk2.csv"
        else:
            fname_shape_test = "NACA5\\shape_equidistant_all_closed.csv"
            
    elif "crowd" in mode.lower():
        fname_shape_train = "NACA4\\shape_crowd_0.1_0.15_30_50_20_5000_odd_xy_closed.csv"
        if useN5w2:
            fname_shape_test = "NACA5\\shape_crowd_0.1_0.15_30_50_20_all.csv"
        else:
            fname_shape_test = "NACA5\\shape_crowd_all_mk2_xy_closed.csv"
    
    elif "msdf" in mode.lower():
        fname_train = source + "NACA4\\" + str(msdf_res) + "_i_i_4.npz"
        fname_test = source + "NACA5\\" + str(msdf_res) + "_i_i_5.npz"
        
    elif "circum" in mode.lower():
        fname_shape_train = "NACA4\\shape_circumferential_5000_odd.csv"
        if useN5w2:
            fname_shape_test = "NACA5\\shape_circumferential_all_mk2.csv"
        else:
            fname_shape_test = "NACA5\\shape_circumferential_all_mk2.csv"
    else:
        raise ValueError

    if not "msdf" in mode:
        X_train, y_train = read_csv_type3(source, fname_lift_train, fname_shape_train, skip_rate=1)
        X_test, y_test = read_csv_type3(source, fname_lift_test, fname_shape_test, skip_rate=1)
    else:
        loaded_array = np.load(fname_train)
        X_train = loaded_array["x_train"]
        y_train = loaded_array["y_train"]
        loaded_array = np.load(fname_test)
        X_test = loaded_array["x_train"]
        y_test = loaded_array["y_train"]
        del loaded_array
    
    if not __mix:
        x_mixed = X_train
        y_mixed = y_train
        x_test = X_test
    else:
        x_mixed = np.concatenate([X_train, X_test])
        y_mixed = np.concatenate([y_train, y_test])

    if not "msdf" in mode:
        scale_x, rescale_x = scaleAndRescale(x_mixed)
        
    else:
        scale_x = lambda x_data: (x_data / 255.0)# - 127.5) / 127.5
        rescale_x = lambda x_data: (x_data * 255).astype("int64")#127.5 + 127.5)

    # scale_y, rescale_y = scaleAndRescale(y_mixed)

    if shuffle:
        samples = x_mixed.shape[0]
        if n_samples > samples:
            n_samples = samples
        print("data reduct option:{0}".format(__dataReductMethod))
        if __dataReductMethod is None or "random" in __dataReductMethod.lower():
            if type(__dataReductMethod) == type(str(1)):
                n_samples = int(re.sub(r'\D', '', __dataReductMethod))
            index = np.random.choice(samples, n_samples, replace = False)  # 重複なし並び替え
            x_mixed = scale_x(x_mixed)[index]
            # y_mixed = scale_y(y_mixed)[index]
            y_mixed = y_mixed[index]
            if __mix:
                x_train, x_test, y_train, y_test = train_test_split(x_mixed, y_mixed, test_size = __test_size, random_state = 42)
            else:
                x_train = x_mixed
                y_train = y_mixed
                
        elif "kmeans" in __dataReductMethod.lower():
            nearest_indices = reductData(source = source, reductTarget=__dataReductMethod, __x = x_mixed, __mix=__mix, mode=mode)
            if __mix:
                idx = np.ones(x_mixed.shape[0], dtype = bool)
                idx[nearest_indices] = False
                x_test = x_mixed[idx, :]
                y_test = y_mixed[idx]
            x_train = x_mixed[nearest_indices, :]
            y_train = y_mixed[nearest_indices]
        else:
            raise ValueError
    else:
        x_train = scale_x(X_train)[::__step]
        x_test = scale_x(X_test)#[::__step]
        y_train = y_train[::__step]
        y_test = y_test#[::__step]
    
    if not __dimReduce is None:
        n_feat = int(re.sub(r'\D', '', __dimReduce))
        if "pca" in __dimReduce.lower():
            if "kpca" in __dimReduce.lower():
                pca = KernelPCA(n_components = n_feat, kernel = "rbf", n_jobs = -1, random_state = 1)
            else:
                pca = PCA(n_components = n_feat, random_state = 1)
            
            s_train = pca.fit_transform(x_train[:, 1:])
            s_test = pca.transform(x_test[:, 1:])
            x_train = np.concatenate([x_train[:, 0].reshape(-1, 1), s_train], axis = 1)
            x_test = np.concatenate([x_test[:, 0].reshape(-1, 1), s_test], axis = 1)
            # re-normalize
            rescale_x1 = rescale_x
            x_mean2 = x_train.mean(axis = 0)
            x_std2 = x_train.std(axis = 0, ddof = 1)
            scale_x2 = lambda x_data: (x_data - x_mean2) / x_std2
            rescale_x2 = lambda x_data: x_data * x_std2 + x_mean2
            x_train = scale_x2(x_train)
            x_test = scale_x2(x_test)
            def rescale_x12(x_data):
                x_data = rescale_x2(x_data) # pca直後に戻す
                s_data = pca.inverse_transform(x_data[:, 1:])  # shapeのみpca逆変換(次元復元)
                x_data = np.concatenate([x_data[:, 0].reshape(-1, 1), s_data], axis = 1)  # angle & shape concat
                return rescale_x1(x_data) # 元のスケールに戻す
            
            rescale_x = rescale_x12
    scale_y, rescale_y = scaleAndRescale(y_train)
    return x_train, x_test, scale_y(y_train), scale_y(y_test), rescale_x, rescale_y

def check_evRatio():
    modes = ["circum", "equidistant", "crowd", "fourier"]
    names = ["circumferential", "equidistant", "concentrate", "fourier"]
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize = (8, 4))
    ax = fig.add_subplot(111)
    for mode, name in zip(modes, names):
        x_train, x_test, y_train, y_test, rescale_x, rescale_y = load_mixed_and_shuffle(mode = mode, n_samples = 4000000,
                                                                                        shuffle = False, useN5w2 = False,
                                                                                        __dimReduce = None,
                                                                                        __test_size = 0.001,
                                                                                        __dataReductMethod = None,
                                                                                        __mix = True)
        x_mixed = np.concatenate([x_train, x_test])
        pca = PCA()
        pca.fit(x_mixed)
        ev_ratio = pca.explained_variance_ratio_
        ev_ratio = np.hstack([0, ev_ratio.cumsum()])
        ax.plot(ev_ratio, label=name)
    ax.set_title("Cumulative Contribution  Ratio from PCA")
    ax.set_xlabel("number of principal components")
    ax.set_ylabel("Cumulative Contribution ratio".lower())
    ax.legend()
    ax.grid()
    plt.show()
    
def old_skip_angle(mode, skip_rate=4, angleAll=None, n_samples = 200000):
    x_train, x_test, y_train, y_test, rescale_x, rescale_y = load_mixed_and_shuffle(mode=mode, n_samples=n_samples, useN5w2=False,
                                  shuffle=False, __step=1, __dimReduce=None, __test_size=0.01, __dataReductMethod=None, __mix=False)

    x_train, y_train = x_train[::skip_rate], y_train[::skip_rate]
    return x_train, x_test, y_train, y_test, rescale_x, rescale_y

def old_skip_wing(mode, skip_rate=4, angleAll=40, n_samples = 200000):

    tmp_x_train, x_test, tmp_y_train, y_test, rescale_x, rescale_y = load_mixed_and_shuffle(mode=mode, n_samples=n_samples,
                                                                                    useN5w2=False,
                                                                                    shuffle=False, __step=1,
                                                                                    __dimReduce=None, __test_size=0.01,
                                                                                    __dataReductMethod=None,
                                                                                    __mix=False)

    x_train = np.zeros((int(tmp_x_train.shape[0] / skip_rate), tmp_x_train.shape[1]))
    y_train = np.zeros((int(tmp_y_train.shape[0] / skip_rate), 1))
    for i in range(int(n_samples / (angleAll*skip_rate))):
        stop = skip_rate * (i + 1) * angleAll
        start = stop - angleAll
        x_train[i * (angleAll):(i + 1) * (angleAll)] = tmp_x_train[start:stop]
        y_train[i * (angleAll):(i + 1) * (angleAll)] = tmp_y_train[start:stop]
    return x_train, x_test, y_train, y_test, rescale_x, rescale_y

def old_skip_unballanced(mode, skip_rate, angleAll=None, n_samples = 200000):
    x_train, x_test, y_train, y_test, rescale_x, rescale_y = load_mixed_and_shuffle(mode=mode, n_samples=n_samples,
                                                                                    useN5w2=False,
                                                                                    shuffle=False, __step=1,
                                                                                    __dimReduce=None, __test_size=0.01,
                                                                                    __dataReductMethod=None,
                                                                                    __mix=False)
    total = int(n_samples / skip_rate)
    x_train, y_train = x_train[n_samples - total:], y_train[n_samples - total:]
    return x_train, x_test, y_train, y_test, rescale_x, rescale_y

def old_skip_clustering(mode, skip_rate, method="kmeans_norm_pca", n_samples = 200000):
    method += "_" + str(int(n_samples / skip_rate))
    return load_mixed_and_shuffle(mode=mode, n_samples=n_samples, shuffle=True, useN5w2=False, msdf_res=16, __step=1,
                                  __dimReduce=None, __test_size=0.33, __dataReductMethod=method, __mix=False)

def reduceAngle(x_data, rescale_x, updateScaler = False):
    from sklearn.preprocessing import StandardScaler
    x_data = rescale_x(x_data)
    n_sample = x_data.shape[0]
    s_data = x_data[:, 1:].reshape(n_sample, 2, -1)
    a_data = x_data[:, 0] * np.pi / 180.0
    rot = np.array([[np.cos(a_data), -np.sin(a_data)], [np.sin(a_data), np.cos(a_data)]])
    s_data = np.einsum("ijk,jli->ilk", s_data, rot).reshape(n_sample, -1)
    scalar = StandardScaler()
    s_data = scalar.fit_transform(s_data)
    if updateScaler:
        return s_data, scalar.inverse_transform
    else:
        return s_data, rescale_x
    
def create_optional_information(s_data=None, rescale_s=None, x_data=None, rescale_x=None, train=True):
    def get_aspect(s_data):
        params = np.zeros((total,2))
        for i in range(total):
            r_data = s_data[i, :].reshape(2, -1).T
            pca = PCA(n_components = 2)
            pca.fit(r_data)
            r_data = pca.transform(r_data)
            length = (np.max(r_data,axis = 0) - np.min(r_data, axis = 0))
            params[i, 0] = length[1] / length[0]
            # print(np.sum(np.diff(r_data, append = r_data[0].reshape(-1, 2), axis = 0)**2, axis = 1))
            # print(np.sum(np.sqrt(np.sum(np.diff(r_data, append = r_data[0].reshape(-1, 2), axis = 0)**2, axis = 1))))
            # print(np.sqrt(np.sum(np.diff(r_data, append = r_data[0].reshape(-1, 2), axis = 0)**2)))
            params[i, 1] = np.sum(np.sqrt(np.sum(np.diff(r_data, append = r_data[0].reshape(-1, 2), axis = 0)**2, axis = 1)))
        return params

    if s_data is None:
        total = x_data.shape[0]
    elif x_data is None and rescale_s is not None:
        total = s_data.shape[0]
    else:
        raise ValueError
    
    n_feat = 2
    params = np.zeros((total, n_feat))
    from pathlib import Path
    case_name = Path("prm{0}{1}{2}.npy".format(n_feat, total, train))
    
    if s_data is None:
        s_data, scale_s, rescale_s = reduceAngle(x_data, rescale_x, scale_s=None, rescale_s=rescale_s)
        s_data = rescale_s(s_data)
        params[:, 0:2] = get_aspect(s_data)
        return params, s_data, rescale_s
    
    if case_name.exists():
        print("calculated data is exists.")
        params = np.load(str(case_name))
    else:
        print("calculated data is not exists.")
        params[:, 0:2] = get_aspect(rescale_s(s_data))
        np.save(str(case_name), params)
    return params

def transformShapeData(data):
    x_train, x_test, y_train, y_test, rescale_x, rescale_y = data[0], data[1], data[2], data[3], data[4], data[5]
    x_test, rescale_x = reduceAngle(x_test, rescale_x, updateScaler = False)
    x_train, rescale_x = reduceAngle(x_train, rescale_x, updateScaler = True)
    return x_train, x_test, y_train, y_test, rescale_x, rescale_y

def load_with_param_for_GAN(mode, params=2, pname=""):
    def rescaler_py(py_data):
        py_data[:, 0] = old_rescale_y(py_data[:, 0])
        py_data[:, 1:1+params] = scalar.inverse_transform(py_data[:, 1:1+params])
        return py_data
    s_train, s_test, y_train, y_test, rescale_s, rescale_y = transformShapeData(load_mixed_and_shuffle("circum", n_samples=500000, shuffle = False, useN5w2 = True))
    p_train = create_optional_information(s_data = s_train, rescale_s = rescale_s, train = True)
    p_test = create_optional_information(s_data = s_test, rescale_s = rescale_s, train = False)
    if params == 1:
        if pname == "aspect":
            num = 0
        else:
            num = 1
        p_train = p_train[:, num].reshape(-1, 1)
        p_test = p_test[:, num].reshape(-1, 1)
    scalar = StandardScaler()
    p_train = scalar.fit_transform(p_train)
    p_test = scalar.transform(p_test)
    
    x_train, x_test, y_train, y_test, rescale_x, rescale_y = load_mixed_and_shuffle(mode = mode, n_samples = 500000, shuffle = False, useN5w2 = True)
    
    y_train = np.concatenate([y_train.reshape(-1, 1), p_train], axis = 1)
    y_test = np.concatenate([y_test.reshape(-1, 1), p_test], axis = 1)
    old_rescale_y = rescale_y
    rescale_y = rescaler_py
    return x_train, x_test, y_train, y_test, rescale_x, rescale_y

def load_noaoa(n_samples=200000, with_param_for_GAN=False):
    # return transformShapeData(old_skip_clustering("circum", skip_rate, method = "kmeans_norm_pca", n_samples = 200000))
    if not with_param_for_GAN:
        return transformShapeData(load_mixed_and_shuffle("circum", n_samples))
    else:
        def rescaler_py(py_data):
            py_data[:, 0] = old_rescale_y(py_data[:, 0])
            py_data[:, 1:3] = scalar.inverse_transform(py_data[:,1:3])
            return py_data
        
        s_train, s_test, y_train, y_test, rescale_s, rescale_y = transformShapeData(load_mixed_and_shuffle("circum", n_samples))
        p_train = create_optional_information(s_data = s_train, rescale_s=rescale_s, train = True)
        p_test = create_optional_information(s_data = s_test, rescale_s = rescale_s, train=False)
        scalar = StandardScaler()
        p_train = scalar.fit_transform(p_train)
        p_test = scalar.transform(p_test)
        y_train = np.concatenate([y_train.reshape(-1, 1), p_train], axis = 1)
        y_test = np.concatenate([y_test.reshape(-1, 1), p_test], axis = 1)
        old_rescale_y = rescale_y
        rescale_y = rescaler_py
        return s_train, s_test, y_train, y_test, rescale_s, rescale_y
        
if __name__ == '__main__':
    check_evRatio()
    exit()
    X_train, X_test, y_train, y_test, rescale_x, rescale_y = load_mixed_and_shuffle(mode = "circum",
                                                                                    n_samples = 5000,
                                                                                    shuffle = True)
    z_train, s_train, rescale_s = create_optional_information(x_data = X_train, rescale_x=rescale_x)
    z_test, s_test, rescale_s = create_optional_information(x_data = X_test, rescale_x=rescale_x, rescale_s=rescale_s)
    # z_test= create_optional_information(s_train, rescale_s)
    y_test = np.concatenate([y_test.reshape(-1,1), z_test.reshape(-1,2)], axis=1)
    
    
    exit()
    # x_train, x_test, y_train, y_test, rescale_x, rescale_y = old_skip_angle(mode="fourier", skip_rate=8)
    # x_train, x_test, y_train, y_test, rescale_x, rescale_y = old_skip_wing(mode="fourier", skip_rate=4, angleAll=40, n_samples=200000)
    # x_train, x_test, y_train, y_test, rescale_x, rescale_y = old_skip_unballanced(mode="fourier", skip_rate=10)
    print(x_train)
    print(x_train.shape)
    exit()
    check_evRatio()
    exit()
    load_mixed_and_shuffle(mode = "msdf")
    exit()
    # 自宅で作成したのでLaboratory用に書き換える
    # source = "D:\\Dropbox\\shareTH\\program\\keras_training\\"
    source = "G:\\Toyota\\Data\\Incompressible_Invicid\\training_data\\"
    # source = "D:\\Toyota\\Dropbox\\shareTH\\program\\laplace\\"
    # ちなみにNACA4とNACA5は基本動作がほぼ一緒だから使い回せるtype3とtype4は同じやり方でok
    fpath_lift = "NACA4\\s0000_e5000_a040_odd.csv"
    fpath_shape = "NACA4\\shape_fourier_5000_odd.csv"
    shape_odd = 0
    read_rate = 1
    read_csv_type3(source, fpath_lift, fpath_shape, shape_odd, read_rate, param = "thickness")
