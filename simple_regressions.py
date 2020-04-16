# coding: utf-8
import numpy as np
from sklearn import linear_model
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import r2_score
from read_training_data import read_csv_type3, load_mixed_and_shuffle
from read_training_data import old_skip_angle, old_skip_wing, old_skip_unballanced, old_skip_clustering
import pickle
from pathlib import Path


np.seterr(divide = "raise")

def __errors(y_pred, y_true):
    return y_pred.flatten() - y_true.flatten()

def max_error(y_pred, y_true):
    err = __errors(y_pred, y_true)
    return err[np.argmax(np.abs(err))]

def min_error(y_pred, y_true):
    err = __errors(y_pred, y_true)
    return err[np.argmin(np.abs(err))]

def rmsq_error(y_pred, y_true):
    err = __errors(y_pred, y_true)
    return np.sqrt(np.sum(err**2)) / err.shape[0]

def abs_median_error(y_pred, y_true):
    err = __errors(y_pred, y_true)
    return np.median(np.abs(err))

def error_print(y_pred, y_true, csv=False, r2=False):
    rmsE = rmsq_error(y_pred, y_true)
    medE = abs_median_error(y_pred, y_true)
    maxE = max_error(y_pred, y_true)
    minE = min_error(y_pred, y_true)
    if not csv:
        return "rmse:{0}_median:{3}_max:{1}_min:{2}\n".format(rmsE, maxE, minE, medE)
    else:
        if r2:
            return "{4},{0},{1},{2},{3}\n".format(rmsE, medE, maxE, minE,r2_score(y_pred, y_true))
        else:
            return "{0},{1},{2},{3}\n".format(rmsE, medE, maxE, minE)

def linear():
    return linear_model.LinearRegression()

def ridge():
    # reg.alpha_でalphaの値を確認すべし
    return linear_model.RidgeCV(alphas = np.logspace(-6, 6, 13))

def lasso():
    # reg.
    return linear_model.LassoCV(cv = 5, random_state = 0)

def elasticnet():
    return linear_model.ElasticNetCV(cv = 5, random_state = 0)

class NeighborWrap(object):
    fitted = False
    def __init__(self, n_neighbors, weight=1, shapetype=0, n_samples=200000, otherInfo=""):
        self.model = NearestNeighbors(n_neighbors = n_neighbors)
        self.neighbor = n_neighbors
        self.weight = weight
        self.tail = otherInfo
        if n_samples == 200000:
            self.path = Path("knn_ii_{0}_{1}_{2}{3}.pkl".format(n_neighbors, weight, shapetype, otherInfo))
        else:
            self.path = Path("knn_ii_{0}_{1}_{2}_{3}{4}.pkl".format(n_neighbors, weight, shapetype, n_samples, otherInfo))


    def fit(self, x, y):
        if not self.fitted:
            self.y_train = y
            self.feat = 1  # y.shape[1]
            if self.path.exists():
                print("learned model loaded")
                self.model = pickle.load(open(self.path, "rb"))
            else:
                print("learned data not found.")
                self.model.fit(x)
                pickle.dump(self.model, open(self.path, "wb"))
            self.fitted = True

    def check_fit(self):
        if not self.fitted:
            raise ValueError

    def predict(self, x):
        self.check_fit()
        distances, indices = self.model.kneighbors(x)
        test_sample = distances.shape[0]
        test_feat = self.feat
        y_test = np.zeros((test_sample, test_feat))
        i = 0
        
        for index, distance in zip(indices, distances):
            try:
                weight = (1.0/distance)**(self.weight)
            except:
                weight = np.zeros(distance.shape[0])
                weight[np.argmin(distance)] = 1
            if np.any(weight < np.finfo(float).eps):
                weight = np.ones(distance.shape[0])
            summation = np.dot(weight, self.y_train[index])

            y_test[i, :] = summation / (np.sum(weight))
            i += 1
        return y_test
    
    def score(self, x, y):
        self.check_fit()
        y_pred = self.predict(x)
        return r2_score(y, y_pred)
    
    def error_distribution(self, x, y):
        self.check_fit()
        y_pred = self.predict(x)
        error = y - y_pred#np.sqrt((y - y_pred)**2)
        plt.rcParams["font.size"] = 14
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.hist(error, bins = 200, label = "$N_R$={0}, $W_D$={1}".format(self.neighbor, self.weight))
        ax.legend()
        ax.set_title("k-NN $C_L$ error histgram", fontsize=18)
        ax.set_xlabel("$C_L$ error")
        ax.set_ylabel("frequency")
        plt.show()
        print(error_print(y_pred, y))
        

def svr(x_train, y_train, kernel='linear', params_cnt=20, figname="sample.png"):
    # generator for closs validation
    def gen_cv():
        m_train = np.floor(y_train.shape[0]*0.75).astype(int)
        train_indices = np.arange(m_train)
        test_indices = np.arange(m_train, y_train.shape[0])
        yield (train_indices, test_indices)

    params = {'C':np.logspace(-10, 10, params_cnt), 'epsilon':np.logspace(-10, 10, params_cnt)}
    gridsearch = GridSearchCV(SVR(kernel=kernel, gamma="scale"), params, cv=gen_cv(), scoring='r2', return_train_score=True)
    gridsearch.fit(x_train, y_train)
    # 検証曲線
    plt_x, plt_y = np.meshgrid(params["C"], params["epsilon"])
    fig = plt.figure(figsize = (8, 8))
    fig.subplots_adjust(hspace = 0.3)
    for i in range(2):
        if i == 0:
            plt_z = np.array(gridsearch.cv_results_["mean_train_score"]).reshape(params_cnt, params_cnt, order = "F")
            title = "Train"
        else:
            plt_z = np.array(gridsearch.cv_results_["mean_test_score"]).reshape(params_cnt, params_cnt, order = "F")
            title = "Cross Validation"
        ax = fig.add_subplot(2, 1, i + 1)
        CS = ax.contour(plt_x, plt_y, plt_z, levels = [0.6, 0.65, 0.7, 0.75, 0.8, 0.85])
        ax.clabel(CS, CS.levels, inline = True)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("C")
        ax.set_ylabel("epsilon")
        ax.set_title(title)
    plt.suptitle("Validation curve / Gaussian SVR")
    plt.savefig(figname)
    plt.close()

    return SVR(kernel=kernel, C=gridsearch.best_params_["C"], epsilon=gridsearch.best_params_['epsilon'], gamma="scale")

def sample_x_y(kind=1, plot=False):
    def test_func1(x, m=3, b=1):
        return m * x + b * np.random.normal(loc = 0, scale = 1, size = x.shape[0])
        
    x = 0.001*np.random.randint(low = -1000, high = 1000, size = 100)
    if kind == 1:
        y = test_func1(x, b=0.5)
    
    if plot:
        plt.plot(x, y, "x")
        plt.show()
    return x.reshape(-1, 1), y
    

def main():
    #for shape_type in [3]:#range(4):
    shape_type = 3
    mode = "fourier"
    n_feat = 10
    for n_sample in range(10, 410, 2):
        # x, y = sample_x_y()
        # x_train, x_test, y_train, y_test, rescale_x, rescale_y = load_x_y(str(shape_type))
        dimReduce = "PCA" + str(n_feat)

        x_train, x_test, y_train, y_test, rescale_x, rescale_y = load_mixed_and_shuffle(mode=mode,
                                                                                        n_samples=n_sample,
                                                                                        __dimReduce=dimReduce,
                                                                                        __test_size=0.01,
                                                                                        __dataReductMethod="kmeans_norm_pca" + str(
                                                                                            n_sample),
                                                                                        __mix=False)
        y_train = y_train.flatten()
        y_test = y_test.flatten()
        
        regs = [linear(), ridge(), lasso(), elasticnet()]
        weights = [0.1, 0.5, 1.0, 5.0, 10.0]
        case_name = "log\\svr" + str(x_train.shape) + "_" + str(shape_type)
        
        regs = [svr(x_train, y_train, kernel='rbf', figname = case_name + ".png")]#[svr(x_train, y_train), svr(x_train, y_train, kernel='rbf')]
        for reg in regs:
            reg.fit(x_train, y_train)
            print(shape_type, reg, reg.score(x_test, y_test))

            with open("vsSVR.csv", "a") as f:
                # f.write(str(x_train.shape) + "\n")
                # f.write(str(shape_type) + "\n")
                # f.write(str(reg) + "\n")
                # f.write(str(reg.score(x_test, y_test)) + "\n")
                f.write(error_print(reg.predict(x_test), y_test, csv=True, r2=True).replace("\n",""))
                f.write("svr,{0},{1},[2]\n".format(n_feat, n_sample,mode))

            # print(reg.coef_)
            # print(reg.intercept_)
    
def mainKNN():
    def plot_error_detail(shape_type=0, n_r=9, w_d=16):
        x_train, x_test, y_train, y_test, rescale_x, rescale_y = load_x_y(str(shape_type))
        print(x_train.shape, x_test.shape)
        #exit()
        # x_train, y_train = x_train[:1000], y_train[:1000]
        reg = NeighborWrap(n_r, w_d, shape_type, x_train.shape[0])
        reg.fit(x_train, y_train)
        reg.error_distribution(x_test, y_test)
    
    def cname_gen(shape_type, x_train, n_neighbor, weight, r2):
        return "{0},{1},{2},{3},{4},".format(shape_type, x_train.shape, n_neighbor, weight, r2)

    #for shape_type in range(1):
    #plot_error_detail(shape_type=0)
    #exit()
    modes = ["circum"]#, "equidistant", "crowd", "circum"]
    shape_types = [0]#[3, 1, 2, 0]
    skip_styles = [old_skip_clustering, old_skip_unballanced, old_skip_wing, old_skip_angle, ]
    # skip_names = ["unballanced", "wing", "angle", "cluster"]
    third_args = [ "kmeans_norm_pca"]#None, 40, None, "kmeans_norm_pca"]
    otherInfos = ["_cl","_ub", "_wg", "_ag"]
    # for shape_type in range(1):
    for mode, shape_type in zip(modes, shape_types):
        for style, arg, info in zip(skip_styles, third_args, otherInfos):
            #if True:#(shape_type == 3 and info == "_cl") or shape_type != 3:
            for n_sample in range(1001,1016):
                # x_train, x_test, y_train, y_test, rescale_x, rescale_y = load_x_y(str(shape_type))

                # x_train, x_test, y_train, y_test, rescale_x, rescale_y = style(mode, 40, arg, n_sample)
                if n_sample < 201:
                    dimReduce = "PCA" + str(n_sample)
                else:
                    dimReduce = None

                x_train, x_test, y_train, y_test, rescale_x, rescale_y = load_mixed_and_shuffle(mode=mode,
                                                                                                n_samples=n_sample,
                                                                                                __dimReduce=dimReduce,
                                                                                                __test_size=0.01,
                                                                                                __dataReductMethod="kmeans_norm_pca"+str(n_sample),
                                                                                                __mix=False)

                y_train = y_train.flatten()
                y_test = y_test.flatten()
                print(x_train.shape, x_test.shape)
                case_name = "log\\knnW1001" + str(x_train.shape) + "_" + str(shape_type)

                # for i in range(5):
                    # for j in range(5):
                    # regs = [NeighborWrap(1, weight), NeighborWrap(5), NeighborWrap(10), NeighborWrap(50)]
                    # for reg in regs:
                n_neighbor =9# (i + 1)**2
                weight = 16#(j)**2
                if (n_neighbor != 1) or (n_neighbor == 1 and weight == 1):
                    print(n_neighbor, weight, info)
                    reg = NeighborWrap(n_neighbors = n_neighbor, weight = weight, shapetype=shape_type, n_samples=x_train.shape[0], otherInfo=info)
                    reg.fit(x_train, y_train)
                    print(shape_type, reg, reg.score(x_test, y_test))
                    csvLine = cname_gen(shape_type, x_train, n_neighbor, weight, reg.score(x_test, y_test))
                    csvLine += error_print(rescale_y(reg.predict(x_test)), rescale_y(y_test), csv=True)
                    #with open(case_name + ".csv", "a") as f:
                    with open("vsLasso.csv", "a") as f:
                        f.write(csvLine)


def load_x_y(shape_type = str(0), n_samples=200000):
    if shape_type == "0":
        mode = "circumferential"
    elif shape_type == "1":
        mode = "equidistant"
    elif shape_type == "2":
        mode = "crowd"
    elif shape_type == "3":
        mode = "fourier"
    elif shape_type == "4":
        mode = "msdf"
    else:
        raise ValueError
    
    """
    fname_lift_train = "NACA4\\s0000_e5000_a040_odd.csv"
    # fname_lift_test = "NACA5\\s21001_e25199_a040.csv"
    fname_lift_test = "NACA5\\s11001_e65199_a040.csv"
    shape_type = str(int(shape_type) + 1)
    if shape_type == str(0):
        fname_shape_train = "NACA4\\shape_fourier_5000_odd_closed.csv"
        fname_shape_test = "NACA5\\shape_fourier_all_closed.csv"
    elif shape_type == str(1):
        fname_shape_train = "NACA4\\shape_equidistant_5000_odd_closed.csv"
        fname_shape_test = "NACA5\\shape_equidistant_all_mk2.csv"
    elif shape_type == str(2):
        fname_shape_train = "NACA4\\shape_crowd_0.1_0.15_30_50_20_5000_odd_xy_closed.csv"
        fname_shape_test = "NACA5\\shape_crowd_all_mk2_xy_closed.csv"
    elif shape_type == str(3):
        fname_shape_train = "NACA4\\shape_modified_fourier_5000_odd.csv"
        fname_shape_test = "NACA5\\shape_modified_fourier_all_mk2.csv"
    elif shape_type == str(4):
        fname_shape_train = "NACA4\\shape_circumferential_5000_odd.csv"
        fname_shape_test = "NACA5\\shape_circumferential_all_mk2.csv"
    else:
        print(shape_type)
        exit()
    
    source = "G:\\Toyota\\Data\\Incompressible_Invicid\\training_data\\"
    rr = 1
    sr = 1
    s_odd = 0
    s_skiptype = False
    unballanced = False
    reductioning = False
    X_train, y_train = read_csv_type3(source, fname_lift_train, fname_shape_train, shape_odd = s_odd, read_rate = rr,
                                      skip_rate = sr, skip_angle = s_skiptype, unbalance = unballanced)
    step = 74
    X_train, y_train = X_train[::step], y_train[::step]
    print(step, X_train.shape)
    x_test, y_test = read_csv_type3(source, fname_lift_test, fname_shape_test, shape_odd = s_odd, read_rate = rr)
    print(x_test.shape)
    # exit()
    """
    # dimReductMethod = "PCA"
    # dimReduce = dimReductMethod + str(n_feat_dim)
    # if clustering:
    # dataReduceMethod = "kmeans_norm_pca_" + str(2000)

    return load_mixed_and_shuffle(mode = mode,
                                  n_samples = n_samples,
                                  useN5w2 = False,
                                  shuffle = False,
                                  __step = 1,
                                  __dimReduce = None,
                                  __test_size = 0.01,
                                  __dataReductMethod = None,
                                  __mix = False)
    
if __name__ == '__main__':
    mainKNN()
    # main()