# coding: utf-8
import numpy as np
from read_training_data import read_csv_type3
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
from dataset_reduction_inviscid import data_reduction
from scipy.spatial import ConvexHull
from scipy.stats import gaussian_kde, gmean, entropy
from scipy.integrate import trapz
import statsmodels.api as sm

def hullplot_and_calc_area(fname_list, pts_list, color_list):
    hulls = []
    figs = []
    axs = []
    fig_hull = plt.figure()
    ax_hull = fig_hull.add_subplot(111)
    for i, pts in enumerate(pts_list):
        hulls.append(ConvexHull(points = pts))
        figs.append(plt.figure())
        hull = hulls[i]
        fig = figs[i]
        axs.append(fig.add_subplot(111))
        ax = axs[i]
        ax.plot(pts[:,0], pts[:,1], ".", markersize=2, color=color_list[i], label=fname_list[i])
        # ax_hull.plot(pts[:, 0], pts[:, 1], ".", markersize = 1, color = color_list[i], label = fname_list[i])
        for simplex in hull.simplices:
            ax.plot(pts[simplex,0], pts[simplex,1], "k-", color=color_list[i])
            ax_hull.plot(pts[simplex,0], pts[simplex,1], "k-", color=color_list[i])#, label=fname_list[i])
        ax.legend()
        ax_hull.legend()
        ax.grid()
        ax_hull.grid()
        ax.set_xlabel("1st principle component")
        ax.set_ylabel("2nd principle component")
        ax_hull.set_xlabel("1st principle component")
        ax_hull.set_ylabel("2nd principle component")
            

        area = hull.area
        fname_list[i] += str(area)
        
        # plt.savefig(fname)
        print(area)
    plt.show()
    
def load_datasets(sr, rr, s_odd, s_skiptype, source, fname_lift_train, fname_lift_test, fname_shape_train, fname_shape_test, standardized = True):
    # 1つずつ20万件を読み出した全体のデータ
    X_train_1, y_train_1 = read_csv_type3(source, fname_lift_train, fname_shape_train, shape_odd = 0,
                                          read_rate = 1, skip_rate = 1, skip_angle = s_skiptype,
                                          unbalance = False)
    # 角度を16度刻みにして読み出した25000件のデータ
    X_train_2, y_train = read_csv_type3(source, fname_lift_train, fname_shape_train, shape_odd = s_odd,
                                        read_rate = rr, skip_rate = 8, skip_angle = True,
                                        unbalance = False)
    # 形状を16飛ばしにして読み出したデータ
    X_train_3, y_train = read_csv_type3(source, fname_lift_train, fname_shape_train, shape_odd = s_odd,
                                        read_rate = rr, skip_rate = 8, skip_angle = False,
                                        unbalance = False)
    # 末尾だけ抜き出してきたデータ
    X_train_4, y_train = read_csv_type3(source, fname_lift_train, fname_shape_train, shape_odd = s_odd,
                                        read_rate = rr, skip_rate = 8, skip_angle = s_skiptype,
                                        unbalance = True)
    
    # クラスタリングによって抜き出してきたデータ
    X_train_5, y_train = data_reduction(X_train_1, y_train_1, reduction_target = 25000, preprocess = "None")
    
    X_test, y_test = read_csv_type3(source, fname_lift_test, fname_shape_test, shape_odd = s_odd,
                                    read_rate = rr)

    if standardized:
        pass
        """
        scalar = StandardScaler()
        scalar.fit(X_train_1)
        X_train_1 = scalar.transform(X_train_1)
        X_train_2 = scalar.transform(X_train_2)
        X_train_3 = scalar.transform(X_train_3)
        X_train_4 = scalar.transform(X_train_4)
        X_train_5 = scalar.transform(X_train_5)
        X_test = scalar.transform(X_test)
        """
    return X_train_1, X_train_2, X_train_3, X_train_4, X_train_5, X_test

def pca_main(X_train_1, X_train_2, X_train_3, X_train_4, X_train_5, X_test):
    pca = PCA(n_components = 2)
    pca.fit(X_train_1)
    transformed_1 = pca.fit_transform(X_train_1)
    print(pca.explained_variance_ratio_)
    print(sum(pca.explained_variance_ratio_))
    fnames = ["Original", "test", "less_angle", "less_shape", "bias", "clustering"]
    color = ["blue", "red", "green", "purple", "pink", "lime"]
    
    transformed_test = pca.fit_transform(X_test)
    transformed_2 = pca.fit_transform(X_train_2)
    transformed_3 = pca.fit_transform(X_train_3)
    transformed_4 = pca.fit_transform(X_train_4)
    transformed_5 = pca.fit_transform(X_train_5)
    hullplot_and_calc_area(fnames,
                           [transformed_1, transformed_test, transformed_2, transformed_3, transformed_4,
                            transformed_5],
                           color)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(transformed_1[:, 0], transformed_1[:, 1], alpha = 0.3, color = "dodgerblue", label = "ALL_Data")
    
    # ax.scatter(transformed_2[:, 0], transformed_2[:, 1], alpha = 0.5, color="aquamarine", label = "Angle Skipped")
    # ax.scatter(transformed_3[:, 0], transformed_3[:, 1], alpha = 0.5, color="darkorange", label = "Shape Skipped")
    # ax.scatter(transformed_4[:, 0], transformed_4[:, 1], alpha = 0.5, color="limegreen", label = "Near End Only")
    
    ax.scatter(transformed_test[:, 0], transformed_test[:, 1], marker = "*", alpha = 0.5, color = "mediumvioletred",
               label = "Test")
    ax.legend(bbox_to_anchor = (0, 1), loc = "upper left", borderaxespad = 1, fontsize = 12)
    ax.set_title("PCA of Incompressible & Invicid Dataset")
    ax.set_xlabel("1st principal component")
    ax.set_ylabel("2nd principal component")
    ax.set_xlim(-25, 25)
    ax.set_ylim(-35, 35)
    plt.show()
    fig.show()


def pca(source, fname_lift_train, fname_lift_test, fname_shape_train, fname_shape_test, standardized = True):
    # r_rate = [1, 2, 4, 8]
    r_rate = [1]
    # s_rate = [2, 4, 8]
    s_rate = [1]
    # s_skiptype = [True, False]
    s_skiptype = True
    # r_rate = [1, 2]
    # r_rate = [4, 8]
    # r_rate = [16, 32]
    # r_rate = [64, 160]
    
    for sr in s_rate:
        for rr in r_rate:
            if rr == 1:
                s_odd = 0  # 全部読みだす
            elif fname_shape_train.find("fourier") != -1:
                s_odd = 3  # 前方から読み出す(fourier用)
            else:
                s_odd = 4  # 全体にわたって等間隔に読み出す(equidistant, dense用)

            X_train_1, X_train_2, X_train_3, X_train_4, X_train_5, X_test = load_datasets(sr, rr, s_odd, s_skiptype,
                                                                                          source, fname_lift_train,
                                                                                          fname_lift_test,
                                                                                          fname_shape_train,
                                                                                          fname_shape_test,
                                                                                          standardized)
            evaluate_dataset(X_train_1, X_train_2, X_train_3, X_train_4, X_train_5, X_test)
            exit()
            # pca_main(X_train_1, X_train_2, X_train_3, X_train_4, X_train_5, X_test)

def multivariate_cv(data_list, method="AlbeltAndZhang"):
    # multivariate coefficient of variance (MCV)
    # data_list = (n_samples, m_variables)
    cov = np.cov(data_list.T)   # covariance_matrix
    mu = np.average(data_list, axis = 0) # average_vector
    mu_norm = np.dot(mu.T, mu)
    if method == "AlbeltAndZhang":
        mcv = np.sqrt(np.dot(mu.T, np.dot(cov, mu)) / mu_norm ** 2)
    elif method == "VoinovAndNikulin":
        mcv = np.sqrt(1.0 / np.dot(mu.T, np.dot(np.linalg.inv(cov), mu)))
    elif method == "VanValen":
        mcv = np.sqrt(np.trace(cov) / mu_norm)
    elif method == "average":
        mcv1 = multivariate_cv(data_list)
        mcv2 = multivariate_cv(data_list, method = "VoinovAndNikulin")
        mcv3 = multivariate_cv(data_list, method = "VanValen")
        mcv = (mcv1 + mcv2 + mcv3) / 3.0
    else:
        raise ValueError
    return mcv

def evaluate_dataset(X_train_1, X_train_2, X_train_3, X_train_4, X_train_5, X_test):
    def scipy_kde(data):
        kde = gaussian_kde(data)
        return #kde(data)
    
    def kde_pdf(data, library="statsmodels"):
        if library == "statsmodels":
            kde_model = sm.nonparametric.KDEMultivariate(data, var_type = "c"*data.shape[1])
            return kde_model.pdf(data)

        elif library == "statsmodels_aprx":
            dim = data.shape[1]
            size = 2**6
            method = "summation"#"summation"    # "multiply"
            if method == "multiply":
                pdf = np.ones(size)
            elif (method == "summation"):
                pdf = np.zeros(size)
            elif (method == "integration") or ((method == "entropy")):
                pdf = np.zeros(dim)

            for i in range(dim):
                kde_model = sm.nonparametric.KDEUnivariate(data[:, i])
                kde_model.fit(gridsize=size)
                if method == "multiply":
                    # print((kde_model.density)**(1/dim))
                    pdf *= (kde_model.density)**(1/dim)
                    # pdf = np.where(pdf > 0, pdf, 0)
                    tmp = pdf * kde_model.density
                    pdf = np.where(pdf > 0, tmp, np.where(kde_model.density > 0, tmp, -tmp))

                elif method == "summation":
                    pdf += kde_model.density / dim

                elif method == "integration":
                    x = kde_model.support[(kde_model.support > 0) * (kde_model.support < 1)]
                    y = (kde_model.density[(kde_model.support > 0) * (kde_model.support < 1)] - 1)**2
                    error = trapz(y=y, x=x)

                    pdf[i] = np.log(error)
                elif method == "entropy":
                    pk = np.where(kde_model.density>0, kde_model.density, 0)
                    pdf[i] += entropy(pk = pk[(kde_model.support > 0) * (kde_model.support < 1)])
                    """
                if i % 5 == 0:
                    plt.plot(kde_model.support, pdf)
                    plt.show()
                """
            if method == "integration":
                return np.exp(pdf)
            elif method == "entropy":
                return pdf
            else:
                return pdf[(kde_model.support > 0) * (kde_model.support < 1)]
        
    def density_ratio(data, library="statsmodels", log10=False):
        pdf = kde_pdf(data, library)
        dmax = np.max(pdf)
        dmin = np.min(pdf)
        # print(dmax, dmin)
        # print(dmax, dmin)
        if False:   # not dmin > 0.0:
            return np.inf
        ratio = dmax / dmin
        if log10:
            return np.log10(ratio)
        else:
            return ratio
            # return gmean(pdf)

    def control_volume(data):
        # diag = np.sqrt(np.sum((np.max(data, axis=0) - np.min(data, axis=0))**2))
        edge = np.max(data, axis=0) - np.min(data, axis=0)
        cv_r = np.median(edge) / np.average(edge)
        # print((np.max(data, axis=0) - np.min(data, axis=0)).shape)
        return cv_r#diag#np.prod((np.max(data, axis=0) - np.min(data, axis=0)) / data.shape[0])

    def scaling_factor(data, worst=False):
        # dim = data.shape[1]
        # for i in range(dim):
        # dmax = np.max(data, axis=0)
        # dmin = np.min(data, axis=0)
        # std = np.std(data, axis=0)
        # sf = (dmax - dmin) / std
        # print(np.average(dmax), np.average(dmin), np.average(std), np.average(sf))
        sf = np.std(data, axis=0)
        # print(np.max(sf), np.min(sf), np.average(sf), gmean(sf))
        if not worst:
            return gmean(sf)
        else:
            return np.min(sf)

    def univariate_cv(data, worst=False):
        cv = np.std(data, axis=0) / np.average(data, axis=0)
        if not worst:
            return gmean(cv)
        else:
            return np.min(cv)

    def get_index(data, logscale=False, library="statsmodels_aprx", method = "AlbeltAndZhang"):
        cv = control_volume(data)
        print(cv)
        # scalar = MinMaxScaler()
        # data = scalar.fit_transform(data)
        sf = 1#scaling_factor(data)
        dratio = (density_ratio(data, library))
        mcv = univariate_cv(data)   #multivariate_cv(data, method)
        if dratio > 0:
            index = dratio / (mcv * sf)
        else:
            index = np.inf
        # print(sf, mcv, dratio, index)
        # worst_index = dratio / (univariate_cv(data, worst=True) * scaling_factor(data, worst=True))
        # print(index/worst_index)
        if logscale:
            return np.log10(index)
        else:
            return index

    scalar = MinMaxScaler()
    X_train_1 = scalar.fit_transform(X_train_1)
    X_train_2 = scalar.transform(X_train_2)
    X_train_3 = scalar.transform(X_train_3)
    X_train_4 = scalar.transform(X_train_4)
    X_train_5 = scalar.transform(X_train_5)
    X_test = scalar.transform(X_test)

    print("original:" + str(get_index(X_train_1)))
    print("s_angle: " + str(get_index(X_train_2)))
    print("s_shape: " + str(get_index(X_train_3)))
    print("unballance: " + str(get_index(X_train_4)))
    print("cluster: " + str(get_index(X_train_5)))
    print("test : " + str(get_index(X_test)))

 

if __name__ == '__main__':
    source = "G:\\Toyota\\Data\\Incompressible_Invicid\\training_data\\"
    fname_lift_train = "NACA4\\s0000_e5000_a040_odd.csv"
    fname_lift_test = "NACA5\\s21001_e25199_a040.csv"
    fname_shape_train = "NACA4\\shape_fourier_5000_odd_closed.csv"
    fname_shape_test = "NACA5\\shape_fourier_all_closed.csv"
    
    pca(source, fname_lift_train, fname_lift_test, fname_shape_train, fname_shape_test)
