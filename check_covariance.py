# coding: utf-8
import numpy as np
from read_training_data import read_csv_type3
from scipy.linalg import det
from scipy.linalg import eigvals
import scipy.stats as sp
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def main():
    a = 1
    source = "G:\\Toyota\\Data\\Incompressible_Invicid\\training_data\\"
    fname_lift_train = "NACA4\\s0000_e5000_a040_odd.csv"
    fname_lift_test = "NACA5\\s21001_e25199_a040.csv"
    fname_shape_train = "NACA4\\shape_fourier_5000_odd.csv"
    fname_shape_test = "NACA5\\shape_fourier_all.csv"

    # r_rate = [1, 2, 4, 8]
    r_rate = [1]
    # s_rate = [2, 4, 8]
    s_rate = [8]
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

            X_train, y_train = read_csv_type3(source, fname_lift_train, fname_shape_train, shape_odd = s_odd, read_rate = rr, skip_rate=sr, skip_angle = s_skiptype, unbalance = True)
            # X_train[:, 0] /= 180.0
            # X_train = np.concatenate([X_train.T, y_train.T]).T
            scalar = StandardScaler()
            scalar.fit(X_train)
            X_train = scalar.transform(X_train) # scalarに標準偏差等のデータが全部入ってるから，再利用はそこから

            # covariance = np.cov(X_train)
            pca = PCA(n_components = 12)
            pca.fit(X_train)
            transformed = pca.fit_transform(X_train)
            # print(det(covariance))
            # print(eigvals(covariance))
            print(pca.explained_variance_ratio_)
            print(sum(pca.explained_variance_ratio_))
            exit()
            for i in range(len(transformed)):
                plt.scatter(transformed[i,0], transformed[i,1])
            plt.title("(iv)")
            plt.xlabel("pc1")
            plt.ylabel("pc2")
            plt.show()
            
            
    
if __name__ == '__main__':
    main()
    