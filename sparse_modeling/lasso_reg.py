# coding: utf-8
# sample from https://github.com/tksmd/hannari-python-2/blob/master/demo.ipynb
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Lasso
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

from read_training_data import load_mixed_and_shuffle
import os
import pickle

shape_expression = "fourier"
n_samples_total = 300000
n_feat_dim = 10
# poly_dim = 2
alpha = 0.002

dimReductMethod = "PCA"
def main(poly_dim=2, data=20, clustering=True):
    dimReduce = dimReductMethod + str(n_feat_dim)
    if clustering:
        dataReduceMethod = "kmeans_norm_pca_" + str(data)
    else:
        dataReduceMethod = "random_" + str(data)
    
    X_train, X_test, y_train, y_test, rescale_x, rescale_y = load_mixed_and_shuffle(mode = shape_expression,
                                                                                    n_samples = n_samples_total,
                                                                                    __dimReduce = dimReduce,
                                                                                    __test_size = 0.01,
                                                                                    __dataReductMethod = dataReduceMethod,
                                                                                    __mix = False)
    x = X_train
    y = y_train
    t = X_test
    print(x.shape)
    print(t.shape)
    if clustering:
        fname = "lasso_p{0}_f{1}_d{2}".format(poly_dim, n_feat_dim, data)
    else:
        fname = "random_lasso_p{0}_f{1}_d{2}".format(poly_dim, n_feat_dim, data)
    fname_model = "pkl//" + fname + "_model.pkl"
    fname_lasso = "pkl//" + fname + "_lasso.pkl"
    if os.path.exists(fname_model) and os.path.exists(fname_lasso):
        print("load {0}".format(fname_model))
        model = pickle.load(open(fname_model, "rb"))
        lasso = pickle.load(open(fname_lasso, "rb"))
    else:
        # 多項式回帰
        poly_preprocess = PolynomialFeatures(poly_dim, include_bias = False)
        lasso = Lasso(alpha = alpha, max_iter = 500000, tol = 0.000001)
    
        model = make_pipeline(poly_preprocess, lasso)
        model.fit(x, y)
        pickle.dump(model, open(fname_model, "wb"))
        pickle.dump(lasso, open(fname_lasso, "wb"))
    
    y_predicted = model.predict(x)
    t_predicted = model.predict(t)
    r2_train = r2_score(y_train, y_predicted)
    r2_test = r2_score(y_test, t_predicted)
    
    print(r2_train, r2_test)
    with open("log_random.txt", "a") as f:
        f.write("{0},{1},{2},{3},{4},{5},{6}\n".format(poly_dim, alpha, n_feat_dim, data, r2_train, r2_test, fname))
        
    coef = lasso.coef_
    idx, = coef.nonzero()
    # print(coef)
    # print(idx)
    plt.figure(figsize = (8, 4))
    plt.stem(idx, coef[idx], basefmt = 'g-')
    plt.title('Lasso')
    # plt.show()
    plt.savefig("img//" + fname + ".png")
    plt.close()
    
if __name__ == '__main__':
    np.random.seed(1)
    from itertools import product
    for p, i in product(range(2, 4), range(200)):
        print(p, i)
        data = 2 * i + 10
        main(poly_dim = p, data = data, clustering = False)
    