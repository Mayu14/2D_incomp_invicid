# coding: utf-8
# sample from https://github.com/tksmd/hannari-python-2/blob/master/demo.ipynb
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Lasso
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

def main():
    poly_dim = 80
    
    w = np.concatenate(([-1], [0] * (poly_dim - 2), [1, 0]))
    poly_dim = 300
    f = np.poly1d(w)
    t = np.linspace(0, 1, 200)
    x = np.arange(0, 1.1, step = 0.1)
    y = f(x) + np.random.normal(loc = 0.01, scale = 0.01, size = len(x))
    
    def format_plot():
        plt.xlabel('t')
        plt.ylabel('y')
        plt.xlim((0, 1.01))
        plt.ylim((0, 1))
        plt.grid()
        plt.legend()
    
    plt.figure(figsize = (8, 6))
    plt.plot(t, f(t), linestyle = 'dashed', color = 'green')
    plt.scatter(x, y, color = 'green', marker = 'o', label = "observed data")
    format_plot()
    plt.show()
    
    # 多項式回帰
    poly_preprocess = PolynomialFeatures(poly_dim, include_bias = False)
    lasso = Lasso(alpha = 0.002, max_iter = 500000, tol = 0.000001)
    
    def fit_and_predict(predictor):
        model = make_pipeline(poly_preprocess, predictor)
        model.fit(x.reshape(-1, 1), y)
        y_predicted = model.predict(x.reshape(-1, 1))
        t_predicted = model.predict(t.reshape(-1, 1))
        return y_predicted, t_predicted
    
    y_predicted, t_predicted = fit_and_predict(lasso)
    
    plt.figure(figsize = (8, 6))
    plt.plot(t, f(t), linestyle = 'dashed', color = 'green', label = 'original')
    plt.plot(t, t_predicted, label = 'Lasso')
    plt.scatter(x, y_predicted, marker = 'o')
    format_plot()
    plt.show()
    coef = lasso.coef_
    idx, = coef.nonzero()
    plt.figure(figsize = (8, 4))
    plt.stem(idx, coef[idx], basefmt = 'g-')
    plt.title('Lasso')
    plt.show()


if __name__ == '__main__':
    np.random.seed(1)
    main()