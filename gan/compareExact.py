# coding: utf-8
import numpy as np
from dvm_beta import zu2circ
from arc_length_transformer import decode_into_cplx
from read_training_data import load_with_param_for_GAN
import matplotlib.pyplot as plt
from scatter_plot import make_scatter_plot

def main(X_sim, number=0, job=8):
    total = int(X_sim.shape[0] / job)
    L_ex = np.zeros((total, 2))
    geta = total*number
    for n in range(total):
        num = geta + n
        aoa = X_sim[num][0]
        print(num, total, float(n)/total)
        shape_vector = X_sim[num][1:]
        z_aoa0 = decode_into_cplx(bnxy = shape_vector, from_concat_bn_xy = True, angle = 0.0,
                                  deg = True)
        # plt.plot(np.real(z_aoa0), np.imag(z_aoa0), ".")
        # plt.show()
        
        L_ex[n, 0], L_ex[n, 1] = zu2circ(z_aoa0, 1.0, aoa, deg = True)

        print(L_ex[n, :])
        if n % 1000 == 0:
            np.savez_compressed("Lex_mk2_{0}".format(number), L_ex=L_ex)
    return L_ex
    
def load_and_calc_exact():
    fname = "ganN2.npz"
    with np.load(fname, allow_pickle = True) as f:
        X_save, L_save = f["X_save"], f["L_save"]
        Z_test, y_save = f["Z_test"], f["L_test"]
    
    # prepare rescaler
    X_train, X_test, y_train, y_test, rescale_x, rescale_y = load_with_param_for_GAN(mode = "Fourier",
                                                                                     params = 1,
                                                                                     pname = "aspect")
    X_train, X_test = rescale_x(X_train), rescale_x(X_test)
    y_train, y_test = rescale_y(y_train), rescale_y(y_test)
    from sklearn.preprocessing import MinMaxScaler
    
    scalarX = MinMaxScaler(feature_range = (-1, 1))
    scalarX.fit(X_train)
    X_train, X_test = scalarX.transform(X_train), scalarX.transform(X_test)
    rescale_x = scalarX.inverse_transform
    
    scalarY = MinMaxScaler(feature_range = (-1, 1))
    scalarY.fit(y_train)
    y_train, y_test = scalarY.transform(y_train), scalarY.transform(y_test)
    rescale_y = scalarY.inverse_transform
    
    # rescale parameters
    L_save = np.concatenate([L_save.reshape(-1, 1), np.zeros_like(L_save).reshape(-1, 1)], axis = 1)
    X_save = rescale_x(X_save)
    y_save = rescale_y(y_save)[:, 0]
    L_save = rescale_y(L_save)[:, 0]
    # check vars
    print(y_save[:6])
    print(L_save[:6])
    
    print(np.corrcoef(y_save, L_save))
    print(np.sqrt(np.sum((y_save - L_save) ** 2)) / y_save.shape[0])
    # save
    return L_save, X_save, y_save
    

def open_and_scat(fin=1, job=8):
    if fin > 0:
        total = 1000 * fin
    else:
        total = 1
    split = int(X_save.shape[0] / job)
    # load
    L_concat_A = np.zeros((total, job))
    L_concat_B = np.zeros((total, job))
    for i in range(job):
        fname = "Lex_mk2_{0}.npz".format(i)
        with np.load(fname, allow_pickle = True) as f:
            L_ex = f["L_ex"]
        L_concat_A[:, i] = L_ex[:total, 0]
        L_concat_B[:, i] = L_ex[:total, 1]
    
    # test
    y_concat = np.zeros((total, job))
    for i in range(job):
        start = split * i
        stop = start + total
        y_concat[:, i] = y_save[start:stop]
    
    L_concat_A = L_concat_A.flatten()
    L_concat_B = L_concat_B.flatten()
    y_concat = y_concat.flatten()
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, median_absolute_error
    r2A = r2_score(L_concat_A, y_concat)
    r2B = r2_score(L_concat_B, y_concat)
    print(r2A, r2B, y_concat.shape)
    make_scatter_plot(data_a = L_concat_A, data_b = y_concat, label_a = "observed (specified.)",
                      label_b = "generated (exact sol.)",
                      fname = "lift_yyplot_{0}_R2_{1:.5f}".format("A", r2A), dash_line = "no")
    
    make_scatter_plot(data_a = L_concat_B, data_b = y_concat, label_a = "observed (specified.)",
                      label_b = "generated (exact sol.)",
                      fname = "lift_yyplot_{0}_R2_{1:.5f}".format("B", r2B), dash_line = "no", max_value=10, min_value=-10)
    """
    print(np.sqrt(np.sum((y_concat - L_concat_A) ** 2)) / y_concat.shape[0])
    print(np.sqrt(np.sum((y_concat - L_concat_B) ** 2)) / y_concat.shape[0])
    print(np.sum(np.abs(y_concat - L_concat_A)) / y_concat.shape[0])
    print(np.sum(np.abs(y_concat - L_concat_B)) / y_concat.shape[0])
    def error(y_concat, L_concat):
        print(mean_squared_error(y_concat, L_concat))
        print(mean_absolute_error(y_concat, L_concat))
        print(median_absolute_error(y_concat, L_concat))
        print(np.max(np.abs(y_concat - L_concat)))
        print(np.min(np.abs(y_concat - L_concat)))
        print(np.max(L_concat))
        print(np.min(L_concat))

    def makeLabel(L_concat):
        label = (L_concat < 10) * (L_concat > -10)
        y_concat2 = label * y_concat
        L_concat2 = label * L_concat
        error(y_concat2, L_concat2)

    error(y_concat, L_concat_A)
    error(y_concat, L_concat_B)
    # print(np.max(y_concat))
    # print(np.min(y_concat))
    makeLabel(L_concat_A)
    makeLabel(L_concat_B)
    """
if __name__ == '__main__':
    epoch = 100
    # for i in range(epoch + 1):
        # fname = "ganXL{0}.npz".format(i)
    L_save, X_save, y_save = load_and_calc_exact()
    L_ex = main(X_save, number = 7)
    # np.savez_compressed("lex", L_ex = L_ex)
    open_and_scat(fin=3)
    