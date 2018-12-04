# -- coding: utf-8 --
import numpy as np
import os
from keras.models import model_from_json
from read_training_data import read_csv_type3
from scatter_plot import make_scatter_plot
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

def inference(source, x_test, y_test, case_name, scatter=True, anglerplot=False):
    json_name = "learned\\" + case_name + "_mlp_model_.json"
    weight_name = "learned\\" + case_name + "_mlp_weight.h5"

    # あとでこの辺を自由に変更できるようにする
    # fname_lift_train = "NACA4\\s0000_e5000_a040_odd.csv"
    # fname_shape_train = "NACA4\\shape_fourier_5000_odd.csv"
    # X_test, y_test = read_csv_type3(source, fname_lift_train, fname_shape_train, shape_odd=0, read_rate=1)

    model = model_from_json(open(source + json_name).read())
    model.load_weights(source + weight_name)

    model.summary()

    model.compile(loss="mean_squared_error",
                  optimizer='Adam')

    # score = model.evaluate()
    # print('test loss :', score[0])
    # print('test accuracy :', score[1])
    
    y_predict = model.predict(x_test)
    r2 = r2_score(y_test, y_predict)
    rms = np.sqrt(mean_squared_error(y_test, y_predict))
    if scatter:
        case_name += "r2_" + str(r2) + "_rms_" + str(rms)
        make_scatter_plot(y_test, y_predict, "CL(Exact)", "CL(Predict)", path="G:\\Toyota\\Data\\Incompressible_Invicid\\fig_post\\", fname=case_name)

    if anglerplot:
        tekito = (99 + 13) * 40  # 22012
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.plot(x_test[tekito:tekito + 10, 0], y_test[tekito:tekito + 10], marker="o", markersize=10, color="dodgerblue", label = "Exact")
        ax.plot(x_test[tekito:tekito + 10, 0], y_predict[tekito:tekito + 10], marker="o", markersize=10, color="mediumvioletred", label = "Estimate")
        ax.plot(x_test[tekito:tekito + 10, 0], y_test[tekito:tekito + 10], color="dodgerblue", linestyle="dashdot", linewidth=1.0)
        ax.plot(x_test[tekito:tekito + 10, 0], y_predict[tekito:tekito + 10], color="mediumvioletred", linestyle="dashdot", linewidth=1.0)
        ax.legend(bbox_to_anchor=(0, 1), loc="upper left", borderaxespad=1, fontsize=12)
        ax.set_xlabel("Angle of Attack [deg]")
        ax.set_ylabel("$\it{C_{L}}$")
        ax.set_title("NACA22012 Wing $\it{C_{L}}$ distribution")
        ax.grid(True)

        fig.savefig(source + case_name + "_NACA22012.png")


# 保存先を検索し，ありそうなファイル名を検索，発見したらリストに追加
def case_name_list_generator(source, fname_lift_test, some_case_test=False, some_case = [], scatter=True, anglerplot=False):
    casename_list = []
    top = "learned\\"
    head_list = ["fourierSr_", "concertrate_", "equidistant_"]
    mid1_list = [str(200000), str(100000), str(50000), str(25000)]
    mid2_list = ["_less_angle_", "_less_shape_"]
    tail_rate = [1, 2, 4, 8]
    tail_total = 200
    bottom1 = "_mlp_model_.json"
    bottom2 = "_mlp_weight.h5"
    fname_shape_list = ["NACA5\\shape_fourier_all.csv", "NACA5\\shape_crowd_0.1_0.15_30_50_20_all.csv", "NACA5\\shape_equidistant_all.csv"]
    some_case_index = 0
    some_case_total = len(some_case)
    for i in range(3):
        head = head_list[i]
        fname_shape_test = fname_shape_list[i]
        for mid1 in mid1_list:
            for mid2 in mid2_list:
                for rr in tail_rate:
                    tail = str(int(tail_total / rr))
                    fname0 = head + mid1 + mid2 + tail
                    fname1 = source + top + fname0
                    if os.path.exists(fname1 + bottom1):
                        if os.path.exists(fname1 + bottom2):
                            if rr == 1:
                                s_odd = 0  # 全部読みだす
                            elif head == head_list[0]:
                                s_odd = 3  # 前方から読み出す(fourier用)
                                
                            else:
                                s_odd = 4  # 全体にわたって等間隔に読み出す(equidistant, dense用)
                            if some_case_test:
                                if some_case[some_case_index] == fname0:
                                    some_case_index += 1
                                    x_test, y_test = read_csv_type3(source, fname_lift_test, fname_shape_test,
                                                                    shape_odd = s_odd, read_rate = rr)
                                    inference(source, x_test, y_test, fname0, scatter, anglerplot)
                            else:
                                x_test, y_test = read_csv_type3(source, fname_lift_test, fname_shape_test,
                                                                shape_odd = s_odd, read_rate = rr)
                                inference(source, x_test, y_test, fname0)
                            
                            if (some_case_total != 0) and (some_case_index == some_case_total):
                                print("all process finished")
                                exit()
                                
    return casename_list

def some_case_test(source, fname_lift_test):
    # head_list = ["fourierSr_", "concertrate_", "equidistant_"]
    # mid1_list = [str(200000), str(100000), str(50000), str(25000)]
    # mid2_list = ["_less_angle_", "_less_shape_"]
    # tail_rate = [1, 2, 4, 8]
    # tail_total = 200
    some_case = ["fourierSr_100000_less_shape_200", "fourierSr_50000_less_angle_200", "fourierSr_25000_less_angle_200"]
    case_name_list_generator(source, fname_lift_test, some_case_test = True, some_case = some_case, scatter = False, anglerplot = True)
    

if __name__ == '__main__':
    source = "G:\\Toyota\\Data\\Incompressible_Invicid\\training_data\\"
    fname_lift_test = "NACA5\\s21001_e25199_a040.csv"
    some_case_test(source, fname_lift_test)
    # case_name_list_generator(source, fname_lift_test)

    
