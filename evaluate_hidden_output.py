# coding: utf-8
from read_training_data import read_csv_type3
import numpy as np
from matplotlib import pyplot as plt

def plot(data, ho):
    plt.plot(data, ho[:, 0], ".", label="1st component")
    plt.plot(data, ho[:, 1], ".", label="2nd component")
    
    plt.show()


def main():
    source = "G:\\Toyota\\Data\\Incompressible_Invicid\\training_data\\"
    s_odd = 0
    rr = 1
    ho = np.load("hidden_output_2.npy")
    fname_shape_test = "NACA5\\shape_modified_fourier_all.csv"
    fname_lift_test = "NACA5\\s21001_e25199_a040.csv"
    x_test, y_test = read_csv_type3(source, fname_lift_test, fname_shape_test, shape_odd = s_odd, read_rate = rr)
    angle = x_test[:,0]
    thickness = read_csv_type3(source, fname_lift_test, fname_shape_test, shape_odd = s_odd, read_rate = rr, param = "thickness")
    maxCamber = read_csv_type3(source, fname_lift_test, fname_shape_test, shape_odd = s_odd, read_rate = rr,
                               param = "maxCamber")
    maxCamberPosition = read_csv_type3(source, fname_lift_test, fname_shape_test, shape_odd = s_odd, read_rate = rr,
                                       param = "reflect")
    desingedLift = read_csv_type3(source, fname_lift_test, fname_shape_test, shape_odd = s_odd, read_rate = rr,
                                       param = "designedLift")
    naca5 = read_csv_type3(source, fname_lift_test, fname_shape_test, shape_odd = s_odd, read_rate = rr,
                                       param = "NACA")
    naca5 = np.array(naca5, dtype=int)
    """
    plot(data = naca5, ho = ho)
    plot(data = maxCamber, ho=ho)
    plot(data = angle, ho = ho)
    plot(data = thickness, ho = ho)
    """
    
    fig = plt.figure(figsize = (18, 12))
    datas = [naca5, thickness, maxCamber, angle, y_test.flatten()]

    for data in datas:
        print(np.corrcoef(data, ho[:, 0])[0, 1], np.corrcoef(data, ho[:, 1])[0, 1])
    exit()
    nList = ["Wing Number", "Max Thickness", "Max Camber", "AoA", "$C_L$"]
    name_header = "vs. "
    fig.suptitle("Hidden Layer Output vs. Some Parameters", fontsize = 22)
    axisLabelSize = 14
    titleSize = 18
    for i, data in enumerate(datas):
        ax = fig.add_subplot(2, 3, i + 1)
        ax.plot(data, ho[:, 0], ".", label="1st component")
        ax.plot(data, ho[:, 1], ".", label="2nd component")
        ax.legend()
        ax.set_title(name_header + nList[i], fontsize = titleSize)
        ax.set_xlabel(nList[i], fontsize = axisLabelSize)
        ax.set_ylabel("hidden layer output", fontsize = axisLabelSize)
        # ax.set_aspect("equal")
    plt.show()

# 1と4, 2と3とで相関とる
def cv_analysis(fname="hOutCV2.npz"):
    def scatter_plot(x, yList, xLabel, xUnit, yLabelList, titleHeader):
        plt.rcParams["font.size"] = 14
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for j in range(3):
            # ax = fig.add_subplot(3, 1, j+1)
            r1 = str(np.corrcoef(x, yList[:,j])[0,1])
            digit = 4
            if r1[0] == "-":
                digit += 1
            zero = True
            for i in range(digit):
                if (r1[i] != "0") and ((r1[i] != "-")) and ((r1[i] != ".")):
                    zero = False
            if zero:
                r1 = "0.00"
                digit = 4
            ax.plot(x, yList[:, j], ".", alpha=0.3, label=yLabelList[j] + " : $R={0}$".format(r1[:digit]))
        ax.set_xlabel("{0}  {1}".format(xLabel,xUnit))
        ax.set_ylabel("Hidden Layer Output")
        ax.legend()
        # ax.set_aspect("equal")
        ax.set_title(titleHeader + "  vs. " + xLabel, fontsize=18)
        plt.show()
       
    def openNpz(number):
        fname = "hOutCV{0}.npz".format(number)
        with np.load(fname, allow_pickle = True) as f:
            if number == 5:
                x = np.load(fname, mmap_mode='r')
                for k in x.files:
                    print(k)
                exit()
            inputs_train = [f["inputs_train_0"], f["inputs_train_1"]]
            inputs_test  = [f["inputs_test_0"], f["inputs_test_1"]]
            y_train, y_test = f["y_train"], f["y_test"]
            p_train, p_test = f["p_train"], f["p_test"]
            if number == 1 or number == 2:
                hidden_output_cl, hidden_output_cd = f["hidden_output_cl"], f["hidden_output_cd"]
                return hidden_output_cl, hidden_output_cd, p_train, p_test
            else:
                hidden_output = f["hidden_output"]
                return hidden_output, p_train, p_test
    
    def corrcoef(a,b, old=False):
        def cov(a,b):
            return np.dot(a.T - np.mean(a, axis = 0).reshape(a.shape[1], 1), b - np.mean(b, axis = 0)) / n
        n = a.shape[0]
        
        if old:
            cov_aa = cov(a,a)
            print(cov_aa)
            cov_bb = cov(b,b)
            cov_ab = cov(a,b)
            return cov_ab / np.sqrt(np.linalg.det(cov_aa) * np.linalg.det(cov_bb))
        
        else:
            d = a.shape[1]
            rho = np.zeros((d,d))
            for i in range(d):
                for j in range(d):
                    rho[i,j] = np.corrcoef(a[:,i], b[:,j])[1,0]
            return rho
    def print4csv(arr):
        row, col = arr.shape[0], arr.shape[1]
        txt = ""
        for j in range(col):
            for i in range(row):
                txt += str(arr[i, j]) + ","
            txt += "\n"
        print(txt)
    
    hidden_output_cl, hidden_output_cd, p_train, p_test = openNpz(1)
    """
    hidden_output, p_train, p_test = openNpz(4)
    #hidden_output_cl, hidden_output_cd, p_train, p_test = openNpz(2)
    #hidden_output, p_train, p_test = openNpz(3)
    print(hidden_output_cl.shape)
    print(hidden_output_cd.shape)
    print(hidden_output.shape)
    print4csv(corrcoef(hidden_output, hidden_output_cl))
    print4csv(corrcoef(hidden_output, hidden_output_cd))
    print4csv(corrcoef(hidden_output_cl, hidden_output_cd))
    exit()
    print(np.corrcoef(hidden_output_cl.T, hidden_output_cd.T))
    print(corrcoef(hidden_output_cl, hidden_output_cd))
    print4csv(corrcoef(hidden_output_cl, hidden_output_cd))
    exit()
    print(hidden_output_cl.shape)
    print(np.cov(hidden_output_cl.T).shape)
    print((hidden_output_cl.T - np.mean(hidden_output_cl.T, axis=1).reshape(5, -1)).shape)
    exit()
    cov = np.cov(hidden_output_cl.T, hidden_output_cd)
    print(cov.shape)
    corr = np.corrcoef(hidden_output_cl.T, hidden_output_cd.T)
    print(corr.shape)
    exit()
    """
    # AoA, Ma, LogRe, NACA, Cm, Cl, Cm, Re, Thickness, MaxCamberPosition
    names = ["AoA", "Ma", "$log_{10}(Re)$", "Cm", "$C_L$", "$C_D$", "Thickness", "Re", "MaxCamberPosition"]
    units = ["[deg]", "", "", "", "", "", "[%]", "", "[%]"]
    sets = ["$s_{xy}$", "$s_x$ ", "$s_y$ "]
    networks = ["Lift Net:", "Drag Net:"]
    print(hidden_output_cl.shape)
    print(p_test.shape)
    p_test[:, 6] *= 10
    p_test[:, 8] *= 100
    csvL = "HLOlift.csv"
    csvD = "HLOdrag.csv"
    txtL, txtD = "Corr. Coef., sxy, sx, sy\n", "Corr. Coef., sxy, sx, sy\n"
    for i, name in enumerate(names):
        txtL += "{0},".format(name)
        txtD += "{0},".format(name)
        for j, set in enumerate(sets):
            r1L = np.corrcoef(hidden_output_cl[:, j], p_test[:, i])
            r1D = np.corrcoef(hidden_output_cd[:, j], p_test[:, i])
            txtL += "{0},".format(r1L[0,1])
            txtD += "{0},".format(r1D[0,1])
        scatter_plot(x=p_test[:,i], yList=hidden_output_cl[:,:3], xUnit = units[i], xLabel=name, yLabelList=sets, titleHeader=networks[0])
        txtL += "\n"
        scatter_plot(x=p_test[:,i], yList=hidden_output_cd[:,:3], xUnit = units[i], xLabel=name, yLabelList=sets, titleHeader=networks[1])
        txtD += "\n"
        
    with open(csvL, "w") as f:
        f.write(txtL)
    with open(csvD, "w") as f:
        f.write(txtD)
        
        

if __name__ == '__main__':
    cv_analysis()
    # main()