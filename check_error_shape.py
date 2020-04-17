# -- coding: utf-8 --
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
from naca_4digit_test import Naca_4_digit, Naca_5_digit
from arc_length_transformer import ArcLength

class ShapeError(object):
    def __init__(self, source, fname_fr, fname_eq, fname_cr):
        self.source = source
        self.s_fr = np.loadtxt(source + fname_fr, delimiter=",")
        self.s_eq = np.loadtxt(source + fname_eq, delimiter=",")
        self.s_cr = np.loadtxt(source + fname_cr, delimiter=",")
        self.shape_total = self.s_fr.shape[0]
        self.exact_resolution = 800
        self.dimension = 201
        self.half = 101
        self.quarter = 51
        self.split_num = 1
        self.split_step = int(self.shape_total / self.split_num)
        self.error = np.zeros((self.split_step, 4), dtype=float)    # 0:NACA, 1:Fr, 2:Eq, 3:Cr

    def calc_error(self, dec_y_u, dec_y_l, ex_y_u, ex_y_l):
        return np.sum(np.sqrt((dec_y_u - ex_y_u)**2) + np.sqrt((dec_y_l - ex_y_l)**2))

    def decode_modFr(self, obj_vector, plot_resolution):

    def decode_fr(self, bn, plot_resolution):
        def summation_sr(bn, plot_resolution, length=2.0):
            n_max = bn.shape[0]
            x_star = np.linspace(start=0, stop=length, num=plot_resolution * 2)
            decryption = np.zeros(plot_resolution * 2)
            for n in range(n_max):
                decryption += bn[n] * np.sin((n + 1) * np.pi * x_star / length)
            return decryption

        decryption = summation_sr(bn, plot_resolution)
        y_u = decryption[:plot_resolution]
        y_l = decryption[plot_resolution:][-1::-1]
        return y_u, y_l

    def decode_eq(self, x, s_half, plot_resolution):
        func = interpolate.interp1d(x, s_half, kind="linear")
        t = np.linspace(start=0, stop=1, num=plot_resolution)
        return func(t)

    def decode_cr(self, s_half, plot_resolution):
        quarter = self.quarter - 1
        x_cr = s_half[:quarter]
        func = interpolate.interp1d(x_cr, s_half[quarter:], kind="linear",
                                    bounds_error=False, fill_value="extrapolate")
        t = np.linspace(start=0, stop=1, num=plot_resolution)
        return func(t)

    def fourier_error(self):
        # NACA翼型計算する
        # 差分の2乗和を計算する
        for i in range(self.shape_total):
            naca4 = str(int(self.s_fr[i, 0]))
            # naca =  Naca_4_digit(int_4=naca4, attack_angle_deg=0, resolution=self.exact_resolution, quasi_equidistant=True, length_adjust=True)
            dec_y_u, dec_y_l = self.decode_fr(self.s_fr[i, 1:], self.exact_resolution)
            self.error[i, 0] = float(naca4)
            #self.error[i, 1] = self.calc_error(dec_y_u, dec_y_l, naca.y_u...)


    def equidistant_error(self):
        x_l = np.linspace(start=0, stop=1, num=self.exact_resolution)
        x_u = x_l[::-1]

        for i in range(self.shape_total):
            naca4 = str(int(self.s_eq[i, 0]))
            # naca =  Naca_4_digit(int_4=naca4, attack_angle_deg=0, resolution=self.exact_resolution, quasi_equidistant=True, length_adjust=True)
            dec_y_u = self.decode_eq(x_u, self.s_eq[i, 1:self.half], self.exact_resolution)
            dec_y_l = self.decode_eq(x_l, self.s_eq[i, self.half:], self.exact_resolution)
            self.error[i, 0] = float(naca4)
            #self.error[i, 2] = self.calc_error(dec_y_u, dec_y_l, naca...)

    def crowd_error(self):

        for i in range(self.shape_total):
            naca4 = str(int(self.s_cr[i, 0]))
            # naca =  Naca_4_digit(int_4=naca4, attack_angle_deg=0, resolution=self.exact_resolution, quasi_equidistant=True, length_adjust=True)
            dec_y_u = self.decode_cr(self.s_cr[i,1:self.half], self.exact_resolution)
            dec_y_l = self.decode_cr(self.s_cr[i, self.half:], self.exact_resolution)
            self.error[i, 0] = float(naca4)
            # self.error[i, 3] = self.calc_error(dec_y....)

    def all_error(self):
        def disp_statistics(array1d):
            print(np.max(array1d), np.min(array1d), np.average(array1d), np.var(array1d))

        x_l_eq = np.linspace(start=0, stop=1, num=self.half-1)
        x_u_eq = x_l_eq[::-1]
        ex_x = np.linspace(start=0, stop=1, num=self.exact_resolution)
        split_num = self.split_num
        split_step = self.split_step 
        for j in range(0,split_num):
            for i in range(split_step):
                k = 3094#j * split_step + i
                naca4 = str(int(self.s_fr[k, 0])).zfill(4)
                naca4 = "6189"
                naca =  Naca_4_digit(int_4=naca4, attack_angle_deg=0, resolution=self.exact_resolution, quasi_equidistant=True, length_adjust=True)

                ex_y_u = naca.equidistant_y_u - 0.5
                ex_y_l = naca.equidistant_y_l - 0.5
    
                fs_y_u, fs_y_l = self.decode_fr(self.s_fr[k, 1:], self.exact_resolution)

                eq_y_u = self.decode_eq(x_u_eq, self.s_eq[k, 1:self.half], self.exact_resolution)
                eq_y_l = self.decode_eq(x_l_eq, self.s_eq[k, self.half:], self.exact_resolution)
                
                cr_y_u = self.decode_cr(self.s_cr[k,1:self.half], self.exact_resolution)
                cr_y_l = self.decode_cr(self.s_cr[k, self.half:], self.exact_resolution)
                self.error[i, 0] = float(naca4)
                self.error[i, 1] = self.calc_error(fs_y_u, fs_y_l, ex_y_u, ex_y_l)
                self.error[i, 2] = self.calc_error(eq_y_u, eq_y_l, ex_y_u, ex_y_l)
                self.error[i, 3] = self.calc_error(cr_y_u, cr_y_l, ex_y_u, ex_y_l)
                print(self.error[i])
                print(self.s_fr[k,0],
                      self.s_eq[k,0],
                      self.s_cr[k,0],
                      )
                cmap = plt.get_cmap("tab10")
                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.plot(ex_x, fs_y_u, color=cmap(0), label="Fourier")
                ax.plot(ex_x, fs_y_l, color=cmap(0))
                ax.plot(ex_x, eq_y_u, color=cmap(1), label="Equidistant")
                ax.plot(ex_x, eq_y_l, color=cmap(1))
                ax.plot(ex_x, cr_y_u, color=cmap(2), label="Concentrate")
                ax.plot(ex_x, cr_y_l, color=cmap(2))
                ax.plot(naca.x_u, naca.y_u-0.5, color=cmap(3), label="Exact")
                ax.plot(naca.x_l, naca.y_l-0.5, color=cmap(3))
                ax.legend()
                ax.set_title("Shape of NACA6189")
                ax.set_xlabel("x/c")
                ax.set_ylabel("y/c")
                plt.show()
                #plt.show()
            np.save(self.source + "shapeError" + str(j).zfill(3), self.error)
        for i in range(1,4):
            disp_statistics(self.error[:,i])

        self.plot_error(shapetype=[1,2,3])

    def plot_error(self, shapetype=[1]):
        datalabel = ["Fourier", "Equidistant", "Concentrate"]
        self.error[:, 1:4] /= 800.0
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for kind in shapetype:
            ax.plot(self.error[:, 0], self.error[:, kind], label=datalabel[kind-1], marker=".", linewidth="0", markersize=2)
        ax.legend()
        ax.set_xlabel("NACA Wing number")
        ax.set_ylabel("Error")
        ax.set_title("Shape error vs. NACA wing number")
        ax.set_ylim([0,0.025])
        plt.rcParams["font.size"] = 18
        plt.show()

def main(source, fname_fourier, fname_equidistant, fname_concentrate):
    error = ShapeError(source, fname_fourier, fname_equidistant, fname_concentrate)
    error.error = np.load(source + "shapeError000.npy")
    error.plot_error(shapetype=[1,2,3])


if __name__ == '__main__':
    source = "G:\\Toyota\\Data\\Incompressible_Invicid\\training_data\\NACA4\\"
    fname_fourier = "shape_fourier_5000_odd_closed.csv"
    fname_equidistant = "shape_equidistant_5000_odd_closed.csv"
    fname_concentrate = "shape_crowd_0.1_0.15_30_50_20_5000_odd_xy_closed.csv"
    fname_mod_fourier = "shape_modified_fourier_5000_odd.csv"
    #fname_fourier = "shape_fourier_all_closed.csv"
    #fname_equidistant = "shape_equidistant_all_closed.csv"
    #fname_concentrate = "shape_crowd_all_mk2_xy_closed.csv"
    main(source, fname_fourier, fname_equidistant, fname_concentrate)
    #exit()
    error = ShapeError(source, fname_fourier, fname_equidistant, fname_concentrate)
    error.all_error()