import numpy as np
import fourier_expansion as fe
from naca_4digit_test import Naca_4_digit, Naca_5_digit
from math import gcd, floor, ceil    # 最大公約数を求めるメソッド(集中計算用)
import matplotlib.pyplot as plt
from itertools import product

lcm = lambda a, b: (a * b) // gcd(a, b)

def odd_number(index):
    return 2 * index + 1

def even_number(index):
    return 2 * index

def make_shape_data_for_NACA4DIGIT_fourier(path, data_Number, odd=True, old_style=True):
    resolution = 10000
    if old_style:
        kind_of_wing = 5000
    
        if odd == True:
            nextwing = odd_number
            fname = path + "NACA4\\shape_fourier_5000_odd.csv"
            pattern = kind_of_wing
        else:
            nextwing = even_number
            fname = path + "NACA4\\shape_fourier_5000_even.csv"
            pattern = int(kind_of_wing * 0.9)
    
        save_data = np.zeros((pattern, data_Number))
        data_id = 0
        for wing in range(kind_of_wing):
            if ((odd == True) or (nextwing(wing) % 100 != 0)):
                save_data[data_id, 0] = float(nextwing(wing))
                naca4 = str(int(nextwing(wing))).zfill(4) # only odd
                naca = Naca_4_digit(int_4=naca4, attack_angle_deg=0.0, resolution=resolution, quasi_equidistant=False, length_adjust = True)
                fex = fe.fourier_expansion(naca.x_u, naca.y_u, naca.x_l, naca.y_l, n=data_Number - 2)
                save_data[data_id, 1:] = fex.bn
                data_id += 1
    
    else:
        fname = path + "NACA4\\shape_fourier_1112_9988_s04.csv"
        save_data = np.zeros((9 * 9 * 20, dataNumber))
        data_id = 0
        for i1 in range(1,10):
            for i2 in range(1,10):
                for i34 in range(12, 92, 4):
                    naca4 = str(i1).zfill(1) + str(i2).zfill(1) + str(i34).zfill(2)
                    naca = Naca_4_digit(int_4=naca4, attack_angle_deg=0.0, resolution=resolution, quasi_equidistant=False, length_adjust = True)
                    fex = fe.fourier_expansion(naca.x_u, naca.y_u, naca.x_l, naca.y_l, n=data_Number - 2)
                    save_data[data_id, 0] = float(naca4)
                    save_data[data_id, 1:] = fex.bn
                    data_id += 1

    np.savetxt(fname, save_data, delimiter=",")


def make_shape_data_for_NACA4DIGIT_equidistant(path, data_Number, odd=True, old_style=True):
    def plot_test(for_poster=False):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        if for_poster:
            x_u = np.linspace(start=0, stop=1, num=resolution)[::-1]
            x_l = np.linspace(start = 0, stop = 1, num = resolution)
            ax.plot(x_u, save_data[wing, 1:half], ".", color = "mediumspringgreen")
            ax.plot(x_l, save_data[wing, half:], ".", color = "mediumspringgreen")
            ax.set_xlim(-0.01, 1.01)
            ax.set_ylim(-0.2, 0.2)
            ax.set_title("Equidistnant Sampling (NACA2612)")
            ax.set_xlabel("x/L")
            ax.set_ylabel("y/L")
        else:
            x = np.linspace(start=-1, stop=1, num=2*resolution)
            ax.plot(x[:resolution], save_data[wing, 1:half], "x", color="blue")
            ax.plot(x[resolution:], save_data[wing, half:], "x", color="red")
        plt.show()
        exit()

    resolution = int((dataNumber - 1)/2)
    half = int((dataNumber + 1) / 2)
    if old_style == True:
        kind_of_wing = 5000
        if odd == True:
            nextwing = odd_number
            fname = path + "NACA4\\shape_equidistant_5000_odd.csv"
            pattern = kind_of_wing
        else:
            nextwing = even_number
            fname = path + "NACA4\\shape_equidistant_5000_even.csv"
            pattern = int(kind_of_wing * 0.9)
    
        data_id = 0
        save_data = np.zeros((pattern, data_Number))
        for wing in range(kind_of_wing):
            if ((odd == True) or (nextwing(wing) % 100) == 0):
                save_data[data_id, 0] = float(nextwing(wing))
                naca4 = str(int(nextwing(wing))).zfill(4) # only odd
                # naca4 = "2612"    # for poster
                naca = Naca_4_digit(int_4=naca4, attack_angle_deg=0.0, resolution=resolution, quasi_equidistant=True, length_adjust=True)
                # 後縁から反時計まわりに格納
                save_data[data_id, 1:half] = naca.equidistant_y_u[::-1] - 0.5
                save_data[data_id, half:] = naca.equidistant_y_l - 0.5
                wing = data_id
                # plot_test(True)
                data_id += 1

    elif old_style == "CV":
        fname = path + "NACA4\\shape_equidistant_CV_240.csv"
        i1_list = [0, 2, 5, 7, 9]
        i2_list = [0, 1, 3, 4, 6, 7]
        i3_list = [12, 22, 32, 42, 52, 62, 72, 82]
        wing_sample = 240
        save_data = np.zeros((wing_sample, data_Number))
        data_id = 0
        for i1, i2, i3 in product(i1_list, i2_list, i3_list):
            if ((i1 == 0) and (i2 == 0)) or (i1 != 1):
                naca4 = str(i1) + str(i2) + str(i3)
                save_data[data_id, 0] = float(naca4)
                naca = Naca_4_digit(int_4 = naca4, attack_angle_deg = 0.0, resolution = resolution,
                                    quasi_equidistant = True, length_adjust = True)
                save_data[data_id, 1:half] = naca.equidistant_y_u[::-1] - 0.5
                save_data[data_id, half:] = naca.equidistant_y_l - 0.5
                data_id += 1

    else:
        fname = path + "NACA4\\shape_equidistant_1112_9988_s04.csv"
        save_data = np.zeros((9 * 9 * 20, dataNumber))
        data_id = 0
        for i1 in range(1,10):
            for i2 in range(1,10):
                for i34 in range(12, 92, 4):
                    naca4 = str(i1).zfill(1) + str(i2).zfill(1) + str(i34).zfill(2)
                    naca = Naca_4_digit(int_4=naca4, attack_angle_deg=0.0, resolution=resolution, quasi_equidistant=True, length_adjust = True)
                    save_data[data_id, 0] = float(naca4)
                    save_data[data_id, 1:half] = naca.equidistant_y_u[::-1] - 0.5
                    save_data[data_id, half:] = naca.equidistant_y_l - 0.5
                    data_id += 1

        
    np.savetxt(fname, save_data, delimiter=",")


def make_shape_data_for_NACA4DIGIT_circumferential(path, data_Number, odd = True, old_style = True):
    def plot_test(naca4="2612"):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        half = 100
        xy = make_circum_arc_from_naca(naca4, resolution, digit = 4)
        print(xy)
        ax.plot(xy[:half]+1, xy[half:], ".", label="circumferential")
        ax.set_xlim(-0.01, 1.01)
        ax.set_ylim(-0.2, 0.2)
        ax.set_title("Circumferential Sampling (NACA2612)")
        ax.set_xlabel("x/L")
        ax.set_ylabel("y/L")
        ax.legend()
        plt.show()
        exit()

    from arc_length_transformer import make_circum_arc_from_naca
    resolution = max(int((dataNumber - 1) / 2), 10000)
    typeName = "circumferential"
    plot_test()
    if old_style == True:
        kind_of_wing = 5000
        if odd == True:
            nextwing = odd_number
            fname = path + "NACA4\\shape_{0}_5000_odd.csv".format(typeName)
            pattern = kind_of_wing
        else:
            nextwing = even_number
            fname = path + "NACA4\\shape_{0}_5000_even.csv".format(typeName)
            pattern = int(kind_of_wing * 0.9)
    
        save_data = np.zeros((pattern, data_Number))
        data_id = 0
        for wing in range(kind_of_wing):
            if ((odd == True) or (nextwing(wing) % 100 != 0)):
                save_data[data_id, 0] = float(nextwing(wing))
                naca4 = str(int(nextwing(wing))).zfill(4)  # only odd
                save_data[data_id, 1:] = make_circum_arc_from_naca(naca4, resolution, digit = 4)
                data_id += 1

    elif old_style == "CV":
        fname = path + "NACA4\\shape_{0}_CV_240.csv".format(typeName)
        i1_list = [0, 2, 5, 7, 9]
        i2_list = [0, 1, 3, 4, 6, 7]
        i3_list = [12, 22, 32, 42, 52, 62, 72, 82]
        wing_sample = 240
        save_data = np.zeros((wing_sample, data_Number))
        data_id = 0
        for i1, i2, i3 in product(i1_list, i2_list, i3_list):
            if ((i1 == 0) and (i2 == 0)) or (i1 != 1):
                naca4 = str(i1) + str(i2) + str(i3)
                save_data[data_id, 0] = float(naca4)
                save_data[data_id, 1:] = make_circum_arc_from_naca(naca4, resolution, digit = 4)
                data_id += 1

    else:
        fname = path + "NACA4\\shape_{0}_1112_9988_s04.csv".format(typeName)
        save_data = np.zeros((9 * 9 * 20, dataNumber))
        data_id = 0
        for i1 in range(1, 10):
            for i2 in range(1, 10):
                for i34 in range(12, 92, 4):
                    naca4 = str(i1).zfill(1) + str(i2).zfill(1) + str(i34).zfill(2)
                    save_data[data_id, 0] = float(naca4)
                    save_data[data_id, 1:] = make_circum_arc_from_naca(naca4, resolution, digit = 4)
                    data_id += 1

    np.savetxt(fname, save_data, delimiter = ",")

def prepare_for_crowd_front_and_back(dataNumber, len_front, len_back, percent_front, percent_back):
    default_resolution = int((dataNumber - 1) / 2)
    percent_center = 100 - (percent_front + percent_back)
    len_center = 1.0 - (len_front + len_back)

    # 各区間の点数
    point_front = int(default_resolution * percent_front / 100)
    point_back = int(default_resolution * percent_back / 100)
    point_center = default_resolution - (point_front + point_back)

    # 各区間での分割数を全区間に換算
    divide_front = floor(point_front / len_front)
    divide_center = floor(0.5 * (point_center + 2) / len_center) * 2
    divide_back = floor(point_back / len_back)

    side_margin = int((divide_center - point_center) / 2)
    front_index = side_margin
    back_index = divide_center - side_margin

    half = int((dataNumber + 1) / 2)
    return divide_front, divide_center, divide_back, point_front, point_center, point_back,\
           percent_center, front_index, back_index, half

def make_shape_data_for_NACA4DIGIT_crowd_front_and_back(path, data_Number, odd=True, len_front=0.1, len_back=0.15, percent_front=30, percent_back=20, old_style=True, only_y=True):
    # 前縁からlen_front*100%までの位置にpercent_front%の点，後縁からlen_back*100%までの位置にpercent_back%の点を配置する
    def plot_test(for_poster=False, only_y=True):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        if only_y:
            if for_poster:
                xl = np.linspace(start = 0, stop = 1, num = divide_front)[::-1]
                xc = np.linspace(start = 0, stop = 1, num = divide_center)[::-1]
                xb = np.linspace(start = 0, stop = 1, num = divide_back)[::-1]
            else:
                xl = np.linspace(start=-1, stop=0, num=divide_front)
                xc = np.linspace(start=-1, stop=0, num=divide_center)
                xb = np.linspace(start=-1, stop=0, num=divide_back)
    
            ax.plot(xl[divide_front - point_front:],
                     save_data[wing, point_back + point_center + 1:half], ".", label="High Density", color="crimson")
            ax.plot(xb[:point_back],
                    save_data[wing, 1:point_back + 1], ".", label = "Middle Density", color = "lime")
            ax.plot(xc[front_index + 1:back_index + 1], save_data[wing, point_back + 1:point_back + point_center + 1], ".", label="Low Density", color="darkblue")
            
    
            xl = np.linspace(start=0, stop=1, num=divide_front)
            xc = np.linspace(start=0, stop=1, num=divide_center)
            xb = np.linspace(start=0, stop=1, num=divide_back)
            ax.plot(xl[:point_front], save_data[wing, half:half + point_front], ".", color="crimson")
            ax.plot(xc[front_index - 1:back_index - 1], save_data[wing, half + point_front:half + point_front + point_center], ".", color="darkblue")
            ax.plot(xb[divide_back - point_back:],
                     save_data[wing, half + point_front + point_center:2 * half - 1], ".", color="lime")
        
        else:
            quarter = int((half - 1) / 2 + 1)
            # ax.plot(save_data[wing, 1:quarter], save_data[wing, quarter:half], ".")
            ax.plot(save_data[wing, 1:point_back + 1],
                    save_data[wing, quarter:quarter + point_back], ".", label = "Middle Density", color = "lime")
            ax.plot(save_data[wing, point_back + 1:point_back + point_center + 1], 
                    save_data[wing, quarter+point_back:quarter + point_back + point_center], ".", label="Low Density", color="darkblue")
            ax.plot(save_data[wing, point_back + point_center + 1:quarter],
                    save_data[wing, quarter + point_back + point_center:half], ".", label="High Density", color="crimson")
            # ax.plot(save_data[wing, half:half+quarter-1], save_data[wing, half+quarter-1:], ".")
            ax.plot(save_data[wing, half:half + point_front],
                    save_data[wing, half + quarter - 1:half + quarter - 1 + point_front], ".", color="crimson")
            ax.plot(save_data[wing, half + point_front:half + point_front + point_center],
                    save_data[wing, half + quarter - 1 + point_front:half + quarter - 1 + point_front + point_center], ".", color="darkblue")
            ax.plot(save_data[wing, half + point_front + point_center:half + quarter - 1],
                    save_data[wing, half + quarter - 1 + point_front + point_center:], ".", color = "lime")
        ax.set_xlim(-0.01, 1.01)
        ax.set_ylim(-0.2, 0.2)
        ax.set_title("Concentrate Sampling (NACA2612)")
        ax.set_xlabel("x/L")
        ax.set_ylabel("y/L")
        ax.legend(bbox_to_anchor = (0, 1), loc = "upper left", borderaxespad = 1, fontsize = 12)
        plt.show()
        exit()

    def main_crowd(naca4, only_y):
        nacaf = Naca_4_digit(int_4 = naca4, attack_angle_deg = 0.0, resolution = divide_front,
                             quasi_equidistant = True, length_adjust = True)
        nacac = Naca_4_digit(int_4 = naca4, attack_angle_deg = 0.0, resolution = divide_center,
                             quasi_equidistant = True, length_adjust = True)
        nacab = Naca_4_digit(int_4 = naca4, attack_angle_deg = 0.0, resolution = divide_back,
                             quasi_equidistant = True, length_adjust = True)
    
        y_uf = nacaf.equidistant_y_u[::-1] - 0.5
        y_uc = nacac.equidistant_y_u[::-1] - 0.5
        y_ub = nacab.equidistant_y_u[::-1] - 0.5
    
        if only_y:
            # save_data[data_id, 1:half] = np.concatenate(
            one2half = np.concatenate(
                [y_ub[:point_back], y_uc[front_index + 1:back_index + 1],
                 y_uf[divide_front - point_front:]])
        
            # save_data[data_id, half:] = np.concatenate([(nacaf.equidistant_y_l - 0.5)[:point_front],
            half2end = np.concatenate([(nacaf.equidistant_y_l - 0.5)[:point_front],
                                                        (nacac.equidistant_y_l - 0.5)[
                                                        front_index - 1:back_index - 1],
                                                        (nacab.equidistant_y_l - 0.5)[
                                                        divide_back - point_back:]])
        else:
            x_f = nacaf.equidistant_x
            x_c = nacac.equidistant_x
            x_b = nacab.equidistant_x
        
            one2half = np.concatenate(
                [x_b[::-1][:point_back], x_c[::-1][front_index + 1:back_index],
                 x_f[::-1][divide_front - point_front:],
                 y_ub[:point_back], y_uc[front_index + 1:back_index], y_uf[divide_front - point_front:]])
        
            half2end = np.concatenate(
                [x_f[:point_front], x_c[front_index:back_index - 1], x_b[divide_back - point_back:],
                 (nacaf.equidistant_y_l - 0.5)[:point_front],
                 (nacac.equidistant_y_l - 0.5)[front_index:back_index - 1],
                 (nacab.equidistant_y_l - 0.5)[divide_back - point_back:]])
        
        return one2half, half2end

    if not only_y:
        dataNumber = 100 + 1
    divide_front, divide_center, divide_back, point_front, point_center, point_back, \
    percent_center, front_index, back_index, half = prepare_for_crowd_front_and_back(dataNumber, len_front, len_back, percent_front, percent_back)
    if not only_y:
        half = (half - 1) * 2 + 1
    casename = str(len_front) +"_" + str(len_back) +"_" +  str(percent_front) + "_" + str(percent_center) + "_" + str(percent_back)
    
    if old_style == True:
        kind_of_wing = 5000
        if odd == True:
            nextwing = odd_number
            fname = path + "NACA4\\shape_crowd_" + casename + "_5000_odd.csv"
            pattern = kind_of_wing
        else:
            nextwing = even_number
            fname = path + "NACA4\\shape_crowd_" + casename + "_5000_even.csv"
            pattern = int(0.9 * kind_of_wing)
            
        data_id = 0
        save_data = np.zeros((pattern, data_Number))
        for wing in range(kind_of_wing):
            if ((odd == True) or (nextwing(wing) % 100 != 0)):
                save_data[data_id, 0] = float(nextwing(wing))
                naca4 = str(int(nextwing(wing))).zfill(4) # only odd
                # naca4 = "2612"  # for poster
                save_data[data_id, 1:half], save_data[data_id, half:] = main_crowd(naca4, only_y)
                data_id += 1
                wing = data_id - 1
                #plot_test(True, only_y=only_y)
    
    elif old_style == "CV":
        fname = path + "NACA4\\shape_crowd_" + casename + "_CV_240.csv"
        i1_list = [0, 2, 5, 7, 9]
        i2_list = [0, 1, 3, 4, 6, 7]
        i3_list = [12, 22, 32, 42, 52, 62, 72, 82]
        wing_sample = 240
        save_data = np.zeros((wing_sample, data_Number))
        data_id = 0
        for i1, i2, i3 in product(i1_list, i2_list, i3_list):
            if ((i1 == 0) and (i2 == 0)) or (i1 != 1):
                naca4 = str(i1) + str(i2) + str(i3)
                save_data[data_id, 0] = float(naca4)
                save_data[data_id, 1:half], save_data[data_id, half:] = main_crowd(naca4, only_y)
                data_id += 1
                
    else:
        fname = path + "NACA4\\shape_crowd_" + casename + "_1112_9988_d4.csv"
        kind_of_wing = 9 * 9 * 20
        pattern = kind_of_wing
    
        save_data = np.zeros((pattern, data_Number))
        save_data = np.zeros((9 * 9 * 20, dataNumber))
        data_id = 0
        for i1 in range(1, 10):
            for i2 in range(1, 10):
                for i34 in range(12, 92, 4):
                    naca4 = str(i1).zfill(1) + str(i2).zfill(1) + str(i34).zfill(2)
                    save_data[data_id, 0] = float(naca4)
                    save_data[data_id, 1:half], save_data[data_id, half:] = main_crowd(naca4, only_y)
                    data_id += 1

    if not only_y:
        fname = fname.replace(".csv","") + "_xy.csv"
    np.savetxt(fname, save_data, delimiter=",")

def make_shape_data_for_NACA4DIGIT_modified_fourier(path, data_Number, odd=True, old_style=True):
    resolution = 10000
    from arc_length_transformer import make_mod_fourier_from_naca
    if old_style == True:
        kind_of_wing = 5000

        if odd == True:
            nextwing = odd_number
            fname = path + "NACA4\\shape_modified_fourier_5000_odd.csv"
            pattern = kind_of_wing
        else:
            nextwing = even_number
            fname = path + "NACA4\\shape_modified_fourier_5000_even.csv"
            pattern = int(kind_of_wing * 0.9)

        save_data = np.zeros((pattern, data_Number))
        data_id = 0
        for wing in range(kind_of_wing):
            if ((odd == True) or (nextwing(wing) % 100 != 0)):
                save_data[data_id, 0] = float(nextwing(wing))
                naca4 = str(int(nextwing(wing))).zfill(4)  # only odd
                save_data[data_id, 1:] = make_mod_fourier_from_naca(naca4, resolution, digit=4)
                data_id += 1
    
    elif old_style == "CV":
        fname = path + "NACA4\\shape_modified_fourier_CV_240.csv"
        i1_list = [0, 2, 5, 7, 9]
        i2_list = [0, 1, 3, 4, 6, 7]
        i3_list = [12, 22, 32, 42, 52, 62, 72, 82]
        wing_sample = 240
        save_data = np.zeros((wing_sample, data_Number))
        data_id = 0
        for i1, i2, i3 in product(i1_list, i2_list, i3_list):
            if ((i1 == 0) and (i2 == 0)) or (i1 != 1):
                naca4 = str(i1) + str(i2) + str(i3)
                save_data[data_id, 0] = float(naca4)
                save_data[data_id, 1:] = make_mod_fourier_from_naca(naca4, resolution, digit = 4)
                data_id += 1

        
    else:
        fname = path + "NACA4\\shape_modified_fourier_1112_9988_s04.csv"
        save_data = np.zeros((9 * 9 * 20, dataNumber))
        data_id = 0
        for i1 in range(1, 10):
            for i2 in range(1, 10):
                for i34 in range(12, 92, 4):
                    naca4 = str(i1).zfill(1) + str(i2).zfill(1) + str(i34).zfill(2)
                    save_data[data_id, 0] = float(naca4)
                    save_data[data_id, 1:] = make_mod_fourier_from_naca(naca4, resolution, digit=4)
                    data_id += 1

    np.savetxt(fname, save_data, delimiter=",")


def make_shape_data_for_NACA5DIGIT_fourier(path, data_Number, old_style):
    resolution = 10000
    if old_style == True:
        kind_of_wing = 9 * 99
    
        fname = path + "NACA5\\shape_fourier_all.csv"
        pattern = kind_of_wing

        save_data = np.zeros((pattern, data_Number))
        head_int3 = [210, 220, 230, 240, 250, 221, 231, 241, 251]
        data_id = 0
        for int3 in head_int3:
            for int2 in range(1, 100):
                naca5 = str(int3) + str(int2).zfill(2)
                save_data[data_id, 0] = float(naca5)
                naca = Naca_5_digit(int_5=naca5, attack_angle_deg=0.0, resolution=resolution, quasi_equidistant=False, length_adjust = True)
                fex = fe.fourier_expansion(naca.x_u, naca.y_u, naca.x_l, naca.y_l, n=data_Number - 2)
                save_data[data_id, 1:] = fex.bn
                data_id += 1

    elif old_style == "all":
        kind_of_wing = 6 * 9 * 99
    
        fname = path + "NACA5\\shape_fourier_all_mk2.csv"
        pattern = kind_of_wing
        print(fname)
        save_data = np.zeros((pattern, data_Number))
        mid_int2 = [10, 20, 30, 40, 50, 21, 31, 41, 51]
        data_id = 0
        for i1 in range(1,7):
            for i23 in mid_int2:
                for i45 in range(1, 100):
                    naca5 = str(i1) + str(i23) + str(i45).zfill(2)
                    save_data[data_id, 0] = float(naca5)
                    naca = Naca_5_digit(int_5 = naca5, attack_angle_deg = 0.0, resolution = resolution,
                                        quasi_equidistant = False, length_adjust = True)
                    fex = fe.fourier_expansion(naca.x_u, naca.y_u, naca.x_l, naca.y_l, n = data_Number - 2)
                    save_data[data_id, 1:] = fex.bn
                    data_id += 1


    else:
        kind_of_wing = 7 * 99
    
        fname = path + "NACA5\\shape_fourier_560.csv"
        pattern = kind_of_wing
        
        save_data = np.zeros((pattern, data_Number))
        head_int3 = [210, 220, 230, 240, 250, 241, 251]
        data_id = 0
        for int3 in head_int3:
            for int2 in range(11, 91):
                naca5 = str(int3) + str(int2).zfill(2)
                save_data[data_id, 0] = float(naca5)
                naca = Naca_5_digit(int_5 = naca5, attack_angle_deg = 0.0, resolution = resolution,
                                    quasi_equidistant = False, length_adjust = True)
                fex = fe.fourier_expansion(naca.x_u, naca.y_u, naca.x_l, naca.y_l, n = data_Number - 2)
                save_data[data_id, 1:] = fex.bn
                data_id += 1

    np.savetxt(fname, save_data, delimiter=",")


def make_shape_data_for_NACA5DIGIT_equidistant(path, data_Number, old_style=True):
    resolution = int((dataNumber - 1) / 2)
    half = int((dataNumber + 1) / 2)
    if old_style == True:
        kind_of_wing = 9 * 99
    
        fname = path + "NACA5\\shape_equidistant_all.csv"
        pattern = kind_of_wing
    
        save_data = np.zeros((pattern, data_Number))
        head_int3 = [210, 220, 230, 240, 250, 221, 231, 241, 251]
        data_id = 0
        for int3 in head_int3:
            for int2 in range(1, 100):
                naca5 = str(int3) + str(int2).zfill(2)
                save_data[data_id, 0] = float(naca5)
                naca = Naca_5_digit(int_5=naca5, attack_angle_deg=0.0, resolution=resolution, quasi_equidistant=True, length_adjust = True)
                # 後縁から反時計まわりに格納
                save_data[data_id, 1:half] = naca.equidistant_y_u[::-1] - 0.5
                save_data[data_id, half:] = naca.equidistant_y_l - 0.5
                data_id += 1

    elif old_style == "all":
        kind_of_wing = 6 * 9 * 99
    
        fname = path + "NACA5\\shape_equidistant_all_mk2.csv"
        pattern = kind_of_wing
        print(fname)
        save_data = np.zeros((pattern, data_Number))
        mid_int2 = [10, 20, 30, 40, 50, 21, 31, 41, 51]
        data_id = 0
        for i1 in range(1, 7):
            for i23 in mid_int2:
                for i45 in range(1, 100):
                    naca5 = str(i1) + str(i23) + str(i45).zfill(2)
                    save_data[data_id, 0] = float(naca5)
                    naca = Naca_5_digit(int_5=naca5, attack_angle_deg=0.0, resolution=resolution, quasi_equidistant=True, length_adjust = True)
                    # 後縁から反時計まわりに格納
                    save_data[data_id, 1:half] = naca.equidistant_y_u[::-1] - 0.5
                    save_data[data_id, half:] = naca.equidistant_y_l - 0.5
                    data_id += 1

    else:
        kind_of_wing = 7 * 99
    
        fname = path + "NACA5\\shape_equidistant_560.csv"
        pattern = kind_of_wing
    
        save_data = np.zeros((pattern, data_Number))
        head_int3 = [210, 220, 230, 240, 250, 241, 251]
        data_id = 0
        for int3 in head_int3:
            for int2 in range(11, 91):
                naca5 = str(int3) + str(int2).zfill(2)
                save_data[data_id, 0] = float(naca5)
                naca = Naca_5_digit(int_5 = naca5, attack_angle_deg = 0.0, resolution = resolution,
                                    quasi_equidistant = True, length_adjust = True)
                # 後縁から反時計まわりに格納
                save_data[data_id, 1:half] = naca.equidistant_y_u[::-1] - 0.5
                save_data[data_id, half:] = naca.equidistant_y_l - 0.5
                data_id += 1

    np.savetxt(fname, save_data, delimiter=",")

def make_shape_data_for_NACA5DIGIT_circumferential(path, data_Number, odd = True, old_style = True):
    from arc_length_transformer import make_circum_arc_from_naca
    resolution = 10000
    typeName = "circumferential"
    if old_style == True:
        kind_of_wing = 9 * 99

        fname = path + "NACA5\\shape_{0}_all.csv".format(typeName)
        pattern = kind_of_wing

        save_data = np.zeros((pattern, data_Number))
        head_int3 = [210, 220, 230, 240, 250, 221, 231, 241, 251]
        data_id = 0
        for int3 in head_int3:
            for int2 in range(1, 100):
                naca5 = str(int3) + str(int2).zfill(2)
                save_data[data_id, 0] = float(naca5)
                save_data[data_id, 1:] = make_circum_arc_from_naca(naca5, resolution, digit=5)
                data_id += 1

    elif old_style == "all":
        kind_of_wing = 6 * 9 * 99

        fname = path + "NACA5\\shape_{0}_all_mk2.csv".format(typeName)
        pattern = kind_of_wing
        print(fname)
        save_data = np.zeros((pattern, data_Number))
        mid_int2 = [10, 20, 30, 40, 50, 21, 31, 41, 51]
        data_id = 0
        for i1 in range(1, 7):
            for i23 in mid_int2:
                for i45 in range(1, 100):
                    naca5 = str(i1) + str(i23) + str(i45).zfill(2)
                    save_data[data_id, 0] = float(naca5)
                    save_data[data_id, 1:] = make_circum_arc_from_naca(naca5, resolution, digit=5)
                    data_id += 1


    else:
        kind_of_wing = 7 * 99

        fname = path + "NACA5\\shape_{0}_560.csv".format(typeName)
        pattern = kind_of_wing

        save_data = np.zeros((pattern, data_Number))
        head_int3 = [210, 220, 230, 240, 250, 241, 251]
        data_id = 0
        for int3 in head_int3:
            for int2 in range(11, 91):
                naca5 = str(int3) + str(int2).zfill(2)
                save_data[data_id, 0] = float(naca5)
                save_data[data_id, 1:] = make_circum_arc_from_naca(naca5, resolution, digit=5)
                data_id += 1

    np.savetxt(fname, save_data, delimiter=",")


def make_shape_data_for_NACA5DIGIT_crowd_front_and_back(path, data_Number, len_front=0.1, len_back=0.15,
                                                        percent_front=30, percent_back=20, old_style=True, only_y=True):
    
    if not only_y:
        dataNumber = 100 + 1
    divide_front, divide_center, divide_back, point_front, point_center, point_back, \
    percent_center, front_index, back_index, half = prepare_for_crowd_front_and_back(dataNumber, len_front, len_back,
                                                                                     percent_front, percent_back)
    if not only_y:
        half = (half - 1) * 2 + 1

    casename = str(len_front) + "_" + str(len_back) + "_" + str(percent_front) + "_" + str(percent_center) + "_" + str(
        percent_back)

    if old_style == True:
        kind_of_wing = 9 * 99
        fname = path + "NACA5\\shape_crowd_" + casename + "_all.csv"
        head_int3 = [210, 220, 230, 240, 250, 221, 231, 241, 251]
        int2_list = np.arange(start = 1, stop = 100, step = 1)

    elif old_style == "all":
        kind_of_wing = 6 * 9 * 99
    
        fname = path + "NACA5\\shape_crowd_all_mk2.csv"
        pattern = kind_of_wing
    
        head_int3 = [110, 120, 130, 140, 150, 121, 131, 141, 151,
                     210, 220, 230, 240, 250, 221, 231, 241, 251,
                     310, 320, 330, 340, 350, 321, 331, 341, 351,
                     410, 420, 430, 440, 450, 421, 431, 441, 451,
                     510, 520, 530, 540, 550, 521, 531, 541, 551,
                     610, 620, 630, 640, 650, 621, 631, 641, 651]
        int2_list = np.arange(start = 1, stop = 100, step = 1)
    else:
        kind_of_wing = 7 * 80
        fname = path + "NACA5\\shape_crowd_" + casename + "_560_new.csv"
        head_int3 = [210, 220, 230, 240, 250, 241, 251]
        int2_list = np.arange(start = 11, stop = 90, step = 1)
    pattern = kind_of_wing

    if not only_y:
        fname = fname.replace(".csv","") + "_xy.csv"
    save_data = np.zeros((pattern, data_Number))
    
    data_id = 0
    for int3 in head_int3:
        for int2 in int2_list:
            naca5 = str(int3) + str(int2).zfill(2)
            save_data[data_id, 0] = float(naca5)
            nacaf = Naca_5_digit(int_5=naca5, attack_angle_deg=0.0, resolution=divide_front, quasi_equidistant=True,
                                length_adjust=True)
            nacac = Naca_5_digit(int_5=naca5, attack_angle_deg=0.0, resolution=divide_center, quasi_equidistant=True,
                                 length_adjust=True)
            nacab = Naca_5_digit(int_5=naca5, attack_angle_deg=0.0, resolution=divide_back, quasi_equidistant=True,
                                 length_adjust=True)

            y_uf = nacaf.equidistant_y_u[::-1] - 0.5
            y_uc = nacac.equidistant_y_u[::-1] - 0.5
            y_ub = nacab.equidistant_y_u[::-1] - 0.5

            if only_y:
                save_data[data_id, 1:half] = np.concatenate(
                    [y_ub[:point_back], y_uc[front_index + 1:back_index + 1], y_uf[divide_front - point_front:]])
    
                save_data[data_id, half:] = np.concatenate([(nacaf.equidistant_y_l - 0.5)[:point_front],
                                                            (nacac.equidistant_y_l - 0.5)[front_index - 1:back_index - 1],
                                                            (nacab.equidistant_y_l - 0.5)[divide_back - point_back:]])
            else:
                x_f = nacaf.equidistant_x
                x_c = nacac.equidistant_x
                x_b = nacab.equidistant_x
    
                save_data[data_id, 1:half] = np.concatenate(
                    [x_b[::-1][:point_back], x_c[::-1][front_index + 1:back_index],
                     x_f[::-1][divide_front - point_front:],
                     y_ub[:point_back], y_uc[front_index + 1:back_index], y_uf[divide_front - point_front:]])
    
                save_data[data_id, half:] = np.concatenate(
                    [x_f[:point_front], x_c[front_index:back_index - 1], x_b[divide_back - point_back:],
                     (nacaf.equidistant_y_l - 0.5)[:point_front],
                     (nacac.equidistant_y_l - 0.5)[front_index:back_index - 1],
                     (nacab.equidistant_y_l - 0.5)[divide_back - point_back:]])

            data_id += 1
            # plot_test()

    np.savetxt(fname, save_data, delimiter=",")


def make_shape_data_for_NACA5DIGIT_modified_fourier(path, data_Number, old_style):
    resolution = 10000
    from arc_length_transformer import make_mod_fourier_from_naca
    if old_style == True:
        kind_of_wing = 9 * 99

        fname = path + "NACA5\\shape_modified_fourier_all.csv"
        pattern = kind_of_wing

        save_data = np.zeros((pattern, data_Number))
        head_int3 = [210, 220, 230, 240, 250, 221, 231, 241, 251]
        data_id = 0
        for int3 in head_int3:
            for int2 in range(1, 100):
                naca5 = str(int3) + str(int2).zfill(2)
                save_data[data_id, 0] = float(naca5)
                save_data[data_id, 1:] = make_mod_fourier_from_naca(naca5, resolution, digit=5)
                data_id += 1

    elif old_style == "all":
        kind_of_wing = 6 * 9 * 99

        fname = path + "NACA5\\shape_modified_fourier_all_mk2.csv"
        pattern = kind_of_wing
        print(fname)
        save_data = np.zeros((pattern, data_Number))
        mid_int2 = [10, 20, 30, 40, 50, 21, 31, 41, 51]
        data_id = 0
        for i1 in range(1, 7):
            for i23 in mid_int2:
                for i45 in range(1, 100):
                    naca5 = str(i1) + str(i23) + str(i45).zfill(2)
                    save_data[data_id, 0] = float(naca5)
                    save_data[data_id, 1:] = make_mod_fourier_from_naca(naca5, resolution, digit=5)
                    data_id += 1


    else:
        kind_of_wing = 7 * 99

        fname = path + "NACA5\\shape_modified_fourier_560.csv"
        pattern = kind_of_wing

        save_data = np.zeros((pattern, data_Number))
        head_int3 = [210, 220, 230, 240, 250, 241, 251]
        data_id = 0
        for int3 in head_int3:
            for int2 in range(11, 91):
                naca5 = str(int3) + str(int2).zfill(2)
                save_data[data_id, 0] = float(naca5)
                save_data[data_id, 1:] = make_mod_fourier_from_naca(naca5, resolution, digit=5)
                data_id += 1

    np.savetxt(fname, save_data, delimiter=",")


if __name__ == '__main__':
    path = "G:\\Toyota\\Data\\Compressible_Invicid\\training_data\\"
    dataNumber = 200 + 1
    odd = True
    #print("fsr4")
    #make_shape_data_for_NACA4DIGIT_fourier(path, dataNumber, odd, old_style = True)
    #print("equid4")
    #make_shape_data_for_NACA4DIGIT_equidistant(path, dataNumber, odd, old_style = True)
    # make_shape_data_for_NACA4DIGIT_equidistant(path, dataNumber, odd, old_style="CV")
    #print("conc4")
    #make_shape_data_for_NACA4DIGIT_crowd_front_and_back(path, dataNumber, odd, old_style = True, only_y=False)
    # make_shape_data_for_NACA4DIGIT_crowd_front_and_back(path, dataNumber, odd, old_style = "CV", only_y = False)
    # print("fsr5")
    # make_shape_data_for_NACA5DIGIT_fourier(path, dataNumber, old_style = True)
    # print("equid5")
    # make_shape_data_for_NACA5DIGIT_equidistant(path, dataNumber, old_style = "all")
    # print("conc5")
    # make_shape_data_for_NACA5DIGIT_crowd_front_and_back(path, dataNumber, old_style = "all", only_y=False)
    # make_shape_data_for_NACA4DIGIT_modified_fourier(path, dataNumber, odd, old_style=True)
    # make_shape_data_for_NACA4DIGIT_modified_fourier(path, dataNumber, odd, old_style = "CV")

    # make_shape_data_for_NACA5DIGIT_modified_fourier(path, dataNumber, old_style="all")
    oStyle = ["all"]
    for style in oStyle:
        make_shape_data_for_NACA4DIGIT_circumferential(path, dataNumber, odd, old_style = style)
        make_shape_data_for_NACA5DIGIT_circumferential(path, dataNumber, odd, old_style = style)
    # 没プログラムのコーナー
    # 3つの小数について，丸めた際の最小公倍数が最小になるように切り上げ､切り捨てを判定するプログラム
    def find_minimum_lcm(pf, pc, pb):
        round2 = lambda  type, real_num: type * floor(real_num) + (1 - type) * ceil(real_num)
        count = 0
        check = 0
        d2 = ceil(pf * pc * pb)
        for i in range(2):
            a = round2(i, pf)
            for j in range(2):
                b = round2(i, pc)
                d1 = lcm(a, b)
                for k in range(2):
                    c = round2(i, pb)
                    tmp = lcm(d1, c)
                    if (d2 > tmp):
                        d2 = tmp
                        check = count
                    count += 1

        pf = round2(check // 4, pf)
        if ((check == 0) or (check == 1) or (check == 4) or (check == 5)):
            pc = round2(0, pc)
        else:
            pc = round2(1, pc)
        pb = round2(check % 2, pb)

        return pf, pc, pb, d2

    # 逆数が小数になる際に、適当な素数を掛けて整数の形にできるか試すプログラム
    def find_integer_coef(real_num):
        prime_number = np.array([1, 2, 3, 5, 7, 11, 13, 17, 19])
        epsilon = 0.001
        coef = 0
        for pn in prime_number:
            inverse = pn / real_num
            if abs(inverse - round(inverse)) < epsilon:
                coef = pn
                break

        if coef == 0:
            print("can't make integer inverse!")
            exit()

        return coef

    #
    def find_integer_inverse(lf, lc, lb):
        pnf = find_integer_coef(lf)
        pnc = find_integer_coef(lc)
        pnb = find_integer_coef(lb)
        coef = lcm(pnf, lcm(pnc, pnb))
        # rf = int(coef / lf)
        # rc = int(coef / lc)
        # rb = int(coef / lb)
        # return lcm(rf, lcm(rc, rb))
        return coef
