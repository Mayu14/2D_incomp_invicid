# coding: utf-8
import numpy as np
import time
from numba import jit
import dvm_beta as dvm
from itertools import product

def odd_number(index):
    return 2 * index + 1

def even_number(index):
    return 2 * index
    

def step4(index):
    ans = divmod(index, 20)
    headlist = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]
    p = product(headlist, repeat = 2)
    for i, v in enumerate(p):
        if i == ans[0]:
            n12 = list(v)
            break
    return int(n12[0] + n12[1] + str((12 + 4 * ans[1])))

def original_number(index):
    return index

# ここでは空力係数のデータを作成する
def stop_watch(func):
    def wrapper(*args, **kargs):
        start = time.time()
        result = func(*args, **kargs)
        elapsed_time = time.time() - start
        print(f"{func.__name__}は{elapsed_time}秒かかりました")
        return result

    return wrapper

@stop_watch
@jit
def calc_lift_coef(odd, number, inflow, type, fixed_size, kind_of_wing, kind_of_angle, step_of_angle, min_angle=None):
    # save_data[id, 0] = wing_type  # 1:joukowski, 2:karman-trefftz, 3:naca4digit
    # save_data[id, 1] = shape_data_1 : if type=1 or type=2 => center_x, elseif type=3 => NACA-4-DigitNumber
    # save_data[id, 2] = shape_data_2 : if type=1 or type=2 => center_y, elseif type=3 => Not use
    # save_data[id, 3] = attack_angle
    # save_data[id, 4] = lift_coefficient

    if min_angle == None:
        min_angle = - (2.0 * kind_of_angle) / step_of_angle

    if odd == True:
        nextnum = odd_number
        endword = "_odd"
    elif odd == False:
        nextnum = even_number
        endword = "even"
        kind_of_wing = int(0.9 * kind_of_wing)  # NACAxx00の分を除外
    elif odd == "step4":
        nextnum = step4
        endword = "mixed"
    """
    else:
        nextnum = original_number
        endword = "mixed"
        startnum = 1
    """
    split = 1
    split_kind_of_wing = int(kind_of_wing / split)
    split_pattern = split_kind_of_wing * kind_of_angle
    save_data = np.zeros((split_pattern, number), dtype = float)
    save_data[:, 0] = type

    angle_and_lift = np.zeros((kind_of_angle, 2))
    for file in range(split):
        data_id = 0
        start = split_kind_of_wing * file

        end = split_kind_of_wing * (file + 1)
        print("split " + str(file).zfill(3) + " / " + str(split) + " started")
        for wing in range(start, end):
            if ((odd == True) or (nextnum(wing) % 100 != 0)):
                save_data[data_id:data_id + kind_of_angle, 1] = nextnum(wing)
                naca4 = str(nextnum(wing)).zfill(4)  # only use odd number
                z, size = dvm.get_complex_coords(type = type, size = fixed_size, naca4 = naca4)
                copy_data_id = data_id
                for angle in range(kind_of_angle):
                    # save_data[data_id, 3] = step_of_angle * angle - kind_of_angle  # attack_angle_deg
                    angle_and_lift[angle, 0] = step_of_angle * angle - min_angle # attack_angle_deg
                    complex_U = dvm.get_complex_U(inflow_velocity = inflow, attack_angle_degree = angle_and_lift[angle, 0])
                    circulation = dvm.get_circulation(z, complex_U, gamma_output = False)
                    # save_data[data_id, 4] = dvm.get_lift_coefficient(z, circulation, complex_U)
                    angle_and_lift[angle, 1] = dvm.get_lift_coefficient(z, circulation, complex_U)
                    data_id += 1
    
                save_data[copy_data_id:copy_data_id + kind_of_angle, 3:5] = angle_and_lift
                
        fname = path + "NACA4\\s" + str(start).zfill(4) + "_e" + str(end).zfill(4) + "_a" + str(kind_of_angle).zfill(3) + endword + ".csv"
        np.savetxt(fname, save_data, delimiter = ",")
    

def make_training_data_for_NACA4DIGIT(path):
    type = 3
    size = int(500/2)   # NACA-4-digit内部で2倍されるため
    inflow = 1.0
    
    kind_of_wing = 5000    # 5000
    
    kind_of_angle = 40   # 40
    step_of_angle = 2
    
    data_number_per_wing = 5
    odd = False
    calc_lift_coef(odd, data_number_per_wing, inflow, type, size, kind_of_wing, kind_of_angle, step_of_angle)


def make_training_data_for_NACA5DIGIT(path, old_style=True):
    type = 4
    size = int(500/2)   # NACA-5-digit内部で2倍されるため
    inflow = 1.0
    if old_style:
        kind_of_wing = 99 * 9
    else:
        kind_of_wing = 99 * 9 * 6

    kind_of_angle = 40  # 40
    step_of_angle = 0.75
    min_angle = 0

    data_number_per_wing = 5
    calc_lift_coef_NACA5DIGIT(path, data_number_per_wing, inflow, type, size, kind_of_wing, kind_of_angle, step_of_angle, min_angle, old_style)


def calc_lift_coef_NACA5DIGIT(path, number, inflow, type, fixed_size, kind_of_wing, kind_of_angle, step_of_angle, min_angle=None, old_style=True):
    # save_data[id, 0] = wing_type  # 1:joukowski, 2:karman-trefftz, 3:naca4digit
    # save_data[id, 1] = shape_data_1 : if type=1 or type=2 => center_x, elseif type=3 => NACA-4-DigitNumber
    # save_data[id, 2] = shape_data_2 : if type=1 or type=2 => center_y, elseif type=3 => Not use
    # save_data[id, 3] = attack_angle
    # save_data[id, 4] = lift_coefficient
    if min_angle == None:
        min_angle = - (2.0 * kind_of_angle) / step_of_angle

    save_data = np.zeros((kind_of_wing * kind_of_angle, number), dtype=float)
    save_data[:, 0] = type

    angle_and_lift = np.zeros((kind_of_angle, 2))
    if old_style:
        head_int3 = [210, 220, 230, 240, 250, 221, 231, 241, 251]
        data_id = 0
        for int3 in head_int3:
            for int2 in range(1, 100):
                naca5 = str(int3) + str(int2).zfill(2)
                save_data[data_id:data_id + kind_of_angle, 1] = int(naca5)
    
                z, size = dvm.get_complex_coords(type=type, size=fixed_size, naca4=naca5)
    
                copy_data_id = data_id
                for angle in range(kind_of_angle):
                    angle_and_lift[angle, 0] = step_of_angle * angle - min_angle  # attack_angle_deg
                    complex_U = dvm.get_complex_U(inflow_velocity=inflow, attack_angle_degree=angle_and_lift[angle, 0])
                    circulation = dvm.get_circulation(z, complex_U, gamma_output=False)
                    angle_and_lift[angle, 1] = dvm.get_lift_coefficient(z, circulation, complex_U)
                    data_id += 1
    
                save_data[copy_data_id:copy_data_id + kind_of_angle, 3:5] = angle_and_lift

        fname = path + "NACA5\\s21001_e25199_a" + str(kind_of_angle).zfill(3) + ".csv"
    else:
        mid_int2 = [10, 20, 30, 40, 50, 21, 31, 41, 51]
        data_id = 0
        for int1 in range(1,7):
            for int23 in mid_int2:
                for int45 in range(1,100):
                    naca5 = str(int1) + str(int23) + str(int45).zfill(2)
                    save_data[data_id:data_id + kind_of_angle, 1] = int(naca5)
                    z, size = dvm.get_complex_coords(type = type, size = fixed_size, naca4 = naca5)
                    z = z[:-1]

                    copy_data_id = data_id
                    for angle in range(kind_of_angle):
                        angle_and_lift[angle, 0] = step_of_angle * angle - min_angle  # attack_angle_deg
                        complex_U = dvm.get_complex_U(inflow_velocity = inflow,
                                                      attack_angle_degree = angle_and_lift[angle, 0])
                        circulation = dvm.get_circulation(z, complex_U, gamma_output = False)
                        angle_and_lift[angle, 1] = dvm.get_lift_coefficient(z, circulation, complex_U)
                        data_id += 1
                        
                    save_data[copy_data_id:copy_data_id + kind_of_angle, 3:5] = angle_and_lift
    
        fname = path + "NACA5\\s11001_e65199_a" + str(kind_of_angle).zfill(3) + ".csv"
    np.savetxt(fname, save_data, delimiter=",")


def validate_training_data_for_NACA4DIGIT(path):
    type = 3
    size = int(500 / 2)  # NACA-4-digit内部で2倍されるため
    inflow = 1.0
    
    kind_of_wing = 1620  # 5000
    
    kind_of_angle = 14  # 40
    step_of_angle = 3
    min_angle = 0
    data_number_per_wing = 5
    odd = "step4"
    calc_lift_coef(odd, data_number_per_wing, inflow, type, size, kind_of_wing, kind_of_angle, step_of_angle, min_angle)

def load_pattern_list_inv(naca4=True):
    i3_list = [12, 22, 32, 42, 52, 62, 72, 82]
    # from naca_4digit_test import Naca_4_digit, Naca_5_digit
    if naca4:
        i2_len = 1
        i1_list = [0, 2, 5, 7, 9]
        i2_list = [0, 1, 3, 4, 6, 7]
        # naca_wing = Naca_4_digit
    else:
        i2_len = 2
        i1_list = [1, 2, 3, 4, 5]
        i2_list = [10, 21, 30, 41, 50]

    angleList = [-7.5, -5.0, -2.5, 0.0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 17.5, 20.0, 22.5, 25.0, 27.5, 30.0, 32.5,
                 35.0]
    return angleList, i1_list, i2_list, i3_list, i2_len

def make_training_data_for_NACAwing(naca4=True):
    shapeHeader = "NACA"
    angleList, i1_list, i2_list, i3_list, i2_len = load_pattern_list_inv(naca4)
    for i1 in i1_list:
        for i2 in i2_list:
            if ((i1 == 0) and (i2 == 0)) or (i1 != 0):
                for i34 in i3_list:
                    int_4 = str(i1) + str(i2).zfill(i2_len) + str(i34).zfill(2)
                    shape = shapeHeader + int_4
                    for angle in angleList:

                        # naca = naca_wing(int_4, attack_angle_deg = angle, resolution = 100, quasi_equidistant=False)
                        # dir = 'D:\\Toyota\\work2\\obj\\'
                        # fname = shapeHeader + int_4 + "_" + str(angle)
                        # naca.generate_obj(dir + fname)

                        con = Config(reynolds, mach, shape, angle)
                        # print(con.casename)


if __name__ == '__main__':
    path = "G:\\Toyota\\Data\\Incompressible_Invicid\\training_data\\"
    # make_training_data_for_NACA4DIGIT(path)
    make_training_data_for_NACA5DIGIT(path, old_style = False)
    #validate_training_data_for_NACA4DIGIT(path)