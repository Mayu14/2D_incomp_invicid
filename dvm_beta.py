# -- coding: utf-8 --
import numpy as np
from numba import jit
from joukowski_wing import joukowski_wing_complex, karman_trefftz_wing_complex
from legacy_vtk_writer import MyStructuredVTKWriter
from naca_4digit_test import Naca_4_digit, Naca_5_digit

def get_lift(density, circulation, complex_U):
    return density * np.abs(complex_U) * circulation

def get_lift_coefficient(z, circulation, complex_U):
    length = np.max(np.real(z)) - np.min(np.real(z))
    return 2.0 * circulation / (np.abs(complex_U) * length)

def get_complex_U(inflow_velocity, attack_angle_degree):
    return inflow_velocity * np.exp(-1j * attack_angle_degree * np.pi / 180.0)

def get_complex_coords(type, size, center_x = -0.08, center_y = 0.08, naca4="0012"):
    if type == 0:
        t = np.linspace(start = 0, stop = 2.0 * np.pi, num = size + 1)
        z = np.exp(1j * t)[:size]
    elif type == 1:
        z = joukowski_wing_complex(size, center_x, center_y)
    elif type == 2:
        z = karman_trefftz_wing_complex(size, center_x, center_y)
    elif type == 3:
        naca = Naca_4_digit(int_4 = naca4, attack_angle_deg = 0.0, resolution = size, quasi_equidistant = False)
        z = naca.transform2complex()
    elif type == 4:
        naca = Naca_5_digit(int_5= naca4, attack_angle_deg = 0.0, resolution = size, quasi_equidistant = False, length_adjust=True)
        z = naca.transform2complex()
    else:
        print("type error")
        exit()
        
    if type != 3:
        return reshape_z(z)
    else:
        return z, z.shape[0]
        

def reshape_z(z):
    if z[0] != z[z.shape[0] - 1]:
        return np.concatenate([z, z[0].reshape(-1)]), z.shape[0] + 1
    else:
        return z, z.shape[0]


@jit
def get_circulation(z, complex_U, gamma_output=True):
    # z: complex coordinate information (counter clock wise from endpoint of wing)
    # complex_U: complex velocity of inflow including angle information
    # setting for calculate
    size = z.shape[0]
    ref = 0.5 * (z[:size - 1] + z[1:])
    delta = z[1:] - z[:size - 1]
    len = np.abs(delta)
    normal = delta / (1j * len)

    # outflow angle
    beta = 0.5 * np.angle((z[1] - z[0]) * (z[size - 2] - z[0]))  # for kutta condition
    
    size1 = size - 1
    S1 = np.zeros((size1, size1), dtype = complex)
    S2 = np.zeros((size1, size1), dtype = complex)
    
    for i in range(size1):
        for j in range(size1):
            coef = len[j] / delta[j]
            log = np.log((z[j + 1] - ref[i]) / (z[j] - ref[i]))
            S1[i, j] = coef * (-1.0 + ((z[j + 1] - ref[i]) / delta[j]) * log)
            S2[i, j] = coef * (1.0 - ((z[j] - ref[i]) / delta[j]) * log)

    # make coefficient matrix
    A = np.zeros((size, size), dtype = float)
    for i in range(size):
        if i < size - 1:
            for j in range(size):
                if j == 0:
                    A[i, j] = np.imag(S1[i, j] * normal[i])
                elif j == size - 1:
                    A[i, j] = np.imag(S2[i, j - 1] * normal[i])
                else:
                    A[i, j] = np.imag((S1[i, j] + S2[i, j - 1]) * normal[i])
    
    A[size - 1, 0] = np.real(-np.abs(z[1] - z[0]) / (z[1] - z[0]) * np.exp(1j * beta))
    A[size - 1, size - 1] = np.real(
        np.abs(z[size - 1] - z[size - 2]) / (z[size - 1] - z[size - 2]) * np.exp(1j * beta))

    # make constant vector
    B = np.zeros(size)
    for i in range(size):
        if i < size - 1:
            B[i] = 2.0 * np.pi * np.real(complex_U * normal[i])

    # solve equation for "gamma" (distribution vertex)
    gamma = np.linalg.solve(A, B)
    if gamma_output:
        return gamma, (- np.sum(0.5 * (gamma[:size - 1] + gamma[1:]) * len))
    else:
        return (- np.sum(0.5 * (gamma[:size - 1] + gamma[1:]) * len))

def get_velocity(z, gamma, complex_U, path=None, fname=None, plot_resolution=500, detail=False, square_plot=True):
    area_size = 2.0
    # setting of plot area
    x_max = np.max(np.real(z))
    x_min = np.min(np.real(z))
    y_max = np.max(np.imag(z))
    y_min = np.min(np.imag(z))
    plot_center = 0.5 * ((x_max + x_min) + 1j * (y_max + y_min))
    plot_width = (x_max - x_min) + 1j * (y_max - y_min)
    plot_min = plot_center - area_size * plot_width
    plot_max = plot_center + area_size * plot_width
    if square_plot == True:
        min_sq = min(np.real(plot_center - area_size * plot_width), np.imag(plot_center - area_size * plot_width))
        max_sq = max(np.real(plot_center + area_size * plot_width), np.imag(plot_center + area_size * plot_width))
        plot_min = min_sq + 1j * min_sq
        plot_max = max_sq + 1j * max_sq
    x = np.linspace(start=np.real(plot_min), stop=np.real(plot_max), num=plot_resolution).reshape(-1, 1)
    y = np.linspace(start=np.imag(plot_min), stop=np.imag(plot_max), num=plot_resolution).reshape(1, -1)
    ref = x + 1j * y
    w = np.zeros((plot_resolution, plot_resolution), dtype=complex)

    size = z.shape[0]
    delta = z[1:] - z[:size - 1]
    len = np.abs(delta)
    d_gamma = gamma[1:] - gamma[:size - 1]

    u_j = lambda z_ref: np.sum(len / delta * (
                d_gamma + (d_gamma / delta * (z_ref - z[:size - 1]) + gamma[:size - 1]) * np.log(
            (z[1:] - z_ref) / (z[:size - 1] - z_ref))))

    # x, yどちらかの小さな方に合わせて調整

    eps = 0.5 * min(np.min(np.real(z[1:] - z[:size-1])), np.min(np.imag(z[1:] - z[:size-1])))

    for x_ref in range(plot_resolution):
        for y_ref in range(plot_resolution):
            if (np.min(np.abs(z - ref[x_ref, y_ref]))) > eps:
                w[x_ref, y_ref] = complex_U + 1j / (2.0 * np.pi) * u_j(ref[x_ref, y_ref])

    velocity = np.zeros((plot_resolution, plot_resolution, 2), dtype=float)
    velocity[:, :, 0] = np.real(w)
    velocity[:, :, 1] = - np.imag(w)

    if ((path != None) and (fname != None)):
        plot_vtk(velocity, plot_resolution, plot_min, plot_max, path, fname)
    
    if detail == True:
        return velocity, plot_min, plot_max
    else:
        return velocity

def plot_vtk(velocity, plot_resolution, plot_min, plot_max, path, fname):
    # plot_min, and plot_max are complex variable
    # real part => x_min and x_max, image part => y_min and y_max, respectively.
    x = np.linspace(start=np.real(plot_min), stop=np.real(plot_max), num=plot_resolution + 1)
    y = np.linspace(start=np.imag(plot_min), stop=np.imag(plot_max), num=plot_resolution + 1)
    z = np.array([0.0])

    writer = MyStructuredVTKWriter(xcoords=x, ycoords=y, zcoords=z, data_dimension=2)  # vtkwriterの初期化
    writer.filename = fname
    writer.casename = "Discrete_Vertex_Method"
    writer.append_data(name="velocity", SCALARS=False, data=velocity)
    writer.output(path)


def validation(type, center_x, center_y, complex_U):
    U = np.real(complex_U)
    angle = -np.angle(complex_U)
    if type == 0:
        return 0, 0
    elif type == 1:
        R = np.sqrt((1.0 - center_x) ** 2 + center_y ** 2)
        lift = 4.0 * np.pi * np.abs(complex_U) * R * np.sin(angle + np.arcsin(center_y / R))
        lift_coef = 2.0 * np.pi * R * np.sin(angle + np.arcsin(center_y / R))
        return lift, lift_coef
    else:   # type == 2:
        print("type error")
        exit()

def zu2circ(z_aoa0, v_in, aoa, deg=False, getLift=True):
    angle_d = aoa
    if not deg:
        angle_d *= 180.0 / np.pi

    complex_U = get_complex_U(inflow_velocity=v_in, attack_angle_degree=angle_d)
    gamma, circulation = get_circulation(z=z_aoa0, complex_U=complex_U)
    if getLift:
        lift = get_lift(density=1.0, circulation=circulation, complex_U=complex_U)
        lift_coef = get_lift_coefficient(z_aoa0, circulation, complex_U)
        return lift, lift_coef
    else:
        return gamma, circulation

def get_naca_gamma(naca4="0012"):
    fixed_size = 250
    z, size = get_complex_coords(type=type, size=fixed_size, naca4=naca4)


def main():
    size = 200
    center_x = -0.08
    center_y = 1.0
    naca4 = "6633"
    inflow = 1.0
    alpha = 0.0
    type = 0
    complex_U = get_complex_U(inflow_velocity = inflow, attack_angle_degree = alpha)

    z, size = get_complex_coords(type, size, center_x, center_y, naca4)

    gamma, circulation = get_circulation(z = z, complex_U = complex_U)
    lift = get_lift(density = 1.0, circulation = circulation, complex_U = complex_U)
    lift_coef = get_lift_coefficient(z, circulation, complex_U)
    print(lift, lift_coef)
    exit()
    # print(lift, lift_coef, validation(type, center_x, center_y, complex_U))

    path = "D:\\Toyota\\Data\\DVM_mk7\\"
    fname = "dvm_mk7_test_type_" + str(type).zfill(2)
    velocity = get_velocity(z, gamma, complex_U, path, fname)
    
    exit()
    
if __name__ == '__main__':
    main()