# -- coding: utf-8 --
import numpy as np
import scipy.linalg.decomp_qr as qr
from dvm_beta import get_complex_coords, get_complex_U
from naca_4digit_test import Naca_4_digit, Naca_5_digit

"""
参考文献
[1] 三上, 宮本, 安藤, "渦法によるはく離流れの解析 -回転円柱および回転円筒部をもつ翼まわりの流れ-", ながれ 12, (1993), pp.31-45
"""

# 渦要素
# class Vortex(object):



def initialize(z):
    # 参考文献[1]の(16)式について
    # 物体形状からAを算出してQR分解，QとRを返す

    # 準備
    p_size = z.shape[0] - 1 # z[0] = z[size - 1]に注意されたし 始点と終点が同じ点を示しているため，点の数は-1)
    ref = 0.5 * (z[:p_size] + z[1:])  # パネル中点の参照座標の取得(論文中の添字i)
    delta = z[1:] - z[:p_size]    # 各パネル単点間の距離
    len = np.abs(delta) # パネルの長さ(論文中のlj)

    normal = delta / (1j * len) # パネルの単位法線ベクトル(論文中のn)

    x = np.real(z[:p_size]) # 始点=終点のため終点を除く
    y = np.imag(z[:p_size])
    nx = np.real(normal)
    ny = np.imag(normal)
    # Aの計算
    A = np.zeros((p_size + 1, p_size))    # 行列は(辺の数+1 × 辺の数)

    for i in range(p_size):  # 最後の行の方程式は別
        h_1 = ((x[0] - x[p_size - 1]) * ny[i] - (y[0] - y[p_size - 1]) * nx[i]) / len[p_size - 1]   # 終点=始点のため端だけ別処理
        h = ((x[1:] - x[:p_size - 1]) * ny[i] - (y[1:] - y[:p_size - 1]) * nx[i]) / len[:p_size - 1] # j = 1 to end

        A[i, 0] = 1.0 / (2.0 * np.pi) * (- h_1 + h[0])

        for j in range(1, p_size - 1):
            A[i, j] = 1.0 / (2.0 * np.pi) * (- h[j - 1] + h[j])

        A[i, p_size - 1] = 1.0 / (2.0 * np.pi) * (- h[p_size - 2] + h_1)

    A[p_size] = len

    # QR分解して直交行列Qの転置と上三角行列Rを返す
    Q, R = qr.qr(np.dot(A.T, A))
    return Q.T, R


def main():
    # 計算条件
    size = 200
    center_x = -0.08
    center_y = 1.0
    naca4 = "6633"
    inflow = 1.0
    alpha = 0.0
    type = 0

    # 一様流の複素速度
    complex_U = get_complex_U(inflow_velocity = inflow, attack_angle_degree = alpha)

    z, size = get_complex_coords(type, size, center_x, center_y, naca4) # 離散的な物体形状(パネル端点)の複素座標の取得(論文中の添字j)

    invQ, R = initialize(z)

    print(invQ)


if __name__ == '__main__':
    main()
