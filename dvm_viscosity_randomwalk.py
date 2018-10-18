# -- coding: utf-8 --
import numpy as np
import scipy.sparse.linalg as spla
from scipy import fftpack
from dvm_beta import get_complex_coords, get_complex_U, plot_vtk
from math import sqrt, ceil
from cmath import phase
from random import gauss
import matplotlib.pyplot as plt
"""
参考文献
[1] 三上, 宮本, 安藤, "渦法によるはく離流れの解析 -回転円柱および回転円筒部をもつ翼まわりの流れ-", ながれ 12, (1993), pp.31-45
"""

# 渦要素オブジェクト
class VortexElement(object):
    def __init__(self, circulation, coordinate, radius):
        self.circulation = circulation
        self.coordinate_z0 = coordinate # old_coordinate
        self.coordinate_z1 = coordinate # new_coordinate

        self.radius = radius

        self.coordinate = self.coordinate_z1
        self.locus_center = 0.0 + 1.0j * 0.0

        # 渦要素の情報を一括で返す
    def get_vortex_data(self):
        return self.circulation, self.coordinate, self.radius

    # 更新したい座標(現在の座標からの修正値)を登録する
    def store_new_coordinate(self, correction_value):
        # self.coordinate_z0 = self.coordinate_z1 # 現在座標を旧座標として登録
        self.coordinate_z0 = self.coordinate  # 現在座標を旧座標として登録
        self.coordinate_z1 += correction_value # 新規座標を登録
        self.locus_center = self.coordinate + 0.5 * correction_value    # z0とz1を結ぶ軌跡の中間点の座標
        self.distance = np.abs(correction_value)

    # 登録された座標を反映する
    def update_new_coordinate(self):
        self.coordinate = self.coordinate_z1

# 渦要素オブジェクト制御用プログラム群
class VortexControl(object):
    def __init__(self, z, complex_U, path, fname):
        self.number = 0 # 渦要素の総数
        self.child_list_num = 0 # 子リストの総数
        self.current_child = 0  # 現在の子リスト位置(削除するかも)

        self.parent_list = [[]]   # child_listのリスト
        self.len_list = [0]  # child_listの現在の要素数
        self.add_list = []  # 追加対象のリスト
        self.deletion_list = [[]]   # 削除対象のリスト
        self.merge_list = []    # 結合対象のリスト

        self.set_config()

        self.complex_U = complex_U
        # 幾何形状関連
        self.z = z  # 物体形状
        self.p_size = z.shape[0] - 1  # z[0] = z[size - 1]に注意されたし 始点と終点が同じ点を示しているため，点の数は-1)
        self.ref = 0.5 * (z[:self.p_size] + z[1:])  # パネル中点の参照座標の取得(論文中の添字i)
        self.delta = (z[1:] - z[:self.p_size])  # 各パネル単点間の距離(物体を左回りする際の接線ベクトルとしている点に注意)
        self.len = np.abs(self.delta)  # パネルの長さ(論文中のlj)
        self.normal = self.delta / (1j * self.len)  # パネルの単位法線ベクトル(論文中のn)
        self.max_len = np.max(self.len) # パネル長のうち最大のもの
        self.obj_len = np.max(np.real(z)) - np.min(np.real(z))  # 空力係数計算用の面積
        self.outer_area = 2.0 * np.abs(max(np.max(np.real(z)) - np.min(np.real(z)), np.max(np.imag(z)) - np.min(np.imag(z)))) # 物体の長軸方向の2倍
        self.object_center = 0.5 * ((np.max(np.real(z)) + np.min(np.real(z))) + 1.0j * (np.max(np.imag(z)) + np.min(np.imag(z))))   # 物体の中心座標

        eps = 0.1
        self.left_bound = np.min(np.real(z)) - eps
        self.right_bound = np.max(np.real(z)) + eps
        self.top_bound = np.max(np.imag(z)) + eps
        self.bottom_bound = np.min(np.imag(z)) - eps
        # パネルごとの単位時間あたりの循環生成量
        self.time_derivative_of_circulation = np.zeros(self.p_size)

        self.set_tAA_and_tA()

        # 初期化が必要な変数
        self.machine_zero = 10.0**(-14)
        self.lift_history = []
        self.drag_history = []
        self.lift = 100
        self.drag = 100
        self.gamma = np.zeros(self.p_size)
        # plotの設定
        self.plot_area_size = 2.0
        self.square_plot = True
        self.vtk_plot_resolution = 500
        self.vtk_path = path
        self.vtk_name = fname

    # 計算設定の読み込み(本処理を開始する前に外部から別途呼び出すことを想定している)
    def set_config(self, reynolds_number = 5000, induced_velocity_limit = 10.0**(2), limit_for_element_list = 1000, vortex_limit = 100000, min_edge = 0.001, time_step_convection=0.01, time_division_diffusion=2, tolerance = 0.005):
        self.reynolds_number = reynolds_number
        self.viscosity = 1.0 / reynolds_number
        self.standard_deviation = sqrt(2.0 * self.viscosity * time_step_convection / time_division_diffusion)

        self.limit_of_induced_velocity = induced_velocity_limit    # 1つの渦要素が誘導できる速度の最大値
        self.limit_for_elements_of_list = limit_for_element_list  # 子リスト内要素の上限に関する初期値
        self.vortex_limit = vortex_limit    # 渦要素の数上限

        self.limit_of_minimum_radius = min_edge / np.pi # min_edgeはzの差分の絶対値(z_len)の最小値を想定(初期値はかなり適当)
        self.limit_of_circulation = self.limit_of_induced_velocity * (2.0 * np.pi * self.limit_of_minimum_radius)

        self.timestep_convection = time_step_convection
        self.time_division_diffusion = time_division_diffusion
        self.timestep_diffusion = time_step_convection / time_division_diffusion

        self.tolerance = tolerance

    def test_case(self):
        # 初期条件の影響を消すための慣らし区間
        self.timestep_convection = 0.05
        for i in range(int(5.0/self.timestep_convection)):
            print(float(i+1)*self.timestep_convection, self.number)
            self.main_process()

        self.timestep_convection = 0.01
        for i in range(int(10.0/self.timestep_convection)):
            self.main_process()
            self.get_aero_characteristics(output_residual=False)
            print(5.0+float(i+1)*self.timestep_convection, self.number, self.lift, self.drag)
            self.plot_detail_velocity_field()

    def main_process(self):
        self.get_circulation_on_panel() # 非粘性仮定でパネル上の循環を求める
        self.generate_vortex()  # 物体表面の接線方向速度を消すために渦要素を発生させる

        self.get_circulation_on_panel() # 追加された渦要素を含めて非粘性仮定で循環値を修正する
        self.get_move_convection()  # 渦要素の移流先を計算する

        for step in range(self.time_division_diffusion):    # 拡散計算の分割処理
            self.get_move_diffusion()   # 渦要素の拡散先を計算する

        self.search_for_deletion_target()  # 接触判定を行う
        self.del_Vortex()  # 物体と接触した渦要素を削除する
        self.complete_move_vortex()  # 移動を反映する

        self.merge_vortex() # 渦要素を結合する


    # 渦要素の循環の合計値を求めるメソッド(O(Nv))
    def get_circulation_sum(self):
        circulation = 0
        for child_list in self.parent_list:    # 全部の子リストについてループ
            for vortex in child_list:    # 子リスト内全要素についてループ
                circulation += vortex.circulation

        return circulation

    # 複素座標z_refに渦要素が誘導する速度を求めるメソッド(複素dist_vector方向速度を求めるオプション有)(O(Nv + Np))
    # include_panel = Trueのときパネル上の循環も考慮に入れる(パネル上の分布を未知とするときのみFalse)
    def get_velocity_from_vortex(self, z_ref, dist_vector = None, include_panel = True):
        def velocity_from_vorticity(z_ref, z_vortex, r_vortex, g_vortex, outer):
            difference = z_ref - z_vortex
            distance = np.abs(difference)  # [1](7)式のr
            gr = min(distance / r_vortex, 1.0)  # [1](7)式のg(r)

            if distance > machine_eps:
                return (g_vortex * gr / distance**2) * (difference * outer)  # [1](5)式

            else:
                return 0

        velocity = 0
        machine_eps = self.machine_zero   # 参照点と渦要素の距離がこの値以下であるとき同一座標とみなす
        outer = -1.0j   # 平面ベクトルと垂直なベクトルとの外積は複素平面における-90度回転と一致する
        for child_list in self.parent_list:    # 全部の子リストについてループ
            for vortex in child_list:    # 子リスト内全要素についてループ
                # velocity += velocity_from_vorticity(z_ref, vortex.coordinate, vortex.radius, vortex.circulation, outer)
                dv = velocity_from_vorticity(z_ref, vortex.coordinate, vortex.radius, vortex.circulation, outer)
                velocity += dv

        # パネル上の渦要素からの寄与
        if include_panel:
            velocity += velocity_from_vorticity(z_ref, self.z[0], self.len[0] / (10000.0 * np.pi), 0.5 * (self.len[self.p_size - 1] + self.len[0]) * self.gamma[0], outer)
            for i in range(1, self.p_size):
                velocity += velocity_from_vorticity(z_ref, self.z[i], self.len[i]/(10000.0*np.pi), 0.5 * (self.len[i - 1] + self.len[i]) * self.gamma[i], outer)

        if dist_vector != None:
            return np.real((velocity / (2.0 * np.pi) + self.complex_U) * np.conjugate(dist_vector))    # dist_vector方向成分を返す(返り値は実数)

        return velocity / (2.0 * np.pi) + self.complex_U # (返り値は複素数)

    # 移流による移動を計算して，次の座標を登録する(O(Nv^2))
    def get_move_convection(self):
        for child_list in self.parent_list:    # 全部の子リストについてループ
            for vortex in child_list:    # 子リスト内全要素についてループ
                # vortex.store_new_coordinate(self.get_velocity_from_vortex(vortex.coordinate) * self.timestep_convection)
                delta_z = self.get_velocity_from_vortex(vortex.coordinate) * self.timestep_convection
                vortex.store_new_coordinate(delta_z)


    # 拡散による移動を計算して次の座標を登録する(O(Nv))
    def get_move_diffusion(self):
        for child_list in self.parent_list:    # 全部の子リストについてループ
            for vortex in child_list:    # 子リスト内全要素についてループ.
                vortex.store_new_coordinate(min(gauss(0.0, self.standard_deviation), 0.1) + 1j * min(gauss(0.0, self.standard_deviation), 0.1))

    # 登録された移動を反映する(O(Nv))
    def complete_move_vortex(self):
        for child_list in self.parent_list:    # 全部の子リストについてループ
            for vortex in child_list:    # 子リスト内全要素についてループ
                vortex.update_new_coordinate()

    # パネル上から渦要素を生成するメソッド(O(Np*Nv))
    def generate_vortex(self):
        p_size = self.p_size  # z[0] = z[size - 1]に注意されたし 始点と終点が同じ点を示しているため，点の数は-1)
        ref = self.ref  # パネル中点の参照座標の取得(論文中の添字i)
        # inv_delta = - self.delta  # 各パネル単点間の距離(物体を右回りする際の接線ベクトルとしている点に注意)
        len = self.len  # パネルの長さ(論文中のlj)
        inv_unit_tangent_vector = (- self.delta / self.len)

        generate_point = self.delta / (1j * np.pi)  # 渦要素を配置する座標(パネル中心からの相対距離)  [1](20)式に基づく

        for i in range(p_size): # 全パネルに関するループ
            tangent_velocity = self.get_velocity_from_vortex(z_ref=ref[i], dist_vector=inv_unit_tangent_vector[i])    # 右回り接線方向速度

            if abs(tangent_velocity) > self.machine_zero:
                tmpCirculation = tangent_velocity * len[i]  # 必要な循環値
                self.time_derivative_of_circulation[i] += tmpCirculation    # 単位時間当たりのパネルから生じる循環の強さ [1](21)式右辺第2項

                number = ceil(abs(tmpCirculation) / self.limit_of_circulation)  # 配置する渦要素の数
                new_circulation = tmpCirculation / number   # 新しい要素1つあたりの循環値
                new_coordinate = ref[i] + generate_point[i] # 渦要素を生成する座標

                # 恐らく，追加と登録を分けないと要素の追加の順番に影響を受ける
                for new_vortex in range(number):    # 新規追加する渦要素の数だけループ
                    self.register_add_target(new_circulation, new_coordinate, np.abs(generate_point[i])) # 半径として参照点と渦要素の初期位置との距離をそのまま採用(次ステップで物体から離れる方向へ移動しなければ渦要素は消滅する)

        self.add_registered_vortex()


    # 子リストの追加メソッド
    def make_new_child_list(self):
        self.child_list_num += 1
        self.current_child += 1
        self.parent_list.append([])
        self.len_list.append(0)
        self.deletion_list.append([])

    # 渦要素の追加メソッド
    def add_Vortex(self, circulation, coordinate, radius):
        self.number += 1
        if self.len_list[self.current_child] >= self.limit_for_elements_of_list:    # リスト内の渦要素数が上限に達していた場合
            self.make_new_child_list()  # 新たに子リストを作成する
        self.len_list[self.current_child] += 1
        # list = self.parent_list[self.current_child]対象
        # list.append(VortexElement(circulation, coordinate, radius))
        self.parent_list[self.current_child].append(VortexElement(circulation, coordinate, radius))

    def add_registered_vortex(self):
        add_list = np.array(self.add_list, dtype=complex).reshape(-1, 3)

        for i in range(add_list.shape[0]):
            self.add_Vortex(np.real(add_list[i, 0]), add_list[i, 1], np.real(add_list[i, 2]))
        self.add_list = []  # 初期化

    # 子リスト・idを指定して渦要素を削除対象として登録するメソッド
    def register_deletion_target(self, child, id):
        self.deletion_list[child].append(id)

    def register_add_target(self, circulation, coordinate, radius):
        self.add_list.append(circulation)
        self.add_list.append(coordinate)
        self.add_list.append(radius)

    # 子リスト・idを指定して渦要素を結合対象として登録するメソッド
    def register_merge_target(self, child, id, angle):
        self.merge_list.append(child)
        self.merge_list.append(id)
        self.merge_list.append(angle)    # 方向

    # 削除対象として登録された渦要素を一括で削除するためのメソッド
    def del_Vortex(self):
        # if self.deletion_list != []:
        for child in range(self.child_list_num + 1):    # 全部の子リストについてループ
            if not self.deletion_list[child] == False:   # 削除対象の渦要素が1つ以上含まれるとき
                target_id = np.sort(np.array(self.deletion_list[child]))[::-1]   # ndarrayに変換し，idの大きい順に並び替える

                for id in target_id:    # 大きい順に並べ替えたので後ろから削除するだけ
                    del self.parent_list[child][id]
                    self.number -= 1
                    self.len_list[child] -= 1

                # 削除対象登録の初期化
                self.deletion_list[child] = []

    # 2つの渦要素を結合するメソッド(結合前の2要素を削除して新規追加する)
    def __merge(self, vortex_info1, vortex_info2):
        # 循環の強度は単純な足し合わせ
        def merge_circulation(g1, g2):
            return g1 + g2

        # 渦の中心座標は渦の面積に対する重心位置
        def merge_coordinate(z1, z2, r1, r2):
            s1 = r1 ** 2
            s2 = r2 ** 2
            return (s1 * z1 + s2 * z2) / (s1 + s2)

        # 渦半径は渦の面積を保存するように指定
        def merge_radius(r1, r2):
            return sqrt(r1**2 + r2**2)

        child1 = int(vortex_info1[0])
        child2 = int(vortex_info2[0])
        id1 = int(vortex_info1[1])
        id2 = int(vortex_info2[1])

        # g1, z1, r1 = self.parent_list[child1][id1].get_vortex_data
        # g2, z2, r2 = self.parent_list[child2][id2].get_vortex_data
        g1, z1, r1 = self.parent_list[child1][id1].circulation, self.parent_list[child1][id1].coordinate, self.parent_list[child1][id1].radius
        g2, z2, r2 = self.parent_list[child2][id2].circulation, self.parent_list[child2][id2].coordinate, self.parent_list[child2][id2].radius

        self.register_deletion_target(child1, id1)
        self.register_deletion_target(child2, id2)
        self.register_add_target(merge_circulation(g1, g2), merge_coordinate(z1, z2, r1, r2), merge_radius(r1, r2))

    # 結合処理の対象となる要素を検索する
    def merge_vortex(self):
        self.boundary_sphere = False

        count = 0
        if self.boundary_sphere:
        # 物体から十分離れている要素を削除対象とする(円の外に出たら結合対象)
            for child in range(self.child_list_num + 1):
                child_list = self.parent_list[child]
                for id in range(self.len_list[child]):
                    difference = child_list[id].coordinate - self.object_center
                    if (abs(difference) > self.outer_area):
                        self.register_merge_target(child, id, phase(difference))    # 渦要素が物体遠方に存在するとき，結合可能として登録
                        count += 1
        else:
            def check_box(z):
                if ((np.real(z) < left) or (right < np.real(z)) or (np.imag(z) < bottom) or (top < np.imag(z))):
                    return True
                else:
                    return False

            left = self.left_bound
            right = self.right_bound
            top = self.top_bound
            bottom = self.bottom_bound
            for child in range(self.child_list_num + 1):
                child_list = self.parent_list[child]
                for id in range(self.len_list[child]):
                    if check_box(child_list[id].coordinate):
                        self.register_merge_target(child, id, phase(child_list[id].coordinate - self.object_center))  # 渦要素が指定領域の外に存在するとき，結合可能として登録
                        count += 1
        # numpy_arrayに変換
        merge_list = np.array(self.merge_list).reshape(-1, 3)
        # 角度基準で並び替えを行う
        merge_target = merge_list[np.argsort(merge_list[:, 2], axis=0)] # child, id, angleの構造を保ったままangle基準で並べ替える

        # 角度の近いもの同士を2つずつ結合する
        for i in range(0, count - 1, 2):
            self.__merge(merge_target[i, 0:2], merge_target[i + 1, 0:2])

        # すべての結合が終了したら
        self.del_Vortex()
        self.add_registered_vortex()
        self.merge_list = []    # リストの初期化

    # 削除対象となる要素を検索する
    def search_for_deletion_target(self):
        def find_intersection_line_by_line(p1, p2, p3, p4):
            def det123(z1, z2, z3):
                # a = np.array([[1, 1, 1], [np.real(z1), np.real(z2), np.real(z3)], [np.imag(z1), np.imag(z2), np.imag(z3)]])
                return np.linalg.det(np.array([[1, 1, 1], [np.real(z1), np.real(z2), np.real(z3)], [np.imag(z1), np.imag(z2), np.imag(z3)]]))

            if det123(p1, p2, p3) * det123(p1, p2, p4) < 0:
                if det123(p3, p4, p1) * det123(p3, p4, p2) < 0:
                    return True

            return False

        z = self.z
        z_ref = self.ref
        z_size = self.p_size
        edge_max = self.max_len

        for child in range(self.child_list_num + 1):    # 全部の子リストについてループ
            child_list = self.parent_list[child]
            for id in range(self.len_list[child]):    # 子リスト内全要素についてループ
                vortex = child_list[id]
                # 各辺の中点を中心とした辺の長さを直径とする円と，座標移動の軌跡の中点を中心とした渦直径を半径とする円とが接触する場所を検索
                candidate = np.argwhere(np.abs(z_ref - vortex.locus_center) < 0.5 * edge_max + vortex.radius + max(vortex.radius, vortex.distance))

                # 移動の両端
                p1 = vortex.coordinate_z0
                p2 = vortex.coordinate_z1

                for i in candidate.reshape(-1):
                    p3 = z[i]
                    # パネル辺の格納
                    if i != z_size - 1:
                        p4 = z[i + 1]
                    else:
                        p4 = z[0]   # 端点対策
                    if find_intersection_line_by_line(p1, p2, p3, p4):
                        self.time_derivative_of_circulation[i] -= vortex.circulation    # 単位時間あたりのパネルからの循環生成量
                        self.register_deletion_target(child, id)    # 接触してれば削除要素として登録


    def get_circulation_on_panel(self):
        for i in range(self.p_size):
            self.b[i] = self.get_velocity_from_vortex(self.ref[i], dist_vector=-self.delta[i] / self.len[i], include_panel=False)   # panel上の渦分布は未知

        self.b[self.p_size] = - self.get_circulation_sum()

        self.gamma = spla.bicg(self.tAA, np.dot(self.tA, self.b))[0]
        error = np.dot(self.tA.T, self.gamma) - self.b
        # print(np.dot(error.T, error))
        # self.plot_velocity_field()


    def __get_pressure_on_panel(self):
        self.pressure = np.zeros(self.p_size)

        for i in range(self.p_size - 1):
            self.pressure[i + 1] = self.pressure[i] - self.time_derivative_of_circulation[i] / self.timestep_convection  # [1](21)式

        self.time_derivative_of_circulation = np.zeros(self.p_size)

    def get_aero_characteristics(self, output_residual = True):
        self.lastDrag = self.drag
        self.lastLift = self.lift

        self.__get_pressure_on_panel()

        characteristics = np.sum(self.pressure * self.normal * self.len) / (0.5 * np.abs(self.complex_U) ** 2 * self.obj_len)
        self.drag = np.real(characteristics)
        self.lift = np.imag(characteristics)
        self.drag_history.append(self.drag)
        self.lift_history.append(self.lift)

        if output_residual:
            return max(sqrt((self.drag - self.lastDrag)**2), sqrt((self.lift - self.lastLift)**2))

    def plot_characteristics_history(self):
        drag = np.array(self.drag_history)
        lift = np.array(self.lift_history)
        t = np.arange(drag.shape[0]) * self.timestep_convection
        plt.plot(t, drag)
        plt.plot(t, lift)
        plt.show()

    def set_tAA_and_tA(self):
        # 参考文献[1]の(16)式について
        # QR分解してもQとRが悪条件になってしまうため，正規方程式を反復法で解く
        # 準備
        z = self.z
        p_size = self.p_size # z[0] = z[size - 1]に注意されたし 始点と終点が同じ点を示しているため，点の数は-1)
        len = self.len # パネルの長さ(論文中のlj)
        normal = self.normal # パネルの単位法線ベクトル(論文中のn)

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


        A[p_size, 0] = (len[p_size - 1] + len[0]) / 2.0
        for j in range(1, p_size):
            A[p_size, j] = (len[j - 1] + len[j]) / 2.0

        self.tA = A.T
        self.tAA = np.dot(A.T, A)
        self.b = np.zeros(p_size + 1)

    # other

    def plot_velocity_field(self, circle = True):
        num = 5
        if circle:
            theta = np.linspace(0, 2.0*np.pi, num**2)
            a = np.linspace(1.001, 1.05, num)
            r = a.reshape(-1, 1) * np.exp(1.0j * theta.reshape(1, -1))
        else:
            a = np.linspace(-0.5, 1.5, num)
            r = a.reshape(-1, 1) + 1j * a.reshape(1, -1)

        w = np.zeros_like(r, dtype=complex)
        for i in range(w.shape[0]):
            for j in range(w.shape[1]):
                w[i, j] = self.get_velocity_from_vortex(r[i, j])
        # print(np.max(w))
        # print(np.min(w))
        plt.quiver(np.real(r), np.imag(r), np.real(w), np.imag(w))
        plt.show()

    def plot_object(self):
        plt.plot(np.real(self.z), np.imag(self.z))
        plt.show()

    def plot_vortex_element(self):
        z = np.zeros(self.number, dtype=complex)
        z1 = np.zeros(self.number, dtype=complex)
        i = 0
        for child_list in self.parent_list:
            for vortex in child_list:
                z[i] = vortex.coordinate_z0
                z1[i] = vortex.coordinate_z1

                i += 1

        plt.plot(np.real(z), np.imag(z), "o")
        plt.plot(np.real(z1), np.imag(z1), "o")
        plt.plot(np.real(self.z), np.imag(self.z))
        plt.show()

    def plot_detail_velocity_field(self):
        area_size = self.plot_area_size
        square_plot = self.square_plot
        plot_resolution = self.vtk_plot_resolution
        path = self.vtk_path
        fname = self.vtk_name

        plot_width = (self.right_bound - self.left_bound) + 1j * (self.top_bound - self.bottom_bound)
        plot_min = self.object_center - area_size * plot_width
        plot_max = self.object_center + area_size * plot_width
        if square_plot == True:
            min_sq = min(np.real(plot_min), np.imag(plot_min))
            max_sq = max(np.real(plot_max), np.imag(plot_max))
            plot_min = min_sq + 1j * min_sq
            plot_max = max_sq + 1j * max_sq
        x = np.linspace(start=np.real(plot_min), stop=np.real(plot_max), num=plot_resolution).reshape(-1, 1)
        y = np.linspace(start=np.imag(plot_min), stop=np.imag(plot_max), num=plot_resolution).reshape(1, -1)
        ref = x + 1j * y
        w = np.zeros((plot_resolution, plot_resolution), dtype=complex)

        for i in range(plot_resolution):
            for j in range(plot_resolution):
                w[i, j] = self.get_velocity_from_vortex(ref[i, j])

        velocity = np.zeros((plot_resolution, plot_resolution, 2), dtype=float)
        velocity[:, :, 0] = np.real(w)
        velocity[:, :, 1] = np.imag(w)

        plot_vtk(velocity, plot_resolution, plot_min, plot_max, path, fname)
    """
    def noize_removal(self):
        drag = np.array(self.drag_history)
        lift = np.array(self.lift_history)
        num = int(10.0/self.timestep_convection)
        t = np.arange(0, self.timestep_convection * (num - 1), num)

        fftpack.fftfreq()
    """
    def get_normalize_characteristics(self):
        # 初期条件の影響を消すための慣らし区間
        for i in range(int(5.0/self.timestep_convection)):
            self.main_process()

        for i in range(int(10.0/self.timestep_convection)):
            self.main_process()
            self.get_aero_characteristics(output_residual=False)



def main():
    # 計算条件
    size = 100
    center_x = -0.08
    center_y = 1.0
    naca4 = "6633"
    inflow = 1.0
    alpha = 0.0
    type = 0

    # 一様流の複素速度
    complex_U = get_complex_U(inflow_velocity = inflow, attack_angle_degree = alpha)

    z, size = get_complex_coords(type, size, center_x, center_y, naca4) # 離散的な物体形状(パネル端点)の複素座標の取得(論文中の添字j)

    path = "D:\\Toyota\\Data\\DVM_v_mk2\\"
    fname = "dvm_v_mk1_test_type_" + str(type).zfill(2)

    # print(invQ)
    controller = VortexControl(z, complex_U, path, fname)
    # controller.plot_object()

    controller.test_case()

if __name__ == '__main__':
    main()
