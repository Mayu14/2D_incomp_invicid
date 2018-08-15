# coding: utf-8
import numpy as np
import dvm_beta as dvm

def validation_joukowski(path):
	size = 500
	center_x = -0.08
	center_y = 0.08
	inflow = 1.0
	density = 1.0
	type = 1
	z, size = dvm.get_complex_coords(type, size, center_x, center_y)

	number = 181
	save_data = np.zeros((number, 7))
	for angle in range(number):
		attack_angle_deg = angle - 90

		complex_U = dvm.get_complex_U(inflow, attack_angle_deg)

		gamma, circulation = dvm.get_circulation(z, complex_U)
		lift = dvm.get_lift(density, circulation, complex_U)
		lift_coef = dvm.get_lift_coefficient(z, circulation, complex_U)
		exact_lift, exact_lift_coef = dvm.validation(type, center_x, center_y, complex_U)
		if ((angle== 12 + 90) and (size == 500)) :
			fname = "joukowski_valid_" + str(size).zfill(4) + "_" + str(center_x) + "_" + str(center_y) +"attack_angle_12_sample"
			velocity = dvm.get_velocity(z, gamma, complex_U, path, fname)

		save_data[angle, 0] = attack_angle_deg
		save_data[angle, 1] = lift
		save_data[angle, 2] = exact_lift
		save_data[angle, 3] = lift_coef
		save_data[angle, 4] = exact_lift_coef
		save_data[angle, 5] = np.sqrt((lift - exact_lift)**2)
		save_data[angle, 6] = np.sqrt((lift_coef - exact_lift_coef)**2)
	
	fname = path + "joukowski_valid_" + str(size).zfill(4) + "_" + str(center_x) + "_" + str(center_y) + ".csv"
	np.savetxt(fname, save_data, delimiter = ",")
	
if __name__ == '__main__':
    path = "D:\\Toyota\\Data\\DVM_validation\\"
    validation_joukowski(path)