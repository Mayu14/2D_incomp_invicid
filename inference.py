# -- coding: utf-8 --
from keras.models import model_from_json
from read_training_data import read_csv_type3
from scatter_plot import make_scatter_plot

def inference(source, case_number, env):
    json_name = "learned\\" + case_number + "_mlp_model_.json"
    weight_name = "learned\\" + case_number + "_mlp_weight.h5"

    # あとでこの辺を自由に変更できるようにする
    # fname_lift_train = "NACA4\\s0000_e5000_a040_odd.csv"
    # fname_shape_train = "NACA4\\shape_fourier_5000_odd.csv"
    # X_test, y_test = read_csv_type3(source, fname_lift_train, fname_shape_train, shape_odd=0, read_rate=1)

    source = "G:\\Toyota\\Data\\Incompressible_Invicid\\training_data\\"
    fname_lift_test = "NACA5\\s21001_e25199_a040.csv"
    fname_shape_test = "NACA5\\shape_fourier_all.csv"
    x_test, y_test = read_csv_type3(source, fname_lift_test, fname_shape_test, shape_odd=0, read_rate=1)

    model = model_from_json(open(source + json_name).read())
    model.load_weight(weight_name)

    model.summary()

    model.compile(loss="mean_squared_error",
                  optimizer='Adam')

    score = model.evaluate()
    print('test loss :', score[0])
    print('test accuracy :', score[1])

    y_predict = model.predict(x_test)
    make_scatter_plot(y_test, y_predict, "CL(Exact)", "CL(Predict)", path="G:\\Toyota\\Data\\Incompressible_Invicid\\fig\\", fname="test")


if __name__ == '__main__':
    case_number = 0
    source = ""
    inference(source, case_number, env="lab")
