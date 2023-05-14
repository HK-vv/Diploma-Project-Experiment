import pickle
import matplotlib.pyplot as plt

from adaptive_measurement import decompose_observable_in_pauli_string
from comparison import encode_state, state_code_c, observable_code_beta, encode_observable, orth_basis, nonorth_basis, \
    state_code_a, observable_code_alpha, state_code_b
from pauli_measurement import PauliMeasurement


def plot_var_per_measure(dataname, pauli_var, opt_pauli_var):
    with open("pickle/" + dataname + ".pkl", 'rb') as f:
        adp = pickle.load(f)
    test_times = len(adp)
    adp_data = []
    for i in range(len(adp[0])):
        adp_point = []
        for j in range(len(adp[0][0])):
            adp_point.append(sum([adp[_][i][j] / test_times for _ in range(test_times)]))
        adp_data.append(adp_point)

    s = 1000
    x = [_[4] for _ in adp_data]
    y_adp = [_[1] * s for _ in adp_data]
    y_pauli = [pauli_var] * len(adp_data)
    y_opt_pauli = [opt_pauli_var] * len(adp_data)

    plt.figure()
    plt.plot(x, y_adp, color='r', label='adaptive')
    plt.plot(x, y_pauli, color='g', label='pauli')
    plt.plot(x, y_opt_pauli, color='b', label='adv pauli')
    plt.legend()
    plt.savefig("fig/" + dataname + "__variance_per_measure.png")
    plt.show()


def plot_overall_performance(dataname, pauli_var, opt_pauli_var):
    with open("pickle/" + dataname + ".pkl", 'rb') as f:
        adp = pickle.load(f)
    test_times = len(adp)
    adp_data = []
    for i in range(len(adp[0])):
        adp_point = []
        for j in range(len(adp[0][0])):
            adp_point.append(sum([adp[_][i][j] / test_times for _ in range(test_times)]))
        adp_data.append(adp_point)

    x = [_[4] for _ in adp_data]
    y_adp = [_[3] for _ in adp_data]
    y_pauli = [pauli_var / _ for _ in x]
    y_opt_pauli = [opt_pauli_var / _ for _ in x]

    plt.figure()
    plt.plot(x, y_adp, color='r', label='adaptive')
    plt.plot(x, y_pauli, color='g', label='pauli')
    plt.plot(x, y_opt_pauli, color='b', label='adv pauli')
    plt.legend()
    plt.savefig("fig/" + dataname + "__overall_performance.png")
    plt.show()


def obsolete():
    with open('adp_ori.pkl', 'rb') as f:
        ori_raw = pickle.load(f)
    with open('adp_can.pkl', 'rb') as f:
        can_raw = pickle.load(f)

    ori_data = []
    can_data = []
    test_time = len(ori_raw)
    for i in range(len(ori_raw[0])):
        ori_point = []
        can_point = []
        for j in range(len(ori_raw[0][0])):
            ori_point.append(sum([ori_raw[_][i][j] / test_time for _ in range(test_time)]))
            can_point.append(sum([can_raw[_][i][j] / test_time for _ in range(test_time)]))
        ori_data.append(ori_point)
        can_data.append(can_point)

    # plotting!

    x = [_[4] for _ in ori_data]

    y1 = [_[3] for _ in ori_data]

    y2 = [_[3] for _ in can_data]
    x = x[1:]
    y1 = y1[1:]
    y2 = y2[1:]
    print(y1)
    print(y2)

    plt.figure()
    plt.plot(x, y1, color='r', label='original')
    plt.plot(x, y2, color='g', label='canonical')
    plt.legend()
    plt.savefig('variance.png')
    plt.show()

    # fig, ax = plt.subplots()
    # ax.plot(x, y1, lable='original')
    # ax.plot(x, y2, lable='canonical')
    # ax.set_xlabel('shots')
    # ax.set_ylabel('variance')
    # ax.legend()

    # plt.show()


if __name__ == '__main__':
    n = 3
    initial_state = encode_state(n, state_code_b, nonorth_basis(n))
    o = encode_observable(n, observable_code_alpha, nonorth_basis(n))
    pstr = decompose_observable_in_pauli_string(n, o)
    pauli_measure = PauliMeasurement(n, pstr, initial_state)
    pauli_var = pauli_measure.measure()[1]
    opt_pauli_var = pauli_measure.opt_measure()[1]

    plot_var_per_measure("nonorth_b_alpha", pauli_var, opt_pauli_var)
    plot_overall_performance("nonorth_b_alpha", pauli_var, opt_pauli_var)
