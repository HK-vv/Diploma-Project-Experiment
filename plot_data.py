import pickle
import matplotlib.pyplot as plt

from adaptive_measurement import decompose_observable_in_pauli_string
from comparison import encode_state, state_code_c, observable_code_beta, encode_observable, orth_basis, nonorth_basis, \
    state_code_a, observable_code_alpha, state_code_b
from pauli_measurement import PauliMeasurement


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
    t = len(adp_data)

    s = 1000
    x = [_[4] // s for _ in adp_data]
    y_adp = [_[1] * s for _ in adp_data]
    y_pauli = [pauli_var] * len(adp_data)
    y_opt_pauli = [opt_pauli_var] * len(adp_data)

    plt.figure()
    plt.plot(x, y_adp, color='r', label='adaptive', marker='o')
    plt.plot(x, y_pauli, color='g', label='pauli')
    plt.plot(x, y_opt_pauli, color='b', label='adv pauli')
    plt.legend(fontsize=16)
    plt.xticks(range(5, t + 5, 5), fontsize=24)
    plt.yticks(fontsize=24)
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
    t = len(adp_data)

    x = [_[4] for _ in adp_data]
    y_adp = [_[3] for _ in adp_data]
    y_pauli = [pauli_var / _ for _ in x]
    y_opt_pauli = [opt_pauli_var / _ for _ in x]

    plt.figure()
    plt.yscale('log')
    # plt.xscale('log')
    plt.plot(x, y_adp, color='r', label='adaptive')
    plt.plot(x, y_pauli, color='g', label='pauli')
    plt.plot(x, y_opt_pauli, color='b', label='adv pauli')
    plt.legend(fontsize=16)
    sep = t // 5
    plt.xticks(range(0, (t + sep) * 1000, sep * 1000), fontsize=20)
    plt.yticks(fontsize=20)
    plt.savefig("fig/" + dataname + "__overall_performance.png")
    plt.show()


def plot_data(basis_s, state_s, observable_s):
    basis_map = {'orth': orth_basis, 'nonorth': nonorth_basis}
    state_map = {'a': state_code_a, 'b': state_code_b, 'c': state_code_c}
    observable_map = {'alpha': observable_code_alpha, 'beta': observable_code_beta}

    n = 3
    basis = basis_map[basis_s](n)
    state = state_map[state_s]
    observable = observable_map[observable_s]
    initial_state = encode_state(n, state, basis)  #
    o = encode_observable(n, observable, basis)  #

    pstr = decompose_observable_in_pauli_string(n, o)
    pauli_measure = PauliMeasurement(n, pstr, initial_state)
    pauli_var = pauli_measure.measure()[1]
    opt_pauli_var = pauli_measure.opt_measure()[1]

    fname = basis_s + '_' + state_s + '_' + observable_s
    plot_var_per_measure(fname, pauli_var, opt_pauli_var)  #
    plot_overall_performance(fname, pauli_var, opt_pauli_var)  #


if __name__ == '__main__':
    basis = ['orth', 'nonorth']
    state = ['a', 'b', 'c']
    ob = ['alpha', 'beta']
    for x in basis:
        for y in state:
            for z in ob:
                plot_data(x, y, z)
