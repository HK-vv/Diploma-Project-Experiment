import pickle

from qiskit import QuantumCircuit, BasicAer, transpile, Aer
import matplotlib.pyplot as plt
import numpy as np

from adaptive_measurement import observable_of_ising_model, AdaptiveMeasurement, Observable, \
    OriginalParameterization, CanonicalParameterization, generate_random_observable, generate_random_pauli_string, \
    decompose_observable_in_pauli_string
from pauli_measurement import PauliMeasurement
from qae import calc_by_qae

state_code_a = [1, 1, 1, 1, 1, 1, 1, 1]
state_code_b = [1, 0, 1, 0, 1, 0, 1, 0]
state_code_c = [0.052, 0.705, 0.743, 0.973, 0.562, 0.591, 0.001, 0.573]

observable_code_alpha = [0.333, 0.151, 0.346, 0.953, 0.730, 0.382, 0.471, 0.777]
observable_code_beta = [0.045, 0.182, 0.919, 0.128, 0.664, 0.217, 0.511, 0.880]


class Preparation(QuantumCircuit):
    def __init__(self, n):
        super().__init__(n)
        for i in range(n):
            self.cnot(i, (i + 1) % n)
            self.ry(i / n, i)
            self.rz(i / n / 2, (i + 1) % n)
            self.h(i)
        self.draw('mpl')
        plt.show()


def orth_basis(n):
    basis = []
    for i in range(2 ** n):
        t = np.zeros(2 ** n)
        t[i] = 1
        basis.append(t)
    return basis


def nonorth_basis(n):
    basis = []
    for i in range(2 ** n):
        t = np.zeros(2 ** n)
        for j in range(i):
            t[j] = 1
        basis.append(t / np.linalg.norm(t))
    return basis


def encode_state(n, x: list, basis):
    s = np.zeros((2 ** n))
    for i in range(2 ** n):
        s += x[i] * basis[i]
    s /= np.linalg.norm(s)
    return s


def encode_observable(n, x: list, basis):
    o = np.zeros((2 ** n, 2 ** n))
    for i in range(2 ** n):
        o += x[i] * np.outer(basis[i], basis[i].conj())
    return o


def classical_monte_carlo(n, initial: np.ndarray, o: np.ndarray):
    # assert o diagonal
    average = sum([abs(initial[i]) ** 2 * o[i][i] for i in range(2 ** n)])
    variance = sum([abs(initial[i]) ** 2 * (o[i][i] - average) ** 2 for i in range(2 ** n)])
    return average, variance


def compare(initial: np.ndarray, pstr, o: np.ndarray, n, path):
    # u = Preparation(n)
    # ideal result:
    ideal_state = initial
    ideal_res = ideal_state.conjugate().T @ o @ ideal_state
    ideal_prob = [(x * x.conjugate()).real for x in ideal_state]
    ideal_variance = sum([ideal_prob[i] * (o[i][i] - ideal_res) ** 2 for i in range(2 ** n)])

    classical_average, classical_variance = classical_monte_carlo(n, initial, o)
    print("ideal_res:", ideal_res)
    print("classical single shot variance:", ideal_variance)

    # two method result:

    # QAE
    # ans = calc_by_qae(u, o, 2, 10000)
    # print("qae result:", ans)

    # pauli measure
    pauli_measure = PauliMeasurement(n, pstr, initial)
    print("pauli original method:", pauli_measure.measure())
    print("pauli optimized method:", pauli_measure.opt_measure())

    # adaptive measure
    adp_res = []
    for i in range(3):
        am = AdaptiveMeasurement(n, pstr, initial, OriginalParameterization)
        adp_res.append(am.run(30))
    with open(path, 'wb') as f:
        pickle.dump(adp_res, f)
    pass


if __name__ == '__main__':
    n = 3
    initial_state = encode_state(n, state_code_c, orth_basis(n))
    o = encode_observable(n, observable_code_alpha, orth_basis(n))
    pstr = decompose_observable_in_pauli_string(n, o)

    compare(initial_state, pstr, o, n, "pickle/otrh_c_alpha.pkl")
