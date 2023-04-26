from qiskit.quantum_info import PauliList

import pickle

from qiskit import QuantumCircuit, BasicAer, transpile, Aer
import matplotlib.pyplot as plt
import numpy as np

from adaptive_measurement import observable_of_ising_model, AdaptiveMeasurement, Observable, \
    OriginalParameterization, CanonicalParameterization
from pauli_measurement import PauliMeasurement
from qae import calc_by_qae


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


def generate_random_pauli_string(n):
    ret = {}
    for i in range(4 ** n):
        plist = []
        for j in range(n):
            b = ((i >> (2 * j)) & 3)
            plist.append(b)
        ret[tuple(plist)] = np.random.random()
    return ret


def generate_random_observable(n):
    pauli_map = {0: 'I', 1: 'X', 2: 'Y', 3: 'Z'}
    observable = np.zeros(shape=(2 ** n, 2 ** n), dtype=complex)
    pauli_string = generate_random_pauli_string(n)
    for pl, c in pauli_string.items():
        pstr = ''
        for i in range(n):
            pstr += pauli_map[pl[n - i - 1]]
        observable += c * PauliList([pstr]).to_matrix()[0]

    return pauli_string, observable


def compare(u: QuantumCircuit, pstr, o: np.ndarray, n, shots):
    # ideal result:
    statevector_backend = BasicAer.get_backend('statevector_simulator')
    ideal_job = statevector_backend.run(transpile(u, backend=statevector_backend))
    ideal_state = ideal_job.result().get_statevector()
    ideal_res = ideal_state.conjugate().T @ o @ ideal_state
    ideal_prob = [(x * x.conjugate()).real for x in ideal_state]
    ideal_variance = sum([ideal_prob[i] * (o[i][i] - ideal_res) ** 2 for i in range(2 ** n)])
    s = 1000
    classical_sample = np.random.choice(a=np.asarray(range(2 ** n)), size=s, replace=True, p=ideal_prob)
    classical_average = sum([o[classical_sample[i]][classical_sample[i]] for i in range(s)]) / s
    classical_variance = sum(
        [(o[classical_sample[i]][classical_sample[i]] - classical_average) ** 2 for i in range(s)]) / s

    print("ideal_res:", ideal_res)
    # print("classical single shot variance:", ideal_variance)
    # print("classical simulate variance:", classical_variance)

    # two method result:

    # QAE
    # ans = calc_by_qae(u, o, 2, 10000)
    # print("qae result:", ans)

    # pauli measure
    pauli_measure = PauliMeasurement(pstr, u)
    print("pauli method:", pauli_measure.measure())

    # adaptive measure
    # pstr = Observable(n, o).get_pauli_string_for_diagonal()
    print(pstr)
    adp_res = []
    for i in range(5):
        am = AdaptiveMeasurement(n, pstr, u, OriginalParameterization)
        adp_res.append(am.run(40))
    with open('pickle/adp_can.pkl', 'wb') as f:
        pickle.dump(adp_res, f)
    pass


if __name__ == '__main__':
    n = 2
    u = Preparation(n)
    pstr, o = generate_random_observable(n)

    compare(u, pstr, o, n, 100)
