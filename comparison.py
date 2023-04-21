import pickle

from qiskit import QuantumCircuit, BasicAer, transpile, Aer
import matplotlib.pyplot as plt
import numpy as np

from adaptive_measurement import observable_of_ising_model, AdaptiveMeasurement, Observable, \
    OriginalParameterization, CanonicalParameterization
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


def compare(u: QuantumCircuit, o: np.ndarray, n, shots):
    # ideal result:
    statevector_backend = BasicAer.get_backend('statevector_simulator')
    ideal_job = statevector_backend.run(transpile(u, backend=statevector_backend))
    ideal_state = ideal_job.result().get_statevector()
    ideal_res = ideal_state.conjugate().T @ o @ ideal_state
    ideal_prob = [(x*x.conjugate()).real for x in ideal_state]
    ideal_variance = sum([ideal_prob[i]*(o[i][i]-ideal_res)**2 for i in range(2**n)])
    s=1000
    classical_sample = np.random.choice(a=np.asarray(range(2**n)), size=s, replace=True, p=ideal_prob)
    classical_average = sum([o[classical_sample[i]][classical_sample[i]] for i in range(s)])/s
    classical_variance = sum([(o[classical_sample[i]][classical_sample[i]]-classical_average)**2 for i in range(s)])/s

    print("ideal_res:", ideal_res)
    print("classical single shot variance:", ideal_variance)
    print("classical simulate variance:", classical_variance)

    # two method result:

    # QAE
    ans = calc_by_qae(u, o, 2, 10000)
    print("qae result:", ans)

    # adaptive measure
    pstr = Observable(n, o).get_pauli_string_for_diagonal()
    print(pstr)
    adp_res = []
    for i in range(5):
        am = AdaptiveMeasurement(n, pstr, u, OriginalParameterization)
        adp_res.append(am.run(40))
    with open('pickle/adp_can.pkl', 'wb') as f:
        pickle.dump(adp_res, f)
    pass


if __name__ == '__main__':
    n = 3
    u = Preparation(n)
    o = observable_of_ising_model(n) * 100
    print(np.diag(o))

    print(Aer.backends())
    compare(u, o, n, 100)
