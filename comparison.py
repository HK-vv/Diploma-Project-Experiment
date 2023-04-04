from qiskit import QuantumCircuit, BasicAer, transpile, Aer
import matplotlib.pyplot as plt
import numpy as np

from adaptive_measurement import observable_of_ising_model, AdaptiveMeasurement, Observable, OriginalParameterization, \
    CanonicalParameterization
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


def compare(u: QuantumCircuit, o: np.ndarray, shots):
    # ideal result:
    statevector_backend = BasicAer.get_backend('statevector_simulator')
    ideal_job = statevector_backend.run(transpile(u, backend=statevector_backend))
    ideal_state = ideal_job.result().get_statevector()
    ideal_res = ideal_state.conjugate() @ o @ ideal_state
    print(ideal_res)

    # two method result:

    # QAE
    ans = calc_by_qae(u, o, 2, 10000)
    print("qae result:", ans)

    # adaptive measure
    pstr = Observable(n, o).get_pauli_string_for_diagonal()
    print(pstr)
    am = AdaptiveMeasurement(n, pstr, u, CanonicalParameterization)
    print(am.run(20))

    pass


if __name__ == '__main__':
    n = 3
    u = Preparation(n)
    o = observable_of_ising_model(n) * 10

    print(Aer.backends())
    compare(u, o, 100)
