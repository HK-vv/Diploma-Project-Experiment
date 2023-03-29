from qiskit import QuantumCircuit, BasicAer, transpile
import matplotlib.pyplot as plt
import numpy as np

from adaptive_measurement_for_test import observable_of_ising_model, AdaptiveMeasurement, Observable


class Preparation(QuantumCircuit):
    def __init__(self, n):
        super().__init__(n)
        for i in range(n):
            self.cnot(i, (i + 1) % n)
            # self.ry(i / n, i)
            # self.rz(i / n / 2, (i + 1) % n)
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
    # adaptive measure
    pstr = Observable(n, o).get_pauli_string_for_diagonal()
    am = AdaptiveMeasurement(n, pstr, u)
    print(am.run(20))

    pass


if __name__ == '__main__':
    n = 4
    u = Preparation(n)
    o = observable_of_ising_model(n)

    prep = QuantumCircuit(n)
    # prep.cnot(0, 1)
    # prep.rx(0.4, 1)
    # prep.cnot(1, 2)
    # prep.ry(0.3, 2)
    # prep.cnot(2, 3)
    # prep.rz(0.26, 3)

    for i in range(n):
        prep.h(i)
    for i in range(n):
        prep.cnot(i,(i+1)%n)
    prep.draw('mpl')
    plt.show()

    compare(prep, o, 100)
