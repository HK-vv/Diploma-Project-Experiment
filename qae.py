import datetime

import numpy as np
from qiskit.circuit import QuantumCircuit, Qubit, Operation
import matplotlib.pyplot as plt
from qiskit.algorithms import EstimationProblem
from qiskit import BasicAer
from qiskit.circuit.bit import Bit
from qiskit.extensions import UnitaryGate
from qiskit.primitives import Sampler
from qiskit.utils import QuantumInstance
from qiskit.algorithms import AmplitudeEstimation
from qiskit import transpile

from adaptive_measurement import Observable, observable_of_ising_model


def convert_to_qae_problem(u: QuantumCircuit, h: list):
    n = u.num_qubits
    a = u.copy()
    a.add_bits([Qubit()])
    R = QuantumCircuit(n + 1)
    # define R
    r = np.zeros([2 ** (n + 1), 2 ** (n + 1)], dtype=complex)
    hb = (1 << n)
    for i in range(2 ** n):
        r[i][i] = np.sqrt(1 - h[i])
        r[i][i|hb] = np.sqrt(h[i])
        r[i|hb][i] = -np.sqrt(h[i])
        r[i|hb][i|hb] = np.sqrt(1 - h[i])
    R.append(UnitaryGate(r), range(n + 1))
    a.compose(R, range(n + 1), inplace=True)
    return EstimationProblem(
        state_preparation=a,
        objective_qubits=n
    )


def calc_by_qae(u: QuantumCircuit, o, num_eval):
    h = np.diagonal(o).copy()
    k = max(0, -min(h))
    h = h + np.ones_like(h, dtype=float) * k
    c = max(h)
    h /= c

    problem = convert_to_qae_problem(u, list(h))
    backend = BasicAer.get_backend('qasm_simulator')
    instance = QuantumInstance(backend, shots=100)

    ae = AmplitudeEstimation(
        num_eval_qubits=num_eval,  # the number of evaluation qubits specifies circuit width and accuracy
        quantum_instance=instance
    )
    # ae.construct_circuit(problem).decompose().draw('mpl', style='iqx')
    # plt.savefig('tmp.jpg')
    ae_result = ae.estimate(problem)

    print(ae_result.estimation, ae_result.mle)
    res = ae_result.mle

    return res * c - k


if __name__ == '__main__':
    n = 4
    prep = QuantumCircuit(n)
    prep.cnot(0, 1)
    prep.rx(0.4, 1)
    prep.cnot(1, 2)
    prep.ry(0.3, 2)
    prep.cnot(2, 3)
    prep.rz(0.26, 3)

    o = observable_of_ising_model(n)*1000
    ans = calc_by_qae(prep, o, 2)
    print(ans)
    print(datetime.datetime.now())

    # a=QuantumCircuit(1)
    # theta_p = 2 * np.arcsin(np.sqrt(0.3))
    # a.ry(theta_p, 0)
    # problem = EstimationProblem(
    #     state_preparation=a,  # A operator
    #     objective_qubits=[0],  # the "good" state Psi1 is identified as measuring |1> in qubit 0
    # )
    # sampler = Sampler()
    # ae = AmplitudeEstimation(
    #     num_eval_qubits=3,  # the number of evaluation qubits specifies circuit width and accuracy
    #     sampler=sampler,
    # )
    # ae_result = ae.estimate(problem)
    # print(ae_result.mle)
    #
    # ae_circuit = ae.construct_circuit(problem)
    # ae_circuit.draw(
    #     "mpl", style="iqx"
    # )
    # plt.show()
