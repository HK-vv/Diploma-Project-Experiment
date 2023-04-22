from qiskit import QuantumCircuit

from util import ideal_probability_distribution
import matplotlib.pyplot as plt


class PauliMeasurement(object):
    def __init__(self, pauli_strings, init_circuit: QuantumCircuit):
        self.pauli_strings = pauli_strings
        self.init_circuit = init_circuit
        self.num_qubits = init_circuit.num_qubits

    def measure(self, shots=None):
        average = 0
        variance = 0
        for single_string, coefficient in self.pauli_strings.items():
            tave, tvar = self.measure_single_string(single_string)
            average += tave * coefficient
            variance += tvar * coefficient ** 2
        return average, variance * len(self.pauli_strings)

    def measure_single_string(self, single_pauli_string, shots=None):
        circ = self.construct_circuit(single_pauli_string)
        d = ideal_probability_distribution(circ)

        average = 0
        variance = 0
        for s, p in d:
            sample = self.calculate_sample(single_pauli_string, s)
            average += sample * p
        for s, p in d:
            sample = self.calculate_sample(single_pauli_string, s)
            variance += (sample - average) ** 2 * p
        return average, variance

    def calculate_sample(self, single_pauli_string, str):
        res = 1
        for i in range(self.num_qubits):
            t = (-1 if str[i] == '1' else 1)
            t = (1 if single_pauli_string[i] == 0 else t)
            res *= t
        return res

    def construct_circuit(self, single_pauli_string):
        meas = QuantumCircuit(self.num_qubits)
        for i in range(self.num_qubits):
            p = single_pauli_string[i]
            if p == 1:
                meas.h(i)
            elif p == 2:
                meas.sdg(i)
                meas.h(i)
        meas.compose(self.init_circuit, front=True, inplace=True)
        return meas
