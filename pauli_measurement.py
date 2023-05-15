from qiskit import QuantumCircuit

from util import ideal_probability_distribution
import matplotlib.pyplot as plt
import numpy as np


class PauliMeasurement(object):
    def __init__(self, n, pauli_strings, init_state: np.ndarray):
        self.pauli_strings = pauli_strings
        self.init_state = init_state
        self.num_qubits = n

    def measure(self, shots=None):
        average = 0
        variance = 0
        for single_string, coefficient in self.pauli_strings.items():
            tave, tvar = self.measure_single_string(single_string)
            average += tave * coefficient
            variance += tvar * coefficient ** 2
        return average, variance * len(self.pauli_strings)

    def opt_measure(self):
        def ismatch(b, pstr):
            for i in range(self.num_qubits):
                if b[i] in [1, 2] and b[i] != pstr[i]:
                    return False
                if b[i] == 0 and pstr[i] not in [0, 3]:
                    return False
                if b[i] not in [0, 1, 2]:
                    raise Exception("parameter b should be in [3]^n")
            return True

        average = 0
        variance = 0
        cosum = sum([abs(c) for _, c in self.pauli_strings.items()])
        for x in range(3 ** self.num_qubits):
            b = []
            t = x
            for _ in range(self.num_qubits):
                b.append(t % 3)
                t //= 3

            current_cosum = sum([abs(c) if ismatch(b, p) else 0 for p, c in self.pauli_strings.items()])
            current_pstrs = {}
            for single_string, coefficient in self.pauli_strings.items():
                if ismatch(b, single_string):
                    current_pstrs[single_string] = coefficient
            if current_cosum>0:
                tave, tvar = self.measure_multiple_string(current_pstrs, current_cosum)
                average += tave
                variance += tvar
        return average, variance * cosum

    def measure_multiple_string(self, pauli_strings: dict, shots=None):
        keys = list(pauli_strings.keys())
        circ = self.construct_circuit(list(pauli_strings.keys())[0])
        d = ideal_probability_distribution(circ)

        average = 0
        variance = 0
        for s, p in d:
            sample = sum([c * self.calculate_sample(p, s) for p, c in pauli_strings.items()])
            average += sample * p
        for s, p in d:
            sample = sum([c * self.calculate_sample(p, s) for p, c in pauli_strings.items()])
            variance += (sample - average) ** 2 * p
        if shots is not None:
            variance /= shots
        return average, variance

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
        if shots is not None:
            variance /= shots
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
        meas.initialize(self.init_state, range(self.num_qubits))
        for i in range(self.num_qubits):
            p = single_pauli_string[i]
            if p == 1:
                meas.h(i)
            elif p == 2:
                meas.sdg(i)
                meas.h(i)
        return meas
