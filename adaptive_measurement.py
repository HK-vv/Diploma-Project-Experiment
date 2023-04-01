from qiskit import QuantumCircuit, QuantumRegister, transpile, BasicAer, Aer
from qiskit.quantum_info import Operator
from qiskit.circuit.quantumregister import AncillaRegister
from qiskit.quantum_info import PauliList
import numpy as np
import matplotlib.pyplot as plt
import n_sphere
from sympy.matrices import Matrix, GramSchmidt
from qiskit.extensions import UnitaryGate
from qiskit_aer import AerSimulator
from qiskit.visualization import *

h = 10 ** -3
delta = 0.05
S = 1000

"""
notations:
    n: 2^qbits
    k: observable length written in pauli string
    s: shots
    t: iterations
"""


class ParameterizedUnitary(object):
    def __init__(self, obj=None):
        self.parameters = [0.6] * 8  # all 8 parameters should be in (0,1)
        # self.parameters = list(np.random.rand(8))
        # print(self.parameters)
        self.unitary = None
        if obj is not None:
            self.parameters = obj.parameters

    def generate_gate(self):
        u = self.get_unitary()
        return UnitaryGate(u)

    def coefficient_on_effects(self, b: np.ndarray):
        """
        O(64)
        :param b:
        :return:
        """
        b = b.flatten().reshape((1, 4))
        effects = self.get_effects()
        for i in range(len(effects)):
            effects[i] = effects[i].flatten()
        effects = np.asarray(effects)
        result = np.linalg.solve(effects.T, b.T)
        return result.flatten()

    def get_effects(self):
        """
        O(16)
        :return:
        """
        u = self.get_unitary()
        effects = []
        for k in range(4):
            eff = np.zeros([2, 2], dtype=complex)
            for i in range(2):
                for j in range(2):
                    eff[i][j] = u[k][i].conjugate() * u[k][j]
            effects.append(eff)
        # the effects should have sum one and independent
        # GramSchmidt([Matrix(x) for x in effects])  # assert effects is independent
        return effects

    def get_unitary(self, update=False):
        """
        O(1)
        :return:
        """
        if self.unitary is not None and update is False:
            return self.unitary
        x = self.parameters
        t1 = [1.0]
        for i in range(2):
            t1.append(x[i] * np.pi)
        t1.append(x[2] * np.pi * 2)
        u0 = np.array(n_sphere.convert_rectangular(t1, digits=10))
        # choice of orthobasis still remains doubt
        tu = [Matrix(u0), Matrix([1, 0, 0, 0]), Matrix([0, 1, 0, 0]), Matrix([0, 0, 1, 1])]
        orth = GramSchmidt(tu, orthonormal=True)
        t2 = [1.0]
        for i in range(3, 7):
            t2.append(x[i] * np.pi)
        t2.append(x[7] * np.pi * 2)
        r = n_sphere.convert_rectangular(t2, digits=10)
        u1 = np.zeros([4, 1])
        for k in range(3):
            u1 += (r[2 * k] + r[2 * k + 1] * 1j) * orth[k + 1]
        u1 = np.array(u1)
        tu = [Matrix(u0), Matrix(u1), Matrix([1, 0, 3, 0]), Matrix([0, 2, 0, 4])]
        tu = GramSchmidt(tu, orthonormal=True)
        u = np.empty([4, 4], dtype=complex)
        for k in range(4):
            u[k] = (np.array(tu[k].T))
        self.unitary = u
        return u


class IcPOVM(object):
    def __init__(self, n, init_circuit):
        self.bits = n
        self.pu = [ParameterizedUnitary() for _ in range(n)]
        self.init = init_circuit

    def construct_circuit(self):
        n = self.bits
        circ = QuantumCircuit()
        input_state = QuantumRegister(n)
        ancilla_state = AncillaRegister(n)
        circ.add_register(input_state, ancilla_state)
        circ.compose(self.init, qubits=range(n), front=True, inplace=True)
        for i in range(n):
            circ.append(self.pu[i].generate_gate(), [i, i + n])  # change to PU
        meas = QuantumCircuit(2 * n, 2 * n)
        meas.measure(range(2 * n), range(2 * n))
        circ = circ.compose(meas, range(2 * n))
        circ.draw('mpl')
        plt.show()
        return circ

    def update_parameters(self, paras):
        for i in range(self.bits):
            for j in range(8):
                # print(paras[i][j])
                self.pu[i].parameters[j] += paras[i][j]
                self.pu[i].parameters[j] = min(self.pu[i].parameters[j], 1 - delta)
                self.pu[i].parameters[j] = max(self.pu[i].parameters[j], delta)
                self.pu[i].parameters[j] = self.pu[i].parameters[j].real
            self.pu[i].get_unitary(update=True)

    def simulate_measure(self, s):
        ms = []

        # todo: simulate circuit and store results in ms
        be = Aer.get_backend('qasm_simulator')
        compiled_circuit = transpile(self.construct_circuit(), be)
        job_sim = be.run(compiled_circuit, shots=s)
        counts = job_sim.result().get_counts()
        plot_histogram(counts)
        plt.show()
        for x, y in counts.items():
            m = []
            for i in range(self.bits):
                m.append(int(x[i] + x[i + self.bits], 2))
            ms.append((m, y))

        return ms


class AdaptiveMeasurement(object):
    def __init__(self, n, observable: dict, init_circuit):
        # assert len(observable[0][0]) == n
        self.N = n
        self.observable = observable  # performed by the pauli string of observable
        self.meas = IcPOVM(n, init_circuit)

    def omega(self, m: list):
        """
        O(64nk)
        :param m:
        :return:
        """
        sigma = PauliList(['I', 'X', 'Y', 'Z']).to_matrix()
        o = self.observable
        r = 0
        for k, v in o.items():
            # k is a vec contains [0,3], and v denotes c_k
            p = v
            for i in range(len(k)):
                p *= self.meas.pu[i].coefficient_on_effects(sigma[k[i]])[m[i]]
            r += p
        return r

    def update_povm(self, ms, nu):
        """
        O(2048n^2sk)
        :param ms:
        :param nu:
        :return:
        """
        gradient = self.variance_gradient(ms)
        diff = []
        for x in gradient:
            mx = max([abs(t) for t in x])
            td = []
            for y in x:
                # print("changes:", -nu * y / mx)
                td.append(-nu * y / mx)
            diff.append(td)
        self.meas.update_parameters(diff)

    def variance_gradient(self, ms):  # ms=[[m_1], [m_2]], m_i denotes result of ith shot
        """
        O(2048n^2sk)
        :param ms:
        :return:
        """
        s = sum([c for m, c in ms])
        gradient = []
        original = 0
        for m, c in ms:
            omega = self.omega(m)
            original += c * omega * omega.conjugate()
        original /= s
        for i in range(self.N):
            current_gradient = []
            pu = ParameterizedUnitary(self.meas.pu[i])
            for k in range(8):
                sm = 0
                tpu = pu
                tpu.parameters[k] += h
                tpu.get_unitary(update=True)
                for m, c in ms:
                    # print("new:", tpu.get_effects())
                    # print("old:", self.meas.pu[i].get_effects())
                    for rl in range(4):
                        deco = self.meas.pu[i].coefficient_on_effects(tpu.get_effects()[rl])
                        # print("deco:", deco)
                        d = deco[m[i]]
                        # print("m[i]:", m[i])

                        t = self.meas.pu[i], m[i]
                        m[i] = rl
                        self.meas.pu[i] = tpu
                        omega = self.omega(m)
                        self.meas.pu[i] = t[0]
                        m[i] = t[1]
                        sm += c * d * omega * omega.conjugate()
                        # print("d:", d)
                sm /= s
                # print("sm:", sm)
                current_gradient.append((sm - original) / h)
            gradient.append(current_gradient)
        print("original:", original)
        print("gradient:", gradient)
        return gradient

    def run(self, t):
        """

        :param t:
        :return:
        """
        s = S
        nu = 0.05
        o, v = None, None
        for i in range(t):
            print(f"{i + 1}th iteration started...")
            ms = self.meas.simulate_measure(s)
            # print('im here')
            ot = sum([self.omega(m) * c /s for m, c in ms])
            vt = 0
            for m, c in ms:
                vt += (self.omega(m) - ot) ** 2 * c
            vt /= s
            if (o, v) == (None, None):
                o, v = ot, vt
            else:
                o = (ot * v + o * vt) / (v + vt)
                v = v * vt / (v + vt)
            # print('im here!')
            self.update_povm(ms, nu)
            s += S
            nu /= 1.1
            print("local result:", (ot, vt))
            print("global result:", (o, v))
        return o, v


class Observable(object):
    def __init__(self, n, M):
        self.d = n
        self.matrix = np.array(M)

    def get_pauli_string_for_diagonal(self):
        A = []
        for i in range(2 ** self.d):
            pstr = ''
            for j in range(self.d):
                b = ((i >> j) & 1)
                pstr += ('I' if b == 0 else 'Z')
            pdiag = PauliList([pstr]).to_matrix()[0].diagonal()
            A.append(pdiag)
        A = np.array(A).transpose()
        # print((A, self.matrix.diagonal()))
        res = np.linalg.solve(A, self.matrix.diagonal())
        ans = {}
        for i in range(2 ** self.d):
            plist = []
            for j in range(self.d):
                b = ((i >> j) & 1)
                plist.append(0 if b == 0 else 3)
            if abs(res[i]) > 10 ** -9:
                ans[tuple(plist)] = res[i]
        return ans


def observable_of_ising_model(n):
    d = []
    for i in range(2 ** n):
        bits = [(i >> j) & 1 for j in range(n)]
        d.append(sum([-2 * (bits[j] ^ bits[j + 1]) + 1 for j in range(n - 1)]))
    observable = np.diag(np.asarray(d, dtype=float))
    return observable


if __name__ == '__main__':
    # o = {(1, 2, 3): 2j, (0, 3, 2): 3, (0, 0, 0): 5 - 10j}
    n = 4
    o = Observable(n, observable_of_ising_model(n))
    # o = Observable(n, np.identity(2**n) * 1000)
    print(o.get_pauli_string_for_diagonal())
    prep = QuantumCircuit(n)
    prep.cnot(0, 1)
    prep.rx(0.4, 1)
    prep.cnot(1, 2)
    prep.ry(0.3, 2)
    prep.cnot(2, 3)
    prep.rz(0.26, 3)

    prep.draw('mpl')
    plt.show()

    am = AdaptiveMeasurement(n, o.get_pauli_string_for_diagonal(), prep)
    print(am.run(30))
