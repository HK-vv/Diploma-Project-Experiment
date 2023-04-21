from decimal import Decimal

from qiskit import QuantumCircuit, QuantumRegister, transpile, BasicAer, Aer
from qiskit.quantum_info import Operator
from qiskit.circuit.quantumregister import AncillaRegister
from qiskit.quantum_info import PauliList, Operator
import numpy as np
import matplotlib.pyplot as plt
import n_sphere
from sympy.matrices import Matrix, GramSchmidt
from qiskit.extensions import UnitaryGate
from qiskit.visualization import *

h = 10 ** -3
delta = 0.05
S = 10000

"""
notations:
    n: 2^qbits
    k: observable length written in pauli string
    s: shots
    t: iterations
"""


class ParameterizedUnitary(object):
    num_parameter = 0

    def __init__(self):
        self.parameters = None
        self.unitary = None

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
        raise Exception("abstract method should never be called")

    def update_diff(self, diffs):
        raise Exception("abstract method should never be called")


class OriginalParameterization(ParameterizedUnitary):
    num_parameter = 8

    def __init__(self, obj=None):
        super(OriginalParameterization, self).__init__()
        # self.parameters = [0.6] * type(self).num_parameter  # all parameters should be in (0,1)
        self.parameters = list(np.random.rand(type(self).num_parameter))
        if obj is not None:
            if isinstance(obj, type(self)):
                self.parameters = obj.parameters.copy()
            else:
                raise Exception("Inconsistent type of parameterization!")

    def get_unitary(self, update=False):
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
        return self.unitary

    def update_diff(self, diffs):
        for i in range(type(self).num_parameter):
            upd = self.parameters[i] + diffs[i].real
            upd = min(upd, 1 - delta)
            upd = max(upd, delta)
            self.parameters[i] = upd


class CanonicalParameterization(ParameterizedUnitary):
    num_parameter = 15

    def __init__(self, obj=None):
        super(CanonicalParameterization, self).__init__()
        # self.parameters = [0.3] * type(self).num_parameter  # all parameters should be in (0,1)
        self.parameters = list(np.random.rand(type(self).num_parameter))
        if obj is not None:
            if isinstance(obj, type(self)):
                self.parameters = obj.parameters.copy()
            else:
                raise Exception("Inconsistent type of parameterization!")

    def get_unitary(self, update=False):
        if self.unitary is not None and update is False:
            return self.unitary
        vcirc = QuantumCircuit(2)
        c = 4 * np.pi
        for i in range(2):
            bias = 0
            vcirc.rx(c * self.parameters[bias + i * 3 + 0], i)
            vcirc.ry(c * self.parameters[bias + i * 3 + 1], i)
            vcirc.rx(c * self.parameters[bias + i * 3 + 2], i)
        vcirc.rxx(c * self.parameters[6], 0, 1)
        vcirc.ryy(c * self.parameters[7], 0, 1)
        vcirc.rzz(c * self.parameters[8], 0, 1)
        for i in range(2):
            bias = 9
            vcirc.rx(c * self.parameters[bias + i * 3 + 0], i)
            vcirc.ry(c * self.parameters[bias + i * 3 + 1], i)
            vcirc.rx(c * self.parameters[bias + i * 3 + 2], i)
        self.unitary = Operator(vcirc).data
        # vcirc.draw('mpl')
        # plt.show()
        return self.unitary

    def update_diff(self, diffs):
        for i in range(type(self).num_parameter):
            upd = self.parameters[i] + diffs[i].real
            upd = upd - np.floor(upd)
            self.parameters[i] = upd


class IcPOVM(object):
    def __init__(self, n, init_circuit, parameterization):
        self.bits = n
        self.parameterization = parameterization
        self.pu = [self.parameterization() for _ in range(n)]
        self.init = init_circuit

    def construct_circuit(self):
        n = self.bits
        circ = QuantumCircuit()
        input_state = QuantumRegister(n)
        ancilla_state = AncillaRegister(n)
        circ.add_register(input_state, ancilla_state)
        circ.compose(self.init, qubits=range(n), front=True, inplace=True)
        for i in range(n):
            circ.append(self.pu[i].generate_gate(), [i, i + n])  # apply PU
        # circ.draw('mpl')
        # plt.show()

        meas = QuantumCircuit(2 * n, 2 * n)
        # meas.measure(range(2 * n), range(2 * n))
        circ = circ.compose(meas, range(2 * n))
        # meas.draw('mpl')
        # plt.show()
        return circ

    def update_parameters(self, diff):
        for i in range(self.bits):
            self.pu[i].update_diff(diff[i])
            self.pu[i].get_unitary(update=True)

    def simulate_measure(self, s):
        ms = []

        # simulate circuit and store results in ms
        # earlier version (using QASM simulator)
        # be = Aer.get_backend('aer_simulator_statevector')
        # compiled_circuit = transpile(self.construct_circuit(), be)
        # job_sim = be.run(compiled_circuit, shots=s)
        # counts = job_sim.result().get_counts(compiled_circuit)
        # plot_histogram(counts)
        # plt.show()
        # for x, y in counts.items():
        #     m = []
        #     for i in range(self.bits):
        #         m.append(int(x[i + self.bits] + x[i], 2))
        #     ms.append((m, y))

        be = BasicAer.get_backend('statevector_simulator')
        circ = self.construct_circuit()
        compiled_circuit = transpile(circ, be)
        job_sim = be.run(compiled_circuit)
        state = job_sim.result().get_statevector()
        probs = [(x * x.conjugate()).real for x in state]

        d = []
        for i in range(2 ** (2 * self.bits)):
            b = bin(i)[2:]
            while len(b) < self.bits * 2:
                b = '0' + b
            b = b[::-1]  # [::-1] for reverse
            m = []
            for j in range(self.bits):
                m.append(int(b[j + self.bits] + b[j], 2))
            d.append((m, probs[i] * s))

        # make sampling
        # sample = np.random.choice(a=np.asarray(range(len(d))), size=s, replace=True, p=[c for m, c in d])
        # for m, c in d:
        #     ms.append((m, sum([1 if d[x][0] == m else 0 for x in list(sample)])))
        return d


class AdaptiveMeasurement(object):
    def __init__(self, n, observable: dict, init_circuit, parameterization):
        self.N = n
        self.observable = observable  # performed by the pauli string of observable
        self.parameterization = parameterization
        self.meas = IcPOVM(n, init_circuit, self.parameterization)

    def omega(self, m: list):
        """
        O(64nk)
        :param m:
        :return:
        """
        sigma = PauliList(['I', 'X', 'Y', 'Z']).to_matrix()
        o = self.observable
        r = Decimal(0)
        for k, v in o.items():
            # k is a vec contains [0,3], and v denotes c_k
            p = Decimal(v.real)
            for i in range(len(k)):
                p *= Decimal(self.meas.pu[i].coefficient_on_effects(sigma[k[i]])[m[i]].real)
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

    def variance_gradient(self, ms):  # ms=[([m_1], c_1), ([m_2], c_2)], m_i denotes result of ith shot
        """
        O(2048n^2sk)
        :param ms:
        :return:
        """
        if self.parameterization == OriginalParameterization:
            return self.gradient_original(ms)
        elif self.parameterization == CanonicalParameterization:
            return self.gradient_canonical(ms)
        else:
            raise Exception("wrong type of parameterization.")

    def variance_of_biased_parameter(self, qubit, param, bias, ms):
        sm = Decimal(0)
        tpu = self.parameterization(self.meas.pu[qubit])
        tpu.parameters[param] += bias
        tpu.parameters[param] -= np.floor(tpu.parameters[param])
        tpu.get_unitary(update=True)
        for m, c in ms:
            for rl in range(4):
                deco = self.meas.pu[qubit].coefficient_on_effects(tpu.get_effects()[rl])
                d = deco[m[qubit]]
                t = self.meas.pu[qubit], m[qubit]
                m[qubit] = rl
                self.meas.pu[qubit] = tpu
                omega = self.omega(m)
                self.meas.pu[qubit] = t[0]
                m[qubit] = t[1]
                sm += Decimal((c * d).real) * omega ** 2
        sm /= Decimal(sum([_ for __, _ in ms]))
        return float(sm)

    def gradient_original(self, ms):
        gradient = []
        for i in range(self.N):
            current_gradient = []
            for k in range(self.parameterization.num_parameter):
                vp = self.variance_of_biased_parameter(i, k, h, ms)
                vm = self.variance_of_biased_parameter(i, k, -h, ms)
                current_gradient.append((vp - vm) / (2 * h))
            gradient.append(current_gradient)
        print("gradient:", gradient)
        return gradient

    def gradient_canonical(self, ms):
        gradient = []
        for i in range(self.N):
            current_gradient = []
            for k in range(self.parameterization.num_parameter):
                vp = self.variance_of_biased_parameter(i, k, h, ms)
                vm = self.variance_of_biased_parameter(i, k, -h, ms)
                current_gradient.append((vp - vm) / (2 * h))
                # vp = self.variance_of_biased_parameter(i, k, 1 / 8, ms)
                # vm = self.variance_of_biased_parameter(i, k, -1 / 8, ms)
                # current_gradient.append(2 * np.pi * (vp - vm))
            gradient.append(current_gradient)
        print("gradient:", gradient)
        return gradient

    def run(self, t):
        """

        :param t:
        :return:
        """
        output = []
        s = S
        nu = 0.05
        o, v = None, None
        total_shots = 0
        print(f"using {self.parameterization}")
        for i in range(t):
            print(f"{i + 1}th iteration started...")
            ms = self.meas.simulate_measure(s)
            total_shots += s
            # print('im here')
            ot = sum([self.omega(m) * Decimal(c.real / s) for m, c in ms])
            # for m,c in ms:
            # print((self.omega(m),c))
            vt = Decimal(0)
            for m, c in ms:
                vt += (self.omega(m) - ot) ** 2 * Decimal(c)
            vt /= (s - 1) * s
            if (o, v) == (None, None):
                o, v = ot, vt
            else:
                o = (ot * v + o * vt) / (v + vt)
                v = v * vt / (v + vt)
            # print('im here!')
            self.update_povm(ms, nu)
            # s += S
            nu /= 1.05

            output.append([ot, vt, o, v, total_shots])
            print("local result:", (ot, vt))
            print("global result:", (o, v))
            print("standard deviation:", v.sqrt())
            print(f"used {total_shots} shots till now.")
        return output


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
        A = np.array(A).T
        # print((A, self.matrix.diagonal()))
        res = np.linalg.solve(A, self.matrix.diagonal())
        ans = {}
        for i in range(2 ** self.d):
            plist = []
            for j in range(self.d):
                b = ((i >> (self.d - j - 1)) & 1)
                plist.append(0 if b == 0 else 3)
            if abs(res[i]) > 10 ** -9:
                ans[tuple(plist)] = res[i]
        return ans


def observable_of_ising_model(n):
    d = []
    for i in range(2 ** n):
        bits = [(i >> j) & 1 for j in range(n)]
        d.append(sum([-2 * (bits[j] ^ bits[j + 1]) + 1 for j in range(n - 1)]))
    observable = np.diag(np.asarray(d, dtype=np.double))
    observable = np.diag(np.random.random(size=2 ** n))
    # observable = np.diag([2, 1, 0.6, 0.1, 0.8, 9, 3.4, 1.7])
    # print(observable)
    return observable


if __name__ == '__main__':
    pass
