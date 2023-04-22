from qiskit import QuantumCircuit, BasicAer, transpile


def ideal_probability_distribution(circ: QuantumCircuit):
    be = BasicAer.get_backend('statevector_simulator')
    compiled_circuit = transpile(circ, be)
    job_sim = be.run(compiled_circuit)
    state = job_sim.result().get_statevector()
    probs = [(x * x.conjugate()).real for x in state]

    d = []
    n = circ.num_qubits
    for i in range(2 ** n):
        b = bin(i)[2:]
        while len(b) < n * 2:
            b = '0' + b
        b = b[::-1]  # [::-1] for reverse
        d.append((b, probs[i]))
    return d
