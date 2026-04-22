import numpy as np
from qibo import gates
from qibo.backends import NumpyBackend
from qibo.models.circuit import Circuit
from qibo.quantum_info import trace_distance as qibo_trace_distance
from scipy.linalg import sqrtm, svd
from qibo.gates import ReadoutErrorChannel


backend = NumpyBackend()
def analyze_circuit(circuit: Circuit, model, noise_model=None):
    """Analyze the gates added by the given model to a circuit.
    If noise_model is given, compare the gates added by the model and the noise_model.
    
    Returns:    
    original_circuit: dict
        Dictionary containing the number of RX, RZ and CZ gates, the number of moments and the number of gates in the original circuit.
    rl: dict
        Dictionary containing the number of RX, RZ and CZ gates, the number of damping and depolarizing channels and the parameters of the channels added by the model.
    ground_truth: dict (returned if noise_model is not None)
        Dictionary containing the number of RX, RZ and CZ gates, the number of damping and depolarizing channels and the parameters of the channels added by the noise_model.
    """

    initial_rx = 0
    initial_rz = 0
    initial_cz = 0

    for gate in circuit.queue:
        if isinstance(gate, gates.RX):
            initial_rx += 1
        elif isinstance(gate, gates.RZ):
            initial_rz += 1
        elif isinstance(gate, gates.CZ):
            initial_cz += 1
    
    original_circuit = {}
    original_circuit['RX'] = initial_rx
    original_circuit['RZ'] = initial_rz
    original_circuit['CZ'] = initial_cz
    original_circuit['moments'] = len(circuit.queue.moments)
    original_circuit['n_gates'] = len(circuit.queue)

    damping = 0
    damping_params = []
    dep = 0
    dep_params = []
    rx = 0
    rz = 0
    rl_circuit = model.apply(circuit)
    for gate in rl_circuit.queue:
        if isinstance(gate, gates.ResetChannel):
            damping += 1
            damping_params.append(gate.parameters)
        elif isinstance(gate, gates.DepolarizingChannel):
            dep += 1
            dep_params.append(gate.parameters)
        elif isinstance(gate, gates.RX):
            rx += 1
        elif isinstance(gate, gates.RZ):
            rz += 1
    rl = {}
    rl['coh_X'] = rx-initial_rx
    rl['coh_Z'] = rz-initial_rz
    rl['dep'] = dep
    rl['damp'] = damping
    rl['damp_params'] = damping_params
    rl['dep_params'] = dep_params

    if noise_model is None:
        return original_circuit, rl
    
    damping = 0
    damping_params = []
    dep = 0
    dep_params = []
    rx = 0
    rz = 0

    gt_circuit = noise_model.apply(circuit)
    for gate in gt_circuit.queue:
        if isinstance(gate, gates.ResetChannel):
            damping += 1
            damping_params.append(gate.parameters)
        elif isinstance(gate, gates.DepolarizingChannel):
            dep += 1
            dep_params.append(gate.parameters)
        elif isinstance(gate, gates.RX):
            rx += 1
        elif isinstance(gate, gates.RZ):
            rz += 1

    ground_truth = {}
    ground_truth['coh_X'] = rx-initial_rx
    ground_truth['coh_Z'] = rz-initial_rz
    ground_truth['dep'] = dep
    ground_truth['damp'] = damping
    ground_truth['damp_params'] = damping_params
    ground_truth['dep_params'] = dep_params

    return original_circuit, rl, ground_truth


def mms(dim):
    """Return the density matrix of the maximally mixed state."""
    return np.eye(dim) / dim

def mse(a,b):
    """Compute the Mean Squared Error between two matrices."""
    return np.mean(np.abs((a - b)**2))

def mae(a,b):
    """Compute the Mean Absolute Error between two matrices."""
    return np.mean(np.abs(a - b))




def _to_prob_vec(rho, tol: float = 1e-12):
    if rho.ndim == 1:
        p = np.real_if_close(rho, tol)
    else:                      # matrice → diagonale reale
        p = np.real_if_close(np.diag(rho), tol)

    p = np.clip(p, 0.0, None)  # taglia eventuali residui negativi
    s = p.sum()
    if s > 0 and not np.isclose(s, 1.0):
        p /= s
    return p.astype(float, copy=False)


def trace_distance(rho1, rho2):
    if rho1.ndim == 2 and rho2.ndim == 2:
       
        diff = rho1 - rho2
        sing = svd(diff, compute_uv=False)
        return 0.5 * np.sum(np.abs(sing))
        

    
    p, q = _to_prob_vec(rho1), _to_prob_vec(rho2)
    return 0.5 * np.sum(np.abs(p - q))


def compute_fidelity(rho1, rho2):
    if rho1.ndim == 2 and rho2.ndim == 2:
        return np.real(np.trace(sqrtm(rho1 @ rho2)) ** 2)

    p, q = _to_prob_vec(rho1), _to_prob_vec(rho2)
    prod = np.clip(p * q, 0.0, None)       # evita sqrt di numeri <0
    return (np.sum(np.sqrt(prod))) ** 2




def test_avg_fidelity(rho1,rho2):
    fidelity = []
    for i in range(len(rho1)):
        print(i, "fidelity: ", compute_fidelity(rho1[i],rho2[i]))
        fidelity.append(compute_fidelity(rho1[i],rho2[i]))
    avg_fidelity = np.array(fidelity).mean()
    return avg_fidelity

def u3_dec(gate):
    """Decompose a U3 gate into RZ and RX gates."""
    # t, p, l = gate.parameters
    params = gate.parameters
    t = params[0]
    p = params[1]
    l = params[2]
    decomposition = []
    if l != 0.0:
        decomposition.append(gates.RZ(gate.qubits[0], l))
    decomposition.append(gates.RX(gate.qubits[0], np.pi/2, 0))
    if t != -np.pi:
        decomposition.append(gates.RZ(gate.qubits[0], t + np.pi))
    decomposition.append(gates.RX(gate.qubits[0], np.pi/2, 0))
    if p != -np.pi:
        decomposition.append(gates.RZ(gate.qubits[0], p + np.pi))
    return decomposition

def grover():
    """Creates a Grover circuit with 3 qubits.
    The circuit searches for the 11 state, the last qubit is ancillary"""
    circuit = Circuit(3, density_matrix=True)
    circuit.add(gates.RZ(0, np.pi/2))
    circuit.add(gates.RX(0, np.pi/2))
    circuit.add(gates.RZ(0, np.pi/2))
    circuit.add(gates.RZ(1, np.pi/2))
    circuit.add(gates.RX(1, np.pi/2))
    circuit.add(gates.RZ(1, np.pi/2))
    circuit.add(gates.RX(2, np.pi))
    circuit.add(gates.RZ(2, np.pi/2))
    circuit.add(gates.RX(2, np.pi/2))
    circuit.add(gates.RZ(2, np.pi/2))
    #Toffoli
    circuit.add(gates.CZ(1, 2))
    circuit.add(gates.RX(2, -np.pi / 4))
    circuit.add(gates.CZ(0, 2))
    circuit.add(gates.RX(2, np.pi / 4))
    circuit.add(gates.CZ(1, 2))
    circuit.add(gates.RX(2, -np.pi / 4))
    circuit.add(gates.CZ(0, 2))
    circuit.add(gates.RX(2, np.pi / 4))
    circuit.add(gates.RZ(1, np.pi / 4))
    circuit.add(gates.RZ(1, np.pi/2))
    circuit.add(gates.RX(1, np.pi/2))
    circuit.add(gates.RZ(1, np.pi/2))
    circuit.add(gates.CZ(0, 1))
    circuit.add(gates.RZ(0, np.pi / 4))
    circuit.add(gates.RX(1, -np.pi / 4))
    circuit.add(gates.CZ(0, 1))
    ###
    circuit.add(gates.RZ(0, np.pi/2))
    circuit.add(gates.RX(0, np.pi/2))
    circuit.add(gates.RZ(0, np.pi/2))
    circuit.add(gates.RX(0, np.pi))
    circuit.add(gates.RX(1, np.pi))
    circuit.add(gates.CZ(0, 1))
    circuit.add(gates.RX(0, np.pi))
    circuit.add(gates.RX(1, np.pi))
    circuit.add(gates.RZ(0, np.pi/2))
    circuit.add(gates.RX(0, np.pi/2))
    circuit.add(gates.RZ(0, np.pi/2))
    circuit.add(gates.RZ(1, np.pi/2))
    circuit.add(gates.RX(1, np.pi/2))
    circuit.add(gates.RZ(1, np.pi/2))
    return circuit

def qft():
    circuit = Circuit(3, density_matrix=True)
    circuit.add(gates.RZ(0, np.pi/2))
    circuit.add(gates.RZ(1, np.pi/2))
    circuit.add(gates.RZ(2, np.pi/2))

    circuit.add(gates.RX(0, np.pi/2))
    circuit.add(gates.RX(1, np.pi/2))
    circuit.add(gates.RX(2, np.pi/2))

    circuit.add(gates.RZ(0, np.pi/2))
    circuit.add(gates.RZ(1, np.pi/2))
    circuit.add(gates.RZ(2, 3*np.pi/2))

    circuit.add(gates.CZ(1,2))
    circuit.add(gates.RX(1, -np.pi/4))

    circuit.add(gates.CZ(1,2))
    circuit.add(gates.RX(1, np.pi/4))
    circuit.add(gates.RZ(2, np.pi/8))
    circuit.add(gates.RZ(1, np.pi/4))

    circuit.add(gates.CZ(0,2))
    circuit.add(gates.RX(0, -np.pi/8))

    circuit.add(gates.CZ(0,2))
    circuit.add(gates.RX(0, np.pi/8))

    circuit.add(gates.CZ(0,1))
    circuit.add(gates.RX(0, -np.pi/4))
    
    circuit.add(gates.CZ(0,1))
    circuit.add(gates.RX(0, -np.pi/4))
    
    return circuit





def numerical_partial_derivative(rho, sigma, epsilon, i, j):
    perturbed_rho = np.copy(rho)
    perturbed_rho[i, j] += epsilon
    return (trace_distance(perturbed_rho, sigma) - trace_distance(rho, sigma)) / epsilon


def get_x_to_z(nqubits):
    x_to_z_1q = gates.H(0).matrix(backend=backend)
    x_to_z = x_to_z_1q
    for _ in range(nqubits - 1):
        x_to_z = np.kron(x_to_z, x_to_z_1q)
    return x_to_z

def get_y_to_z(nqubits):
    y_to_z_1q = gates.S(0).matrix(backend=backend) @ gates.H(0).matrix(backend=backend)
    y_to_z = y_to_z_1q
    for _ in range(nqubits - 1):
        y_to_z = np.kron(y_to_z, y_to_z_1q)
    return y_to_z

def get_delta_rho(rho, lam, nshots, readout_f):
    nqubits = int(np.log2(rho.shape[0]))

    X = np.array([[0, 1], [1, 0]])
    Y = np.array([[0, -1j], [1j, 0]])
    Z = np.array([[1, 0], [0, -1]])

    Xn, Yn, Zn = X, Y, Z
    for _ in range(nqubits - 1):
        Xn = np.kron(Xn, X)
        Yn = np.kron(Yn, Y)
        Zn = np.kron(Zn, Z)

    readout = ReadoutErrorChannel(0, [[readout_f, 1-readout_f], [1-readout_f, readout_f]])
    x_to_z = get_x_to_z(nqubits)
    y_to_z = get_y_to_z(nqubits)

    rho_x = readout.apply_density_matrix(state=x_to_z @ rho @ x_to_z, nqubits=nqubits, backend=backend)
    rho_y = readout.apply_density_matrix(state=y_to_z.conjugate().transpose() @ rho @ y_to_z, nqubits=nqubits, backend=backend)
    rho_z = readout.apply_density_matrix(state=rho, nqubits=nqubits, backend=backend)

    a_x = np.trace(rho @ Xn)/(2**nqubits)
    a_y = np.trace(rho @ Yn)/(2**nqubits)
    a_z = np.trace(rho @ Zn)/(2**nqubits)

    delta_ax = np.sqrt(1 - a_x**2) / np.sqrt(nshots) + lam * np.abs(a_x) + np.abs(np.trace(rho_x @ Zn)/(2**nqubits) - a_x)
    delta_ay = np.sqrt(1 - a_y**2) / np.sqrt(nshots) + lam * np.abs(a_y) + np.abs(np.trace(rho_y @ Zn)/(2**nqubits) - a_y)
    delta_az = np.sqrt(1 - a_z**2) / np.sqrt(nshots) + np.abs(np.trace(rho_z @ Zn)/(2**nqubits) - a_z)

    delta_rho = delta_ax * Xn + delta_ay * Yn + delta_az * Zn
    return delta_rho


def estimate_hardware_noise(dms_rl, dms_true, lam = 0.004, readout_f = 0.96, nshots = 4000):
    """Estimate the hardware noise due to measurements and shot noise from the density matrices obtained with the model.
    Args:
        dms_rl (np.array): density matrices obtained with the model
        dms_true (np.array): density matrices btained with state tomography
        lam (float): depolarizing parameter
        readout_f (float): readout fidelity
        nshots (int): number of shots
    """
    epsilon = 1e-6

    delta_dms_true = get_delta_rho(dms_true, lam, nshots, readout_f)

    D = trace_distance(dms_true, dms_rl)

    partial_derivatives_dms_true = np.zeros_like(dms_true)
    for i in range(dms_true.shape[0]):
        for j in range(dms_true.shape[1]):
            partial_derivatives_dms_true[i, j] = numerical_partial_derivative(
                dms_true, dms_rl, epsilon, i, j)
            
    delta_D = np.sum(np.abs(partial_derivatives_dms_true * delta_dms_true))
    return delta_D



