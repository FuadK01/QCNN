import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, transpile, assemble
from qiskit.tools.visualization import circuit_drawer, plot_histogram
from qiskit.quantum_info import state_fidelity
from qiskit_aer import AerSimulator
import Benchmarking
from autograd.numpy.numpy_boxes import ArrayBox
from pennylane.numpy import tensor as qml_tensor

# Very unexpressive
def U_TTN(qc, params, qregister):  # 2 params
    qc.ry(params[0], qregister[0])
    qc.ry(params[1], qregister[1])
    qc.cx(qregister[0], qregister[1])

# Very expressive
def U_5(qc, params, qregister):  # 10 params
    qc.rx(params[0], qregister[0])
    qc.rx(params[1], qregister[1])
    qc.rz(params[2], qregister[0])
    qc.rz(params[3], qregister[1])
    qc.crz(params[4], qregister[1], qregister[0])
    qc.crz(params[5], qregister[0], qregister[1])
    qc.rx(params[6], qregister[0])
    qc.rx(params[7], qregister[1])
    qc.rz(params[8], qregister[0])
    qc.rz(params[9], qregister[1])

# Very expressive
def U_6(qc, params, qregister):  # 10 params
    qc.rx(params[0], qregister[0])
    qc.rx(params[1], qregister[1])
    qc.rz(params[2], qregister[0])
    qc.rz(params[3], qregister[1])
    qc.crx(params[4], qregister[1], qregister[0])
    qc.crx(params[5], qregister[0], qregister[1])
    qc.rx(params[6], qregister[0])
    qc.rx(params[7], qregister[1])
    qc.rz(params[8], qregister[0])
    qc.rz(params[9], qregister[1])

# Somewhat expressive
def U_9(qc, params, qregister):  # 2 params
    qc.h(qregister[0])
    qc.h(qregister[1])
    qc.cz(qregister[0], qregister[1])
    qc.rx(params[0], qregister[0])
    qc.rx(params[1], qregister[1])

# Somewhat expressive
def U_13(qc, params, qregister):  # 6 params
    qc.ry(params[0], qregister[0])
    qc.ry(params[1], qregister[1])
    qc.crz(params[2], qregister[1], qregister[0])
    qc.ry(params[3], qregister[0])
    qc.ry(params[4], qregister[1])
    qc.crz(params[5], qregister[0], qregister[1])

# Somewhat expressive
def U_14(qc, params, qregister):  # 6 params
    qc.ry(params[0], qregister[0])
    qc.ry(params[1], qregister[1])
    qc.crx(params[2], qregister[1], qregister[0])
    qc.ry(params[3], qregister[0])
    qc.ry(params[4], qregister[1])
    qc.crx(params[5], qregister[0], qregister[1])

# Somewhat expressive
def U_15(qc, params, qregister):  # 4 params
    qc.ry(params[0], qregister[0])
    qc.ry(params[1], qregister[1])
    qc.cx(qregister[1], qregister[0])
    qc.ry(params[2], qregister[0])
    qc.ry(params[3], qregister[1])
    qc.cx(qregister[0], qregister[1])

# Somewhat expressive
def U_SO4(qc, params, qregister):  # 6 params
    qc.ry(params[0], qregister[0])
    qc.ry(params[1], qregister[1])
    qc.cx(qregister[0], qregister[1])
    qc.ry(params[2], qregister[0])
    qc.ry(params[3], qregister[1])
    qc.cx(qregister[0], qregister[1])
    qc.ry(params[4], qregister[0])
    qc.ry(params[5], qregister[1])
# Very expressive
def U_SU4(qc, params, qregister): # 15 params
    qc.u(params[0], params[1], params[2], qregister[0])
    qc.u(params[3], params[4], params[5], qregister[1])
    qc.cx(qregister[0], qregister[1])
    qc.ry(params[6], qregister[0])
    qc.rz(params[7], qregister[1])
    qc.cx(qregister[1], qregister[0])
    qc.ry(params[8], qregister[0])
    qc.cx(qregister[0], qregister[1])
    qc.u(params[9], params[10], params[11], qregister[0])
    qc.u(params[12], params[13], params[14], qregister[1])

def U4(qc, params, qregister, initial_rotate = True): # 8 params
    if initial_rotate:
        for i in range(4):
            qc.ry(params[i], qregister[i])
        qc.cry(params[5], qregister[0], qregister[1])
        qc.cry(params[6], qregister[1], qregister[2])
        qc.cry(params[7], qregister[2], qregister[3])
        qc.cry(params[4], qregister[3], qregister[0])
    else:
        qc.cry(params[0], qregister[0], qregister[1])
        qc.cry(params[1], qregister[1], qregister[2])
        qc.cry(params[2], qregister[2], qregister[3])
        qc.cry(params[3], qregister[3], qregister[0])

def U2(qc, params, qregister): # 4 params
    # Initial Rotations
    qc.ry(params[0], qregister[0])
    qc.ry(params[1], qregister[1])
    qc.cx(qregister[0], qregister[1])
    qc.ry(params[2], qregister[0])
    qc.ry(params[3], qregister[1])
    qc.cx(qregister[1], qregister[0])

def U3(qc, params, qregister):
    # Initial Rotations
    qc.ry(params[0], qregister[0])
    qc.ry(params[1], qregister[1])
    qc.ry(params[2], qregister[2])
    qc.ry(params[0], qregister[3])
    qc.ry(params[1], qregister[4])
    qc.ry(params[2], qregister[5])
    qc.ry(params[0], qregister[6])
    qc.ry(params[1], qregister[7])

    # 4 qubit operations
    U4(qc, params[3:7], [qregister[0], qregister[1], qregister[2], qregister[3]], initial_rotate=False)
    U4(qc, params[3:7], [qregister[4], qregister[5], qregister[6], qregister[7]], initial_rotate=False)

    # Rotations with U4 params
    qc.ry(params[4], qregister[0])
    qc.ry(params[5], qregister[1])
    qc.ry(params[3], qregister[2])
    qc.ry(params[4], qregister[3])
    qc.ry(params[5], qregister[4])
    qc.ry(params[3], qregister[5])
    qc.ry(params[4], qregister[6])
    qc.ry(params[5], qregister[7])

    # More 4 qubit operations
    U4(qc, params[3:7], [qregister[2], qregister[3], qregister[4], qregister[5]], initial_rotate=False)
    qc.cry(params[4], qregister[6], qregister[7])
    qc.cry(params[5], qregister[7], qregister[0])
    qc.cry(params[6], qregister[0], qregister[1])
    qc.cry(params[3], qregister[1], qregister[6])


def regular_layer(rotation, qc, params, qregister): # 12 params
    if rotation == 'RY':
        for i in range(4):
            qc.ry(params[i], qregister[i])
        qc.cry(params[5], qregister[0], qregister[1])
        qc.cry(params[6], qregister[1], qregister[2])
        qc.cry(params[7], qregister[2], qregister[3])
        qc.cry(params[4], qregister[3], qregister[0])
        qc.cry(params[8], qregister[1], qregister[2])
        qc.cry(params[9], qregister[2], qregister[3])
        qc.cry(params[10], qregister[3], qregister[0])
        qc.cry(params[11], qregister[0], qregister[1])
    
    if rotation == 'RX':
        for i in range(4):
            qc.rx(params[i], qregister[i])
        qc.crx(params[5], qregister[0], qregister[1])
        qc.crx(params[6], qregister[1], qregister[2])
        qc.crx(params[7], qregister[2], qregister[3])
        qc.crx(params[4], qregister[3], qregister[0])
        qc.crx(params[8], qregister[1], qregister[2])
        qc.crx(params[9], qregister[2], qregister[3])
        qc.crx(params[10], qregister[3], qregister[0])
        qc.crx(params[11], qregister[0], qregister[1])
    
    if rotation == 'RZ':
        for i in range(4):
            qc.rz(params[i], qregister[i])
        qc.crz(params[5], qregister[0], qregister[1])
        qc.crz(params[6], qregister[1], qregister[2])
        qc.crz(params[7], qregister[2], qregister[3])
        qc.crz(params[4], qregister[3], qregister[0])
        qc.crz(params[8], qregister[1], qregister[2])
        qc.crz(params[9], qregister[2], qregister[3])
        qc.crz(params[10], qregister[3], qregister[0])
        qc.crz(params[11], qregister[0], qregister[1])


# Pooling Layers
def Pooling_ansatz1(qc, params, qregister): #2 params
    qc.crz(params[0], qregister[0], qregister[1])
    qc.x(qregister[0])
    qc.crx(params[1], qregister[0], qregister[1])

def Pooling_ansatz2(qc, qregister):  # 0 params
    qc.crz(qregister[0], qregister[1])


# Quantum Circuits for Convolutional layers
def conv_layer1(U, qc, params, qregister):
    U(qc, params, [qregister[0], qregister[7]])
    for i in range(0, 8, 2):
        U(qc, params, [qregister[i], qregister[i + 1]])
    for i in range(1, 7, 2):
        U(qc, params, [qregister[i], qregister[i + 1]])

def conv_layer2(U, qc, params, qregister):
    U(qc, params, [qregister[0], qregister[6]])
    U(qc, params, [qregister[0], qregister[2]])
    U(qc, params, [qregister[4], qregister[6]])
    U(qc, params, [qregister[2], qregister[4]])

def conv_layer3(U, qc, params, qregister):
    U(qc, params, [qregister[0], qregister[4]])

def Dense_Layer(qc, params, qregister):
    qc.crx(params[0], qregister[0], qregister[4])
    qc.crz(params[1], qregister[4], qregister[0])
    #qc.ry(params[2], qregister[0])
    #qc.ry(params[3], qregister[4])

# Quantum Circuits for Pooling layers
def pooling_layer1(V, qc, params, qregister):
    for i in range(0, 8, 2):
        V(qc, params, [qregister[i+1], qregister[i]])

def pooling_layer2(V, qc, params, qregister):
    V(qc, params, [qregister[2], qregister[0]])
    V(qc, params, [qregister[6], qregister[4]])

def pooling_layer3(V, qc, params, qregister):
    V(qc, params, [qregister[0], qregister[4]])

def QCNN_structure(U, qc, params, U_params, qregister):
    param1 = params[0:U_params]
    param2 = params[U_params: 2 * U_params]
    param3 = params[2 * U_params: 3 * U_params]
    param4 = params[3 * U_params: 3 * U_params + 2]
    param5 = params[3 * U_params + 2: 3 * U_params + 4]
    param6 = params[3 * U_params + 4: 3 * U_params + 19]

    conv_layer1(U, qc, param1, qregister)
    pooling_layer1(Pooling_ansatz1, qc, param4, qregister)
    conv_layer2(U, qc, param2, qregister)
    pooling_layer2(Pooling_ansatz1, qc, param5, qregister)
    conv_layer3(U, qc, param3, qregister)
    U_SU4(qc, param6, [qregister[0], qregister[4]])

def QCNN_structure_without_pooling(U, qc, params, U_params, qregister):
    param1 = params[0:U_params]
    param2 = params[U_params: 2 * U_params]
    param3 = params[2 * U_params: 3 * U_params]

    conv_layer1(U, qc, param1, qregister)
    conv_layer2(U, qc, param2, qregister)
    conv_layer3(U, qc, param3, qregister)

def QCNN_1D_circuit(U, qc, params, U_params, qregister):
    param1 = params[0: U_params]
    param2 = params[U_params: 2*U_params]
    param3 = params[2*U_params: 3*U_params]

    for i in range(0, 8, 2):
        U(qc, param1, [qregister[i], qregister[i + 1]])
    for i in range(1, 7, 2):
        U(qc, param1, [qregister[i], qregister[i + 1]])
    
    U(qc, param2, [qregister[2], qregister[3]])
    U(qc, param2, [qregister[4], qregister[5]])
    U(qc, param3, [qregister[3], qregister[4]])

def Large_QCNN_structure(qc, params, qregister): # 146 params # Uniform entaglement variables
    param_U4 = params[:8]
    param_U2 = params[8:12]
    param_U3 = params[12:19]
    param_pooling1 = params[19:21]
    param_reg_1 = params[21:33]
    param_reg_2 = params[33:45]
    param_reg_3 = params[45:57]
    param_reg_4 = params[57:69]
    param_reg_5 = params[69:81]
    param_reg_6 = params[81:93]
    param_reg_7 = params[93:105]
    param_reg_8 = params[105:117]
    param_reg_9 = params[117:129]
    param_pooling2 = params[129:131]
    param_dense = params[131:]

    U4(qc, param_U4, qregister[:4])
    U4(qc, param_U4, qregister[4:])
    U2(qc, param_U2, qregister[:2])
    U4(qc, param_U4, qregister[2:6])
    U2(qc, param_U2, qregister[6:])
    U3(qc, param_U3, qregister)

    for l in range(2):
        for i in range(2, 10, 2):
            U2(qc, param_U2, qregister[i-2:i])
        
        for i in range(2, 8, 2):
            U2(qc, param_U2, qregister[i-1:i+1])
    
    pooling_layer1(Pooling_ansatz1, qc, param_pooling1, qregister)

    regular_layer("RY", qc, param_reg_1, [qregister[0], qregister[2], qregister[4], qregister[6]])
    regular_layer("RY", qc, param_reg_2, [qregister[0], qregister[2], qregister[4], qregister[6]])
    regular_layer("RY", qc, param_reg_3, [qregister[0], qregister[2], qregister[4], qregister[6]])
    regular_layer("RX", qc, param_reg_4, [qregister[0], qregister[2], qregister[4], qregister[6]])
    regular_layer("RX", qc, param_reg_5, [qregister[0], qregister[2], qregister[4], qregister[6]])
    regular_layer("RX", qc, param_reg_6, [qregister[0], qregister[2], qregister[4], qregister[6]])
    regular_layer("RZ", qc, param_reg_7, [qregister[0], qregister[2], qregister[4], qregister[6]])
    regular_layer("RZ", qc, param_reg_8, [qregister[0], qregister[2], qregister[4], qregister[6]])
    regular_layer("RZ", qc, param_reg_9, [qregister[0], qregister[2], qregister[4], qregister[6]])

    pooling_layer2(Pooling_ansatz1, qc, param_pooling2, qregister)

    U_SU4(qc, param_dense, [qregister[0], qregister[4]])
    
    return 0

def QCNN(X, params, U, U_params):
    
    '''
    Pass data through QCNN
    ARGS: X- Data
          params- tunable parameters of entire quantum circuits
          U- Unitary ansatz
          U_params- tunable parameters of single circuit
    RETURNS: Measurement of final dual qubit probablility
    '''

    Benchmarking.data_embedding(X) #, embedding_type=embedding_type)
    if isinstance(params, ArrayBox):
        params = [float(param._value) for param in params]
    elif isinstance(params, qml_tensor):
        params = np.array(params)

    # Iniatilising quantum circuit
    qregister = QuantumRegister(8, name='q')
    cregister = ClassicalRegister(2, name='c')
    simulator = AerSimulator()
    qc = QuantumCircuit(qregister, cregister)

    # Quantum Convolutional Neural Network
    if U == 'U_TTN':
        QCNN_structure(U_TTN, qc, params, U_params, qregister)
    elif U == 'U_5':
        QCNN_structure(U_5, qc, params, U_params, qregister)
    elif U == 'U_6':
        QCNN_structure(U_6, qc, params, U_params, qregister)
    elif U == 'U_9':
        QCNN_structure(U_9, qc, params, U_params, qregister)
    elif U == 'U_13':
        QCNN_structure(U_13, qc, params, U_params, qregister)
    elif U == 'U_14':
        QCNN_structure(U_14, qc, params, U_params, qregister)
    elif U == 'U_15':
        QCNN_structure(U_15, qc, params, U_params, qregister)
    elif U == 'U_SO4':
        QCNN_structure(U_SO4, qc, params, U_params, qregister)
    elif U == 'U_SU4':
        QCNN_structure(U_SU4, qc, params, U_params, qregister)
    elif U == 'U_SU4_no_pooling':
        QCNN_structure_without_pooling(U_SU4, qc, params, U_params, qregister)
    elif U == 'U_SU4_1D':
        QCNN_1D_circuit(U_SU4, qc, params, U_params, qregister)
    elif U == 'U_9_1D':
        QCNN_1D_circuit(U_9, qc, params, U_params, qregister)
    elif U == 'Large':
        Large_QCNN_structure(qc, params, qregister)
    else:
        print("Invalid Unitary Ansatz")
        return False

    qc.measure([0, 4], [0, 1])

    # Transpiling qcircuit
    tqc = transpile(qc, simulator)

    # Assemble the transpiled circuit for the simulator
    tqc_job = assemble(tqc)

    # Simulate the circuit
    result = simulator.run(tqc_job).result()

    # Get the counts from the result
    counts = result.get_counts(qc)

    #print("Measurement Results: ", counts)
    # Calculate the probability distribution
    prob_dist = np.zeros(4)
    for key, value in counts.items():
        index = int(key, 2)
        prob_dist[index] = value / 1000


    #print("Probability distribution:", prob_dist)

    # Convert the counts to a list of integers
    #measurement_list = [int(key, 2) for key in counts.keys()]

    # Find the index of the maximum value
    #max_index = measurement_list.index(max(measurement_list))
    #print("Index of maximum value:", max_index)

    return prob_dist

if __name__ == "__main__":
    total_params = 38
    params = np.random.randn(total_params) # Randomly initialises circuit parameters
    # Iniatilising quantum circuit
    qregister = QuantumRegister(8, name='q')
    cregister = ClassicalRegister(2, name='c')
    simulator = AerSimulator()
    qc = QuantumCircuit(qregister, cregister)

    Large_QCNN_structure(qc, params, qregister)
    qc.measure([0, 4], [0, 1])

    qc.draw("mpl")
    plt.show()
