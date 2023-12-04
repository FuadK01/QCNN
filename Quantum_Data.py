import numpy as np
import matplotlib.pyplot as plt
np.random.seed(2664)
import random
from sklearn.preprocessing import normalize

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute, Aer, IBMQ
from qiskit.quantum_info import Pauli
from qiskit.tools.visualization import circuit_drawer
import itertools as it
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import sin_data_generator
# import Benchmarking
from tqdm import tqdm

#matplotlib.rcParams['mathtext.fontset'] = 'stix'
#matplotlib.rcParams['font.family'] = 'STIXGeneral'
width=0.75
color='black'
fontsize=28
ticksize=22
figsize=(14,10)


def depolarisation_channel(circ, qreg, p, wrap=False, inverse=False, label='depol_channel'):
    """
    Given a quantum circuit, applies a depoloarisation channel to a given register with probability p.
    
    circ:    Qiskit circuit object.
    qreg:    Qiskit circuit quantum register object.
    p:       Probability of applying Pauli operation to a given qubit, otherwise apply identity (float).
    wrap:    Wrap operation into a single gate (bool) default=False.
    inverse: Apply the inverse operation (bool) default=False.
    label:   Name given to wrapped operation (str) default='depol_channel'.
    """
    
    n = qreg.size

    if inverse:
        wrap = True

    if wrap:
        qreg = QuantumRegister(n, 'q_reg')
        circ = QuantumCircuit(qreg)

    num_terms = 4**n
    max_param = num_terms / (num_terms - 1)

    if p < 0 or p > max_param:
        raise NameError("Depolarizing parameter must be in between 0 "
                         "and {}.".format(max_param))

    prob_iden = 1 - p / max_param
    prob_pauli = p / num_terms
    probs = [prob_iden] + (num_terms - 1) * [prob_pauli]

    paulis = [Pauli("".join(tup)) for tup in it.product(['I', 'X', 'Y', 'Z'], repeat=n)]
    #print(paulis)
    gates_ind = np.random.choice(num_terms, p=probs, size=1)
    #print('gi',gates_ind,p)
    gates_ind = gates_ind[0]
    #gates = paulis[gates_ind]
    #print(gates_ind)
    gates = paulis[gates_ind]
    #print(gates)
    #print(gates,qreg[:])
    #jj=0
    #for gate in gates:
    #    circ.append(gate, [qreg[jj]])
    #    jj=jj+1
    
    circ.append(gates, qreg[:])

    if wrap:
        circ = circ.to_gate()
        circ.label = label

    if inverse:
        circ = circ.inverse()
        circ.label = label+'†'

    return circ

#data=sin_gen(5, 10000)
qdata=[]

#main file needed
def quantum_data(p,freq_1, freq_2, freq_3):

    #p = 0.8
    #was 0.5 by me
    pauilProb=0
    n = 8
    print('generating data')
    data=sin_data_generator.multi_plot_gen(p,freq_1, freq_2, freq_3, 10000)
    print('generated data')

    outputs=[] #add example to array
    labels=[] #add corresponding label to array
    #Creates a file with a unique name, based on variance and frequency
    #opens it
    f = open('C:\\Users\\Fuad K\\Desktop\\Physics\\5th Year\\My_QCNN\\Rewritten Code\\Quantum_Data\\Qdata3'+str(p)+str(freq_1[0])+str(freq_1[1])+str(freq_2[0])+str(freq_2[1])+'.txt', 'a')
    
    pbar = tqdm(total=10000)
    #print(range(0,len(data[0])))
    for i in range(0,len(data[0])):
        #print(i)
        #if i%100 == 0:
            #print(i)
        wave=data[0][i]
        wave = wave/np.sqrt(np.sum(np.abs(wave)**2))
        #print(wave)
        label=data[1][i]
        q_reg = QuantumRegister(n, 'q_reg')
        circ = QuantumCircuit(q_reg)
        circ.initialize(wave, q_reg)
        backend = Aer.get_backend('statevector_simulator')
        circ = depolarisation_channel(circ, q_reg, pauilProb)
        job = execute(circ, backend)
        result = job.result()
        out_state = np.array(result.get_statevector(circ, decimals=5))
        labels.append(label)
        outputs.append(out_state)
        pbar.update(1)
    f_out=[outputs,labels]
    #np.savetxt('outfile.txt', array.view(float))
    #writes the data to the file
    f.write(str(f_out))
    f.close()

    return f_out

if __name__ == "__main__":
    #You can essentially ignore all of these, they just keep track of what data I had
    #or had not generated
    #to generate data run this file after you put your file
    #quantum_data(variance,[start frequency,end frequency])
    quantum_data(1,[10,11], [50,51], [30,31])
    quantum_data(1,[10,11], [10,11], [10,11])
    quantum_data(1,[120,150], [120,150], [120,150])
    quantum_data(1,[50,70], [70,90], [120,150])


    #quantum_data(0.8,[100,101], [150, 151], [130,131])
    #quantum_data(0.8,[120,121], [170, 171], [150,151])