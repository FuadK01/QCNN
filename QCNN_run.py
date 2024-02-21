import Benchmarking
import numpy as np
import datetime
import os

"""
Here are possible combinations of benchmarking user could try.
Unitaries: ['U_TTN', 'U_5', 'U_6', 'U_9', 'U_13', 'U_14', 'U_15', 'U_SO4', 'U_SU4', 'U_SU4_no_pooling', 'U_SU4_1D', 'U_9_1D']
U_num_params: [2, 10, 10, 2, 6, 6, 4, 6, 15, 15, 15, 2]

circuit: 'QCNN' 
cost_fn: 'cross_entropy'
Note: when using 'mse' as cost_fn binary="True" is recommended, when using 'cross_entropy' as cost_fn must be binary="False".
"""

# Constant declarations
EPOCHS = 400
LEARNING_RATE = 0.01
BATCH_SIZE = 40

if __name__ == "__main__":
    # Choosing Unitary
    Unitaries= ['U_6', 'U_5', 'U_9', 'U_13', 'U_14', 'U_15', 'U_SO4', 'U_SU4', 'Large']#, 'U_SU4_no_pooling', 'U_SU4_1D', 'U_9_1D']
    U_num_params = {'U_6': 10, 'U_5': 10, 'U_9': 2, 'U_13': 6, 'U_14': 6, 'U_15': 4, 'U_SO4': 6, 'U_SU4': 15, 'Large': 146}#, 15, 15, 2]{'U_6': 10, 'U_5': 10, 'U_9': 2, 'U_13': 6, 'U_14': 6, 'U_15': 4, 'U_SO4': 6, 'U_SU4': 15}
    
    # Set filename for saving results of training
    filename = "Result"+str(datetime.datetime.now().date()) + '_' + str(datetime.datetime.now().time()).replace(':', '.')
    #Input name of quantum data file to be trained on
    Qdata = "Qdata31202150517071"
    fname = r"C:\\Users\\Fuad K\\Desktop\\Physics\\5th Year\\My_QCNN\\Rewritten Code\\Quantum_Data\\"+ Qdata + ".txt"
    testName = "G" + Qdata
    
    with open(fname) as f:
        lines = f.readlines()
        filedata=''.join(lines)

        #data handelling
        allstates=filedata.split(')], ')
        print(len(allstates))

        labels=allstates[1]
        labels=labels[:-1]
        labels=labels.strip(']')
        labels=labels.strip('[')
        labels=labels.split(',')
        labels=[eval(i) for i in labels]

        filedata = filedata.split('),')
        #print(filedata)
        print(len(filedata))
        counter1=0
        filedata_new=[]
        for data in filedata:
            #print(data)
            data = data.replace('[array(','')
            data = data.replace('rray([','')
            data=data[2:-1]
            data=data.replace('\n       ','')
            data=data.replace('  ','')
            data=data.replace(' ','')
            data=data.split(',')
            #print("data length",len(data))

            if counter1 == len(filedata)-1:  
                data=';'.join(data)
                data=data.split(')')
                data=data[0]
                data=data[:-1]
                data=data.split(';')
                #print('new data length',len(data))
            #length of the file
            if len(data) == 256:
                complexData=[]
                for i in data:
                    h=np.complex128(i)
                    complexData.append(h)
                counter1=counter1+1
            complexData=np.array(complexData,dtype=complex)
            filedata_new.append(complexData)

    #print(len(filedata_new))
    #print(type(filedata_new[0]),labels[0])
    print('counter1:',counter1)
    dataset=[filedata_new,labels]

    #snr? Frequences
    #was 0.1
    freqs=[1]
    print('Data reading complete...')
    #print(freqs)
    print("test",testName,"\n")
    #print("Running QCNN...")

    for U in U_num_params:
        print("\nRunning QCNN with ", U, " and ", str(U_num_params[U]), " params.")
        accuracy = Benchmarking.Benchmarking(dataset, U, U_num_params[U], filename, testName, LEARNING_RATE, BATCH_SIZE, circuit='QCNN', steps = EPOCHS, snr=1)
        best_accuracy = 0.001
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            Architecture = U
    
    print("Best circuit architecture used the ", Architecture, " ansatz, with a test set accuracy of ", str(best_accuracy * 100), "%")
    

    #accuracy = Benchmarking.Benchmarking(dataset, 'Large', U_num_params['Large'], filename, testName, LEARNING_RATE, BATCH_SIZE, circuit='QCNN', steps = EPOCHS, snr=1)

    #train pnoise network with gnoise
