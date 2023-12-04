import Training
import QCNN_circuit
import numpy as np
import pennylane as qml
from pennylane.templates.embeddings import AmplitudeEmbedding
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time
import sin_data_generator 
import pickle
import datetime
import os

# This is an implementation of data_embedding function used for 8 qubits Quantum Convolutional Neural Network (QCNN)
# and Hierarchical Quantum Classifier circuit.

def data_load_and_process(dataset):
    '''
    args:
    dataset-[outputs, labels] labelled data (either noise or sin wave)

    returns:
    randomly shuffled training and test data ot feed into QCNN/CNN
    '''
    
    #x_train, x_test, y_train, y_test = train_test_split(dataset[0], dataset[1], test_size=0.2, random_state=0, shuffle=True)    
    train_ratio = 0.75
    validation_ratio = 0.15
    test_ratio = 0.10

    # train is now 75% of the entire data set
    x_train, x_test, y_train, y_test = train_test_split(dataset[0], dataset[1], test_size=1 - train_ratio)

    # test is now 10% of the initial data set
    # validation is now 15% of the initial data set
    x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=test_ratio/(test_ratio + validation_ratio)) 

    #print(x_train, x_val, x_test)

    return (x_train,x_val, x_test, y_train,y_val, y_test)

def data_embedding(X): #encode input X with 8 qubits with amplitude encoding
    '''
    Embeds 256 data points into 8 qubit and normalises the data, with L2 normalisation

    args: X - the data
    '''
    AmplitudeEmbedding(X, wires=range(8), normalize=True)

def accuracy_test(predictions, labels, binary = True):
    '''
    This functions calculates the accuracy of the preedicitons
    args: predictions - is the label outputed by neural network (QCNN/CNN)
          labels - Y_test/Y_train datta
          binary True/Flase (not used)
    '''
    #QE add meaningful varaible names
    acc = 0
    #for i in len(predictions):       
    for l,p in zip(labels, predictions):
        if np.argmax(p) != l: 
            P = 0
        else:
            P = 1
        if P == l:
            acc = acc + 1
    return acc / len(labels)

def Benchmarking(dataset, U, U_params, filename, testName, circuit, steps, snr, binary=True):
    """
    This function benchmarks the QCNN
        Parameters
        ----------
        dataset : [outputs, labels]
            Labelled data
        U : function_name
            Unitary to base the circuit on
        U_num_params : int
            Number of params required for chosen unitary
        filename : string
            Filename where results from QCNN benchmark will be saved
        testName : string
            Name of folder where results will be saved
        circuit : function
            Quantum Circuit used to run QCNN

        Raises
        ------
        Exception
            If labels generate incorrectly

        Returns
        -------
        list
            Returns generated data with accompanying labels for each set of data points

        """
    
    # Make folder for saving results
    try:
        path = "C:\\Users\\Fuad K\\Desktop\\Physics\\5th Year\My_QCNN\\Rewritten Code\\QCNN_Results\\QResults" + str(datetime.datetime.now().date())+str(testName)
        os.mkdir(path)
    except:
        print("File "+path+"already created")
    
    start = time.time()

    # Initalising QCNN - Opening results file and choosing embedding method
    f = open(path + filename + '.txt', 'a')
    Emebdding = 'Amplitude'

    X_train, X_val, X_test, Y_train, Y_val, Y_test = data_load_and_process(dataset)
    currentData = (X_train, X_val, X_test, Y_train, Y_val, Y_test)

    # Create directory to save QCNN data from above
    try:
        datapath = "C:\\Users\\Fuad K\\Desktop\\Physics\\5th Year\\My_QCNN\\Rewritten Code\\QCNN_Data" + str(datetime.datetime.now().date())+str(testName)
        os.mkdir(datapath)
    except:
        print("File " + path + "already created")
    
    currentfile = datapath + "\\data" + str(testName) + ".pkl"
    print("Saving current Data:",currentfile)
    pickle.dump(currentData, open(currentfile,'wb'))
    print("\n")

    data_length = str(len(dataset[0]))
    print("Loss History for " + circuit + " circuits, " + str(U) + " Amplitude with " +'cross entropy' + ' trained with: ' + data_length + ' with snr: ' +str(snr))

    for i in range(2):
        loss_history, trained_params = Training.circuit_training(X_train, X_val, Y_train, Y_val, U, U_params, steps, testName)
        Valpredictions = [QCNN_circuit.QCNN(x, trained_params, U, U_params) for x in X_val]
        Trainingpredictions = [QCNN_circuit.QCNN(x, trained_params, U, U_params) for x in X_train]
        #calculate accuracy
        accuracy = accuracy_test(Trainingpredictions, Y_train, binary)
        #for that epoch looks at the accuracy for the model and looks at the current accuracy for both seen (training)
        print("Training Dataset Accuracy for " + U + " Amplitude :" + str(accuracy))
        #and partially seen data (validation)
        accuracy = accuracy_test(Valpredictions, Y_val, binary)
        print("Validation Dataset Accuracy for " + U + " Amplitude :" + str(accuracy))
    
    print("Trained Paramters: ", trained_params)
    # Plots the QCNN Loss Graph
    plt.plot(loss_history, label=U)
    plt.xlabel('Epochs')
    plt.ylabel('Training Loss')
    plt.title('Loss History across '+ str(steps) + 'epochs.')
    plt.savefig(path+'\QCNN_Loss'+str(snr)+'.png')
    plt.show()

    #makes predictions of test set with trained parameters
    #Now training Off Validation data sets 
    testPredictions = [QCNN_circuit.QCNN(x, trained_params, U, U_params) for x in X_test]
    testAccuracy = accuracy_test(Trainingpredictions, Y_test, binary)
    print("test Accuracy for " + U + " Amplitude :" + str(accuracy))

    #Writes file
    f.write("Loss History for " + circuit + " circuits, " + U + " Amplitude with " +'cross entropy' + ' trained with: ' + data_length + ' with snr: ' +str(snr))
    f.write("\n")
    f.write(str(loss_history))
    f.write("\n")
    f.write("Total time: "+ str(time.time() - start)+ "seconds.cc")
    f.write("\n")
    f.write("Training Accuracy for " + U + " Test Acuracy :" + str(testAccuracy))
    f.write("Test Accuracy for " + U + " Test Acuracy :" + str(testAccuracy))
    f.write("\n")
    f.write("Accuracy for " + U + " Test Acuracy :" + str(testAccuracy))
    f.write("\n")
    f.write("\n")

    f.close()