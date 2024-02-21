import QCNN_circuit
import pennylane as qml
from pennylane import numpy as np
import numpy as np0
import autograd.numpy as anp
from pennylane.templates.embeddings import AmplitudeEmbedding
import pickle
import datetime
import os
from tqdm import tqdm
import Benchmarking


def cross_entropy(labels, predictions):
    '''
    Finds cross entropy loss for multiclass classification
    ARGS:
          labels: True labels (ground truth)
          predictions: Model predictions (output from QCNN)
    RETURNS: 
          cross entropy loss
    '''
    epsilon = 1e-15  # small constant to avoid log(0)
    # Clip predictions to avoid log(0) or log(1)
    predictions = np0.clip(predictions, epsilon, 1 - epsilon)
    labels = np0.eye(4)[np0.asarray(labels)]
    # print("\n LABEL LENGTH:")
    # print(len(labels))

    # Compute cross entropy
    loss = np0.sum(labels * np0.log(predictions)) / len(labels)
    
    return -1 * loss

def least_squares(labels, predictions):
    '''
    Finds least squares loss for multiclass classification
    ARGS:
          labels: True labels (ground truth)
          predictions: Model predictions (output from QCNN)
    RETURNS: 
          least squares loss
    '''

    predictions = np.array(predictions)
    labels = np.eye(4)[anp.asarray(labels)]

    loss = np.sum((predictions - labels)**2) / len(labels)

    return loss


def cost(params, X, Y, U, U_params):
    '''
    computes total cost of QCNN over training set
    ARGS: params- tunable parameters of QCNN
          X - embedded input
          Y - labeled output
          U - Unitaries used in qcnn
          U_params - unitary cirucit tunable parameters
    RETURNS: Total loss 
    '''
    #print("params",params)
    predictions = [QCNN_circuit.QCNN(x, params, U, U_params) for x in X]

    loss = cross_entropy(Y, predictions)

    return loss

def circuit_training(X_train, X_val, Y_train, Y_val, U, U_params, steps, testName, learning_rate, batch_size):
    '''
    Trains QCNN on training data
    ARGS: X_train- training data
          Y_train- training labels
          U- circuit architecture of qcnn
          U_params- tunable parameters of qcnn to be learned
    RETURNS: loss history and learned parameters
    '''
    smallest = float('inf')

    if U == 'U_SU4_no_pooling' or U == 'U_SU4_1D' or U == 'U_9_1D':
        total_params = U_params * 3
    elif U == 'Large':
        total_params = U_params
    else:
        total_params = U_params * 3 + 2 * 3 + 15
    
    params = np.random.randn(total_params, requires_grad=True) # Randomly initialises circuit parameters

    # Defining optimizer
    opt = qml.AdamOptimizer(stepsize=learning_rate)
    loss_history = []

    # Saving model to QCNN_Models folder
    try:
        path = "C:\\Users\\Fuad K\\Desktop\\Physics\\5th Year\\My_QCNN\\Rewritten Code\\QCNN_Models"
        os.mkdir(path)
    except:
        print("File "+path+"already created")
    
    pbar = tqdm(total=steps)
    # epoch = len(X_train)/batch_size

    no_improvement = 1
    best_params = 0
    # QCNN Training Loop
    for it in range(steps):
        """
        Calculate loss for each epoch
        """
        batch_index = np.random.randint(0, len(X_train), (batch_size,))

        X_batch = [X_train[i] for i in batch_index]
        Y_batch = [Y_train[i] for i in batch_index]

        """if type(best_params) != int:
            params = best_params"""

        # Cost function is called which then calls the quantum cicuit
        params, cost_new = opt.step_and_cost(lambda v: cost(v, X_batch, Y_batch, U, U_params), params)
        loss_history.append(cost_new)

        if cost_new < smallest:
            currentfile = path+"\model"+str(testName)+str(it)+"C"+str(cost_new)+".pkl"
            print(" Saving current parameters:",currentfile)
            pickle.dump(params, open(currentfile,'wb'))
            smallest = cost_new
            no_improvement = 1
            best_params = params
            print("iteration: ", it, " cost: ", cost_new)
        else:
            #print("No cost improvement")
            no_improvement += 1    

        """if it % 10 == 0:
            print("iteration: ", it, " cost: ", cost_new)
            if cost_new < smallest:
                currentfile = path+"\model"+str(testName)+str(it)+"C"+str(cost_new)+".pkl"
                print("Saving current parameters:",currentfile)
                pickle.dump(params, open(currentfile,'wb'))
                smallest = cost_new
                no_improvement = 1
                #best_params = params
            else:
                print("No cost improvement")
                no_improvement += 1"""
        
        if no_improvement % 50 == 0:
            print(" Decreasing learning rate... \n ")
            learning_rate /= 10
            opt = qml.AdamOptimizer(stepsize=learning_rate)
            no_improvement = 1
        
        pbar.update(1)
    
    return loss_history, best_params
