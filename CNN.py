import numpy as np
from keras.models import Sequential
from keras.utils import to_categorical
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt
import sin_data_generator
from sklearn.model_selection import train_test_split
import pickle
import datetime
import os
import time
# Imports for deprecated funtion
import torch.nn as nn
import torch

# Deprecated
def get_n_params(model):
    np=0
    for p in list(model.parameters()):
        np += p.nelement()
    return np

# Deprecated
def accuracy_test(predictions, labels):
    acc = 0
    for (p,l) in zip(predictions, labels):
        if p[0] >= p[1]:
            pred = 0 
        else:
            pred = 1

        if pred == l:
            acc = acc + 1
    acc = acc / len(labels)
    return acc

# Deprecated Benchmarking Function
def Old_Benchmarking_CNN(dataset,filename, input_size, optimizer, smallest, steps=300, n_feature=2, batch_size=20):
    final_layer_size = int(input_size / 4)
    train_ratio = 0.75
    validation_ratio = 0.15
    test_ratio = 0.10
    X_train, X_test, Y_train, Y_test = train_test_split(dataset[0], dataset[1], test_size=1 - train_ratio)
    #X_train, X_test, Y_train, Y_test = train_test_split(dataset[0], dataset[1], test_size=0.2, random_state=0, shuffle=True)
    X_val, X_test, Y_val, Y_test = train_test_split(X_test, Y_test, test_size=test_ratio/(test_ratio + validation_ratio)) 
    currentData = (X_train, X_val, X_test, Y_train, Y_val, Y_test)
    currentfile = "D:\CNN\CNN_Data\data"+str(datetime.datetime.now().date()) + '_' + str(datetime.datetime.now().time()).replace(':', '.') +".pkl"
    print("Saving current data:",currentfile)
    path = "D:\CNN\CNN_Models"+str(datetime.datetime.now().date())
    try:
        os.mkdir(path)
    except:
        print("File "+path+"already created")
    pickle.dump(currentData, open(currentfile,'wb'))
    CNN = nn.Sequential(
        nn.Conv1d(in_channels=1, out_channels=n_feature, kernel_size=2, padding=1),
        nn.ReLU(),
        nn.MaxPool1d(kernel_size=2),
        nn.Conv1d(in_channels=n_feature, out_channels=n_feature, kernel_size=2, padding=1),
        nn.ReLU(),
        nn.MaxPool1d(kernel_size=2),
        nn.Flatten(),
        nn.Linear(n_feature * final_layer_size, 2))

    loss_history = []
    for it in range(steps):
        batch_idx = np.random.randint(0, len(X_train), batch_size)
        X_train_batch = np.array([X_train[i] for i in batch_idx])
        Y_train_batch = np.array([Y_train[i] for i in batch_idx])

        X_train_batch_torch = torch.tensor(X_train_batch, dtype=torch.float32)
        X_train_batch_torch.resize_(batch_size, 1, input_size)
        Y_train_batch_torch = torch.tensor(Y_train_batch, dtype=torch.long)

        criterion = nn.CrossEntropyLoss()
        if optimizer == 'adam':
            opt = torch.optim.Adam(CNN.parameters(), lr=0.01, betas=(0.9, 0.999))
        elif optimizer == 'nesterov':
            opt = torch.optim.SGD(CNN.parameters(), lr=0.1, momentum=0.9, nesterov=True)

        Y_pred_batch_torch = CNN(X_train_batch_torch)

        loss = criterion(Y_pred_batch_torch, Y_train_batch_torch)
        loss_history.append(loss.item())
        if it % 10 == 0:
            print("[iteration]: %i, [LOSS]: %.10f" % (it, loss.item()))
            if loss.item() < smallest:
                currentfile = path+"\model"+str(it)+"C"+str(loss.item())+".pkl"
                #currentfile = "CNN_Models\model"+str()+"C"+str(loss.item())+".pkl"
                print("Saving current parameters:",currentfile)
                pickle.dump(CNN, open(currentfile,'wb'))
                smallest = loss.item()   
        opt.zero_grad()
        loss.backward()
        opt.step()

        X_test_torch = torch.tensor(X_val, dtype=torch.float32)
        X_test_torch.resize_(len(X_val), 1, input_size)
        Y_pred = CNN(X_test_torch).detach().numpy()
        accuracy = accuracy_test(Y_pred, Y_val)
        N_params = get_n_params(CNN)
 

    filename1=filename+'.txt'
    f = open(filename1, 'a')
    f.write("Loss History for CNN: " )
    f.write("\n")
    f.write(str(loss_history))
    f.write("\n")
    f.write("Accuracy for CNN with " +optimizer + ": " + str(accuracy))
    f.write("\n")
    f.write("Number of Parameters used to train CNN: " + str(N_params))
    f.write("\n")
    f.write("\n")

    f.close()
    plt.plot(loss_history)
    plt.title('CNN Loss History with '+ optimizer+ ' Optimiser')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.savefig(filename+'CNN Loss History with qdata'+optimizer+' Optimiser.png')


def Benchmarking_CNN(dataset, filename, input_size, optimizer, steps=300, batch_size=20):
    """
    Takes as input, noisy data and trains a CNN model to identify if a Sine wave is in the data. Model paramters 
    are saved to CNN_Models folder when validation loss is minimised. Data used to train model and and the resulting plots are 
    saved in CNN_Data and CNN_Results folders respectively.

        Parameters
        ----------
        dataset : list
            Generated noisy sin data with accompanying labels for each set of data points
        filename : str
            File location where results of CNN are outputted
        input_size : int
            Number of data points in each value in dataset.
        optimizer : str
            Which type of optimizer is used while training the CNN.
        steps: int
            Number of epochs used while training.
        batch_size: int
            Size of batches used while training model.

        Raises
        ------
        Exception
            If labels generate incorrectly

        Returns
        -------
        list
            Returns generated data with accompanying labels for each set of data points

    """
 
    # Setting ratios for training, validation and test sets
    train_ratio, validation_ratio, test_ratio = 0.75, 0.15, 0.1

    # Splitting data into aformentioned sets and saving into a folder for future testing
    X_train, X_test, Y_train, Y_test = train_test_split(dataset[0], dataset[1], test_size = 1 - train_ratio)
    X_val, X_test, Y_val, Y_test = train_test_split(X_test, Y_test, test_size=test_ratio/(test_ratio + validation_ratio)) 
    currentData = (X_train, X_val, X_test, Y_train, Y_val, Y_test)

    # Preprocessing Data arrays to be appropiate for CNN
    X_train = np.array(X_train).reshape(len(X_train),  input_size, 1)   ;   Y_train = to_categorical(np.array(Y_train))
    X_val = np.array(X_val).reshape(len(X_val),  input_size, 1)     ;   Y_val = to_categorical(np.array(Y_val))
    X_test = np.array(X_test).reshape(len(X_test),  input_size, 1)      ;   Y_test = to_categorical(np.array(Y_test))

    # Saving preprocessed data used
    currentfile = "C:\\Users\Fuad K\\Desktop\Physics\\5th Year\\My_QCNN\\Rewritten Code\\CNN_Data\\data" + str(datetime.datetime.now().date()) + '_' + str(datetime.datetime.now().time()).replace(':', '.') +".pkl"
    print("Saving current data:",currentfile)
    pickle.dump(currentData, open(currentfile, 'wb'))

    # Estabilishing path to save model
    path = "C:\\Users\\Fuad K\Desktop\\Physics\\5th Year\\My_QCNN\\Rewritten Code\\CNN_Models\\"+str(datetime.datetime.now().date())
    try:
        os.mkdir(path)
    except:
        print("File " + path + "already created")
    

    # Building the CNN model for multiclass classification of sequential data.
    model = Sequential()
    model.add(Conv1D(1, 2, activation='relu', input_shape = (input_size, 1)))
    model.add(MaxPooling1D(2))
    model.add(Conv1D(1, 2, activation='relu'))
    model.add(MaxPooling1D(2))
    model.add(Flatten())
    model.add(Dense(12, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4, activation='softmax'))

    # Choosing optimizer for CNN
    if optimizer == 'nesterov':
        opt=SGD(learning_rate=0.001, momentum=0.9, nesterov=True)
    elif optimizer == 'adam':
        opt=Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999)
    
    # Compiling CNN
    model.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics = ['accuracy'])
    
    # Initialising book-keeping variable
    loss_history = []

    # Setting up model checkpoints to find minimum value for validation loss. The best model is saved and the weights are recorded.
    checkpoint_path = path + "_best_val_loss"
    checkpoint = ModelCheckpoint(checkpoint_path, save_weights_only=True, monitor='val_loss', save_best_only=True, mode='min', verbose=1)
    lr_optimisation = ReduceLROnPlateau(monitor='val_loss',
                                        factor=0.1,
                                        patience=50)

    # Keeping track of training time
    start = time.time()

    # Beginning training loop
    history = model.fit(X_train, Y_train,
                        epochs=steps,
                        batch_size=batch_size,
                        validation_data=(X_val, Y_val),
                        callbacks=[checkpoint, lr_optimisation],
                        verbose=1)
    
    # Printing CNN training time
    end = time.time()

    print("Training time for Classical CNN Model is " + str(end-start) + " seconds")

    # Loading the best model
    model.load_weights(checkpoint_path)

    # Evaluating model on the validation set
    loss_history = history.history['loss']
    test_loss, accuracy = model.evaluate(X_test, Y_test)
    print("Test accuracy of model is " + str(accuracy))

    # Saving the number of parameters used in model
    N_params = model.count_params()

    # Finally, saving loss history to a file
    filename += '.txt'
    f = open(filename, 'a')
    f.write("Loss History for CNN: " )
    f.write("\n")
    f.write(str(loss_history))
    f.write("\n")
    f.write("Accuracy for CNN with " + optimizer + ": " + str(accuracy))
    f.write("\n")
    f.write("Number of Parameters used to train CNN: " + str(N_params))
    f.write("\n")
    f.write("\n")
    f.close()

    plt.plot(loss_history)
    plt.title('CNN Loss History with '+ optimizer+ ' Optimiser')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.savefig(filename+'CNN Loss History with qdata'+optimizer+' Optimiser.png')
    plt.show()

    return 1

"""
Function to test running this algorithm
"""        
if __name__ == "__main__":
    p = 0.9
    freq_1=[10,30]
    freq_2=[80,100]
    freq_3=[50,70]
    frequencies = [freq_1, freq_2, freq_3]
    filename="C:\\Users\\Fuad K\\Desktop\\Physics\\5th Year\\My_QCNN\\Rewritten Code\\CNN_Results\\CNN_Result"+str(datetime.datetime.now().date()) + '_' + str(datetime.datetime.now().time()).replace(':', '.')
    dataset=sin_data_generator.multi_plot_gen(p, *frequencies, 10000)
    #for i in range(0,10):
    print('running')
        #dataset = sin_generator.sin_gen3(i,10000)
        #dataset =sin_generator.sin_genn(5,10000)
        
    Benchmarking_CNN(dataset, filename, input_size = 256, optimizer='nesterov', steps=300, batch_size=50)
    Old_Benchmarking_CNN(dataset, filename, input_size = 256, optimizer='nesterov', steps=300, batch_size=50)


