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
def Benchmarking_CNN(dataset, filename, input_size, optimizer, steps=300, batch_size=20):
 
    # Setting ratios for training, validation and test sets
    train_ratio, validation_ratio, test_ratio = 0.75, 0.15, 0.1

    # Splitting data into aformentioned sets and saving into a folder for future testing
    X_train, X_test, Y_train, Y_test = train_test_split(dataset[0], dataset[1], test_size = 1 - train_ratio)
    X_val, X_test, Y_val, Y_test = train_test_split(X_test, Y_test, test_size=test_ratio/(test_ratio + validation_ratio)) 
    currentData = (X_train, X_val, X_test, Y_train, Y_val, Y_test)

    # Preprocessing Data arrays to be appropiate for CNN
    X_train = np.array(X_train).reshape(len(X_train),  input_size, 1)   ;   Y_train = np.array(Y_train)
    X_val = np.array(X_val).reshape(len(X_val),  input_size, 1)     ;   Y_val = np.array(Y_val)
    X_test = np.array(X_test).reshape(len(X_test),  input_size, 1)      ;   Y_test = np.array(Y_test)

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
    

    # Building the CNN model for classification of sequential data.
    model = Sequential()
    model.add(Conv1D(4, 2, activation='relu', input_shape = (input_size, 1)))
    model.add(MaxPooling1D(2))
    model.add(Conv1D(3, 2, activation='relu'))
    model.add(MaxPooling1D(2))
    model.add(Flatten())
    model.add(Dense(4, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # Choosing optimizer for CNN
    if optimizer == 'nesterov':
        opt=SGD(learning_rate=0.005, momentum=0.9, nesterov=True)
    elif optimizer == 'adam':
        opt=Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999)
    
    # Compiling CNN
    model.compile(optimizer=opt,
                  loss='binary_crossentropy',
                  metrics = ['accuracy'])
    
    # Initialising book-keeping variable
    loss_history = []

    # Setting up model checkpoints to find minimum value for validation loss. The best model is saved and the weights are recorded.
    checkpoint_path = path + "_best_val_loss"
    checkpoint = ModelCheckpoint(checkpoint_path, save_weights_only=True, monitor='val_loss', save_best_only=True, mode='min', verbose=1)

    # Keeping track of training time
    start = time.time()

    # Beginning training loop
    history = model.fit(X_train, Y_train,
                        epochs=steps,
                        batch_size=batch_size,
                        validation_data=(X_val, Y_val),
                        callbacks=[checkpoint],
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


    """
    # Training loop
    for it in range(steps):
        # Creating randomised batch based on optional `batch_size` paramter
        batch_index = np.random.randint(0, len(X_train), batch_size)
        X_train_batch = np.array([X_train[i] for i in batch_index])
        Y_train_batch = np.array([Y_train[i] for i in batch_index])
        # Reshaping input batch array
        X_train_batch = X_train_batch.reshape((batch_size, input_size, 1))

        # Training model on the batch
        history = model.train_on_batch(X_train_batch, Y_train_batch)

        # Keeping track of loss
        loss = history[0]
        loss_history.append(loss)
        
        # Printing progress every 10 iterations. Also checking if loss has reached minimum,
        # and saving model when the loss is minimised
        if it % 10 == 0:
            print("[iteration]: %i, [LOSS]: %.10f" % (it, loss))
            if loss < smallest:
                currentfile = path + "/model" + str(it) + "C" + str(loss) + ".pkl"
                print("Saving current parameters:", currentfile)
                model.save(currentfile)
                smallest = loss

        # Evaluating the model on the validation set
        X_val = np.array(X_val)
        Y_val = np.array(Y_val)
        X_val_reshaped = X_val.reshape((len(X_val), input_size, 1))
        accuracy = model.evaluate(X_val_reshaped, Y_val)[1]
    """
    
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


def new_Benchmarking_CNN(dataset, filename, input_size, optimizer, steps=300, batch_size=20):
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
    

    # Building the CNN model for binary classification of sequential data.
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
    p = 0.5
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
        
    new_Benchmarking_CNN(dataset, filename, input_size = 256, optimizer='nesterov', steps=300, batch_size=50)




    