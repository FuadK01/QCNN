import pickle
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint
import numpy as np

def loadModel(filename):
    paramfile = open(filename, 'rb')
    params = pickle.load(paramfile)
    paramfile.close()
    return params

def testModel(CNN, filename):
    input_size = 256
    datafile = open(filename, 'rb')
    currentdata = pickle.load(datafile)
    #gets the specific data that it was trained on
    X_train, X_val, X_test, Y_train, Y_val, Y_test = currentdata
    datafile.close()

    test_loss, accuracy = CNN.evaluate(X_test, Y_test)
    print("Accuracy:" + str(accuracy))

if __name__ == "__main__":
    name = input("Enter file name of the model you would like to load: ")
    model =loadModel(name)
    print(model)
    oldData = input("Enter file name of data: ")
    testModel(model,oldData)
    #Use if you wish to test lots of models at once
    '''name = ['CNN_Models\\2023-08-28A0.31011\model280C0.6872777342796326.pkl','CNN_Models\\2023-08-28B0.32021\model220C0.6766580939292908.pkl','CNN_Models\\2023-08-28C0.51040\model340C0.6793094873428345.pkl','CNN_Models\\2023-08-28D52021\model420C0.6618232727050781.pkl','CNN_Models\\2023-08-282E12021\model0C0.6652181148529053.pkl','CNN_Models\\2023-08-28F0.3120121\model440C0.6839501261711121.pkl','CNN_Models\\2023-08-28G0.32080\model200C0.6803209781646729.pkl','CNN_Models\\2023-08-28H0.340120\model260C0.6852693557739258.pkl','CNN_Models\\2023-08-28I0.12021\model200C0.6902934908866882.pkl','CNN_Models\\2023-08-28J0.82021\model460C0.6699413061141968.pkl','CNN_Models\\2023-08-28K0.5100101\model20C0.6859658360481262.pkl','CNN_Models\\2023-08-28L0.31020\model400C0.6666512489318848.pkl','CNN_Models\\2023-08-28M0.51011\model80C0.6830105781555176.pkl','CNN_Models\\2023-08-28N0.52021\model20C0.6791728138923645.pkl','CNN_Models\\2023-08-28O0.31040\model180C0.6803804636001587.pkl','CNN_Models\\2023-08-28P0.540120\model420C0.6803244352340698.pkl','CNN_Models\\2023-08-28Q0.3100101\model280C0.6823588609695435.pkl','CNN_Models\\2023-08-28R0.840120\model240C0.6839205622673035.pkl','CNN_Models\\2023-08-28U0.81040\model260C0.6432438492774963.pkl']
    oldData =['CNN_Data\\2023-08-28A0.31011\dataA0.31011.pkl','CNN_Data\\2023-08-28B0.32021\dataB0.32021.pkl','CNN_Data\\2023-08-28C0.51040\dataC0.51040.pkl','CNN_Data\\2023-08-28D52021\dataD52021.pkl','CNN_Data\\2023-08-282E12021\data2E12021.pkl','CNN_Data\\2023-08-28F0.3120121\dataF0.3120121.pkl','CNN_Data\\2023-08-28G0.32080\dataG0.32080.pkl','CNN_Data\\2023-08-28H0.340120\dataH0.340120.pkl','CNN_Data\\2023-08-28I0.12021\dataI0.12021.pkl','CNN_Data\\2023-08-28J0.82021\dataJ0.82021.pkl','CNN_Data\\2023-08-28K0.5100101\dataK0.5100101.pkl','CNN_Data\\2023-08-28L0.31020\dataL0.31020.pkl','CNN_Data\\2023-08-28M0.51011\dataM0.51011.pkl','CNN_Data\\2023-08-28N0.52021\dataN0.52021.pkl','CNN_Data\\2023-08-28O0.31040\dataO0.31040.pkl','CNN_Data\\2023-08-28P0.540120\dataP0.540120.pkl','CNN_Data\\2023-08-28Q0.3100101\dataQ0.3100101.pkl','CNN_Data\\2023-08-28R0.840120\dataR0.840120.pkl','CNN_Data\\2023-08-28U0.81040\dataU0.81040.pkl']
    for i in range(20):
        print(name[i])
        print(oldData[i])
        model =loadModel(name[i])
        testModel(model,oldData[i])'''
        #CNN_Data\2023-08-28B0.32021\dataB0.32021.pkl
        #CNN_Models\2023-08-28B0.32021\model120C0.489306777715683.pkl
    #accuracy = accuracy_test(Y_pred, Y_val)


