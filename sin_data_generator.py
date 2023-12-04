import numpy as np
import matplotlib.pyplot as plt
import statistics as stat

"""
Generates data from a given sin function with added noise. Simulates the kind of data
recorded in GravWave experiements (LIGO).
    Parameters
    ----------
    snr : float
        The variance that is used on the gaussian to decided how much noise to add.
    freq : list, float
        Frequencies, a list of two values, [start frequency, last frequency]..
    length : int, optional
        Number of outputs generated. Limited by storage space of qubits.

    Raises
    ------
    Exception
        If labels generate incorrectly

    Returns
    -------
    list
        Returns generated data with accompanying labels for each set of data points

"""
def sin_gen(snr, freq, length):
    # Number of data points in each graph
    data_length = 256

    outputs = []
    labels = []

    for i in range(0, length):
        # Randomly decides if the data will be signal (1) or noise (0)
        label = np.random.randint(0,2)
        if label == 0:
            x = np.linspace(0,0,data_length)
            labels.append(label)
            noise = np.random.normal(0, snr, data_length) # Generates random gaussian noise with sigma=snr
            output = x + noise
        
        elif label == 1:
            x = np.linspace(0, 2*np.pi, data_length)
            labels.append(label)
            # Randomised Frequency of the sine wave
            frequency = np.random.randint(freq[0], freq[1])
            phase = np.random.randint(0, data_length) # Randomised phase
            signal = np.sin((frequency * x) + phase)
            noise = np.random.normal(0, snr, data_length)
            output = signal + noise

        else:
            raise Exception("Error encountered whilst producing sin plots. Should be impossible, check random label generator")
        
        output /= np.sqrt(np.sum(np.abs(output)**2))
        outputs.append(output)

    dataset = [outputs, labels]

    return dataset

def multi_sin_gen(snr, freq_1, freq_2, length):
    # Number of data points in each graph
    data_length = 256

    outputs = []
    labels = []

    for i in range(0, length):
        # Randomly decides if the data will be noise (0), signal with freq_1 (1), signal with freq_2 (2)
        label = np.random.randint(0,3)

        if label == 0:
            x = np.linspace(0,0,data_length)
            labels.append(label)
            noise = np.random.normal(0, snr, data_length) # Generates random gaussian noise with sigma=snr
            output = x + noise
        
        elif label == 1:
            x = np.linspace(0, 2*np.pi, data_length)
            labels.append(label)
            # Randomised Frequency of the sine wave
            frequency = np.random.randint(freq_1[0], freq_1[1])
            phase = np.random.randint(0, data_length) # Randomised phase
            signal = np.sin((frequency * x) + phase)
            noise = np.random.normal(0, snr, data_length)
            output = signal + noise
        
        elif label == 2:
            x = np.linspace(0, 2*np.pi, data_length)
            labels.append(label)
            # Randomised Frequency of the sine wave
            frequency = np.random.randint(freq_2[0], freq_2[1])
            phase = np.random.randint(0, data_length) # Randomised phase
            signal = np.sin((frequency * x) + phase)
            noise = np.random.normal(0, snr, data_length)
            output = signal + noise

        else:
            raise Exception("Error encountered whilst producing sin plots. Should be impossible, check random label generator")
        
        output /= np.sqrt(np.sum(np.abs(output)**2))
        outputs.append(output)

    dataset = [outputs, labels]

    return dataset


def multi_plot_gen(snr, freq_1, freq_2, freq_3, length):
    # Number of data points in each graph
    data_length = 256

    outputs = []
    labels = []

    for i in range(0, length):
        # Randomly decides if the data will be noise (0), sinusoidal signal with freq_1 (1), combined sinusoidal signal with freq_2 (2),
        # and exponential signal with freq_3 (3)
        label = np.random.randint(0,4)

        if label == 0:
            x = np.linspace(0,0,data_length)
            labels.append(label)
            noise = np.random.normal(0, snr, data_length) # Generates random gaussian noise with sigma=snr
            output = x + noise
        
        elif label == 1:
            x = np.linspace(0, 2*np.pi, data_length)
            labels.append(label)
            # Randomised Frequency of the sine wave
            frequency = np.random.randint(freq_1[0], freq_1[1])
            phase = np.random.randint(0, data_length) # Randomised phase
            signal = np.sin((frequency * x) + phase)
            noise = np.random.normal(0, snr, data_length)
            output = signal + noise
        
        elif label == 2:
            x = np.linspace(0, 2*np.pi, data_length)
            labels.append(label)
            # Randomised Frequency of the sine wave
            frequency_1 = np.random.randint(freq_2[0], freq_2[1])
            frequency_2 = np.random.randint(freq_2[0], freq_2[1])
            phase_1 = np.random.randint(0, data_length) # Randomised phase
            phase_2 = np.random.randint(0, data_length) # Randomised phase
            signal = np.sin((frequency_1 * x) + phase_1) + np.sin((frequency_2 * x) + phase_2)
            noise = np.random.normal(0, snr, data_length)
            output = signal + noise

        elif label == 3:
            x = np.linspace(0, 2*np.pi, data_length)
            labels.append(label)
            # Randomised Frequency of the sine wave
            frequency_1 = np.random.randint(freq_3[0], freq_3[1])
            frequency_2 = np.random.randint(freq_3[0], freq_3[1])
            phase_1 = np.random.randint(0, data_length) # Randomised phase
            phase_2 = np.random.randint(0, data_length) # Randomised phase
            signal = np.cos(frequency_1 * x + frequency_2 * x ** 2 + phase_1)
            noise = np.random.normal(0, snr, data_length)
            output = signal + noise

        else:
            raise Exception("Error encountered whilst producing sin plots. Should be impossible, check random label generator")
        
        output /= np.sqrt(np.sum(np.abs(output)**2))
        outputs.append(output)

    dataset = [outputs, labels]

    return dataset


if __name__ == "__main__":
    sin_gen(0.5, [10,30], 10)
    multi_sin_gen(0.5, [10, 30], [60, 80], 10)
    multi_plot_gen(0.5, [10, 30], [60, 80], [100, 120], 10)
