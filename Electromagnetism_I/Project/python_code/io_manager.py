import numpy as np
import time

def readdata_numpy(file, performance=False):
    """
    file: file to be loaded
    performance: if True prints the programm and process time
    """

    start_time = time.time()
    start_process = time.clock()

    with open(file, 'r') as f:
        parameters = np.loadtxt(file)
        f.close()

    if performance:
        print("programm time: ", time.time() - start_time)
        print("process time: ", time.clock() - start_process)

    return parameters

def readdata(file, performance=False):
    """
    file: file to be loaded
    performance: if True prints the programm and process time
    """

    start_time = time.time()
    start_process = time.clock()

    parameters = []

    with open(file, 'r') as f:
        f.readline()
        parameters = np.loadtxt(file)
        for i in range(len(parameters)):
            parameters[i] = float(f.readline().split(' ')[0])
        f.close()

    if performance:
        print("programm time: ", time.time() - start_time)
        print("process time: ", time.clock() - start_process)

    return parameters


""""
Little performance analysis (for one small documents, i.e. example.txt):

    - Standard python run:
        > first run:
            programm time:  0.005000591278076172
            process time:  0.005152900000000016
        > second run:
            programm time:  0.004998922348022461
            process time:  0.005046500000000009
        > third run:
            programm time:  0.0040035247802734375
            process time:  0.005018999999999996
    - Numpy run:
        > first run:
            programm time:  0.005001068115234375
            process time:  0.004916300000000012
        > second run:
            programm time:  0.004993438720703125
            process time:  0.005008699999999977
        > third run:
            programm time:  0.005002260208129883
            process time:  0.005027400000000015

Conclusion:
    No big difference can be found. Standard python has a small better performance at the programm time while numpy has a small better performance at process time. 
"""