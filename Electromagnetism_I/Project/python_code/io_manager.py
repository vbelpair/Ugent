import numpy as np

def readdata_numpy(file, performance=False):
    """
    file: file to be loaded
    """

    with open(file, 'r') as f:
        parameters = np.loadtxt(file)
        f.close()

    return parameters