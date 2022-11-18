# python modules
import argparse
import os

# self made modules
import leapfrog_scheme as ls
import io_manager as io
import plot_manager as pm

# define argument parsers
parser = argparse.ArgumentParser(description='Input data file to run programm.')
parser.add_argument('file', type=str, help='Enter the file in which the data is contained to run the leapfrog scheme.', nargs='?', default=None)

if __name__ == "__main__":

    args = parser.parse_args()
    file = args.file

    if file is None:
        print("No file argument was given.")
        file = input("Please enter a file: ")

    parameters = io.readdata_numpy(file)
    I, V, z, t = ls.leapfrog(parameters[:-2])

    """
    Z:  position of the sensor [m]
    T: snapshot time
    """
    Z, T = parameters[-2], parameters[-1]
    n, m = int(Z/z[1]), int(T)
    pm.plot_V_at_t(V,z,t[1]*T,m)
    pm.plot_V_at_z(V,t,Z,n)

# hello this is the end of the code




