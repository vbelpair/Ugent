import matplotlib.pyplot as plt
import numpy as np

def plot_V_at_z(V, t, Z, n):

    v = V[n,:]

    fig, ax = plt.subplots()
    ax.plot(t, v)
    ax.set_xlabel('time [s]')
    ax.set_ylabel('Voltage [V]')
    ax.set_title(f'Voltage at z = {Z}m')
    plt.savefig("exampletime.png")


def plot_V_at_t(V, z, T, m):

    v = V[:,m]

    fig, ax = plt.subplots()
    ax.plot(z, v)
    ax.set_xlabel('z-position [m]')
    ax.set_ylabel('Voltage [V]')
    ax.set_title(f'Voltage at t = {T}s')
    plt.savefig("exampleposition.png")

if __name__ == "__main__":

    # plotting source signal

    H = lambda t: 1*np.greater(t, 0) # heavie side function
    def B(a,b):
        # block function: a < b
        return lambda t: H(t-a) - H(t-b)

    # define source signal

    t1, t2, t3, t4 = 5, 6, 10, 11
    A, tr = 3, 1
    eg = lambda t: A/tr*(t-t1)*B(t1, t2)(t) + A*B(t2,t4)(t) - A/tr*(t-t3)*B(t3,t4)(t)

    t = np.arange(30)
    plt.plot(t, eg(t))
    plt.show()



