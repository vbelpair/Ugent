import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

def plot_time(V, t, Z=0, n=None, name="exampletime.png"):

    if n:
        V = V[n,:]

    fig, ax = plt.subplots()
    ax.plot(t, V)
    ax.set_xlabel('time [s]')
    ax.set_ylabel('Voltage [V]')
    ax.set_title(f'Voltage at z = {Z}m')
    plt.savefig(name)


def plot_space(V, z, T, m, name="exampleposition.png"):

    fig, ax = plt.subplots()
    ax.plot(z, V[:,m])
    ax.set_xlabel('z-position [m]')
    ax.set_ylabel('Voltage [V]')
    ax.set_title(f'Voltage at t = {T}s')
    plt.savefig(name)


def plot_animation(s, z, t):
    """
    s: signal
    z: space coordinate
    t: time
    """
    fig, ax = plt.subplots()
    ax.set_xlabel('z-position [m]')
    ax.set_ylabel('Voltage [V]')
    M = len(t)

    # function that draws each frame of the animation
    def animate(m):

        ax.clear()
        ax.plot(s[:,m], z)
        ax.set_ylim([np.min(s)-abs(np.std(s)),np.max(s)+abs(np.std(s))])

    anim = FuncAnimation(fig, animate, frames=20, interval=500, repeat=False)

    anim.save('animation')

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

    


