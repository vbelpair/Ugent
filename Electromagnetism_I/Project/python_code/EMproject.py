#=============================================================#
# Welcome to EMproject.py!
# The purpose of this python file is to calculate the currents 
# and voltages on a transmission line given certain boundary 
# conditions.
# This file is created by: - Bram Popelier
#                          - Constantijn Coppers
#                          - Vincent Belpaire
#=============================================================#

#==================#
# importing modules
#==================#

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os 

#=========================#
# defining argument parser
#=========================#

parser = argparse.ArgumentParser(description = 'Input data file to run programm.')
parser.add_argument('file', type=str, help = 'Enter the file in which the data is contained to run the leapfrog scheme.', nargs='?', default=None)

#===========================#
# creating usefull functions
#===========================#

# defining some mathematical functions

H = lambda t: 1*np.greater(t, 0) # heavie side function
def B(a,b):
    # block function (a < b)
    return lambda t: H(t-a) - H(t-b)

# define source signal (the bit)

def bit(A, tr, th, tf, D=0):
    """
    A: voltage amplitude [V]
    tr: rise time [s]
    th: half-amplitude pulse width [s]
    tf: fall time [s]
    D: delay [s]
    """
    t1, t2, t3, t4 = D, D+tr, D+1/2*tr+th-1/2*tf, D+1/2*tr+th+1/2*tf
    return lambda t: A/tr*(t-t1)*B(t1, t2)(t) + A*B(t2,t4)(t) - A/tr*(t-t3)*B(t3,t4)(t)

# allow more LaTeX in plots
# make sure LaTeX is installed on your computer, if not uncheck this

update = {"text.usetex": True, 'text.latex.preamble': r'\usepackage{cmbright}'}
plt.rcParams.update(update)

# define plotter
def splot(X, Y, title = '', lb = None, axis = ['', ''], sname = '', tsp = True):
    '''
    Simple Plot Function
    
    X : List of arrays containing plotting data on x-axis
    Y : Corresponding y values
    aadede
    title      : title of plot
    axis       : list with two elements containing name of x- and y axis 
    sname      : name of saved plot file
    tsp        : transparent background (default : True)
    '''
    
    ## make a figure with axes
    fig, ax = plt.subplots(1, figsize=(8, 4))
    
    ## make plot title
    ax.set_title(title, size = 16)
    
    ## lable axis
    plt.xlabel(axis[0], size = 14)
    plt.ylabel(axis[1], size = 14)
    
    ## plot the data
    if type(X) == list and len(X) > 1:
        if lb == None:
            lb = ['']*len(X)
        for i in range(len(X)):
            ax.plot(X[i], Y[i], lw = 2, label = lb[i])
    else:
        ax.plot(X, Y, lw = 2, label = lb)
    
    plt.legend()
    plt.grid()
    plt.tight_layout()
    
    if sname != '':
        plt.savefig(sname, transparent = tsp, dpi = 300)
    plt.show()

#=========================#
# creating leapfrog scheme
#=========================#

def leapfrog_scheme(alpha, Rc, Eg, Rg, Rl, Cl, dt, M, N):
    """
    alpha: courant factor of the transmission line
    Rc: characteristic impedance of the transmission line
    Eg: source signal
    Rg: generator impedance
    Rl: load impedance
    Cl: load capacitance
    M: number of time steps
    N: number of cells
    """

    # define adjusted current and voltage
    V = np.zeros((N+1,M+1))
    i = np.zeros((N,M)) # remark i = Rc*I

    # define constants for leapfrog scheme

    rg, Z1, Z2 = Rc/Rg, Rc*(1/Rl-2*Cl/dt), Rc*(1/Rl+2*Cl/dt) # adjusted Rc with respect to Rg and Rl
    C1, C2 = (1-alpha*rg)/(1+alpha*rg), 2*alpha/(1+alpha*rg) # constants for BC at z = 0
    C3, C4 = (1-alpha*Z2)/(1+alpha*Z1), 2*alpha/(1+alpha*Z1) # constants for BC at z = N*dz

    for m in range(M-1):

        i[:,m] = i[:,m-1] + alpha*(V[:N,m] - V[1:,m])
        V[0,m+1] = C1*V[0,m] + C2*(rg*Eg[m] - i[0,m])
        V[1:N,m+1] = V[1:N,m] + alpha*(i[0:N-1,m] - i[1:N,m])
        V[N,m+1] = C3*V[N,m] + C4*i[N-1,m]

    I = 1/Rc*i # readjust i to I
    return V, I

#============================#
# creating plotting functions
#============================#
 
def plot_time(V, t, Z=0, n = None, name = "exampletime.png"):

    if n:
        V = V[n,:]

    laxis = ['time [ns]', 'Voltage [V]']
    ttl = f'Voltage at $z = {Z}$ m' 
    path = os.path.dirname('__file__')  + 'figures/'  + name
    splot(t*10**9, V, axis = laxis, title = ttl, sname = path)

def plot_space(V, z, T, m, name = "exampleposition.png"):

    laxis = ['$z$-position [m]', 'Voltage [V]']
    ttl = f'Voltage at $t = {T*10**9:.2f}$ ns' 
    path = path = os.path.dirname('__file__') + 'figures/' + name
    splot(z, V[:,m], axis = laxis, title = ttl, sname = path) 

def plot_animation(V, z, t):
    # uncheck this if the video does not run (mac)
    #plt.rcParams["backend"] = "TkAgg"
    t *= 10**9

    fig = plt.figure(figsize=(8,4))

    def plot_initialize():
        plt.ylim(-1, 1)
        plt.xlabel('$z$-coordinate [m]', fontsize=12)
        plt.ylabel('Voltage [V]', fontsize=12)

    def animate(i):
        
        plt.clf()
        plot_initialize()
        plt.plot(z, V[:,i], label = f'|{np.max(np.abs(V[:, i])):.2f}|')
        plt.title(f'Voltage at {t[i]:.2f} ns', fontsize=16)
        plt.legend()
        plt.grid()
    
    ani = animation.FuncAnimation(fig, animate, interval = 50, frames = np.arange(len(t)))
    ani.save('animation.mp4')


#=====================#
# running the programm
#=====================# 

if __name__ == "__main__":

    args = parser.parse_args()
    file = args.file

    if file is None:
        print("No file argument was given.")
        file = input("Please enter a file: ")

    with open(file, 'r') as f:
        parameters = np.loadtxt(file)
        f.close()

    # unpacking the parameters

    d = parameters[0]
    v = parameters[1]
    Rc = parameters[2]
    Rg = parameters[3]
    Rl = parameters[4]
    Cl = parameters[5]
    A = parameters[6]
    Tbit = parameters[7]
    tr = parameters[8]
    D = parameters[9]
    N = int(parameters[10])
    dt = parameters[11]
    M = int(parameters[12])
    z_sensor = parameters[13]
    m_snapchot = int(parameters[14])

    # creating other constants

    dz = d/N
    alpha = v*dt/dz

    # creating time and space coordinates

    t = dt*np.arange(M+1)
    z = dz*np.arange(N+1)

    # executing the leapfrog scheme

    Eg = bit(A, tr, Tbit, tr, D)(t[:-1]+dt/2)
    V, I = leapfrog_scheme(alpha, Rc, Eg, Rg, Rl, Cl, dt, M, N)

    # plotting the output

    plot_time(V, t, z_sensor, int(z_sensor/dz), name=file[:-4]+"time.png")
    plot_space(V, z, dt*m_snapchot, m_snapchot, name=file[:-4]+"position.png")
    plot_animation(V,z,t)

