import numpy as np

def leapfrog(parameters):

    P = parameters
    """
    l:  length of the line [m]
    v:  signal speed [m/s]
    Zc: characteristic impedance of the line [Ohm]
    Zg: generator impedance [Ohm]
    Zl: load impedance [Ohm]
    Cl: load capacitance [F]
    A:  amplitude of the voltage source [V]
    Tb: bit duration [s]
    tr: bit rise time [s]
    d:  bit delay [s]
    N:  number of cells 
    dt: time step [s]
    M:  number of time steps
    """
    l, v, Zc, Zg, Zl, Cl = P[0], P[1], P[2], P[3], P[4], P[5]
    A, Tb, tr, d = P[6], P[7], P[8], P[9]
    N, dt, M = int(P[10]), P[11], int(P[12])

    # define adjusted current and voltage
    V = np.zeros((N+1,M+1))
    i = np.zeros((N,M)) # remark i = Zc*I

    # define constants for leapfrog scheme

    dz = l/N
    alpha = v*dt/dz
    print(f'alpha value: {alpha}')
    rg, rl = Zc/Zg, Zc/Zl# adjusted Zc with respect to Zg and Zl
    C1, C2 = (1-alpha*rg)/(1+alpha*rg), 2*alpha/(1+alpha*rg) # constants for BC at z = 0
    C3, C4 = (1-alpha*rl)/(1+alpha*rl), 2*alpha/(1+alpha*rl) # constants for BC at z = N*dz

    # define mathematical expressions

    H = lambda t: 1*np.greater(t, 0) # heavie side function
    def B(a,b):
        # block function: a < b
        return lambda t: H(t-a) - H(t-b)

    # define source signal

    t1, t2, t3, t4 = d, d+tr, d+Tb, d+tr+Tb
    eg = lambda t: A/tr*(t-t1)*B(t1, t2)(t) + A*B(t2,t4)(t) - A/tr*(t-t3)*B(t3,t4)(t)

    for m in range(M-1):

        V[0,m+1] = C1*V[0,m] + C2*(rg*eg((m+0.5)*dt) - i[0,m])
        i[:,m] = i[:,m-1] + alpha*(V[:N,m] - V[1:,m])
        V[1:N,m+1] = V[1:N,m] + alpha*(i[0:N-1,m] - i[1:N,m])
        V[N,m+1] = C3*V[N,m] + C4*i[N-1,m]

    I = 1/Zc*i # readjust i to I
    t = dt*np.arange(M+1) # time for plotting
    z = dz*np.arange(N+1) # space for plotting
    
    return I, V, z, t, eg
