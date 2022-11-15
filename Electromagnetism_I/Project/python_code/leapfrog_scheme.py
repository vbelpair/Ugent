import numpy as np

def leapfrog(I,V,Rl,Rg,Eg,L,C,dz,dt):
    """
    I: current
    V: voltage
    Rl: resistance at load
    Rg: resistance at generator
    L: p.u.l. inductance
    C: p.u.l. capacitance
    dz: space step
    dt: time step
    """

    # define constants for leapfrog scheme

    v = 1/(L*C)**(1/2)
    alpha = v*dt/dz
    Rc = (L/C)**(1/2)
    rg, rl = Rc/Rg, Rc/Rl# adjusted Rc with respect to Rg and RL
    C1, C2 = (1-alpha*rg)/(1+alpha*rg), 2*alpha(1+alpha*rg) # constants for BC at z = 0
    C3, C4 = (1-alpha*rl)/(1+alpha*rl), 2*alpha(1+alpha*rl) # constants for BC at z = N*dz

    N, M = I.shape()
    i = Rc*I # adjusted I

    for m in range(M-1):

        V[0,m+1] = C1*V[0,m] + C2*(rg*Eg[m] - i[0,m])
        i[:,m] = i[:,m-1] + alpha*(V[:N,m] - V[1:,m])
        V[1:N,m+1] = V[1:N,m] + alpha*(i[0:N-1,m] - i[1:N,m])
        V[N,m+1] = C3*V[N,m] + C4*i[N-1,m]

    I = 1/Rc*i # readjust i to I

    return I, V
