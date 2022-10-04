import numpy as np

def R(angle):
    C, S = np.cos(angle), np.sin(angle)
    return np.array([[C,-S],[S,C]])

class Field:

    def __init__(self, h, dim, n=1):
        self.dim = dim
        self.h = h
        self.F = np.full(dim, n, dtype=float)

    def add_material(self, geometry, n):

        h = self.h
        Ny, Nx = self.dim
        
        
        for j in range(Nx):
            for i in range(Ny):
                x, y = h*j, h*(Ny-i)
                if geometry(x,y):
                    self.F[i,j] = n

    def get(self):
        return self.F


class Lightbeam:

    def __init__(self, v, y0, tau):
        self.v = v/(v@v)**(0.5) # normalization
        self.y0 = y0
        self.tau = tau

    def propagate(self, F, T):
        v, y0, tau = self.v, self.y0, self.tau
        N, h = F.get(), F.h
        Ny, Nx = F.dim

        L = np.zeros((T+1, 2))
        L[0] = [0, y0]

        for t in range(T):
            L[t+1] = v*tau + L[t]
            j1, i1 = int(L[t,0]/h), int(L[t,1]/h)
            j2, i2 = int(L[t+1,0]/h), int(L[t+1,1]/h )
            if (j1 >= Nx) | (j2 >= Nx) | (i1 >= Ny) | (i2 >= Ny):
                L = L[:t+1]
                break
            if N[i1,j1] != N[i2,j2]:
                ni, nr = N[i1,j1], N[i2,j2]
                D = 20*int(Nx/Ny)
                for k in range(1,100):
                    Un = N[i2+D, j2-k:j2+1+k] # upper row
                    #print('U', Un)
                    if (Un[0] == ni) & (nr in Un):
                        Uj = k-np.min(np.where(Un == nr))
                        break
                for k in range(1,100):
                    Ln = N[i2-D, j2-k:j2+1+k] # lower row
                    #print('L', Ln)
                    if (Ln[0] == ni) & (nr in Ln):
                        Lj = k-np.min(np.where(Ln == nr))
                        break
                n = np.array([1, -(Uj-Lj)/(2*D)])
                #print(n)
                n = n/(n@n)**(0.5) # normalization
                c = n@v
                nr = ni/nr
                if c < 0:
                    n = -n
                    c = -c
                sinr = (nr**2)*(1-(c**2))
                if sinr > 1:
                    v = 0
                else:
                    v = nr*v + (nr*c-(1-sinr)**0.5)*n
                    v = v/(v@v)**(0.5)
        return L


import matplotlib.pyplot as plt

a, b = 0.5, 2
H, B = 1440, 3440
h = 10**(-2)
x0, y0 = B/4*h, H/2*h
#geometry = lambda x, y: (abs(((x-x0)/a)**2-((y-y0)/b)**2) <= 1) & (abs(y-y0) <= 0.35*H*h)
#geometry = lambda x, y: abs((y-y0)-2*(x-x0)) <= 2
#geometry = lambda x, y: abs(((x-x0)/a)**2+((y-y0)/b)**2) < 0.35*H*h

def ellips(a,b,x0,y0):
    return lambda x,y: abs(((x-x0)/a)**2+((y-y0)/b)**2) < 0.35*H*h

def hyperbole(a,b,x0,y0):
    return lambda x,y: (abs(((x-x0)/a)**2-((y-y0)/b)**2) <= 1) & (abs(y-y0) <= 0.5*H*h)

N = Field(h, (H,B))
n = 1.7
#N.add_material(hyperbole(a,b,x0,y0), n)
N.add_material(ellips(a,b,x0,y0), n)

for i in range(0, 2):
    light = Lightbeam(np.array([np.cos(-np.pi*i/9),np.sin(-np.pi*i/9)]), H/2*h+200*h, h)
    L = light.propagate(N, B)
    x, y = L[:, 0]/h, L[:, 1]/h
    plt.plot(x,y, color="white", lw=2)
N = N.get()
N[np.where(N==n)] = 2

plt.plot(np.arange(0,B,1), np.full(B,H/2), ls='--')
plt.imshow(N, cmap='RdBu', origin='lower')
plt.savefig("N.png")