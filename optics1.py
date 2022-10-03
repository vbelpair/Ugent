import numpy as np

def R(angle):
    C, S = np.cos(angle), np.sin(angle)
    return np.array([[C,-S],[S,C]])

class Space:

    def __init__(self, h, dim, n=1):
        self.dim = dim
        self.h = h
        self.N = np.full(dim, n, dtype=float)

    def add_material(self, geometry, n):

        h = self.h
        Ny, Nx = self.dim
        
        
        for j in range(Nx):
            for i in range(Ny):
                x, y = h*j, h*(Ny-i)
                if geometry(x,y):
                    self.N[i,j] = n


class Lightbeam:

    def __init__(self, v, y0, tau):
        self.v = v/(v@v)**(0.5) # normalization
        self.y0 = y0
        self.tau = tau

    def propagate(self, space, T):
        v, y0, tau = self.v, self.y0, self.tau
        N, h = space.N, space.h
        Ny, Nx = space.dim

        L = np.zeros((T+1, 2))
        L[0] = [0, y0]

        for t in range(T):
            L[t+1] = v*tau + L[t]
            j1, i1 = int(L[t,0]/h), int(Ny-L[t,1]/h)
            j2, i2 = int(L[t+1,0]/h), int(Ny-L[t+1,1]/h )
            if (j1 >= Nx) | (j2 >= Nx) | (i1 >= Ny) | (i2 >= Ny):
                break
            if N[i1,j1] != N[i2,j2]:
                ni, nr = N[i1,j1], N[i2,j2]
                for k in range(1,100):
                    Un = N[i2-1, j2-k:j2+1+k] # upper row
                    print("U", Un)
                    if (Un[0] == ni) & (nr in Un):
                        Uj = k-np.min(np.where(Un == nr))
                        break
                for k in range(1,100):
                    Ln = N[i2+1, j2-k:j2+1+k] # lower row
                    print("L", Ln)
                    if (Ln[0] == ni) & (nr in Ln):
                        Lj = k-np.min(np.where(Ln == nr))
                        break
                n = np.array([1, (Lj-Uj)/2])
                n = n/(n@n)**(0.5) # normalization
                cosi = n@v
                thetai = np.degrees(np.arccos(cosi))
                if thetai > 90:
                    thethai = thetai%90
                    n = -n
                sinr = ni/nr*np.sin(np.radians(thetai))
                thetar = np.arcsin(sinr)
                print(thetai % 90, sinr, np.degrees(thetar))
                v = R(thetar)@n
        return L


import matplotlib.pyplot as plt

a, b = 1, 2
H, B = 480, 1000
h = 10**(-2)
x0, y0 = B/4*h, H/2*h
geometry = lambda x, y: (abs(((x-x0)/a)**2-((y-y0)/b)**2) <= 1) & (abs(y) <= H*h)
#geometry = lambda x, y: abs((y-y0)-2*(x-x0)) <= 2

space = Space(h, (H,B))
n = 1.4
space.add_material(geometry, n)

light = Lightbeam(np.array([1,0]), y0+70*h, h)
L = light.propagate(space, B)
x, y = L[:, 0]/h, L[:, 1]/h
N = space.N
N[np.where(N==n)] = 2

plt.plot(x,y, color="red", lw=2)
plt.imshow(N)
plt.savefig("N.png")