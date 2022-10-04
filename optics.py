import numpy as np

P = lambda t: [np.cosh(t), 10*np.sinh(t)]
t = np.linspace(-1,1,20)

x = np.linspace(-5,5,500)
y = np.linspace(-5,5,500)

X, Y = np.meshgrid(x,y)

def R(angle):
    C, S = np.cos(angle), np.sin(angle)
    return np.array([[C, -S], [C, S]])

class Space2D:

    def __init__(self, dim, n=1): 
        self.y = np.linspace(0,1,dim[0])
        self.x = np.linspace(0,1,dim[1])
        self.N = np.full(dim, n)
        self.deltax = x[1] - x[0]
        self.deltay = y[1] - y[0]

    def add_layer(self, condition, n):
        R, C = self.N.shape

        for i in range(R):
            for j in range(C):
                if condition(self.x[j], self.y[i]):
                    self.N[i,j] = n
    

class Lightbeam:

    def __init__(self, y0, l=np.array([1,0])):
        self.l = l/np.sqrt(l@l) # normalizing vector
        self.y0 = y0

    def propagate(self, space, T):
        
        N = space.N
        x, y = space.x, space.y
        Nx, Ny = len(x), len(y)
        h, k = space.deltax, space.deltay

        L = np.zeros((T+1,2))
        L[0] = [x[0], self.y0]
        l = self.l

        for t in range(T-1):
            r = l*t+L[t]
            i1, j1 = int(Ny - L[t,1]/k), int(L[t,0]/h)
            i2, j2 = int(Ny - r[1]/k), int(r[0]/h)
            L[t+1] = r
            if N[i1, j1] != N[i2, j2]:
                
                theta_r = N[i1,j1]/N[i2, j2]*(1-l@n)**(0.5)
                
                

import matplotlib.pyplot as plt


condition = lambda x, y: abs(y-1.5*x) < 0.4
space = Space2D((480,480))
N = space.N
space.add_layer(condition, n=2)
light = Lightbeam(0)
L = light.propagate(space)


#plt.imshow(L+N)
#plt.show()

#x = np.linspace(-1,1,10)
#y = np.linspace(-1,1,10)
#X, Y = np.meshgrid(x,y)

#print(X)
#print(Y)