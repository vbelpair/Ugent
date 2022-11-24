import matplotlib.pyplot as plt


import numpy as np
import matplotlib.pyplot as plt

update = {"text.usetex": True, 'text.latex.preamble': r'\usepackage{cmbright}'}
#update2 = {"text.usetex": True}
plt.rcParams.update(update)

def splot(X, Y, title = '', axis = ['', ''], sname = ''):
    '''
    Simple Plot Function
    
    X : List of arrays containing plotting data on x-axis
    Y : Corresponding y values
    
    title      : title of plot
    axis       : list with two elements containing name of x- and y axis 
    sname      : name of saved plot file
    '''
    fig, ax = plt.subplots(1, figsize=(8, 4), dpi = 250)
    
    ax.set_title(title, size = 14)
    
    plt.xlabel(axis[0], size = 12)
    plt.ylabel(axis[1], size = 12)
    
    for x, y in zip(X, Y):
        ax.plot(x, y, lw = 2)
        
    plt.grid()

    plt.tight_layout()
    if sname != '':
        plt.savefig(sname, transparent = True)
    plt.show()

### TEST CODE ###
import numpy as np
x1 = np.linspace(-np.pi, np.pi, 100)
y1 = (lambda x: x**2)(x1)
y2 = (lambda x: x)(x1)

X = np.array([x1, x1])
Y = np.array([y1, y2])

import os
path = os.path.dirname(__file__)
path += '/figures/test.png'
print(path)

splot(X, Y, title = 'Amazing project '+ r'$\nabla \varphi = \vec{\partial}_{x}\varphi + \vec{\partial}_y\varphi + \vec{\partial}_z\varphi$', sname = path)