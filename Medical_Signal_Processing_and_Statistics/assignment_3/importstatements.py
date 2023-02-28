# Basic imports that are required for the smooth use of Python
import numpy as np                # Absolutely necessary
from scipy import signal          # For signal processing tools
from scipy import fftpack as fft  # Fourier transform and spectral analysis
import math as math               # Not always required. But can simplify the matrix/linear algebraical calculations 
import matplotlib.pyplot as plt

#import itertools                  # used to iterate over all combinations of parameters for a certain plot
# alternative: plotly - also for jupyterLab

## WAV files
from scipy.io import wavfile
import warnings                   # used to ignore some warnings in WAV-file reading
warnings.simplefilter(action='ignore', category=wavfile.WavFileWarning)

## Loading Matlab files
from scipy.io import loadmat  # To load a matlab data file.

## Breakpoints (if needed) for debugging
import pdb     # Setting breakpoints for debugging

## HTML/CSS styling
from IPython.display import HTML
def css_styling():
    return HTML(open("../assets/styles/custom.css",'r').read())


## Plot parameters
#plt.ioff()                        # interactive mode off -- this means you need plt.show() to show the plots
#plt.rcParams['figure.figsize'] = [12, 8]
#plt.rcParams.update({'font.size': 22})

# Define the Pole-Zero plot function here.

def PoleZeroPlot(B,A=(1,)):
    zeroes = np.roots(np.array(B))
    poles = np.roots(np.array(A))

    #print('The zeroes of H(z) are: '+'\t'.join(['{:3.3f}'.format(x) for x in zeroes]))
    #print('The poles of H(z) are: '+'\t'.join(['{:3.3f}'.format(x) for x in poles]))
    print('Poles are plotted using ''x'' and zeros as filled in circles ''o''.') 
    fig = plt.figure(); fig.clf()
    ax = fig.add_subplot(111)
    ax.plot(np.real(zeroes),np.imag(zeroes),'o',color='b',markersize=8)
    ax.plot(np.real(poles),np.imag(poles),'x',color='r',markersize=8)
    ax.set_xlim((-1.5,1.5))
    ax.set_ylim((-1.5,1.5))
    # Plot the unit circle
    ax.plot(np.cos(np.arange(0,2*np.pi,np.pi/100)),np.sin(np.arange(0,2*np.pi,np.pi/100)),\
            color='k')
    ax.set_aspect('equal')
    #plt.axes().set_aspect('equal')
    ax.grid('on')
    #fig.show()