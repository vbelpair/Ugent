# Basic imports that are required for the smooth use of Python
import numpy as np                # Absolutely necessary
from scipy import signal          # For signal processing tools
from scipy import fftpack as fft  # Fourier transform and spectral analysis
import math as math               # Not always required. But can simplify the matrix/linear algebraical calculations 
import matplotlib.pyplot as plt

import IPython.display as ipd     # used to construct display elements. ipd.display == display (which is imported by default in Jupyter Lab)

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


def IIRFiltOrdN(filtSpec=(1/6,1/np.sqrt(2)),filtOrd=1,lpfFlag=True):
    
    FTs = filtSpec[0]
    B = 2*np.cos(2*np.pi*FTs)
    A = (2 + B)/4
    D = (filtSpec[1])**(2/filtOrd)
    
    rts = np.roots((A-D, -(2*A-B*D), (A-D)))
    (rts,) = rts[np.where(np.abs(rts)<1)]

    if (rts.size>1):
        rts = np.min(rts)
    
    
    b = np.ones(1)
    a = np.ones(1)
    G = ((1 -2*rts + rts**2)/4)**(filtOrd/2) # Gain of the filter (b0 in the notation)
    
    for ordC in np.arange(filtOrd):
        b = np.convolve(b,(1,1))
        a = np.convolve(a,(1,-rts))
    
    if not(lpfFlag):
        b = b*(-np.ones(filtOrd+1))**(np.arange(filtOrd+1))
        a = a*(-np.ones(filtOrd+1))**(np.arange(filtOrd+1))
    b *= G 
    return b,a