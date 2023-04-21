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


## Plot parameters
#plt.ioff()                        # interactive mode off -- this means you need plt.show() to show the plots
#plt.rcParams['figure.figsize'] = [12, 8]
#plt.rcParams.update({'font.size': 22})