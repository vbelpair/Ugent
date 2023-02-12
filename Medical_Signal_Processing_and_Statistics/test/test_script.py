# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft
#from scipy.signal.windows import hann

srate = 400
dt = 1/srate
duration = 10
t = np.arange(0,duration,dt)

s1 = np.sin(50 * np.pi * t)
s2 = 2 * np.sin(25 * np.pi* t)
s3 = 10 * np.sin(10 * np.pi* t)
s = s1 + s2 + s3

#w = hann(len(s))
#s_w = s * w

df = 1/(len(s)/srate)

s_fft = np.fft.fftshift(np.abs(fft(s))) * 1/len(s)
freqs = np.fft.fftshift(np.fft.fftfreq(len(s_fft), 1/srate))

plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"  
fig, ax = plt.subplots(2, figsize=(9,6)) 

ax[0].plot(t,s,'b', linewidth=1) 
ax[0].set_xlim(0,10) 
ax[0].set_ylim([-40,40]) 
ax[0].set_xticks([0, 2, 4, 6, 8, 10]) 
ax[0].set_xticklabels([0, 2, 4, 6, 8, 10]) 
ax[0].xaxis.set_tick_params(width=2) 
ax[0].tick_params(axis="x", labelsize=14) 
ax[0].tick_params(axis="y", labelsize=14) 
ax[0].spines['top'].set_visible(False)
ax[0].spines['right'].set_visible(False) 
ax[0].spines['bottom'].set_linewidth(2) 
ax[0].spines['left'].set_linewidth(2) 
ax[0].set_ylabel('Amplitude', fontsize=16, fontweight='bold') 
ax[0].set_xlabel('Time (s)', fontsize=16, fontweight='bold') 

ax[1].plot(freqs,s_fft,'b', linewidth=1) 
ax[1].set_xlim(-200,200)
ax[1].set_yticks([0, 5, 10]) 
ax[1].set_yticklabels([0, 5, 10]) 
ax[1].xaxis.set_tick_params(width=2)
ax[1].tick_params(axis="x", labelsize=14)
ax[1].tick_params(axis="y", labelsize=14)
ax[1].spines['top'].set_visible(False)
ax[1].spines['right'].set_visible(False)
ax[1].spines['bottom'].set_linewidth(2)
ax[1].spines['left'].set_linewidth(2)

ax[1].set_ylabel('Magnitude', fontsize=16, fontweight='bold')
ax[1].set_xlabel('Frequency (Hz)', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('testplot.png', dpi=300)





