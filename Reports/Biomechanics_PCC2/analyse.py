from audioop import mul
import os
from symbol import parameters
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dirname = os.path.dirname(__file__)


def retrieve_data(filename, delimiter=',', dirname=dirname):

    file = os.path.join(dirname, filename)

    with open(file, 'r') as f:
        data = pd.read_csv(f, delimiter=delimiter, index_col=0)
        f.close()

    return data

CM = retrieve_data('Coarse-Mesh')
CM['mesh'] = 'Coarse'
MM = retrieve_data('Medium-Mesh')
MM['mesh'] = 'Medium'
FM = retrieve_data('Fine-Mesh')
FM['mesh'] = 'Fine'


'''
M = pd.concat([CM, MM, FM])
P = M.keys()[-3]
print(P)

fig, ax = plt.subplots(figsize=(10,7))

M.boxplot(column=P, by='mesh', ax=ax)
plt.savefig(os.path.join(dirname, 'figures/TP_boxplot.png'))
'''

# mesh sensitivity analysis

M = [CM, MM, FM]
P = CM.keys()[-3]

N = [len(m) for m in M]
q05 = [m[P].quantile(0.05) for m in M]
mu = [m[P].mean() for m in M]
q95 = [m[P].quantile(0.95) for m in M]

'''
figfig, ax = plt.subplots(figsize=(10,7))

ax.scatter(N,q05,marker='o',c='red',label='0.05-percentile')
ax.scatter(N,mu,marker='o',c='green',label='average')
ax.scatter(N,q95,marker='o',c='blue',label='0.95-percentile')

ax.plot(N,q05,ls='--',c='red')
ax.plot(N,mu,ls='--',c='green')
ax.plot(N,q95,ls='--',c='blue')

ax.set_xlabel('Number of Elements', size=15) 
ax.set_ylabel(P, size=15)
plt.legend()
plt.savefig(os.path.join(dirname, f"figures/{P.replace(' ', '')}_MSA.png"))
'''

print(P)
for i in range(2):
    deltaN = N[i+1] - N[i]
    eps05 = abs(q05[i+1]-q05[i])/deltaN
    epsmu = abs(mu[i+1]-mu[i])/deltaN
    eps95 = abs(q95[i+1]-q95[i])/deltaN
    print(i, eps05, epsmu, eps95)