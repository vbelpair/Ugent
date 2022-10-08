import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

filename = 'data.txt'

dirname = os.path.dirname(__file__)
file = os.path.join(dirname, filename)

with open(file, 'r') as f:
    data = pd.read_csv(f, delimiter=' ')
    f.close()
column_samples = data.keys()[1:13]
for i in column_samples:
    data[i] = pd.to_numeric(data[i], errors='coerce')
data = data.replace(np.nan, 0, regex=True)

# question 1.1

C = data[data['label'] == 'C0'][column_samples[:-1]]
A = data[data['label'] == 'M1'][column_samples[:-1]]

x = C.values.reshape((C.size))
y = A.values.reshape((A.size))

b, a = np.polyfit(x,y,1)
f = lambda x: b*x + a
c = np.linspace(x.min(), x.max(), 10)
y_cap = f(c)

fig, ax = plt.subplots(1, figsize=(10, 5))
ax.scatter(x,y, lw=0.5, c='black', label='measurement')
ax.plot(c, y_cap, ls='--', c='red', label=f'trendline y = bx + a (b={b:.2f}, a={a:.2f})')
ax.set_xlabel('concentration [mM]', size=20)
ax.set_ylabel('absorbance', size=20)
ax.set_title("Standard curve for nitrophenol", size=25)
plt.legend()
plt.savefig(dirname + '/fig1.png')

    

