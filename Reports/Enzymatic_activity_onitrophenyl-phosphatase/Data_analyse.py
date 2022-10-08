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

A = data[data['label'] == 'M1'][column_samples[-1]]
P = lambda z: z/b
P0 = P(A)

fig, ax = plt.subplots(1, figsize=(10, 5))
ax.scatter(x,y, lw=0.5, c='black', label='measurement')
ax.scatter(P0,A+a, lw=0.5, c='green', label='P0')
ax.plot(c, y_cap, ls='--', c='red', label=f'trendline y = bx + a (b={b:.2f}, a={a:.2f})')
ax.set_xlabel('concentration [mM]', size=20)
ax.set_ylabel('absorbance', size=20)
ax.set_title("Standard curve for nitrophenol", size=25)
plt.legend()
plt.savefig(dirname + '/fig1.png')


#Question 3.1

C = np.array([0.4,0.8, 1.6, 2, 4, 6, 8, 10, 12, 14, 16, 20])
A = np.array(data[data['label'] == 'M3'][column_samples])
A[:, [0, 11]] = A[:, [11, 0]]
A_S = A[0:4]
A_B = A[4:]


x = np.zeros(A_S.size)

for i in range(4):
    x[len(C)*i:len(C)*(1+i)] = C
c = np.linspace(x.min(), x.max(), 10)
y_S = A_S.reshape((A_S.size))

c_S, b_S, a_S = np.polyfit(x,y_S,2)
f_S = lambda x:  c_S*x**2 + b_S*x + a_S
y_cap_S = f_S(c)

y_B = A_B.reshape((A_B.size))
b_B, a_B = np.polyfit(x,y_B,1)
f_B = lambda x: b_B*x + a_B
c = np.linspace(x.min(), x.max(), 10)
y_cap_B = f_B(c)
R_2 = 1 - np.sum((y_S - f_S(x))**2)/np.sum((y_S - np.mean(y_S))**2)
fig, ax = plt.subplots(1, figsize=(10, 5))
ax.scatter(x,y_S, lw=0.5, c='black', label='measurement')
ax.scatter(x,y_B, lw=0.5, c='black', label='measurement')
ax.plot(c, y_cap_S, ls='--', c='red', label = f'trendline squared = {R_2:.2f}')
ax.plot(c, y_cap_B, ls='--', c='red', label = 'trendline control')
ax.set_xlabel('concentration [mM]', size=20)
ax.set_ylabel('absorbance', size=20)
ax.set_title("M3", size=25)
plt.legend()
plt.savefig(dirname + '/M3fig1.png')
plt.show()


