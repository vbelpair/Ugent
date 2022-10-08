import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

filename = 'data.txt'

dirname = os.path.dirname(__file__)
file = os.path.join(dirname, filename)

# retrieving data:

with open(file, 'r') as f:
    data = pd.read_csv(f, delimiter=' ')
    f.close()
column_samples = data.keys()[1:13]
for i in column_samples:
    data[i] = pd.to_numeric(data[i], errors='coerce')
data = data.replace(np.nan, 0, regex=True)

# defining analyzing function: 

def analyze(x, y, xlabel, ylabel, title, filename, g=lambda x: x):
    '''
    x, y: arrays with corresponding values
    xlabel, ylabel, title, filename: strings used for plot formation
    g: function used to define linear regression technique
    '''
    
    b, a = np.polyfit(g(x), y, 1)
    f = lambda x: b*g(x) + a
    z = np.linspace(x.min(), x.max(), 10)
    Y = f(z)

    R2 = 1 - np.sum((y-f(x))**2)/np.sum((y-np.mean(y))**2)

    fig, ax = plt.subplots(1, figsize=(10, 5))
    ax.scatter(x,y, lw=0.5, c='black', label='measurements')
    #ax.scatter(P0,A+a, lw=0.5, c='green', label='P0')
    ax.plot(z, Y, ls='--', c='red', label=f'trendline, R squared = {R2:.2f}')
    ax.set_xlabel(xlabel, size=20)
    ax.set_ylabel(ylabel, size=20)
    ax.set_title(title, size=25)
    plt.legend()

    dirname = os.path.dirname(__file__)
    file = os.path.join(dirname, filename)
    plt.savefig(file)

    return a, b

# question 1.1

C = data[data['label'] == 'C0'][column_samples[:-1]]
A = data[data['label'] == 'M1'][column_samples[:-1]]

x = C.values.reshape((C.size))
y = A.values.reshape((A.size))

a, b = analyze(x, y, 'concentration', 'absorbance', 'Standard curve for nitrophenol', 'fig1.png')

A = data[data['label'] == 'M1'][column_samples[-1]]
P = lambda z: z/b
P0 = np.mean(P(A))
print(f'q1.1: P0 = {P0:.2f}')

# question 2.1

C = np.array([0.4, 0.8, 1.6, 2, 4, 6, 8, 10, 12, 14, 16, 20])
A = data[data['label'] == 'M2'][column_samples]
A_S = A[:4]
A_B = A[4:8]

x = np.zeros(A_S.size)
for i in range(4):
    x[len(C)*i:len(C)*(1+i)] = C

y_S = A_S.values.reshape((A_S.size))
a, b = analyze(x, y_S, 'concentration (mM)', 'absorbance', '', 'fig2_1_1.png', g=lambda x: np.log(x))

y_B = A_B.values.reshape((A_B.size))
a, b = analyze(x, y_B, 'concentration (mM)', 'absorbance', '', 'fig2_1_2.png')

# question 2.2

A = data[data['label'] == 'M2'][column_samples]
A_S = A[:4]

s = np.zeros(A_S.size)
for i in range(4):
    s[len(C)*i:len(C)*(1+i)] = C
v = A_S.values.reshape((A_S.size))/(1200*b)

a, b = analyze(1/s, 1/v, r'$rac{1}{[S]}$ (mM$^{-1}$)', r'$\frac{1}{v}$ (s mM$^{-1}$)', 'Lineweaver-Burk without inhibitor', 'fig2_2.png')
Vmax = 1/a
Km = b*Vmax
print(f'q2.2: Vmax = {Vmax:.2f}, Km = {Km:.2f}')

# question 2.3

a, b = analyze(v/s, v, r'$\frac{v}{[S]} (s$^{-1}$)', r'$v (mM s$^{-1}$)', 'Eadie-Hofstee without inhibitor', 'fig2_3.png')
Vmax = a
Km = -b
print(f'q2.3: Vmax = {Vmax:.2f}, Km = {Km:.2f}')

# question 2.4

a, b = analyze(s, s/v, r'$[S] (mM)', r'$\frac{[S]}{v}$ (s)', 'Hanes-Woolf without inhibitor', 'fig2_4.png')
Vmax = 1/b
Km = a*Vmax
print(f'q2.4: Vmax = {Vmax:.2f}, Km = {Km:.2f}')

#Question 3.1

A = np.array(data[data['label'] == 'M3'][column_samples])
A[:, [0, 11]] = A[:, [11, 0]]
A_S = A[0:4]
A_B = A[4:]

x = np.zeros(A_S.size)
for i in range(4):
    x[len(C)*i:len(C)*(1+i)] = C
c = np.linspace(x.min(), x.max(), 10)
y = A_S.reshape((A_S.size))

a, b = analyze(x, y, 'concentration [mM]', 'absorbance', '', 'fig3_1.png', g=lambda x: np.log(x))

#Question 3.2

C = np.array([0.4, 0.8, 1.6, 2, 4, 6, 8, 10, 12, 14, 16, 20])
A = np.array(data[data['label'] == 'M3'][column_samples])
A[:, [0, 11]] = A[:, [11, 0]]
A_S = A[:4]
A_B = A[4:8]

s = np.zeros(A_S.size)
for i in range(4):
    s[len(C)*i:len(C)*(1+i)] = C
v = A_S.reshape(A_S.size)/1200

a, b = analyze(1/s, 1/v, r'$\frac{v}{[S]}$ s$^{-1}$)', r'$v (mM} s$^{-1}$)', "LineaWeaver-Burk with inhibitor", 'fig3_2.png')
Vmax = 1/a
Km = b*Vmax
print(f'q3.2: Vmax = {Vmax:.2f}, Km = {Km:.2f}')

#Question 3.3

a, b = analyze(v/s, v, r'$\frac{v}{[S]} (s$^{-1}$)', r'$v (mM s$^{-1}$)', "Eadie-Hofstee with inhibitor", 'fig3_3.png')
Vmax = a
Km = -b
print(f'q3.3: Vmax = {Vmax:.2f}, Km = {Km:.2f}')


#Question 3.4

a, b = analyze(s, s/v, r'$[S]$ (mM)', r'$\frac{[S]}{v}$ (s)', "Hanes-Woolf with inhibitor", 'fig3_4.png')
Vmax = 1/b
Km = a*Vmax
print(f'q3.4: Vmax = {Vmax:.6f}, Km = {Km:.2f}')
