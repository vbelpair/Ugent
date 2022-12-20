import numpy as np
import matplotlib.pyplot as plt
import sympy as sy
from scipy.optimize import fsolve


class solve_1D:

    def vp(x0, g, n=5, pt=False, plot=False, eps=0):
        '''
        Iteraties voor vinden van een vast punt

        x0    : Startpunt
        g     : Hulpfunctie
        n     : Aantal iteraties
        pt    : Printen van iteraties
        plot  : Printen van itratieplots

        returnvalue: Array met iteratiewaarden
        '''
        x = np.zeros(n+1)
        x[0] = x0

        for i in range(1, n+1):
            if pt == True:
                print('iteratie' + str(i) + ':', x0)

            x[i] = g(x[i-1])
            Eps = abs(x[i] - x[i-1])

            if Eps <= eps:
                x = x[:i]
                break

        if plot == True:
            solve_1D.plot_it(x)

        return x

    def Newton(x0, f, plot=False, pt=False, n=10, eps=0):
        '''
        Newton methode voor het vinden van een vast punt

        x0    : Startwaarde
        f     : Functie
        pt    : Printen van iteraties
        plot  : Printen van itratieplots 
        '''
        # Berekenen afgeleide
        t = sy.symbols('t')
        df = sy.lambdify(t, sy.diff(f(t)), 'numpy')

        # Hulpfunctie
        def g(x): return x - f(x)/df(x)

        return solve_1D.vp(x0, g, n=n, plot=f if plot == True else False, pt=pt, eps=eps)

    def bisectie(f, I, eps=0, delta=0, n=np.inf, pt=False):
        '''
        Bisectiemethode voor het vinden van een nulpunt

        f      : Functie voor te vinden nulpunt
        I      : Interval (a, b) met nulpunt inbegrepen
        eps    : Kleinste intervallengte
        delta  : Fout Y-as
        pt     : Printen van iteraties
        '''

        # Initialisatie
        a, b = I
        F = np.array([f(a), f(b)])
        X = []

        # Input checken
        if F[0]*F[1] >= 0:
            raise Exception('f(a) en f(b) dienen tegengesteld teken te hebben')

        Delta, Eps, N = np.inf, np.inf, 0

        while Delta > delta and Eps > eps and N < n:
            # Bisectie
            x = (a + b)/2
            f_x = f(x)

            # Aantal iteraties aanpassen
            N += 1

            # Nulpunt gevonden
            if f_x == 0:
                Eps = 0
                Delta = 0
                print('Fout (Y-as): \t', Eps)
                print('Fout (X-as): \t', Delta)
                print('Aantal iteraties: \t', N)
                X.append(x)
                return np.array(X)

            # Grenzen aanpassen
            if f_x*F[0] < 0:
                F[1] = f_x
                b = x

            else:
                F[0] = f_x
                a = x

            # Fouten berekenen
            i = np.argmin(np.abs(F))
            Delta = abs(F[i])
            Eps = np.abs(b - a)

            # Nauwkeurigste nulpunt kiezen
            x = a if i == 0 else b

            # Opslaan iteraties
            X.append(x)
            if pt == True:
                print('Iteratie', N, '\na, b:\t', (a, b), '\tf-waarden:\t', F,
                      '\nNulpunt:\t', x, '\nEps:\t', Eps, '\tDelta:\t', Delta, '\n')

        print('Fout (Y-as): \t', Eps)
        print('Fout (X-as): \t', Delta)
        print('Aantal iteraties: \t', N)
        return X

    def Regula_falsi(f, I, eps=0, delta=0, n=np.inf, pt=False):
        '''
        Regula-falsi methode voor het vinden van een nulpunt

        f      : Functie voor te vinden nulpunt
        I      : Interval (a, b) met nulpunt inbegrepen
        eps    : Kleinste intervallengte
        delta  : Fout Y-as
        pt     : Printen van iteraties
        '''

        # Initialisatie
        a, b = I
        F = np.array([f(a), f(b)])
        X = []

        # Input checken
        if F[0]*F[1] >= 0:
            raise Exception('f(a) en f(b) dienen tegengesteld teken te hebben')

        Delta, Eps, N = np.inf, np.inf, 0

        while Delta > delta and Eps > eps and N < n:
            # Regula-falsi
            x = (a*F[1] - b*F[0])/(F[1] - F[0])
            f_x = f(x)

            # Aantal iteraties aanpassen
            N += 1

            # Nulpunt gevonden
            if f_x == 0:
                Eps = 0
                Delta = 0
                print('Fout (Y-as): \t', Eps)
                print('Fout (X-as): \t', Delta)
                print('Aantal iteraties: \t', N)
                X.append(x)
                return np.array(X)

            # Grenzen aanpassen
            if f_x*F[0] < 0:
                F[1] = f_x
                b = x

            else:
                F[0] = f_x
                a = x

            # Fouten berekenen
            i = np.argmin(np.abs(F))
            Delta = abs(F[i])
            Eps = np.abs(b - a)

            # Nauwkeurigste nulpunt kiezen
            x = a if i == 0 else b

            # Opslaan iteraties
            X.append(x)
            if pt == True:
                print('Iteratie', N, '\na, b:\t', (a, b), '\tf-waarden:\t', F,
                      '\nNulpunt:\t', x, '\nEps:\t', Eps, '\tDelta:\t', Delta, '\n')

        print('Fout (Y-as): \t', Eps)
        print('Fout (X-as): \t', Delta)
        print('Aantal iteraties: \t', N)
        return X

    def secans(x0, f, n=5, eps=0):
        '''
        Secansmethode voor vinden van een nulpunt van de functie f

        x0   : tuple met (x0, x1)
        f    : functie voor te vinden nulpunt
        n    : aantal iteraties
        eps  : fout
        '''
        x = np.zeros(n+1)
        x[0], x[1] = x0[0], x0[1]

        for i in range(1, n):

            x[i+1] = x[i] - f(x[i])/((f(x[i]) - f(x[i-1]))/(x[i]-x[i-1]))

            Eps = abs(x[i] - x[i-1])

            if Eps <= eps:
                x = x[:i]
                break

        # if plot == True:
            # solve_1D.plot_it(x)

        return x

    def plot_it(xi, x=False):
        '''
        Plotten van opeenvolgende iteraties

        xi  : Iteratiewaarden
        x   : Werkelijke waarde
        '''
        fig = plt.figure(figsize=(8, 4))
        axes = fig.add_axes([0, 0, 1, 1])

        # Iteraties
        plt.plot(xi, linewidth=3, marker='o', markersize=10, label='$x_{i}$')

        # Werkelijke waarde

        if x != False:
            plt.axhline(y=x, linestyle='--', color='red',
                        linewidth=3, label='$x$')

        # Benoemen assen
        plt.title('Convergentie iteratieschema',
                  fontsize=14, fontweight="bold")
        plt.xlabel('Iteraties')
        plt.ylabel('Iteratiewaarde $x_n$')

        # Lay-out
        plt.legend()
        plt.grid()

        plt.show()

    def plot_np(f, x=False):
        '''
        Grafische voorstelling van de functie voor het vinden van een nulpunt

        f  : Functie
        x  : Oplossing
        '''
        # Vinden van nulpunt via fsolve
        if x == False:
            x = fsolve(f, 0)[-1]
        print('Nulpunt: x =', x)
        X = np.arange(min([x - 1.5*x, x + 1.5*x]),
                      max([x - 1.5*x, x + 1.5*x]), 0.01)

        # Plotten
        fig = plt.figure(figsize=(8, 4))
        axes = fig.add_axes([0, 0, 1, 1])

        plt.title('Voorstellen van nulpunt', fontsize=14, fontweight="bold")
        plt.xlabel('$X$-as')
        plt.ylabel('$Y$-as')

        # Plotten functie
        plt.plot(X, f(X), linewidth=3, label='$y=f(x)$')

        # Plotten nulpunt
        plt.plot(x, 0, markersize=10, marker='o', color='red')
        plt.axvline(x=x, linewidth=3, color='red',
                    linestyle='--', label=f'$x=${x:f}')

        # Plotten x-as
        plt.axhline(y=0, linewidth=2, color='black')

        plt.legend()
        plt.grid()

        plt.show()

    def plotf(y, I, h=0.001):
        a, b = I
        t = np.arange(a, b + h, h)

        fig = plt.figure(figsize=(8, 4))
        axes = fig.add_axes([0, 0, 1, 1])

        plt.plot(t, y(t), lw=3, label='$y=f(x)$')

        plt.legend()
        plt.grid()
        plt.show()
