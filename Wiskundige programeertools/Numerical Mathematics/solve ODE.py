import numpy as np
from scipy.optimize import fsolve
from scipy.integrate import trapz as int_t
from scipy.interpolate import lagrange


class solve_ODE:

    def fEuler(y0, f, I, h=0.01):
        '''
        Voorwaartse Euler methode (expliciet)
            Oplossen van het Cauhy probleem
                y'(x) = f(x, y(x)),  y(a) = y0

            f    : Functie 
            y0   : Beginvoorwaarde 
            I    : Interval (a, b)
            h    : Staplengte
        '''

        # Begin- en eindpunt
        a, b = I

        # x-waarden
        x = np.arange(a, b + h, h)

        # Oplossingsvector
        if type(y0) == float:
            y0 = [y0]
        m = len(y0)
        y = np.zeros((m, len(x)))
        y[:, 0] = y0

        # Itereren
        for i in range(1, len(x)):
            y[:, i] = y[:, i-1] + h*f(x[i-1], y[:, i-1])
            # print(y[:,i])

        return (x, y)

    def mEuler(y0, f, I, h=0.01):
        '''
        Geavanceerde Euler methode
            Oplossen van het Cauhy probleem
                y'(x) = f(x, y(x)),  y(a) = y0

            f    : Functie 
            y0   : Beginvoorwaarde 
            I    : Interval (a, b)
            h    : Staplengte
        '''
        # Begin- en eindpunt
        a, b = I

        # x-waarden
        x = np.arange(a, b + h, h)

        # Oplossingsvector
        if type(y0) == float:
            y0 = [y0]
        m = len(y0)
        y = np.zeros((m, len(x)))
        y[:, 0] = y0

        # Itereren
        for i in range(1, len(x)):
            k1 = f(x[i-1], y[:, i-1])
            k2 = f(x[i-1] + h/2, y[:, i-1] + h/2*k1)
            y[:, i] = y[:, i-1] + h*k2

        return (x, y)

    def bEuler(y0, f, I, h=0.01):
        '''
        Backward Euler methode (impliciet)
            Oplossen van het Cauhy probleem
                y'(x) = f(x, y(x)),  y(a) = y0

            f    : Functie 
            y0   : Beginvoorwaarde 
            I    : Interval (a, b)
            h    : Staplengte
        '''
        # Begin- en eindpunt
        a, b = I

        # x-waarden
        x = np.arange(a, b + h, h)

        # Oplossingsvector
        if type(y0) == float:
            y0 = [y0]
        m = len(y0)
        y = np.zeros((m, len(x)))
        y[:, 0] = y0

        # Itereren
        for i in range(1, len(x)):
            def g(t): return y[:, i-1] + h*f(x[i], t)
            def F(t): return y[:, i-1] + h*f(x[i], t) - t

            # Gebruik fsolve of vaste-punt iteratie uit solve_1D
            y[:, i] = fsolve(F, y[:, i-1])

        return (x, y)

    def Runge_Kutta(y0, f, I, h=0.01, orde=2, omega=1.):
        '''
        Runge-Kutta methode
            Oplossen van het Cauhy probleem
                y'(x) = f(x, y(x)),  y(a) = y0

            f    : Functie 
            y0   : Beginvoorwaarde 
            I    : Interval (a, b)
            h    : Staplengte
        '''
        # Begin- en eindpunt
        a, b = I

        # x-waarden
        x = np.arange(a, b + h, h)

        # Oplossingsvector
        if type(y0) == float:
            y0 = [y0]
        m = len(y0)
        y = np.zeros((m, len(x)))
        y[:, 0] = y0

        # Itereren
        for i in range(1, len(x)):
            if orde == 2:
                k1 = f(x[i-1], y[:, i-1])
                k2 = f(x[i-1] + h/(2*omega), y[:, i-1] + h/(2*omega)*k1)
                y[:, i] = y[:, i-1] + h*((1 - omega)*k1 + omega*k2)

            if orde == 4:
                k1 = f(x[i-1], y[:, i-1])
                k2 = f(x[i-1] + h/2, y[:, i-1, ] + h/2*k1)
                k3 = f(x[i-1] + h/2, y[:, i-1] + h/2*k2)
                k4 = f(x[i-1] + h, y[:, i-1] + h*k3)
                y[:, i] = y[:, i-1] + h/6*(k1 + 2*k2 + 2*k3 + k4)

        return (x, y)

    def Adams_Bashforth(y0, f, I, m=0, h=0.1):
        '''
        Adams-Bashforth formules voor het oplossen van ODE's
            Oplossen van het Cauhy probleem
                y'(x) = f(x, y(x)),  y(a) = y0

            f    : Functie 
            y0   : Beginvoorwaarde 
            I    : Interval (a, b)
            h    : Staplengte
            m    : m = orde - 1
        '''

        # Begin- en eindpunt
        a, b = I

        # x-waarden
        x = np.arange(a, b, h)

        # Oplossingsvector
        if type(y0) == float:
            y0 = [y0]
        r = len(y0)
        y = np.zeros((r, len(x)))

        # Bepalen van overige m punten via
        iv = (I[0], I[0] + (m+1)*h)
        y_ex = solve_ODE.mEuler(y0, f, iv, h=h)
        if m == 0:
            return y_ex

        y[:, 0: m + 1] = y_ex[-1][:, 0:m+1]

        for i in range(m + 1, len(x)):
            if m == 1:
                k1 = f(x[i-1], y[:, i-1])
                k2 = f(x[i-2], y[:, i-2])
                y[:, i] = y[:, i-1] + h/2*(3*k1 - k2)

            if m == 2:
                k1 = f(x[i-1], y[:, i-1])
                k2 = f(x[i-2], y[:, i-2])
                k3 = f(x[i-3], y[:, i-3])
                y[:, i] = y[:, i-1] + h/12*(23*k1 - 16*k2 + 5*k3)

            if m == 3:
                k1 = f(x[i-1], y[:, i-1])
                k2 = f(x[i-2], y[:, i-2])
                k3 = f(x[i-3], y[:, i-3])
                k4 = f(x[i-4], y[:, i-4])
                y[:, i] = y[:, i-1] + h/24*(55*k1 - 59*k2 + 37*k3 - 9*k4)

        return (x, y)
