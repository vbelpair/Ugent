import numpy as np
import matplotlib.pyplot as plt
import sympy as sy
from scipy.optimize import fsolve
from autograd import grad, jacobian


class solve_nD:

    def vp(x0, C, g, n=5, pt=False, eps=0):
        '''
        Vastepuntiteratie in nD voor het probleem
            x_{n+1} = X*c_n + g

        x0    : Beginwaarde
        n     : Aantal iteraties
        pt    : Printen van iteraties

        return : Oplossingen als rijen (oplossig na iteratie i = x[i])
        '''
        nx = len(x0)
        ng = len(g)
        nC = len(C)
        mC = len(C[0])

        # Checken convergentievoorwaarden
        if nC != mC or nC != nx or ng != nx:
            raise Exception('Wrong size data')

        if np.max(np.abs(np.linalg.eigvals(C))) > 1:
            raise Exception('No convergence possible')

        # Aanmaken hulpfunctie
        def G(x): return C@x + g

        # Aanmaken oplossingsvariabele
        x = np.zeros((n+1, len(x0)))
        x[0] = x0

        # Itereren
        for i in range(1, n+1):

            # Printen iteraties
            if pt == True:
                print('iteratie' + str(i) + ':', x0)

            x[i] = G(x[i-1])

            # Stoppen bij fout
            Eps = np.linalg.norm(x[i] - x[i-1])
            if Eps < eps:
                x = x[:i]
                break
        print('Aantal iteraties:\t', i, '\tRelatieve fout:\t', Eps)
        return x

    def Newton(X0, F, n=10, eps=0, pt=False):
        '''
        Newtonmethode in nD:
            x_{n+1} = x_n + Phi^{-1}(x_n)*F(x_n)

            X0   : Beginwaarde
            n    : Aantal iteraties
            F    : Lijst(!) van functies
        '''

        # Berekenen jacobiaan
        def Phi(x): return jacobian(F)(x)

        # Oplossingsvector
        x = np.zeros((n+1, len(X0)))
        x[0] = X0

        # Functie
        def G(x): return x - np.linalg.inv(Phi(x))@F(x)

        for i in range(1, n+1):
            x[i] = G(x[i-1])

            # Printen iteraties
            if pt == True:
                print('iteratie' + str(i) + ':', x[i])

            # Stoppen bij fout
            Eps = np.linalg.norm(x[i] - x[i-1])
            if Eps < eps:
                x = x[:i]
                break

        print('Aantal iteraties:\t', i, '\tRelatieve fout:\t', Eps)

        return x

    def Jacobi(x0, A, b, n=5, pt=False, eps=0):
        '''
        Jacobimethode voor het oplossen van stelsel van de vorm
            Ax = b

        x0    : Beginwaarde
        n     : Aantal iteraties
        pt    : Printen van iteraties

        return : Oplossingen als rijen (oplossig na iteratie i = x[i])
        '''
        n2, m = A.shape

        # Vierkante matrix
        if n2 != m:
            raise Exception('Matrix is niet vierkant')

        # Geen nul op diagonaal
        D = np.diag(A)
        if len(D[D == 0]) != 0:
            raise Exception('Diagonaal bezit een nul')

        # Ontbinden matrix
        E = -np.tril(A, -1)
        F = -np.triu(A, +1)
        invD = np.diag(1./D)

        # Opstellen hulpfuncties
        C = np.dot(invD, E+F)
        g = np.dot(invD, b)

        return solve_nD.vp(x0, C, g, n=n, pt=pt, eps=eps)

    def Gauss_Seidel(x0, A, b, n=5, pt=False, eps=0):
        '''
        Gauss-Seidelmethode voor het oplossen van stelsel van de vorm
            Ax = b

        x0    : Beginwaarde
        n     : Aantal iteraties
        pt    : Printen van iteraties

        return : Oplossingen als rijen (oplossig na iteratie i = x[i])
        '''
        n2, m = A.shape

        # Vierkante matrix
        if n2 != m:
            raise Exception('Matrix is niet vierkant')

        # Geen nul op diagonaal
        D = np.diag(A)
        if len(D[D == 0]) != 0:
            raise Exception('Diagonaal bezit een nul')

        # Ontbinden matrix
        D = np.diag(A)
        E = -np.tril(A, -1)
        F = -np.triu(A, 1)

        M = np.linalg.inv(np.diag(D)-E)

        C = np.dot(M, F)
        g = np.dot(M, b)

        return solve_nD.vp(x0, C, g, n=n, pt=pt, eps=eps)
