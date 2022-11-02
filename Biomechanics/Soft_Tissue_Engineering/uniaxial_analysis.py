import numpy as np
import matplotlib.pyplot as plt
import os

"""
Informative website for use of openpyxl: https://www.geeksforgeeks.org/working-with-excel-spreadsheets-in-python/
"""

def Pult_1PK(l, f, A, filename, figdir):

    '''
    l: extension 
    f: load
    A: initial area
    '''

    # get maximum force value: take the force on the first "disruption" => df/dl < 0

    fm = np.mean(f)
    df = np.diff(f)/np.diff(l)
    fmax = f[np.intersect1d(np.where(df < 0), np.where(f > fm))][0]

    # calculating the ultimate stress

    Pult = fmax/A
    
    # plot extension against load

    fig, ax = plt.subplots(1, figsize=(10, 5))
    ax.plot(l,f)
    ax.axhline(fmax, color='black', ls='--')
    ax.set_xlabel('Tensile extension [mm]')
    ax.set_ylabel('Load [N]')
    ax.set_title(filename)
    ax.grid()
    plt.savefig(figdir + f'/{filename}_extension_vs_load.png')

    return Pult

def Pmax_Laplace(cd, T):

    """
    cd: clinical data
    T: initial thickness of specimen
    """

    Ddia, Dsys = cd[2]
    Rdia, Rsys = Ddia/2, Dsys/2
    Pdia, Psys = cd[3]
    rico = (Psys-Pdia)/(Rsys-Rdia)
    Ri = Rsys - Psys/rico
    ri = 1.1*Psys/rico + Ri

    lambda_theta = Ri/ri
    lambda_z = 1.2
    lambda_r = 1/(lambda_z*lambda_theta)
    F = np.array([[lambda_r,0,0],[0,lambda_theta,0],[0,0,lambda_z]])

    H = T
    R0 = Ri + H
    r0 = R0/lambda_theta
    h = r0 - ri
