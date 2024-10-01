import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

""" Module Functions """
def func_powerlaw(temp, k =1,crit_temp = 4.51, exponent = -1):
    return k*np.abs(temp-crit_temp)**exponent

def magnet_curvefit(temperature, magnetization, index):
    x = int(index)
    y = int(index) + 3
    params, covariance = curve_fit(func_powerlaw, temperature[x:y], magnetization[x:y]/(X*Y*Z), maxfev =int(1e9))
    Ks = params[0]
    magnetic_expo = params[1]
    mFit_crittemp = params[2]
    sigma_mag = covariance[1,1]
    mSigma_crittemp = covariance[2,2]
    
    plt.figure()
    plt.plot(temperature[x:y], func_powerlaw(temperature[x:y], Ks, magnetic_expo, mFit_crittemp), 'orange', label='fit')
    plt.plot(temperature[x:y], magnetization[x:y]/(X*Y*Z), 'o', color='blue', label=f"{X}x{Y}x{Z}")
    plt.text(5,0.0001,f'\u03B2 = {-1*magnetic_expo:.3f}\nTc = {mFit_crittemp:.3f}', bbox=dict(facecolor='orange', alpha=0.7), fontsize=10)
    plt.xlabel('Temperature', fontsize = 12)
    plt.ylabel('Magnetization', fontsize = 12)
    plt.title(f'Fit L={Z}',  fontsize = 12, fontweight = "bold")
    plt.legend(loc = 'upper left')
    plt.savefig(f"Magnetization Fit {X}x{Y}x{Z}")
    
    #Kasi d ko mapalabas sa graph mismo
    print(f"Expo = {magnetic_expo:.3f}")
    print(f"Tc = {mFit_crittemp:.3f}")

    plt.show()
    
""" Variables """
X = 10
Y = 10
Z = 10

df = pd.read_csv(f"{X}x{Y}x{Z}_Results.csv")

Temperature = df.iloc[:,0]
Magnet = df.iloc[:,1]
Suscept = df.iloc[:,2]

#For Curve Fittings
Ks = 0.0
magnetic_expo = 0.0
suscept_expo = 0.0
sigma_mag = 0.0
sigma_sus = 0.0
mFit_crittemp = 0.0
sFit_crittemp = 0.0
mSigma_crittemp = 0.0
sSigma_crittemp = 0

crit_index = np.where(Suscept == max(Suscept))
crit_index = crit_index[0];

magnet_curvefit(Temperature,Magnet, crit_index)
