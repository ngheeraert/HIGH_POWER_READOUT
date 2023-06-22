import numpy as np
from scipy.optimize import curve_fit

def KD( a, b ):
    
    if a==b:
        return 1
    else:
        return 0

def ov( f1, f2 ):
    
    overlap = np.exp( -0.5*np.vdot(f1,f1) - 0.5*np.vdot(f2,f2) + np.vdot(f1,f2) )
    return overlap
