import sys
import time
import numpy as np
from .functions import *
from .params import params
from .system_fortran import system_fortran
from .dynamics import dynamics
import matplotlib.pyplot as plt
import os
from scipy import fft

pi = np.pi
