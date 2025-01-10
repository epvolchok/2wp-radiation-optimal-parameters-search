import matplotlib
#matplotlib.use('Qt5Agg')   
import matplotlib.pyplot as plt 
import numpy as np
from matplotlib.widgets import Button, Slider
#import matplotlib as mpl
from matplotlib import gridspec
import multiprocessing as mp
from sympy import symbols, Eq, nsolve
from scipy.optimize import fsolve
from scipy.integrate import quad
from functools import partial

from libradenergy import EnergyDependence as radiation
from libdimparam import *

from matplotlib import rc 


rc('text', usetex=True)
rc('font', family='serif')
rc('text.latex', preamble=r"\usepackage[utf8]{inputenc}")
matplotlib.rcParams.update({'font.size': 14})

fig = plt.figure(figsize=(15, 8))
gs = gridspec.GridSpec(3, 2, hspace=0.)

R, l, tau, z0 = 12., 800 * 10**(-9), 3.48, 0.
Wl = 0.35    # J

init_a01 = 0.7
init_a02 = 0.7

# dimensionless laser energy
def Wlsum(f): return Wl/DimensionVars().Wl0(f)
# limits for integration
def z1(f, d, a0): return 5. * radiation.Rayleigh(f, d, a0, Wlsum(f))
# duration of radiation, based on numerical simulations
def trad(f):
    n = radiation.density(f)
    return 100./radiation.wp(n)

def eta_f(f, d, a01, a02):
    return radiation.eta(d, a01, a02, f, -z1(f, d, a01), z1(f, d, a01), trad(f), Wlsum(f))

fs = np.linspace(0.5, 100., 200) # in THz
ds = np.linspace(0.01, 0.5, 10)  #np.arange(0.1, 0.51, 0.1)

def eta_fs(a01, a02, d, npr=5):
    partial_eta_f = partial(eta_f, a01, a02, d)
    with mp.Pool(processes=npr) as p:
        etas = p.map(partial_eta_f, fs)
    etas = np.array(etas) * 10**4    
    return etas

etas = eta_fs(init_a01, init_a02, d=0.08, npr=5)

ax_eta = plt.subplot(gs[0,0])
ax_eta.plot(fs, etas, label='d= 0.08')

plt.show()
