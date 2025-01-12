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

#a few more additional functions

# dimensionless laser energy
def Wlsum(Wl, f): 
    n = radiation.density(f)
    return Wl/DimensionVars().Wl0(n)
# limits for integration
def z1(rad, f, d, a0): return 5. * rad.Rayleigh(f, d, a0, Wlsum(f))
# duration of radiation, based on numerical simulations
def trad(f):
    n = radiation.density(f)
    return 100./radiation.wp(n)

def eta_f(rad, d, a01, a02, Wl, f):
    return rad.eta(d, a01, a02, f, -z1(rad, f, d, a01), z1(rad, f, d, a01), trad(f), Wlsum(Wl, f))

def eta_fs(rad, a01, a02, d, Wl, fs, npr=5):
    partial_eta_f = partial(eta_f, rad, d, a01, a02, Wl)
    with mp.Pool(processes=npr) as p:
        etas = p.map(partial_eta_f, fs)
    etas = np.array(etas) * 10**4    
    return etas

def axis_labels(ax, ylabel, f1, f2):
    ax.legend()
    ax.set_xlabel(r'Frequency, THz')
    ax.set_ylabel(ylabel)

    ax_twin = ax.twiny()
    ax_twin.set_xlim(radiation.density(f1), radiation.density(f2))
    ax_twin.set_xlabel(r'Density, cm$^{-3}$')

def main():

    fig = plt.figure(figsize=(15, 8))
    gs = gridspec.GridSpec(3, 2, hspace=0.)

    R, l, tau, z0 = 12., 800 * 10**(-9), 3.48, 0.
    Wl = 0.35    # J

    init_a01 = 0.7
    init_a02 = 0.7

    rad = radiation(R, l, tau, z0)

    fs = np.linspace(0.5, 100., 200) # in THz
    ds = np.linspace(0.01, 0.5, 10)  #np.arange(0.1, 0.51, 0.1)


    init_d = 0.08
    #etas = eta_fs(rad, init_a01, init_a02, init_d, Wl, fs, npr=5)

    ax_eta = plt.subplot(gs[0:2,0])
    #ax_eta.plot(fs, etas, label='d= 0.08')
    axis_labels(ax_eta, r'Efficiency, $\times 10^{-4}$', fs[0], fs[-1])

    plt.show()

    return 0

if __name__ == '__main__':

    main()
