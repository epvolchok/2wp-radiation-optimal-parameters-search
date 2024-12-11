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

from libradiation import *

from matplotlib import rc 

font = {'family' : 'serif'}
rc('text', usetex=True)
#rc('text.latex', unicode=True)
rc('font', family='serif')
rc('text.latex', preamble=r"\usepackage[utf8]{inputenc}")
rc('text.latex', preamble=r"\usepackage[T1]{fontenc}")
matplotlib.rcParams.update({'font.size': 14})


fig = plt.figure(figsize=(15, 8))
gs = gridspec.GridSpec(1, 1)


a01 = 0.67
a02 = 0.8
sigma01 = 1.86
N = 3.
sigma02 = N * sigma01
R = 12
l = 800 * 10**(-9)
tau = 3.48
T = 40*10**(-15)
n = 0.25*me/(pi*q*q)*(tau/T)*(tau/T)
z0 = 0
Wlsum = 0.35
d = 0.08

def z1(f, d, a0): return 5. * Rayleigh(l, f, tau, d, a0, Wlsum, Wl0(f))
def trad(f): return 100./wp(f)

fs = np.linspace(0.5, 120., 1000) # in THz
ds = np.linspace(0.01, 0.5, 10)
den_v = np.vectorize(density)
ns = den_v(fs)


def eta_f(f):
    return eta(a01, a02, tau, l, f, d, z0, R, -z1(f, d, a01), z1(f, d, a01), trad(f), Wlsum, Wl0(f))

with mp.Pool(processes=5) as p:
    etas = p.map(eta_f, fs)
etas = np.array(etas) * 10**4


print(Sigma0(tau, d, a01, l, 54, Wlsum, Wl0(54)), Sigma0(tau, 1.-d, a02, l, 54, Wlsum, Wl0(54)))
ax_eta = plt.subplot(gs[0,0])
ax_eta.plot(fs, etas)
ax_eta.set_xlabel(r'Frequency, THz')
ax_eta.set_ylabel(r'Efficiency, $\times 10^{-4}$')
ax_eta.set_xscale('log')

ax_eta_twin = ax_eta.twiny()
ax_eta_twin.plot(ns, etas)
#ax_eta_twin.set_xlim(density(fs[0]), density(fs[-1]))
ax_eta_twin.set_xlabel(r'Density, cm$^{-3}$')
ax_eta_twin.set_xscale('log')

plt.show()