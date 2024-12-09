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
gs = gridspec.GridSpec(1, 2)
gs2 = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=gs[0, 1], hspace=0.1)

# a01, a02, d, f
#def eta_opt(tau: float, d: float, a01: float, a02: float, l: float, f: float, z0: float,  z1: float, z2: float) -> float:
#    func = lambda z: Fsigma(tau, d, a01, a02, l, f, z, z0)*Fsigma(tau, d, a01, a02, l, f, z, z0)

#    return quad(lambda z: func(z), z1, z2)[0]



R = 12
l = 800 * 10**(-9)
tau = 3.48
z0 = 0

def z1(f, d, a0): return 5. * Rayleigh(l, f, tau, d, a0)
def trad(f): return 100./wp(f)

def Wld(a0, d, f): return Wl(a0, tau, d, l, f)*Wl0(f)

def Wlsum(a01, d, f): return (Wld(a01, d, f)/d)*Wl0(f)

fs = np.linspace(0.5, 100., 1000) # in THz
ds = np.linspace(0.01, 0.5, 10)  #np.arange(0.1, 0.51, 0.1)

init_a01 = 0.7
init_a02 = 0.7


def eta_f(a01, a02, d, f):
    return eta(a01, a02, tau, l, f, d, z0, R, -z1(f, d, a01), z1(f, d, a01), trad(f), Wlsum(a01, d, f))

def eta_fs(a01, a02, d, npr=5):
    partial_eta_f = partial(eta_f, a01, a02, d)
    with mp.Pool(processes=npr) as p:
        etas = p.map(partial_eta_f, fs)
    etas = np.array(etas) * 10**4    
    return etas


ax_eta = plt.subplot(gs[0,0])
for d in [0.08, 0.1, 0.2, 0.3, 0.4, 0.5]:
    etas = eta_fs(init_a01, init_a02, d)
    ax_eta.plot(fs, etas, label='d='+str(d) )


ax_eta.legend()
ax_eta.set_xlabel(r'Frequency, THz')
ax_eta.set_ylabel(r'Efficiency, $\times 10^{-4}$')

ax_eta_twin = ax_eta.twiny()
ax_eta_twin.set_xlim(density(fs[0]), density(fs[-1]))
ax_eta_twin.set_xlabel(r'Density, cm$^{-3}$')


def plot_func(ax, fs, *funcs):
    for func, label in funcs:
        ax.plot(fs, np.vectorize(func)(fs), label=label)
    ax.legend()
def Dimsigma1(f): return Dimsigma(Sigma0(tau, d, init_a01, l, f), f) * 10**6 #mkm
def Dimsigma2(f): return Dimsigma(Sigma0(tau, 1.-d, init_a02, l, f), f) * 10**6 #mkm
def Dimtau_f(f): return Dimtau(tau, f) * 10**(15) # fs
def Wl1_f(f): return Wld(init_a01, d, f)
def Wl2_f(f): return Wld(init_a02, d, f)
def Wlsum_f(f): return Wlsum(init_a01, d, f)


ax_sigma = fig.add_subplot(gs2[0])
plot_func(ax_sigma, fs, (Dimsigma1, r'$\sigma_{01}$'), (Dimsigma2, r'$\sigma_{02}$'))
ax_sigma.set_ylim(0., 200)
ax_sigma.set_ylabel(r'Spot sizes, $\mu$m')

ax_sigma_twin = ax_sigma.twiny()
ax_sigma_twin.set_xlim(density(fs[0]), density(fs[-1]))
ax_sigma_twin.set_xlabel(r'Density, cm$^{-3}$')

ax_energy = fig.add_subplot(gs2[1], sharex=ax_sigma)
plot_func(ax_energy, fs, (Wl1_f, r'$\mathcal{W}_{l1}$'), (Wl2_f, r'$\mathcal{W}_{l2}$') , (Wlsum, r'$\mathcal{W}_{\Sigma l}$'))
ax_energy.set_ylim(0., 5.)
ax_energy.set_ylabel(r'Energy, J')

ax_tau = fig.add_subplot(gs2[2], sharex=ax_energy)
plot_func(ax_tau, fs, (Dimtau_f, r'$\tau$'))
ax_tau.set_ylim(0., 100)
ax_tau.set_ylabel(r'Laser duration, fs')

ax_tau.set_xlabel(r'Frequency, THz')

for ax in (ax_sigma, ax_energy, ax_tau):
    
    ax.label_outer()






plt.show()


"""
def eta_fs(a01, a02, d, npr=10):
    partial_eta_f = partial(eta_f, a01, a02, d)
    with mp.Pool(processes=npr) as p:
        etas = p.map(partial_eta_f, fs)
    etas = np.array(etas) * 10**4    
    return etas

ax_eta = plt.subplot(gs[0,0])
extent = [0.01, 0.5, 0.5, 100.]
ax_eta.imshow(eta_fs, cmap='bwr', origin='lower', extent=extent)
ax_eta.set_xlabel(r'Energy share, d')
ax_eta.set_ylabel(r'Frequency, THz')
for d in ds:
    ax_eta.plot(fs, eta_fs(init_a01, init_a02, d), label=r'd='+str(d))
ax_eta.legend()
plt.show()

"""