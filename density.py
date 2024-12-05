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
gs = gridspec.GridSpec(1, 2, width_ratios=[1., 1.])
gs2 = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[0, 1], hspace=0.1)
fs = np.linspace(0.5, 100., 1000) # in THz
eta_v = np.vectorize(eta)
R_v = np.vectorize(Rayleigh)


a01 = 0.67
a02 = 0.8
sigma01 = 1.86
N = 3.
sigma02 = N * sigma01
R = 12
l = 800 * 10**(-9)
tau = 3.48
z0 = 0
def z1(f): return 5. * Rayleigh(l, f, sigma01)
def trad(f): return 100./wp(f)
def Wl1(f): return Wl(a01, tau, sigma01, l, f)
def Wl2(f): return Wl(a02, tau, sigma02, l, f)
def Wlsum(f): return (Wl1(f) + Wl2(f))

print(z1(10.), trad(10.), Wlsum(30.))
print(Power(a01, a02, tau, l, 30., sigma01, N, z0, R, -z1(30.), z1(30.)))
print(wp(30.), Dimlessw0(l, 30.))
print(Wl1(30.), Wl2(30.))
print(eta(a01, a02, tau, l, 30., sigma01, N, z0, R, -z1(30.), z1(30.), trad(30.), Wlsum(30)))
zs = np.linspace(-z1(20.), z1(20.), 1000)
E0_v = np.vectorize(E0)
Es = E0_v(a01, a02, tau, l, 20., sigma01, N, zs, z0, R)



def eta_f(f: float) -> float:
    return eta(a01, a02, tau, l, f, sigma01, N, z0, R, -z1(f), z1(f), trad(f), Wlsum(f))

with mp.Pool(processes=5) as p:
    etas = p.map(eta_f, fs)
etas = np.array(etas) * 10**4

ax_eta = plt.subplot(gs[0:4,0])
ax_eta.plot(fs, etas)
ax_eta.set_xlabel(r'Frequency, THz')
ax_eta.set_ylabel(r'Efficiency, $\cdot 10^{-4}$')

def plot_func(ax, fs, *funcs):
    for func in funcs:
        ax.plot(fs, np.vectorize(func)(fs), label=func.__name__)
    ax.legend()
def Dimsigma1(f): return Dimsigma(sigma01, f) * 10**6 #mkm
def Dimsigma2(f): return Dimsigma(sigma02, f) * 10**6 #mkm
def Dimtau_f(f): return Dimtau(tau, f) * 10**(15) # fs

t1 = 20
t2 = 50


t = (t1,)
tt, = t
print(tt)

def func1(f, *t): 
    tt, = t
    return Dimtau_f(f[0]) - tt
f1 = fsolve(func1, x0 = np.array([20.]), args=(t1,))[0]
f2 = fsolve(func1, x0 = np.array([20.]), args=(t2,))[0]
print(f1, f2)

def frange(ax, f1, f2):
    y1, y2 = ax.get_ylim()
    print(y1, y2)
    ax.plot([f1, f1], [y1, y2], color='red')
    ax.plot([f2, f2], [y1, y2], color='red')


ax_sigma = fig.add_subplot(gs2[0])
plot_func(ax_sigma, fs, Dimsigma1, Dimsigma2)
ax_sigma.set_ylim(0., 200)
ax_sigma.set_ylabel(r'Spot sizes, $\mu$m')
frange(ax_sigma, f1, f2)



ax_sigma_twin = ax_sigma.twiny()

ax_energy = fig.add_subplot(gs2[1], sharex=ax_sigma)
plot_func(ax_energy, fs, Wl1, Wl2, Wlsum)
ax_energy.set_ylim(0., 5.)
ax_energy.set_ylabel(r'Energy, J')
frange(ax_energy, f1, f2)




ax_tau = fig.add_subplot(gs2[2], sharex=ax_energy)
plot_func(ax_tau, fs, Dimtau_f)
ax_tau.set_ylim(0., 50)
ax_tau.set_ylabel(r'Laser duration, fs')
frange(ax_tau, f1, f2)

ax_tau.set_xlabel(r'Frequency, THz')

for ax in (ax_sigma, ax_energy, ax_tau):
    
    ax.label_outer()

plt.show()
