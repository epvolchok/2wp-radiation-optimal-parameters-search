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

"""
den_v = np.vectorize(density)
ns = den_v(fs)

sigmas_mesh, ns_mesh = np.meshgrid(sigmas_v, ns)
ax_F.imshow(etas, cmap='bwr', origin='lower', extent=extent) #, extent=extent
point = ax_F.scatter([init_s0], [init_N], marker='o', color='black')
#ax.contour(sigmas_v, ns, etas, extent=extent)

#np.unravel_index(np.argmax(Fsigmas, axis=None), Fsigmas.shape)
args = np.unravel_index(np.argmax(etas), etas.shape)

text_dn1 = plt.text(-3, 4.8, r'dn1 ='+str(dnw(init_a0, tau, init_s0)), horizontalalignment='left',verticalalignment='center')
text_eta = plt.text(-3, 4.5, r'eta ='+str(Fsigma(init_s0, init_N, 0., 0., 1.)), horizontalalignment='left',verticalalignment='center')


def update(val):
    point.set_offsets(np.array([[s0_slider.val, N_slider.val]]))
    
    text_dn1.set_text(r'dn1 ='+str(dnw(a0_slider.val, tau, s0_slider.val)))
    text_eta.set_text(r'eta ='+str(eta_sigma(s0_slider.val, N_slider.val)))
    fig.canvas.draw_idle()


# register the update function with each slider
a0_slider.on_changed(update)
s0_slider.on_changed(update)
N_slider.on_changed(update)

resetax = fig.add_axes([0.1, 0.1, 0.1, 0.04])
button = Button(resetax, 'Reset', hovercolor='0.975')

def reset(event):
    a0_slider.reset()
    s0_slider.reset()
    N_slider.reset()
button.on_clicked(reset)

ax_s0 = fig.add_axes([0.1, 0.4, 0.3, 0.05])
s0_slider = Slider(
    ax=ax_s0,
    label='s0',
    valmin=1.2,
    valmax=3.,
    valinit=init_s0,
    valstep=0.01,
)
"""

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

def Dimsigma1(rad, d, a0, Wl, f): 
    n = radiation.density(f)
    wp_n = radiation.wp(n)
    sigma0 = rad.Sigma0(d, a0, f, Wlsum(Wl, f))
    return DimensionVars().Dimsigma(sigma0, wp_n) * 10**6 #mkm

def dnw_en(rad, d, a0, Wl, f):
    sigma0 = rad.Sigma0(d, a0, f, Wlsum(Wl, f))
    return rad.dnw(sigma0, a0)

def axis_labels(ax, func, fs, ns, label, ylabel):
    ax.plot(fs, func(fs), label=label)
    ax.legend()
    ax.set_xlabel(r'Frequency, THz')
    ax.set_ylabel(ylabel)
    ax.set_xscale('log')

    ax_twin = ax.twiny()
    ax_twin.plot(ns, func(ns))
    ax_twin.set_xlabel(r'Density, cm$^{-3}$')
    ax_twin.set_xscale('log')

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
    n_v = np.vectorize(radiation.density)
    ns = n_v(fs)
    etas = np.random.rand(len(fs))


    init_d = 0.08
    #etas = eta_fs(rad, init_a01, init_a02, init_d, Wl, fs, npr=5)

    ax_eta = plt.subplot(gs[0:2,0])
    #ax_eta.plot(fs, etas, label='d= 0.08')
    #axis_labels(ax_eta, r'Efficiency, $\times 10^{-4}$', fs[0], fs[-1])
    axis_labels(ax_eta, etas, fs, ns, '', r'Efficiency, $\times 10^{-4}$')

    plt.show()

    return 0

if __name__ == '__main__':

    main()
