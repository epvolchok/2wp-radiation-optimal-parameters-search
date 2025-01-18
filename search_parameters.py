import matplotlib
matplotlib.use('Qt5Agg')   
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
from matplotlib.patches import Rectangle


from libradenergy import EnergyDependence as radiation
from libdimparam import *
import libplot
from libPlot import Plotter

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
"""

#a few more additional functions

def eta_f(rad, d, a01, a02, Wl, f):
    return rad.eta(d, a01, a02, f, -rad.z1(f, d, a01), rad.z1(f, d, a01), radiation.trad(f), radiation.Wlsum(Wl, f))

def eta_fs(rad, a01, a02, d, Wl, fs, npr=5):
    partial_eta_f = partial(eta_f, rad, d, a01, a02, Wl)
    with mp.Pool(processes=npr) as p:
        etas = p.map(partial_eta_f, fs)
    etas = np.array(etas) * 10**4    
    return etas


def main():

    fig = plt.figure(figsize=(15, 8))
    gs = gridspec.GridSpec(3, 2, hspace=0.)

    R, l, tau, z0 = 12., 800 * 10**(-9), 3.48, 0.
    Wl = 0.35    # J
    rad = radiation(R, l, tau, z0)

    init_a01 = 0.7
    init_a02 = 0.7
    init_d = 0.08
    init_f = 28.

    params = [init_a01, init_a02, init_d, init_f, rad, Wl]

    fs = np.linspace(0.5, 100., 200) # in THz
    ds = np.linspace(0.01, 0.5, 10)  #np.arange(0.1, 0.51, 0.1)
    
    ns = Plotter.vectorization(radiation.density, fs)
    etas = np.random.rand(len(fs))

    graphs = Plotter(fig, gs, rad, params)
    graphs.plot('eta', [etas], fs, ns, [r'$d='+str(init_d)+r'$'], params[0:2])

    sigmas1 = Plotter.vectorization(rad.Dimsigma, fs, init_d, init_a01, Wl)
    sigmas2 = Plotter.vectorization(rad.Dimsigma, fs, 1. - init_d, init_a02, Wl)
    graphs.plot('sigma', [sigmas1, sigmas2], fs, ns, [r'$\sigma_{01}$', r'$\sigma_{02}$'])

    dns1 = Plotter.vectorization(rad.dnw_en, fs, init_d, init_a01, Wl)
    dns2 = Plotter.vectorization(rad.dnw_en, fs, 1. - init_d, init_a02, Wl)
    graphs.plot('dn', [dns1, dns2], fs, ns, [r'$\delta n_{w1}$', r'$\delta n_{w2}$'])
    graphs.ax_dn.set_ylim(0., 5.5)

    a1_slider, a2_slider = graphs.sliders(params[0:2])

    for slider in (a1_slider, a2_slider):
            slider.on_changed(graphs.update_wrapper(rad, a1_slider, a2_slider, params[2:]))

    graphs.button.on_clicked(graphs.reset_wrapper(rad, a1_slider, a2_slider, params[2:]))
    
    graphs.plot_show()

    return 0

if __name__ == '__main__':

    main()
