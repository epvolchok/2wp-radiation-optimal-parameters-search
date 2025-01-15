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

    init_a01 = 0.7
    init_a02 = 0.7

    rad = radiation(R, l, tau, z0)

    fs = np.linspace(0.5, 100., 200) # in THz
    ds = np.linspace(0.01, 0.5, 10)  #np.arange(0.1, 0.51, 0.1)
    
    
    ns = libplot.vectorization(radiation.density, fs)
    etas = np.random.rand(len(fs))


    init_d = 0.08
    init_f = 28.
    #etas = eta_fs(rad, init_a01, init_a02, init_d, Wl, fs, npr=5)

    ax_eta = plt.subplot(gs[0:2,0])
    #ax_eta.plot(fs, etas, label='d= 0.08')
    #axis_labels(ax_eta, r'Efficiency, $\times 10^{-4}$', fs[0], fs[-1])
    libplot.plot(ax_eta, [etas], fs, ns, [r'$d='+str(init_d)+r'$'], r'Efficiency, $\times 10^{-4}$')

    
    sigmas1 = libplot.vectorization(rad.Dimsigma, fs, init_d, init_a01, Wl)
    sigmas2 = libplot.vectorization(rad.Dimsigma, fs, 1. - init_d, init_a02, Wl)
    
    ax_sigmas = plt.subplot(gs[1,1])
    libplot.plot(ax_sigmas, [sigmas1, sigmas2], fs, ns, [r'$\sigma_{01}$', r'$\sigma_{02}$'], \
                 r'Laser spot-sizes, $\mu$ m', bottom=False)

    dns1 = libplot.vectorization(rad.dnw_en, fs, init_d, init_a01, Wl)
    dns2 = libplot.vectorization(rad.dnw_en, fs, 1. - init_d, init_a02, Wl)

    ax_dn = plt.subplot(gs[2, 1])
    libplot.plot(ax_dn, [dns1, dns2], fs, ns, [r'$\delta n_{w1}$', r'$\delta n_{w2}$'], \
                r'Level of nonlinearity', top=False)
    ax_dn.set_ylim(0., 5.5)

    ax_a1 = fig.add_axes([0.05, 0.2, 0.3, 0.05])
    ax_a2 = fig.add_axes([0.05, 0.15, 0.3, 0.05])
    
    sliders_param = [(r'$a_{01}$', ax_a1, init_a01), (r'$a_{02}$', ax_a2, init_a02)]
    a1_slider, a2_slider = libplot.create_sliders(*sliders_param)

    resetax = fig.add_axes([0.1, 0.1, 0.1, 0.04])
    button = Button(resetax, 'Reset', hovercolor='0.975')
    button.on_clicked(libplot.reset_wrapper(a1_slider, a2_slider))
    plt.connect('button_press_event', libplot.on_click)



    text_d = plt.figtext(0.65, 0.91, fr'$d = {init_d:.2f}$', horizontalalignment='left',verticalalignment='center', fontsize=16)
    text_f = plt.figtext(0.73, 0.91, fr'$f ={init_f:.2f}$ THz', horizontalalignment='left',verticalalignment='center', fontsize=16)
    params = [init_d, init_a01, init_a02, init_f, -rad.z1(init_f, init_d, init_a01, Wl), \
                rad.z1(init_f, init_d, init_a01, Wl), Wl, radiation.trad(init_f)]
    fig.patches.extend(plt.Rectangle((0.6, 0.75), 0.35, 0.2, color='gray', fill=True, alpha=0.5))
    text_eta = plt.figtext(0.68, 0.87, fr'$\eta = {rad.eta(*params)*10**4: .2f}$'+r'$ \cdot 10^{-4}$', \
                        horizontalalignment='left',verticalalignment='center', fontsize=16)
    text_P = plt.figtext(0.68, 0.83, fr'Power $= {rad.Power(*params[:-1]):.2f}$ GW', \
                        horizontalalignment='left',verticalalignment='center', fontsize=16)
    maxE0 = rad.DimE0(init_d, init_a01, init_a02, init_f, 0, radiation.Wlsum(Wl, init_f))
    text_E0 = plt.figtext(0.68, 0.79, fr'Max $E_0 = {maxE0:.2f} $ MV/cm', \
                        horizontalalignment='left',verticalalignment='center', fontsize=16)
    

    fig.tight_layout()
    plt.show()

    return 0

if __name__ == '__main__':

    main()
