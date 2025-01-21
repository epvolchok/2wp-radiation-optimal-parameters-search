"""
Copyright (c) 2025 VOLCHOK Evgeniia
for contacts e.p.volchok@gmail.com

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0
"""

import matplotlib
matplotlib.use('Qt5Agg')   
import matplotlib.pyplot as plt 
import numpy as np
from matplotlib import gridspec


from libradenergy import EnergyDependence as radiation
from libdimparam import *
from libPlot import Plotter

from matplotlib import rc 


rc('text', usetex=True)
rc('font', family='serif')
rc('text.latex', preamble=r"\usepackage[utf8]{inputenc}")
matplotlib.rcParams.update({'font.size': 14})

def main():

    fig = plt.figure(figsize=(15, 8))
    gs = gridspec.GridSpec(3, 2, hspace=0.)

    R, l, tau, z0 = 12., 800 * 10**(-9), 3.48, 0.
    Wl = 0.35    # J
    rad = radiation(R, l, tau, z0)

    init_a01 = 0.7
    init_a02 = 0.7
    init_d = 0.11
    init_f = 28.

    params = [init_a01, init_a02, init_d, init_f, rad, Wl]

    fs = np.linspace(0.5, 100., 100) # in THz
    ds = np.linspace(0.01, 0.5, 50)  #np.arange(0.1, 0.51, 0.1)
    
    ns = Plotter.vectorization(radiation.density, fs)
    
    graphs = Plotter(fig, gs, rad, params)
    etas = rad.eta_2d(init_a01, init_a02, ds, Wl, fs, 10)
    graphs.plot('eta', [etas], fs, ns, [], params[2:4])

    sigmas1, sigmas2 = Plotter.make_array(fs, [init_d, init_a01, Wl, 1. - init_d, init_a02, Wl], \
                                          rad.Dimsigma, rad.Dimsigma)
    graphs.plot('sigma', [sigmas1, sigmas2], fs, ns, [r'$\sigma_{01}$', r'$\sigma_{02}$'])

    dns1, dns2 = Plotter.make_array(fs, [init_d, init_a01, Wl, 1. - init_d, init_a02, Wl], \
                                    rad.dnw_en, rad.dnw_en)
    graphs.plot('dn', [dns1, dns2], fs, ns, [r'$\delta n_{w1}$', r'$\delta n_{w2}$'])
    graphs.ax_dn.set_ylim(0., 5.5)

    a1_slider, a2_slider = graphs.sliders(params[0:2])

    for slider in (a1_slider, a2_slider):
            slider.on_changed(graphs.update_wrapper(a1_slider, a2_slider, params[2:], fs, ds))

    graphs.button.on_clicked(Plotter.reset_wrapper(a1_slider, a2_slider))
    fig.canvas.mpl_connect('button_press_event', \
                           graphs.onclick_wrapper(a1_slider, a2_slider, params[2:], fs))


    graphs.plot_show()

    return 0

if __name__ == '__main__':

    main()
