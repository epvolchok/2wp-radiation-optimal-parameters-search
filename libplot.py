import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
from matplotlib import gridspec
from matplotlib import rc
from matplotlib.backend_bases import MouseButton
import numpy as np 

rc('text', usetex=True)
rc('font', family='serif')
rc('text.latex', preamble=r"\usepackage[utf8]{inputenc}")
matplotlib.rcParams.update({'font.size': 14})

"""
    Some functions to make visualization shorter and easier
"""

def vectorization(func, fs, *args):
    func_v = np.vectorize(func)
    res=np.empty(len(fs))
    if args:
        res = func_v(fs, *args)
    else: res = func_v(fs)
    return res

def plot(ax, ys, fs, ns, labels, ylabel, top=True, bottom=True):
    for i, y in enumerate(ys):
        line = ax.plot(fs, y, label=labels[i])

    ax.legend()
    ax.set_ylabel(ylabel)
    ax.set_xscale('log')
    
    ax_twin = ax.twiny()
    ax_twin.plot(ns, ys[-1], color=line[0].get_color())
    ax_twin.set_xscale('log')

    if top:
        ax_twin.set_xlabel(r'Density, cm$^{-3}$')
    if bottom:
        ax.set_xlabel(r'Frequency, THz')

    ax.tick_params(bottom=bottom, top=False, )
    ax.tick_params(labelbottom=bottom, labeltop=False)

    ax_twin.tick_params(top=top, bottom=False)
    ax_twin.tick_params(labeltop=top, labelbottom=False)


def create_sliders(*args):
    sliders = ()
    if args:
        for label, ax, init in  args:
            slider = Slider(
                ax=ax,
                label=label,
                valmin=0.2,
                valmax=0.85,
                valinit=init,
                valstep=0.01,)
            sliders += (slider,)
    return sliders

def reset_wrapper(slider1, slider2):
    
    def reset(event):
        slider1.reset()
        slider2.reset()

    return reset

def on_click(event):
    if event.button is MouseButton.LEFT:
        print(event.xdata, event.ydata)