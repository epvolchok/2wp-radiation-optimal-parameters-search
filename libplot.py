import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
from matplotlib import gridspec
from matplotlib import rc 

rc('text', usetex=True)
rc('font', family='serif')
rc('text.latex', preamble=r"\usepackage[utf8]{inputenc}")
matplotlib.rcParams.update({'font.size': 14})

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

    ax.tick_params(bottom=bottom, top=top)
    ax.tick_params(labelbottom=bottom, labeltop=top)

    ax_twin.tick_params(bottom=bottom, top=top)
    ax_twin.tick_params(labelbottom=bottom, labeltop=top)
