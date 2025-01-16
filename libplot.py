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
        ax_twin.set_xlabel(r'Density, cm$^{-3}$', labelpad=6.5)
    if bottom:
        ax.set_xlabel(r'Frequency, THz')

    ax.tick_params(bottom=bottom, top=False, )
    ax.tick_params(labelbottom=bottom, labeltop=False)

    ax_twin.tick_params(top=top, bottom=False)
    ax_twin.tick_params(labeltop=top, labelbottom=False)

    ax.set_zorder(1)  # default zorder is 0 for ax1 and ax2
    ax.set_frame_on(False)


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
def update_wrapper(fig, text_fig, d, f, a1_slider, a2_slider, rad, Wl):
    def update(val):
        #point.set_offsets(np.array([[s0_slider.val, N_slider.val]]))
        params = [d, a1_slider.val, a2_slider.val, f, -rad.z1(f, d, a1_slider.val, Wl), \
                    rad.z1(f, d, a1_slider.val, Wl), Wl, rad.trad(f)]
        maxE0 = rad.DimE0(d, a1_slider.val, a2_slider.val, f, 0, rad.Wlsum(Wl, f))
        updated_text = r'''
        $d = {:.2f} \qquad f ={:.2f}$ THz \\
        $\eta = {:.2f} \cdot 10^{{-4}}$ \\
        Power $= {:.2f}$ GW \\
        Max $E_0 = {:.2f}$ MV/cm
        '''.format(d, f, rad.eta(*params)*10**4, rad.Power(*params[:-1]), maxE0)

        text_fig.set_text(updated_text)
        fig.canvas.draw_idle()
    return update

def onclick_wrapper(ax, fig, text_fig, d, f, a1_slider, a2_slider, rad, Wl):
    def on_click(event):
        if event.inaxes is ax:
            f, d = event.xdata, event.ydata

        print(event.xdata, event.ydata)
    return on_click