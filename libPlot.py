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

class Plotter():
    def __init__(self, fig, gs, rad, params):
        self.figure = fig
        self.gs = gs

        self.ax_eta = plt.subplot(gs[0:2,0])
        self.ax_sigmas = plt.subplot(gs[1,1])
        self.ax_dn = plt.subplot(gs[2, 1])

        self.lines = []

        self.ax_a1 = self.figure.add_axes([0.05, 0.2, 0.3, 0.05])
        self.ax_a2 = self.figure.add_axes([0.05, 0.15, 0.3, 0.05])
        self.reset_ax = self.figure.add_axes([0.1, 0.1, 0.1, 0.04])
        self.button = Button(self.reset_ax, 'Reset', hovercolor='0.975')
        self.text_fig = plt.figtext(0.75, 0.83, Plotter.create_text(rad, params), \
                        horizontalalignment='center',verticalalignment='center', fontsize=16, \
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    @staticmethod
    def create_text(rad, params):
        a01, a02, d, f, rad, Wl = params
        params_forfunc = [d, a01, a02, f, -rad.z1(f, d, a01, Wl), \
                    rad.z1(f, d, a01, Wl), Wl, rad.trad(f)]
        maxE0 = rad.DimE0(d, a01, a02, f, 0, rad.Wlsum(Wl, f))
        text = r'''
        $d = {:.2f} \qquad f ={:.2f}$ THz \\
        $\eta = {:.2f} \cdot 10^{{-4}}$ \\
        Power $= {:.2f}$ GW \\
        Max $E_0 = {:.2f}$ MV/cm
        '''.format(d, f, rad.eta(*params_forfunc)*10**4, rad.Power(*params_forfunc[:-1]), maxE0)
        return text
        
    @staticmethod
    def vectorization(func, fs, *args):
        func_v = np.vectorize(func)
        res=np.empty(len(fs))
        if args:
            res = func_v(fs, *args)
        else: res = func_v(fs)
        return res
    
    @staticmethod
    def plot_func(ax, ys, fs, ns, labels, ylabel, top=True, bottom=True):
        lines = []
        for i, y in enumerate(ys):
            line = ax.plot(fs, y, label=labels[i])
            lines.append(line)

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
        return lines
    
    @staticmethod
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



    def plot(self, name, ys, fs, ns, labels, params=[]):
        
        if 'eta' in name:
            print('eta')
            self.lines += Plotter.plot_func(self.ax_eta, ys, fs, ns, labels, \
                                            r'Efficiency, $\times 10^{-4}$')
            point = self.ax_eta.scatter([params[0]], [params[1]], marker='o', color='red')
            self.lines.append(point)
        elif 'sigma' in name:
            self.lines += Plotter.plot_func(self.ax_sigmas, ys, fs, ns, labels, \
                                            r'Laser spot-sizes, $\mu$ m', bottom=False)
        elif 'dn' in name:
            self.lines += Plotter.plot_func(self.ax_dn, ys, fs, ns, labels, \
                                            r'Level of nonlinearity', top=False)
            
        print(self.lines)
            
    def sliders(self, params):
        init_a01, init_a02= params
        sliders_param = [(r'$a_{01}$', self.ax_a1, init_a01), (r'$a_{02}$', self.ax_a2, init_a02)]
        a1_slider, a2_slider = Plotter.create_sliders(*sliders_param)
        return a1_slider, a2_slider
        
        
    def update_wrapper(self, rad, a1_slider, a2_slider, params):
        def update(val):
            #point.set_offsets(np.array([[s0_slider.val, N_slider.val]]))
            
            updated_text = Plotter.create_text(rad, [a1_slider.val, a2_slider.val]+params)

            self.text_fig.set_text(updated_text)
            self.figure.canvas.draw_idle()
        return update
    
    def reset_wrapper(self, rad, slider1, slider2, params):
    
        def reset(event):
            slider1.reset()
            slider2.reset()
            updated_text = Plotter.create_text(rad, [slider1.val, slider2.val]+params)
            self.text_fig.set_text(updated_text)
        return reset
    
    def onclick_wrapper(self, params):
        def on_click(event):
            if event.inaxes is self.ax_eta:
                f, d = event.xdata, event.ydata

            print(event.xdata, event.ydata)
        return on_click
    
    def plot_show(self):
        self.figure.tight_layout()
        plt.show()

