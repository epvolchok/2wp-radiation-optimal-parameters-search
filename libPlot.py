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

        self.lines = {}

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
    def make_array(fs, params, *funcs):
        arrays = []
        num_funcs = len(funcs)
        chunk_size = len(params) // num_funcs
        for i, func in enumerate(funcs):
            param_chunk = params[i * chunk_size: (i + 1) * chunk_size]
            arrays.append(Plotter.vectorization(func, fs, *param_chunk))
        return arrays

    @staticmethod
    def plot_func(ax, ys, fs, ns, labels, ylabel, top=True, bottom=True):
        lines = []
        for i, y in enumerate(ys):
            line = ax.plot(fs, y, label=labels[i])
            lines.append(line[0])


        ax.legend()
        ax.set_ylabel(ylabel)
        ax.set_xscale('log')
        
        ax_twin = ax.twiny()
        ax_twin.plot(ns, ys[-1], visible=False) #color=line[0].get_color()
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
    def plot_eta(ax, ys, ns, ylabel, params):
        extent = [0.5, 100., 0.01, 0.5]
        map_eta = ax.imshow(ys, origin='lower', aspect='auto', extent=extent)
        ax.set_xlabel(r'Frequency, THz')
        ax.set_ylabel(r'Energy share, $\mathcal{W}_{L1}/\mathcal{W}$')
        ax.set_title(ylabel)
        ax.set_xlim(0.5, 100.)

        #point = ax.scatter([params[0]], [params[1]], marker='o', color='red', ec='white', zorder=3)

        ax_twin = ax.twiny()
        ax_twin.plot(ns, ys[0], visible=False)
        ax_twin.set_ylim(0.01, 0.5)
        ax_twin.set_xlabel(r'Density, cm$^{-3}$', labelpad=6.5)
        
        # Добавление точки
        print(f"Point coordinates: x={params[1]}, y={params[0]}")
        point = ax.scatter(
            [params[1]], [params[0]],
            marker='o', color='red', ec='white', zorder=3
        )

        ax.set_zorder(2)
        ax_twin.set_zorder(1)
        ax_twin.patch.set_visible(False)
        #ax.set_zorder(1)  # default zorder is 0 for ax1 and ax2
        #ax.set_frame_on(False)
        return map_eta, point
    
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
            self.lines['eta'], self.lines['point'] = Plotter.plot_eta(self.ax_eta, ys[0], ns, r'Efficiency, $\times 10^{-4}$', params)
        elif 'sigma' in name:
            self.lines['sigma1'], self.lines['sigma2'] = Plotter.plot_func(self.ax_sigmas, \
                                ys, fs, ns, labels, r'Laser spot-sizes, $\mu$m', bottom=False)
        elif 'dn' in name:
            self.lines['dn1'], self.lines['dn2'] = Plotter.plot_func(self.ax_dn, ys, fs, ns, labels, \
                                            r'Level of nonlinearity', top=False)
            
    def sliders(self, params):
        init_a01, init_a02= params
        sliders_param = [(r'$a_{01}$', self.ax_a1, init_a01), (r'$a_{02}$', self.ax_a2, init_a02)]
        a1_slider, a2_slider = Plotter.create_sliders(*sliders_param)
        return a1_slider, a2_slider
        
    def updater(self, rad, a01, a02, params, d, fs, Wl):

        updated_text = Plotter.create_text(rad, [a01, a02]+params)
        self.text_fig.set_text(updated_text)

        sigmas1, sigmas2 = Plotter.make_array(fs, [d, a01, Wl, 1. - d, a02, Wl], rad.Dimsigma, rad.Dimsigma)   
        self.lines['sigma1'].set_ydata(sigmas1)
        self.lines['sigma2'].set_ydata(sigmas2)

        dns1, dns2 = Plotter.make_array(fs, [d, a01, Wl, 1. - d, a02, Wl], rad.dnw_en, rad.dnw_en)
        self.lines['dn1'].set_ydata(dns1)
        self.lines['dn2'].set_ydata(dns2)

        self.figure.canvas.draw_idle()

    def update_wrapper(self, a1_slider, a2_slider, params, fs, ds):
        
        def update(val):
            rad = params[-2]
            d = self.lines['point'].get_offsets()[0][1]
            f = self.lines['point'].get_offsets()[0][0]
            print(self.lines['point'].get_offsets()[0], d)
            Wl = params[-1]
            a01 = a1_slider.val
            a02 = a2_slider.val
            params2 = params[:]
            params2[0] = d
            params2[1] = f
            etas = rad.eta_2d(a01, a02, ds, Wl, fs, 10)
            self.lines['eta'].set_data(etas)
            self.updater(rad, a01, a02, params2, d, fs, Wl)
            
        return update
    
    @staticmethod
    def reset_wrapper(slider1, slider2):
    
        def reset(event):
            slider1.reset()
            slider2.reset()
        return reset
    
    def onclick_wrapper(self, a1_slider, a2_slider, params, fs):
        def on_click(event):
            if event.inaxes in [self.ax_eta, self.ax_sigmas, self.ax_dn]:
                if event.inaxes is self.ax_eta:
                    f, d = event.xdata, event.ydata
                    rad = params[-2]
                    Wl = params[-1]
                    a01 = a1_slider.val
                    a02 = a2_slider.val
                    self.lines['point'].set_offsets(np.array([[f, d]]))
                    params2 = params[:]
                    params2[0] = d
                    params2[1] = f
                    self.updater(rad, a01, a02, params2, d, fs, Wl)
                print(event.xdata, event.ydata)
        return on_click
    
    def plot_show(self):
        self.figure.tight_layout()
        plt.show()

