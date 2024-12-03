import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider
#import matplotlib as mpl
from matplotlib import gridspec

from libradiation import *

fig = plt.figure(figsize=(10, 6))
gs = gridspec.GridSpec(1, 2)

def eta_sigma(sigma01: float, N: float) -> float:
    return Fsigma(sigma01, N, 0., 0., 1.)*Fsigma(sigma01, N, 0., 0., 1.)/((1. + N*N)*sigma01*sigma01)

eta_v = np.vectorize(eta_sigma)

tau = 3.42


sigmas_v = np.linspace(1., 3.5, 100)
ns = np.linspace(1., 5., 100)

extent = [1., 3.5, 1., 5.]

sigmas_mesh, ns_mesh = np.meshgrid(sigmas_v, ns)

etas = eta_v(sigmas_mesh, ns_mesh)

init_a0 = 0.7
init_s0 = 1.26
init_N = 2.41

ax_F = plt.subplot(gs[0,1])
ax_F.imshow(etas, cmap='bwr', origin='lower', extent=extent) #, extent=extent
point = ax_F.scatter([init_s0], [init_N], marker='o', color='black')
#ax.contour(sigmas_v, ns, etas, extent=extent)

#np.unravel_index(np.argmax(Fsigmas, axis=None), Fsigmas.shape)
args = np.unravel_index(np.argmax(etas), etas.shape)
print(f'maximal arguments= {args}')
print(f'maximal params= {sigmas_v[args[1]], ns[args[0]]}')


text_dn1 = plt.text(-3, 4.8, r'dn1 ='+str(dnw(init_a0, tau, init_s0)), horizontalalignment='left',verticalalignment='center')
text_eta = plt.text(-3, 4.5, r'eta ='+str(eta_sigma(init_s0, init_N)), horizontalalignment='left',verticalalignment='center')

ax_a0 = fig.add_axes([0.1, 0.5, 0.3, 0.05])
a0_slider = Slider(
    ax=ax_a0,
    label='a0',
    valmin=0.5,
    valmax=0.8,
    valinit=init_a0,
    valstep=0.01,
)

ax_s0 = fig.add_axes([0.1, 0.4, 0.3, 0.05])
s0_slider = Slider(
    ax=ax_s0,
    label='s0',
    valmin=1.2,
    valmax=3.,
    valinit=init_s0,
    valstep=0.01,
)

ax_N = fig.add_axes([0.1, 0.3, 0.3, 0.05])
N_slider = Slider(
    ax=ax_N,
    label='N',
    valmin=1.,
    valmax=5.,
    valinit=init_N,
    valstep=0.01,
)

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

plt.show()

