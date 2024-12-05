from math import sqrt, pi, exp, sin, fabs
from scipy.special import j0, j1
from scipy.integrate import quad

"""
    Library for calculating characteristics (amplitude, power, efficiency) of 2wp radiation
    in the scheme with couner-propagating laser pulses and searching for optimal parameters.

    tau - laser pulse duration
    a0 - amplitude of laser field
    omega0 - laser frequency
    sigma0 - laser spot size
    z0 - focus coordinate
    z - longitudinal coordinate
    trad - radiation duration
    Wl - full energy of laser system
    d - fraction of Wl fell on a narrow laser pulse
"""

#physical constants
q = 4.8032 * 10**(-10)
me = 9.1094 * 10**(-28)
c = 2.9979 * 10**(8)


def density(f: float) -> float:
    """
    Plasma density in dependence on radiation frequency
    """
    return 0.25 * pi * me * f * f * 10**(24)/q/q

def wp(f: float)-> float:
    """
    Plasma frequency
    # Math: \omega_p = \sqrt{\dfrac{4 \pi n q^2}{ m_e}}
    """
    return sqrt(4. * pi * density(f) * q * q/ me)

def Dimw0(l: float) -> float:
    """
    Dimension laser frequency
    # Math: \omega_0 = \dfrac{2 \pi c}{\lambda}
    """
    return 2. *pi *c / l

def Dimlessw0(l: float, f: float) -> float:
    """
    Dimensionless laser frequency
    # Math: \omega_0/\omega_p
    """
    return Dimw0(l)/wp(f)

def rad_freq(f: float) -> float:
    """
    Radiation frequency
    # Math: \dfrac{2 \omega_p}{2 \pi}
    """
    return wp(f)/pi

def Dimtau(tau, f):
    return tau/wp(f)

def Dimsigma(sigma, f): 
    return sigma*c/wp(f)

def f_tau(tau: float) -> float:
    """
    # Math: f_{\tau} = \dfrac{\sin(\tau)}{4 - 5 \tau^2/\pi^2+\tau^4/\pi^4}
    """
    return sin(tau)/(4.- 5.*tau * tau /pi /pi + tau * tau * tau * tau /(pi * pi * pi * pi))

def Phi0(a0: float, tau: float) -> float:
    """
    Wakefield amplitude
    # Math: \Phi_0=\dfrac{3}{4} a_0^2 f_{\tau}
    """
    return 0.75 * a0 * a0 * f_tau(tau)

def Rayleigh(l:float, f:float, sigma0: float) -> float:
    """
    Rayleigh lenght
    # Math: \mathcal{R}=\omega_0 \sigma_0^2/2
    """
    return 0.5 * Dimlessw0(l, f) * sigma0 * sigma0

def sigma(sigma0: float, z: float, z0: float, l:float, f:float) -> float:
    """
    Laser diffraction factor
    # Math: \sigma (z)=\sigma_0 \sqrt{1+(z-z_0)^2/\mathcal{R}^2}
    """
    return sigma0 * sqrt(1. + (z - z0) * (z - z0) / (Rayleigh(l, f, sigma0) * Rayleigh(l, f, sigma0)))

def Fsigma(sigma01: float, N: float, z: float, z0: float, l:float, f:float) -> float:
    """
    # Math: \mathcal{F}_{\sigma}=\dfrac{\sigma_{01}^2 \sigma_{02}^2 \left|\sigma_2^2-\sigma_1^2\right|}{(\sigma_1^2+\sigma_2^2)^{2}} \exp\left[-\dfrac{3}{8} \frac{\sigma_1^2 \sigma_2^2}{\sigma_1^2+\sigma_2^2}\right]
    """
    sigma02 = N * sigma01
    sigma1_2 = sigma(sigma01, z, z0, l, f) * sigma(sigma01, z, z0, l, f)
    sigma2_2 = sigma(sigma02, z, z0, l, f) * sigma(sigma02, z, z0, l, f)
    sig_sum = sigma1_2 + sigma2_2
    f1 = sigma01 * sigma01 * sigma02 * sigma02 * fabs(sigma2_2 - sigma1_2)/(sig_sum * sig_sum)
    f2 = exp(-3. * sigma1_2 * sigma2_2/ 8. /sig_sum)
    return f1*f2

def E0(a01: float, a02: float, tau: float, l:float, f:float, sigma01: float, N: float, z: float, \
                                                                 z0: float, R: float) -> float:
    """
    Radiation amplitude
    # Math: \mathcal{E}_0 = \dfrac{ 3 \Phi_{0,1} \Phi_{0, 2} \mathcal{F}_{\sigma}}{2\sqrt{(J_0+2\sqrt{3} R J_1)^2+16 R^2 J_0^2}}
    """

    sigma02 = N * sigma01
    var = sqrt(3.) * R
    divider = sqrt((j0(var) + 2.*sqrt(3.) * R*j1(var))*(j0(var) + 2.*sqrt(3.) * R*j1(var)) + \
                                                                16. * R * R*j0(var) * j0(var))
    
    return 1.5 * Phi0(a01, tau) * Phi0(a02, tau) * Fsigma(sigma01, N, z, z0, l, f) / divider


def Power(a01: float, a02: float, tau: float, l:float, f:float, sigma01: float, N: float, \
                                            z0: float, R: float, z1: float, z2: float) -> float:
    """
    Radiation power
    # Math: \mathcal{P} = \pi R \int_{z1}^{z2} \mathcal{E}_0^2 dz
    """
    e_2 = lambda z: E0(a01, a02, tau, l, f, sigma01, N, z, z0, R) * \
                            E0(a01, a02, tau, l, f, sigma01, N, z, z0, R)
    
    return 0.69 * pi * R * quad(lambda z: e_2(z), z1, z2)[0]

def eta(a01: float, a02: float, tau: float, l:float, f:float, sigma01: float, N: float, \
                    z0: float, R: float, z1: float, z2: float, trad: float, Wl: float) -> float:
    """
    Radiation efficiency
    # Math: \eta = \dfrac{\mathcal{P} \tau_{rad}}{\mathcal{W}_L}
    """
    return Power(a01, a02, tau, l, f, sigma01, N, z0, R, z1, z2) * trad / Wl * 10**(9)

def dnw(a0: float, tau: float, sigma0: float)-> float:
    """
    Level of nonlinearity in a wake
    # Math: \delta n_{\Phi} = \Phi_0 \left(1 + \dfrac{8}{\sigma_0^2}\right)
    """

    return Phi0(a0, tau) * (1. + 8./(sigma0*sigma0))

def Wl0(f: float) -> float:
    return me * c**5 * 10**3 * density(f)/(wp(f) * wp(f) * wp(f))

def Wl(a0: float, tau: float, sigma0: float, l: float, f: float) -> float: # in J
    """
    Energy of a laser pulse
    # Math: \mathcal{W}_L = \dfrac{ 3 \pi}{16} \tau a_{0}^2 \sigma_0^2 \omega_0^2
    """
    return 3. * pi/ 16. * tau * a0 * a0 * sigma0 * sigma0 * Dimlessw0(l, f) * Dimlessw0(l, f) * Wl0(f)