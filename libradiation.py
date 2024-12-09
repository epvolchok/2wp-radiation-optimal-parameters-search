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
q = 4.8032 * 10**(-10) #st.q
me = 9.1094 * 10**(-28) #gr
c = 2.9979 * 10**(8) # m/s


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

def Sigma0(tau: float, d: float, a0: float, l: float, f: float)-> float:
    """
    Laser spot size of a narrower laser pulse in dependence of summary energy a the laser system
    # Math: \sigma_{01}^2 = \dfrac{16 d Wl}{3 \pi \tau a_{01}^2}
    """
    return sqrt(16. * d /(3. * pi * tau * a0 * a0 *Dimlessw0(l, f) * Dimlessw0(l, f)))

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

def Rayleigh(l: float, f: float,  tau: float,  d: float, a0: float) -> float:
    """
    Rayleigh lenght
    # Math: \mathcal{R}=\omega_0 \sigma_0^2/2
    """
    return 0.5 * Dimlessw0(l, f) * Sigma0(tau, d, a0, l, f) * Sigma0(tau, d, a0, l, f)

def sigma(tau: float, d: float, a0: float, z: float, z0: float, l: float, f: float) -> float:
    """
    Laser diffraction factor
    # Math: \sigma (z)=\sigma_0 \sqrt{1+(z-z_0)^2/\mathcal{R}^2}
    """
    return Sigma0(tau, d, a0, l, f) * sqrt(1. + (z - z0) * (z - z0) / (Rayleigh(l, f,  tau,  d, a0) * Rayleigh(l, f,  tau,  d, a0)))

def sigmas(tau: float, d: float, a01: float, a02: float, l: float, f: float)->float:
    sigma01 = Sigma0(tau, d, a01, l, f)
    sigma02 = Sigma0(tau, 1.-d, a02, l, f)

    return sigma01, sigma02

def Fsigma(tau: float, d: float, a01: float, a02: float, l: float, f: float, z: float, z0: float) -> float:
    """
    # Math: \mathcal{F}_{\sigma}=\dfrac{\sigma_{01}^2 \sigma_{02}^2 \left|\sigma_2^2-\sigma_1^2\right|}{(\sigma_1^2+\sigma_2^2)^{2}} \exp\left[-\dfrac{3}{8} \frac{\sigma_1^2 \sigma_2^2}{\sigma_1^2+\sigma_2^2}\right]
    """
    sigma01, sigma02 = sigmas(tau, d, a01, a02, l, f)
    sigma1_2 = sigma(tau, d, a01, z, z0, l, f) * sigma(tau, d, a01, z, z0, l, f)
    sigma2_2 = sigma(tau, 1.-d, a02, z, z0, l, f) * sigma(tau, 1.-d, a02, z, z0, l, f)
    sig_sum = sigma1_2 + sigma2_2
    f1 = sigma01 * sigma01 * sigma02 * sigma02 * fabs(sigma2_2 - sigma1_2)/(sig_sum * sig_sum)
    f2 = exp(-3. * sigma1_2 * sigma2_2/ 8. /sig_sum)
    return f1*f2

def E0(a01: float, a02: float, tau: float, l: float, f: float, d: float, \
                                                        z: float, z0: float, R: float) -> float:
    """
    Radiation amplitude
    # Math: \mathcal{E}_0 = \dfrac{ 3 \Phi_{0,1} \Phi_{0, 2} \mathcal{F}_{\sigma}}{2\sqrt{(J_0+2\sqrt{3} R J_1)^2+16 R^2 J_0^2}}
    """
    #sigma01, sigma02 = sigmas(tau, Wl, d, a01, N)
    var = sqrt(3.) * R
    divider = sqrt((j0(var) + 2.*sqrt(3.) * R*j1(var))*(j0(var) + 2.*sqrt(3.) * R*j1(var)) + \
                                                                16. * R * R*j0(var) * j0(var))
    
    return 1.5 * Phi0(a01, tau) * Phi0(a02, tau) * Fsigma(tau, d, a01, a02, l, f, z, z0) / divider


def Power(a01: float, a02: float, tau: float, l: float, f: float, d: float, \
                                            z0: float, R: float, z1: float, z2: float) -> float:
    """
    Radiation power
    # Math: \mathcal{P} = \pi R \int_{z1}^{z2} \mathcal{E}_0^2 dz
    """
    e_2 = lambda z: E0(a01, a02, tau, l, f, d, z, z0, R) * \
                            E0(a01, a02, tau, l, f, d, z, z0, R)
    
    return 0.69 * pi * R * quad(lambda z: e_2(z), z1, z2)[0]

def eta(a01: float, a02: float, tau: float, l: float, f: float, d: float, \
                    z0: float, R: float, z1: float, z2: float, trad: float, Wl: float) -> float:
    """
    Radiation efficiency
    # Math: \eta = \dfrac{\mathcal{P} \tau_{rad}}{\mathcal{W}_L}
    """
    return Power(a01, a02, tau, l, f, d, z0, R, z1, z2) * trad / Wl * 10**(9)


def dnw(a0: float, tau: float, d: float, l: float, f: float)-> float:
    """
    Level of nonlinearity in a wake
    # Math: \delta n_{\Phi} = \Phi_0 \left(1 + \dfrac{8}{\sigma_0^2}\right)
    """

    return Phi0(a0, tau) * (1. + 8./(Sigma0(tau, d, a0, l, f)*Sigma0(tau, d, a0, l, f)))

def Wl0(f: float) -> float:
    return me * c**5 * 10**3 * density(f)/(wp(f) * wp(f) * wp(f))

def Wl(a0: float, tau: float, d: float, l: float, f: float) -> float: # in J
    """
    Energy of a laser pulse
    # Math: \mathcal{W}_L = \dfrac{ 3 \pi}{16} \tau a_{0}^2 \sigma_0^2 \omega_0^2
    """
    return 3. * pi/ 16. * tau * a0 * a0 * Sigma0(tau, d, a0, l, f) * Sigma0(tau, d, a0, l, f) * Dimlessw0(l, f) * Dimlessw0(l, f) * Wl0(f)