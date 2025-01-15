"""
Copyright (c) 2025 VOLCHOK Evgeniia
for contacts e.p.volchok@gmail.com

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0
"""

from libdimparam import *
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

class RadiationProperties():

    def __init__(self, R: float, l: float, tau: float, z0: float):

        self.R = R        # plasma column radius in c/wp
        self.l = l        # laser wavelength, m
        self.tau = tau    # laser duration, dimensionless
        self.z0 = z0      # focusing point, c/wp (given case, it is the same for both lasers)

    @staticmethod
    def density(f: float) -> float:
        """
        Plasma density in dependence on radiation frequency,
        f in THz
        # Math: \dfrac{\pi m_e f^2}{4 q^2} 10^{24}
        """
        return 0.25 * pi * me * f * f * 10**(24)/q/q
    
    @staticmethod
    def wp(n: float)-> float:
        """
        Plasma frequency
        # Math: \omega_p = \sqrt{\dfrac{4 \pi n q^2}{ m_e}}
        n in cm^-3
        """
        return sqrt(4. * pi * n * q * q/ me)
    
    @staticmethod
    def rad_freq(f: float) -> float:
        """
        Radiation frequency, Hz
        # Math: \dfrac{2 \omega_p}{2 \pi}
        """
        n = RadiationProperties.density(f)
        return RadiationProperties.wp(n)/pi
    
    @staticmethod
    def trad(f):
        """
        Radiation time, in sec.
        """
        n = RadiationProperties.density(f)
        return 100./RadiationProperties.wp(n)
    
    def Dimlessw0(self, f: float ) -> float:
        """
        Dimensionless laser frequency
        # Math: \omega_0/\omega_p
        """
        n = RadiationProperties.density(f)
        return DimensionVars.Dimw0(self.l)/RadiationProperties.wp(n)
    
    def Wl(self, sigma0: float, a0: float, f: float) -> float:
        """
        Dimension energy of a laser pulse, J
        # Math: \mathcal{W}_L = \dfrac{ 3 \pi}{16} \tau a_{0}^2 \sigma_0^2 \omega_0^2
        """
        return 3. * pi/ 16. * self.tau * a0 * a0 * sigma0 * sigma0 * \
                                self.Dimlessw0(f) * self.Dimlessw0(f) * DimensionVars().Wl0(f)
    @staticmethod
    def Wlsum(Wl: float, f: float)-> float:
        """
        Dimensionless laser energy
        """
        n = RadiationProperties.density(f)
        return Wl/DimensionVars.Wl0(n)
    
    def Phi0(self, a0: float) -> float:
        """
        Wakefield amplitude (electrostatic potential)
        # Math: \Phi_0=\dfrac{3}{4} a_0^2 f_{\tau}
        """
        return 0.75 * a0 * a0 * self.f_tau()

    def f_tau(self) -> float:
        """
        Used in Phi0
        # Math: f_{\tau} = \dfrac{\sin(\tau)}{4 - 5 \tau^2/\pi^2+\tau^4/\pi^4}
        """
        t = self.tau
        return sin(t)/(4.- 5.*t * t /pi /pi + t * t * t * t /(pi * pi * pi * pi))
    
    def dnw(self, sigma0: float, a0: float)-> float:
        """
        Level of nonlinearity in a wake
        # Math: \delta n_{\Phi} = \Phi_0 \left(1 + \dfrac{8}{\sigma_0^2}\right)
        """

        return self.Phi0(a0) * (1. + 8./ (sigma0 * sigma0))

    def Rayleigh(self, sigma0: float, f: float) -> float:
        """
        Rayleigh lenght
        # Math: \mathcal{R}=\omega_0 \sigma_0^2/2
        """
        return 0.5 * self.Dimlessw0(f) * sigma0 * sigma0
    
    def sigma(self, sigma0: float, z: float, f: float) -> float:
        """
        Laser diffraction factor
        # Math: \sigma (z)=\sigma_0 \sqrt{1+(z-z_0)^2/\mathcal{R}^2}
        """
        return sigma0 * sqrt(1. + (z - self.z0) * (z - self.z0) / \
                             (self.Rayleigh(sigma0, f) * self.Rayleigh(sigma0, f)))
    
    def Fsigma(self, sigma01: float, sigma02: float, f: float, z: float) -> float:
        """
        # Math: \mathcal{F}_{\sigma}=\dfrac{\sigma_{01}^2 \sigma_{02}^2 \left|\sigma_2^2-\sigma_1^2\right|}{(\sigma_1^2+\sigma_2^2)^{2}} \exp\left[-\dfrac{3}{8} \frac{\sigma_1^2 \sigma_2^2}{\sigma_1^2+\sigma_2^2}\right]
        """
        
        sigma1_2 = self.sigma(sigma01, z, f) * self.sigma(sigma01, z, f)
        sigma2_2 = self.sigma(sigma02, z, f) * self.sigma(sigma02, z, f)
        sig_sum = sigma1_2 + sigma2_2
        f1 = sigma01 * sigma01 * sigma02 * sigma02 * fabs(sigma2_2 - sigma1_2)/(sig_sum * sig_sum)
        f2 = exp(-3. * sigma1_2 * sigma2_2/ 8. /sig_sum)
        return f1*f2


    def E0(self, sigma01: float, sigma02: float, a01: float, a02: float, f: float, z: float) -> float:
        """
        Radiation amplitude
        # Math: \mathcal{E}_0 = \dfrac{ 3 \Phi_{0,1} \Phi_{0, 2} \mathcal{F}_{\sigma}}{2\sqrt{(J_0+2\sqrt{3} R J_1)^2+16 R^2 J_0^2}}
        """
        
        var = sqrt(3.) * self.R
        divider = sqrt((j0(var) + 2.*sqrt(3.) * self.R*j1(var))*(j0(var) + \
                        2.*sqrt(3.) * self.R*j1(var)) + 16. * self.R * self.R*j0(var) * j0(var))
        
        return 1.5 * self.Phi0(a01) * self.Phi0(a02) * self.Fsigma(sigma01, sigma02, f, z)/ divider
    
    def Power(self, sigma01: float, sigma02: float, a01: float, a02: float, f: float, \
                                                                   z1: float, z2: float) -> float:
        """
        Radiation power in GW
        # Math: \mathcal{P} = \pi R \int_{z1}^{z2} \mathcal{E}_0^2 dz
        """
        e_2 = lambda z: self.E0(sigma01, sigma02, a01, a02, f, z) * \
                                self.E0(sigma01, sigma02, a01, a02, f, z)
        
        return 0.69 * pi * self.R * quad(lambda z: e_2(z), z1, z2)[0]
    
    def z1(self, f: float, d: float, a0: float, Wl: float) -> float:
        """
        Integration limits
        """
        return 5. * self.Rayleigh(f, d, a0, RadiationProperties().Wlsum(Wl, f))
    
    def eta(self, sigma01: float, sigma02: float, a01: float, a02: float, f: float, \
                                            z1: float, z2: float, trad: float, Wlsum) -> float:
        """
        Radiation efficiency
        # Math: \eta = \dfrac{\mathcal{P} \tau_{rad}}{\mathcal{W}_L}
        """
        return self.Power(sigma01, sigma02, a01, a02, f, z1, z2, Wlsum) * trad / Wlsum * 10**(9)