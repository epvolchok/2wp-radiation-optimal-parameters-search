"""
Copyright (c) 2025 VOLCHOK Evgeniia
for contacts e.p.volchok@gmail.com

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0
"""
from libradiationproperties import RadiationProperties as RadProp
from math import sqrt, pi

class EnergyDependence(RadProp):

    def __init__(self, R: float, l: float, tau:float, z0: float):
        super().__init__(R, l, tau, z0)

    def Sigma0(self, d: float, a0: float, f: float, Wlsum: float)-> float:
        """
        Laser spot size of a narrower laser pulse in dependence of summary energy a the laser system
        # Math: \sigma_{01}^2 = \dfrac{16 d Wl}{3 \pi \tau a_{01}^2}
        Wlsum is dimensionless.
        """
        return sqrt(16. * d*Wlsum /(3. * pi * self.tau * a0 * a0 * self.Dimlessw0(f) * self.Dimlessw0(f)))
    
    def Rayleigh(self, f: float,  d: float, a0: float, Wlsum: float) -> float:
        """
        Rayleigh lenght
        # Math: \mathcal{R}=\omega_0 \sigma_0^2/2
        """
        sigma0 = self.Sigma0(d, a0, f, Wlsum)
        Rl = super().Rayleigh(sigma0, f)
        return Rl
    
    def Wl(self, a0: float, f: float, d: float, Wlsum: float) -> float:
        """
        Energy of a laser pulse, J
        # Math: \mathcal{W}_L = \dfrac{ 3 \pi}{16} \tau a_{0}^2 \sigma_0^2 \omega_0^2
        """
        sigma0 = self.Sigma0(d, a0, f, Wlsum)
        
        return super().Wl(sigma0, a0, f)
    
    def sigma(self,  d: float, a0: float, z: float, f: float, Wlsum: float) -> float:
        """
        Laser diffraction factor
        # Math: \sigma (z)=\sigma_0 \sqrt{1+(z-z_0)^2/\mathcal{R}^2}
        """
        return super().sigma(self.Sigma0(d, a0, f, Wlsum), z, f)
    
    def sigmas(self, d: float, a01: float, a02: float, f: float, Wlsum: float)->float:
        """
        Useful function, will be used in Fsigma
        """
        sigma01 = self.Sigma0(d, a01, f, Wlsum)
        sigma02 = self.Sigma0(1.-d, a02, f, Wlsum)
        return sigma01, sigma02

    def Fsigma(self, d: float, a01: float, a02: float, f: float, z: float, Wlsum: float) -> float:
       
        sigma01, sigma02 = self.sigmas(d, a01, a02, f, Wlsum)
        
        return super().Fsigma(sigma01, sigma02, f, z)


    def E0(self, d: float, a01: float, a02: float, f: float, z: float, Wlsum: float) -> float:
        """
        Radiation amplitude
        """
        sigma01, sigma02 = self.sigmas(d, a01, a02, f, Wlsum)
        return super().E0(sigma01, sigma02, a01, a02, f, z)
    
    def Power(self, d: float, a01: float, a02: float, f: float, \
                                                    z1: float, z2: float, Wlsum: float) -> float:
        """
        Radiation power in GW
        # Math: \mathcal{P} = \pi R \int_{z1}^{z2} \mathcal{E}_0^2 dz
        """
        sigma01, sigma02 = self.sigmas(d, a01, a02, f, Wlsum)

        return super().Power(sigma01, sigma02, a01, a02, f, z1, z2)
    
    def eta(self, d: float, a01: float, a02: float, f: float, \
                                        z1: float, z2: float, trad: float, Wlsum: float) -> float:
        """
        Radiation efficiency
        # Math: \eta = \dfrac{\mathcal{P} \tau_{rad}}{\mathcal{W}_L}
        """
        return self.Power(d, a01, a02, f, z1, z2, Wlsum) * trad / Wlsum * 10**(9)
