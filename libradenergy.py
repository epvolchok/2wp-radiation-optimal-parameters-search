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
from libdimparam import *
import multiprocessing as mp
from functools import partial
import numpy as np

class EnergyDependence(RadProp):

    def __init__(self, R: float, l: float, tau:float, z0: float):
        super().__init__(R, l, tau, z0)
        self.obj = RadProp(self.R, self.l, self.tau, self.z0)

    def Sigma0(self, d: float, a0: float, f: float, Wlsum: float)-> float:
        """
        Laser spot size of a narrower laser pulse in dependence of summary energy a the laser system
        # Math: \sigma_{01}^2 = \dfrac{16 d Wl}{3 \pi \tau a_{01}^2}
        Wlsum is dimensionless.
        """
        return sqrt(16. * d*Wlsum /(3. * pi * self.tau * a0 * a0 * self.Dimlessw0(f) * self.Dimlessw0(f)))
    
    def Dimsigma(self, f: float, d: float, a0: float, Wl: float) -> float: 
        """
        Dimension sigma, mkm
        """
        n = RadProp.density(f)
        wp_n = RadProp.wp(n)
        sigma0 = self.Sigma0(d, a0, f, RadProp.Wlsum(Wl, f))
        return DimensionVars().Dimsigma(sigma0, wp_n)
    
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
    
    def dnw_en(self, f, d, a0, Wl):
        sigma0 = self.Sigma0(d, a0, f, RadProp.Wlsum(Wl, f))
        return super().dnw(sigma0, a0)
    
    def sigma(self,  d: float, a0: float, z: float, f: float, Wlsum: float) -> float:
        """
        Laser diffraction factor
        # Math: \sigma (z)=\sigma_0 \sqrt{1+(z-z_0)^2/\mathcal{R}^2}
        """
        return self.obj.sigma(self.Sigma0(d, a0, f, Wlsum), z, f)
    
    def sigmas(self, d: float, a01: float, a02: float, f: float, Wlsum: float)->float:
        """
        Useful function, will be used in Fsigma
        """
        sigma01 = self.Sigma0(d, a01, f, Wlsum)
        sigma02 = self.Sigma0(1.-d, a02, f, Wlsum)
        return sigma01, sigma02

    def Fsigma(self, d: float, a01: float, a02: float, f: float, z: float, Wlsum: float) -> float:
       
        sigma01, sigma02 = self.sigmas(d, a01, a02, f, Wlsum)
        
        return self.obj.Fsigma(sigma01, sigma02, f, z)


    def E0(self, d: float, a01: float, a02: float, f: float, z: float, Wlsum: float) -> float:
        """
        Radiation amplitude
        """
        sigma01, sigma02 = self.sigmas(d, a01, a02, f, Wlsum)
        return self.obj.E0(sigma01, sigma02, a01, a02, f, z)
    
    def DimE0(self, d: float, a01: float, a02: float, f: float, z: float, Wlsum: float) -> float:
        """
        Dimension radiation amplitude
        """
        n = RadProp.density(f)
        wp_n = RadProp.wp(n)
        return DimensionVars().DimE0(self.E0(d, a01, a02, f, z, Wlsum), wp_n)
    
    def Power(self, d: float, a01: float, a02: float, f: float, \
                                                    z1: float, z2: float, Wl: float) -> float:
        """
        Radiation power in GW
        # Math: \mathcal{P} = \pi R \int_{z1}^{z2} \mathcal{E}_0^2 dz
        """
        Wlsum = RadProp.Wlsum(Wl, f)
        sigma01, sigma02 = self.sigmas(d, a01, a02, f, Wlsum)

        return self.obj.Power(sigma01, sigma02, a01, a02, f, z1, z2)
    
    def eta(self, d: float, a01: float, a02: float, f: float, \
                                        z1: float, z2: float, Wl: float, trad: float) -> float:
        """
        Radiation efficiency
        # Math: \eta = \dfrac{\mathcal{P} \tau_{rad}}{\mathcal{W}_L}
        """

        P = self.Power(d, a01, a02, f, z1, z2, Wl)

        eff = P * trad / Wl * 10**(9)
        return eff
    
    def eta_f(self, d, a01, a02, Wl, f):
        return self.eta(d, a01, a02, f, -self.z1(f, d, a01, Wl), \
                   self.z1(f, d, a01, Wl), Wl, RadProp.trad(f))

    def eta_fs(self, a01, a02, d, Wl, fs, npr=5):
        partial_eta_f = partial(self.eta_f, d, a01, a02, Wl)
        with mp.Pool(processes=npr) as p:
            etas = p.map(partial_eta_f, fs)
        etas = np.array(etas) * 10**4    
        return etas
    
    def eta_2d(self, a01, a02, ds, Wl, fs, npr=5):
        etas = np.empty((len(ds), len(fs)))
        for id_, d in enumerate(ds):
            etas[id_] = self.eta_fs(a01, a02, d, Wl, fs, npr)
        return etas