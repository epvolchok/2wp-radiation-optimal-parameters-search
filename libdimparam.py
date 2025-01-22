"""
Copyright (c) 2025 VOLCHOK Evgeniia
for contacts e.p.volchok@gmail.com

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0
"""
from math import pi, sqrt

"""
Dimension coefficinets and physical constants
"""

#physical constants
q = 4.8032 * 10**(-10) #st.q
me = 9.1094 * 10**(-28) #gr
c = 2.9979 * 10**(8) # m/s
electron_ch = 1.6 * 10**(-19) # coulomb

class DimensionVars():

    @staticmethod
    def Dimw0(l: float) -> float:
        """
        Dimension laser frequency
        # Math: \omega_0 = \dfrac{2 \pi c}{\lambda}
        returns 1/sec
        """
        return 2. *pi *c / l
    
    @staticmethod
    def Dimtau(tau: float, wp: float) -> float:
        """
        Dimension laser duration
        # Math: \tau_{dim} = \tau \omega_p^{-1}
        returns sec
        """
        return tau/wp

    @staticmethod
    def Dimsigma(sigma: float, wp: float) -> float: 
        """
        Dimension laser spot-size
        # Math: \sigma \dfrac{c}{\omega_p}
        returns micro m
        """
        return sigma * c/wp * 10**6

    @staticmethod
    def Wl0(n: float) -> float:
        """
        Dimension coefficient of energy
        # Math: \mathcal{W}_{L0} = n m_e c^2 \dfrac{c^3}{\omega_p^3} * 10^3
        returns J
        """
        
        wp_n = sqrt(4. * pi * n * q *q/ me)
        return me * c**5 * 10**3 * n/(wp_n * wp_n * wp_n)
    
    @staticmethod
    def DimE0(E0:float, wp: float) -> float:

        return E0 * me * c * wp/ electron_ch * 10**(-11)
