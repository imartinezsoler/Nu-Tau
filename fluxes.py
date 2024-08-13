import numpy as np


""" Astro Flux """


def Astrf(ene, norm=1, gamma=2.37):
    flux = norm * 4 * np.pi * (1.44e-18) * (np.power(ene / 1e5, -gamma))
    return flux
