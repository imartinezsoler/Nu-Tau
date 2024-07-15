import numpy as np


""" Astro Flux """


def Astrf(ene, gamma=-2.37):
    flux = 4 * np.pi * (1.44e-18) * (np.power(ene / 1e5, gamma))
    return flux
