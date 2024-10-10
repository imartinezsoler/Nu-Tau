import numpy as np


""" Astro Flux """


def Astrf(ene, norm=1, gamma=2.37, fraction=1/3):
    cutoff = 0 # GeV
    flux = np.zeros_like(ene)
    flux[ene>cutoff] = fraction * norm * 4 * np.pi * (1.44e-18) * (np.power(ene[ene>cutoff] / 1e5, -gamma))
    return flux
