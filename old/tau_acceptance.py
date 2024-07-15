import matplotlib.pyplot as plt
import numpy as np
# plt.style.use('/usr/share/matplotlib/mpl-data/stylelib/paper.mplstyle')

# Fraction of nu_taus as function of energy split by samples. Obtained
# from SK joint-fit-simulation
bins = np.array(
    [
        20.0,
        23.91626349,
        28.59938297,
        34.19951893,
        40.8962353,
        48.90425696,
        58.48035476,
        69.93157868,
        83.6251031,
        100.0,
    ]
)
e_bins = (bins[1:] + bins[:-1]) / 2
mge_nu = np.array(
    [
        90.57080592,
        89.53571099,
        62.10569549,
        51.49597251,
        39.33360714,
        30.79407401,
        31.82916894,
        21.21944596,
        14.23255522,
    ]
)
mge_nub = np.array(
    [
        105.655345,
        82.91009712,
        57.96369622,
        49.15908413,
        36.68588368,
        31.54985997,
        22.74524788,
        19.07665951,
        18.34294184,
    ]
)
mre_nu = np.array(
    [
        152.53414353,
        128.54000859,
        87.40720584,
        78.83787194,
        47.98826988,
        39.41893597,
        32.56346884,
        18.85253459,
        22.28026816,
    ]
)
mre_nub = np.array(
    [
        71.24224962,
        52.17742226,
        39.13306669,
        30.10235899,
        28.09553506,
        19.06482736,
        13.04435556,
        17.05800343,
        11.0375316,
    ]
)
mro = np.array(
    [
        390.33835562,
        240.20821884,
        315.27328723,
        202.67568465,
        142.62362994,
        82.57157523,
        112.59760258,
        97.5845889,
        22.51952052,
    ]
)

# Normalizing
total = mge_nu + mge_nub + mre_nu + mre_nub + mro
mge_nu /= total
mge_nub /= total
mre_nu /= total
mre_nub /= total
mro /= total


# Plotting samples with no flavour information
plt.plot([], [], color="m", label=r"FC multi-GeV $\nu_{e}$-like", linewidth=5)
plt.plot([], [], color="c", label=r"FC multi-GeV $\bar{\nu_{e}}$-like", linewidth=5)
plt.plot([], [], color="r", label=r"FC multi-ring $\nu_{e}$-like", linewidth=5)
plt.plot([], [], color="b", label=r"FC multi-ring $\bar{\nu_{e}}$-like", linewidth=5)
plt.plot([], [], color="g", label=r"FC multi-ring other", linewidth=5)
plt.stackplot(
    e_bins, mge_nu, mge_nub, mre_nu, mre_nub, mro, colors=["m", "c", "r", "b", "g"]
)
plt.xlabel(r"$E_\nu$ (GeV)")
plt.ylabel(r"$\nu_\tau$ fraction")
plt.legend(loc="upper left")
plt.show()
