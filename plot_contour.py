import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import os

plt.style.use(os.environ["PYNU"] + "/../utils/plot.mplstyle")


#Others
df = pd.read_csv('IC_tracks.csv', sep=", ")
plt.plot(df['gamma'], df['norm'], label='IceCube tracks fit', color="green", linestyle="--")  # Change 'x' and 'y' to your actual column names
df = pd.read_csv('IC_cascades_pp.csv', sep=", ")
plt.plot(df['gamma'], df['norm'], label=r'IceCube cascades fit, $pp$', color="orange", linestyle="--")  # Change 'x' and 'y' to your actual column names
df = pd.read_csv('HESE_pp.csv', sep=", ")
plt.plot(df['gamma'], df['norm'], label='HESE fit', color="crimson", linestyle="--")  # Change 'x' and 'y' to your actual column names
df = pd.read_csv('MESE_pp.csv', sep=", ")
plt.plot(df['gamma'], df['norm'], label='MESE fit', color="cyan", linestyle="--")  # Change 'x' and 'y' to your actual column names

# Load the CSV file
df = pd.read_csv('systs_fit_n=1_g=-2.6_Tau_Ebin9.csv', sep=", ")
# df = pd.read_csv("stats_fit_n=0_noTau_noEbin.csv", sep=", ")
true_norm = 1.0
true_gamma = 2.6

# Pivot the data to get the X, Y, Z format suitable for contour plot
x = df["norm"].values
y = df["gamma"].values
z = df["chi2"].values

# Create a grid for interpolation
xi = np.linspace(min(x), max(x), 100)
yi = np.linspace(min(y), max(y), 100)
xi, yi = np.meshgrid(xi, yi)

# Interpolate the data using griddata
zi = griddata((x, y), z, (xi, yi), method="cubic")


# Create the contour plot
p=plt.contour(yi, xi, zi, levels=(2.71,))
plt.plot(true_gamma, true_norm, "*", label="HK 10-year projection,\n" + r"$\Phi_{Astro}^{\nu_{\tau}+\overline{\nu_{\tau}}}$="+str(true_norm)+r", $\gamma=$"+str(true_gamma))
plt.ylabel("Normalization")
plt.xlabel(r"$\gamma$")
plt.xlim(2.1, 3.1)
plt.ylim(0.5, 4.0)
plt.legend(loc="upper right", fontsize=11)
plt.tight_layout()
plt.show()
