import pandas as pd
import matplotlib.pyplot as plt
import mpltern

# Step 1: Load data from CSV file
data = pd.read_csv('stats_fit_n=1_g=-2.6_composition=1-1-1_cutoff=20GeV_Tau_Ebin9.csv', sep=", ")

# Step 2: Extract the ternary components (e, mu, tau) and function values (chi2)
e = data['e']
mu = data['mu']
tau = data['tau']
chi2 = data['chi2']

# Step 3: Set up the ternary plot
fig, ax = plt.subplots(subplot_kw={'projection': 'ternary'})

# Step 4: Create the filled contour plot using the ternary coordinates
contour_filled = ax.tricontourf(e, mu, tau, chi2, levels=(0.0, 1.0, 3.0), cmap='viridis')

# Step 5: Overlay contour lines
contour_lines = ax.tricontour(e, mu, tau, chi2, levels=(0.0, 1.0, 3.0), colors='black', linewidths=0.7)

# Step 6: Add a colorbar
cbar = fig.colorbar(contour_filled, ax=ax, shrink=0.8)
cbar.set_label("Chi2 Values")

# Step 7: Set corner labels for ternary plot
ax.set_tlabel(r"$\nu_e$ only")
ax.set_llabel(r"$\nu_\mu$ only")
ax.set_rlabel(r"$\nu_\tau$ only")

# Step 8: Show the plot
plt.show()
