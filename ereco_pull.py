import h5py
import numpy as np
import matplotlib.pyplot as plt


with h5py.File("/home/investigator/Pheno/Atmospherics/AtmNuJoint_SimRelease/SuperK.hdf5") as hf:
    pnu = np.asarray(hf["pnu"])
    ereco = np.asarray(hf["evis"])
    nuPDG = np.asarray(hf["ipnu"])
    #weight = np.asarray(hf["tune_weights"]) * np.asarray(hf["inv_flux"])
    sample = np.asarray(hf["itype"])

cut = (sample==12) | (sample==11) | (sample==13)
n, bins, __ = plt.hist((pnu[cut]-ereco[cut])/pnu[cut], bins=100)
plt.show()
print(n/np.sum(n))
print(bins)