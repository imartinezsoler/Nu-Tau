import numpy as np
from scipy.optimize import minimize
from taus_from_SK import HK
from scipy.interpolate import griddata
from scipy.optimize import minimize


def expected(hk, norm=1, gamma=2.5, tauNN=False, bins="default"):
    return hk.binned(norm, gamma, tauNN, bins)


def observed(hk, norm=1, gamma=2.5, tauNN=False, bins="default"):
    return hk.binned(norm, gamma, tauNN, bins)


def chi2(E, O):
    return 2 * np.sum(E - O + O * np.log(O / E))


def chi2_systs(syst, hk, norm, gamma, tauNN, bins, O):
    nominal = np.array([0, 1, 1, 1, 1, 1, 1, 1, 1])
    sigma = np.array([0.15, 0.2, 0.25, 0.05, 0.2, 0.04, 0.04, 0.03, 0.2])

    """ systematics inside the hk class to implement the cuts easily """
    w_all = np.ones(hk.entries)
    w_atm = np.ones(hk.entries)
    
    """ atm. spectral index """
    w_atm = hk.tilt(syst[0])

    """ atm. normalization """
    w_atm = hk.norm(syst[1])


    """ tau xsection """
    w_all = hk.tau_xsec(syst[2])

    """ DIS """
    w_all = hk.dis(syst[3])

    """ NC/CC """
    w_all = hk.nc_cc(syst[4])


    """ multi-ring PID """
    w_all = hk.mr_pid(syst[5])

    """ multi-ring other"""
    w_all = hk.mr_other(syst[6])

    """ multi-ring nu/nubar"""
    w_all = hk.nu_nubar(syst[7])

    """ pc stop/thru """
    w_all = hk.stop_thru(syst[8])

    E = hk.binned(norm, gamma, tauNN, bins, w_all=w_all, w_atm=w_atm)


    return 2 * np.sum(E - O + O * np.log(O / E)) + np.sum(((syst-nominal)/sigma)**2)




if __name__ == "__main__":
    hk = HK()

    print(f"norm, gamma, chi2")
    Observed = observed(hk, norm=1, gamma=2.6, tauNN=True)
    gamma = np.linspace(1.6, 3.1, 30)
    norm = np.linspace(0.0, 4, 20)
    nominal = np.array([0, 1, 1, 1, 1, 1, 1, 1, 1])
    bounds = ((-1,1), (0.5, 2), (0.5, 2), (0.5, 2), (0.5, 2), (0.5, 2), (0.5, 2), (0.5, 2), (0.5, 2))
    for g in gamma:
        for n in norm:
            #Expected = expected(hk, norm=n, gamma=g, tauNN=True)
            #X2 = chi2(Expected[Observed>0], Observed[Observed>0])
            #print(f"{n}, {g}, {X2}")

            X2 = chi2_systs(nominal, hk, n, g, True, "default", Observed)
            res = minimize(chi2_systs, nominal, args=(hk, n, g, True, "default", Observed), method="L-BFGS-B", bounds=bounds, options={'disp':False})
            print(f"{n}, {g}, {res.fun}")
