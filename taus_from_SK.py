import h5py
import numpy as np
import matplotlib.pyplot as plt
from fluxes import Astrf


""" SK/HK part """

sg_ebins = np.array(
    [
        0.1,
        0.25118864315095796,
        0.3981071705534973,
        0.630957344480193,
        1.0,
        1.584893192461114,
    ]
)
mg_4_ebins = np.array([1.0, 2.5118864315095797, 5.011872336272725, 10.0, 100.0])
mg_2_ebins = np.array([1.3, 2.5118864315095797, 100.0])
mr_3_ebins = np.array([1.0, 2.5118864315095797, 5.011872336272725, 100.0])
mr_4_ebins = np.array(
    [0.1, 1.3299998745408388, 2.5118864315095797, 5.011872336272725, 100.0]
)
pcs_ebins = np.array([0.1, 2.5118864315095797, 100.0])
pct_ebins = np.array(
    [0.1, 1.32739445772974, 2.5118864315095797, 5.011872336272725, 100.0]
)
upmus_ebins = np.array(
    [1.584893192461114, 2.4945947269429536, 4.9888448746001215, 100000.0]
)
upmut_ebins = np.array([0.1, 100000.0])
z10bins = np.array(
    [-1, -0.839, -0.644, -0.448, -0.224, 0.0, 0.224, 0.448, 0.644, 0.839, 1.0]
)
z10bins_up = np.array([-1, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0])
z1bins = np.array([-1, 1.0])

EnergyBins = {
    0: sg_ebins,
    1: sg_ebins,
    2: sg_ebins,
    3: sg_ebins,
    4: sg_ebins,
    5: sg_ebins,
    6: sg_ebins,
    7: mg_4_ebins,
    8: mg_4_ebins,
    9: mg_2_ebins,
    10: mr_3_ebins,
    11: mr_3_ebins,
    12: mr_4_ebins,
    13: mg_4_ebins,
    14: pcs_ebins,
    15: pct_ebins,
    16: upmus_ebins,
    17: upmut_ebins,
    18: upmut_ebins,
    19: sg_ebins,
    20: sg_ebins,
    21: sg_ebins,
    22: sg_ebins,
    23: sg_ebins,
    24: mg_4_ebins,
    25: mg_4_ebins,
    26: mg_4_ebins,
    27: mg_2_ebins,
    28: mg_2_ebins,
}
CTBins = {
    0: z10bins,
    1: z1bins,
    2: z1bins,
    3: z10bins,
    4: z10bins,
    5: z1bins,
    6: z1bins,
    7: z10bins,
    8: z10bins,
    9: z10bins,
    10: z10bins,
    11: z10bins,
    12: z10bins,
    13: z10bins,
    14: z10bins,
    15: z10bins,
    16: z10bins_up,
    17: z10bins_up,
    18: z10bins_up,
    19: z10bins,
    20: z10bins,
    21: z10bins,
    22: z10bins,
    23: z10bins,
    24: z10bins,
    25: z10bins,
    26: z10bins,
    27: z10bins,
    28: z10bins,
}


all_sample_names = {
    0: "sk1-3_fc_subgev_1ring_elike_0decaye",
    1: "sk1-3_fc_subgev_1ring_elike_1decaye",
    2: "sk1-5_fc_1ring_ncpi0",
    3: "sk1-3_fc_subgev_1ring_mulike_0decaye",
    4: "sk1-3_fc_subgev_1ring_mulike_1decaye",
    5: "sk1-3_fc_subgev_1ring_mulike_2decaye",
    6: "sk1-5_fc_2ring_ncpi0",
    7: "sk1-3_fc_multigev_1ring_nuelike",
    8: "sk1-3_fc_multigev_1ring_nuebarlike",
    9: "sk1-3_fc_multigev_1ring_mulike",
    10: "sk1-5_fc_multigev_multiring_nuelike",
    11: "sk1-5_fc_multigev_multiring_nuebarlike",
    12: "sk1-5_fc_multigev_multiring_mulike",
    13: "sk1-5_fc_multigev_multiring_other",
    14: "sk1-5_pc_stop",
    15: "sk1-5_pc_thru",
    16: "sk1-5_upmu_stop",
    17: "sk1-5_upmu_thru_nonshowering",
    18: "sk1-5_upmu_thru_showering",
    19: "sk4-5_fc_subgev_1ring_nuelike",
    20: "sk4-5_fc_subgev_1ring_nuebarlike_0neutron",
    21: "sk4-5_fc_subgev_1ring_nuebarlike_1neutron",
    22: "sk4-5_fc_subgev_1ring_numulike",
    23: "sk4-5_fc_subgev_1ring_numubarlike",
    24: "sk4-5_fc_multigev_1ring_nuelike",
    25: "sk4-5_fc_multigev_1ring_nuebarlike_0neutron",
    26: "sk4-5_fc_multigev_1ring_nuebarlike_1neutron",
    27: "sk4-5_fc_multigev_1ring_numulike",
    28: "sk4-5_fc_multigev_1ring_numubarlike",
}
tau_sample_names = {
    7: "sk1-3_fc_multigev_1ring_nuelike",
    8: "sk1-3_fc_multigev_1ring_nuebarlike",
    10: "sk1-5_fc_multigev_multiring_nuelike",
    11: "sk1-5_fc_multigev_multiring_nuebarlike",
    13: "sk1-5_fc_multigev_multiring_other",
    14: "sk1-5_pc_stop",
    15: "sk1-5_pc_thru",
}
hk_tau_sample_names = {
    7: r"HK FC multigev 1ring $\nu_{e}$-like",
    8: r"HK FC multigev 1ring $\overline{\nu_{e}}$-like",
    10: r"HK FC multigev multiring $\nu_{e}$-like",
    11: r"HK FC multigev multiring $\overline{\nu_{e}}$-like",
    13: "HK FC multigev multiring other",
    14: "HK PC stop",
    15: "HK PC thru",
}

# Exposure in units of HK years, i.e. 188 kton
sk_days = 6500
sk123_days = 2795
sk45_days = sk_days - sk123_days
sk_expfv = 27.2  # kton
hk_scale = 6.9  # taking into account expanded FV in SK but not in HK
how_much_skmc = 50
hk_years = 10
tau_sample_exposure = {
    7: hk_years * hk_scale * 365.25 / sk123_days / how_much_skmc,
    8: hk_years * hk_scale * 365.25 / sk123_days / how_much_skmc,
    10: hk_years * hk_scale * 365.25 / sk_days / how_much_skmc,
    11: hk_years * hk_scale * 365.25 / sk_days / how_much_skmc,
    13: hk_years * hk_scale * 365.25 / sk_days / how_much_skmc,
    14: hk_years * hk_scale * 365.25 / sk_days / how_much_skmc,
    15: hk_years * hk_scale * 365.25 / sk_days / how_much_skmc,
}

with h5py.File(
    "/home/investigator/Pheno/SK-DC/tune_mc/final_SuperK_fullMC_3_2.h5"
) as hf:
    pnu = np.asarray(hf["pnu"])
    ipnu = np.asarray(hf["ipnu"])
    ereco = np.asarray(hf["evis"])
    nuPDG = np.asarray(hf["ipnu"])
    weight = np.asarray(hf["tune_weights"]) * np.asarray(hf["inv_flux"])
    sample = np.asarray(hf["itype"])
    w_oscbf = np.asarray(hf["tune_weights"]) * np.asarray(hf["w_no"])
    cosz_reco = np.asarray(hf["recodirZ"])
    cosz_true = np.asarray(hf["dirnuZ"])
    current = np.array(hf["current"][()]).astype(str)


""" Astro Flux """


def Astrf(ene, gamma=-2.37):
    flux = 4 * np.pi * (1.44e-18) * (np.power(ene / 1e5, gamma))
    return flux


""" Prompt flux """
# Ignored for now
# prompt_flux = pd.read_csv('/home/investigator/Pheno/AstroTau/Nu-Tau/NuTauSK/Nu_Tau_prompt_H3a_SIBYLL23C_pr.txt', sep='\s+', engine='python')
gammas = [-2.4, -2.8, -3.0]
# gammas = [-2.4]
for g in gammas:
    """ Plot Ereco histograms """
    for s_id, s_name in hk_tau_sample_names.items():
        cut = sample == s_id
        taucut = cut & (np.abs(ipnu) == 16) & (current == "CC")
        ecut = cut & (np.abs(ipnu) == 12) & (current == "CC")
        mucut = cut & (np.abs(ipnu) == 14) & (current == "CC")
        taunccut = cut & (np.abs(ipnu) == 16) & (current != "CC")
        nccut = cut & (current != "CC")
        plt.figure()
        data = [
            ereco[nccut],
            ereco[mucut],
            ereco[ecut],
            ereco[taucut],
            ereco[taunccut],
            ereco[taucut],
        ]
        w = [
            w_oscbf[nccut] * tau_sample_exposure[s_id],
            w_oscbf[mucut] * tau_sample_exposure[s_id],
            w_oscbf[ecut] * tau_sample_exposure[s_id],
            w_oscbf[taucut] * tau_sample_exposure[s_id],
            weight[taunccut] * Astrf(pnu[taunccut], gamma=g) * tau_sample_exposure[s_id],
            weight[taucut] * Astrf(pnu[taucut], gamma=g) * tau_sample_exposure[s_id],
        ]
        labels = [
            r"Atm. NC",
            r"Atm. $\nu_{\mu}$ CC",
            r"Atm. $\nu_{e}$ CC",
            r"Atm. $\nu_{\tau}$ CC",
            r"Astro. $\nu_{\tau}$ NC",
            r"Astro. $\nu_{\tau}$ CC",
        ]
        plt.hist(
            data,
            bins=EnergyBins[s_id],
            # color="red",
            weights=w,
            label=labels,
            stacked=True
        )

        plt.title(f"{s_name}")
        plt.xlabel(r"$E_{reco} / GeV$")
        plt.ylabel("Events")
        plt.legend()
        plt.grid(True)
        plt.savefig("figs/ereco_" + str(s_id) + "_gamma=" + str(g) + ".png")
        # plt.show()
        plt.close()

    """ Plot Etrue histograms """
    for s_id, s_name in hk_tau_sample_names.items():
        cut = sample == s_id
        taucut = cut & (np.abs(ipnu) == 16) & (current == "CC")
        ecut = cut & (np.abs(ipnu) == 12) & (current == "CC")
        mucut = cut & (np.abs(ipnu) == 14) & (current == "CC")
        taunccut = cut & (np.abs(ipnu) == 16) & (current != "CC")
        nccut = cut & (current != "CC")
        plt.figure()
        data = [
            pnu[nccut],
            pnu[mucut],
            pnu[ecut],
            pnu[taucut],
            pnu[taunccut],
            pnu[taucut],
        ]
        w = [
            w_oscbf[nccut] * tau_sample_exposure[s_id],
            w_oscbf[mucut] * tau_sample_exposure[s_id],
            w_oscbf[ecut] * tau_sample_exposure[s_id],
            w_oscbf[taucut] * tau_sample_exposure[s_id],
            weight[taunccut] * Astrf(pnu[taunccut], gamma=g) * tau_sample_exposure[s_id],
            weight[taucut] * Astrf(pnu[taucut], gamma=g) * tau_sample_exposure[s_id],
        ]
        labels = [
            r"Atm. NC",
            r"Atm. $\nu_{\mu}$ CC",
            r"Atm. $\nu_{e}$ CC",
            r"Atm. $\nu_{\tau}$ CC",
            r"Astro. $\nu_{\tau}$ NC",
            r"Astro. $\nu_{\tau}$ CC",
        ]
        plt.hist(
            data,
            bins=20,
            # color="red",
            weights=w,
            label=labels,
            stacked=True
        )

        plt.title(f"{s_name}")
        plt.xlabel(r"$\log_{10} (E_{\nu} / GeV)$")
        plt.ylabel("Events")
        plt.legend()
        plt.grid(True)
        plt.savefig("figs/etrue_" + str(s_id) + "_gamma=" + str(g) + ".png")
        # plt.show()
        plt.close()

    """ Plot reco zenith histograms """
    for s_id, s_name in hk_tau_sample_names.items():
        cut = sample == s_id
        taucut = cut & (np.abs(ipnu) == 16) & (current == "CC")
        ecut = cut & (np.abs(ipnu) == 12) & (current == "CC")
        mucut = cut & (np.abs(ipnu) == 14) & (current == "CC")
        taunccut = cut & (np.abs(ipnu) == 16) & (current != "CC")
        nccut = cut & (current != "CC")
        plt.figure()
        data = [
            cosz_reco[nccut],
            cosz_reco[mucut],
            cosz_reco[ecut],
            cosz_reco[taucut],
            cosz_reco[taunccut],
            cosz_reco[taucut],
        ]
        w = [
            w_oscbf[nccut] * tau_sample_exposure[s_id],
            w_oscbf[mucut] * tau_sample_exposure[s_id],
            w_oscbf[ecut] * tau_sample_exposure[s_id],
            w_oscbf[taucut] * tau_sample_exposure[s_id],
            weight[taunccut] * Astrf(pnu[taunccut], gamma=g) * tau_sample_exposure[s_id],
            weight[taucut] * Astrf(pnu[taucut], gamma=g) * tau_sample_exposure[s_id],
        ]
        labels = [
            r"Atm. NC",
            r"Atm. $\nu_{\mu}$ CC",
            r"Atm. $\nu_{e}$ CC",
            r"Atm. $\nu_{\tau}$ CC",
            r"Astro. $\nu_{\tau}$ NC",
            r"Astro. $\nu_{\tau}$ CC",
        ]
        plt.hist(
            data,
            bins=CTBins[s_id],
            # color="red",
            weights=w,
            label=labels,
            stacked=True
        )

        plt.title(f"{s_name}")
        plt.xlabel(r"$\cos~\theta^{zen}_{reco}$")
        plt.ylabel("Events")
        plt.legend()
        plt.grid(True)
        plt.savefig("figs/coszreco_" + str(s_id) + "_gamma=" + str(g) + ".png")
        # plt.show()
        plt.close()

    """ Plot true zenith histograms """
    for s_id, s_name in hk_tau_sample_names.items():
        cut = sample == s_id
        taucut = cut & (np.abs(ipnu) == 16) & (current == "CC")
        ecut = cut & (np.abs(ipnu) == 12) & (current == "CC")
        mucut = cut & (np.abs(ipnu) == 14) & (current == "CC")
        taunccut = cut & (np.abs(ipnu) == 16) & (current != "CC")
        nccut = cut & (current != "CC")
        plt.figure()
        data = [
            cosz_true[nccut],
            cosz_true[mucut],
            cosz_true[ecut],
            cosz_true[taucut],
            cosz_true[taunccut],
            cosz_true[taucut],
        ]
        w = [
            w_oscbf[nccut] * tau_sample_exposure[s_id],
            w_oscbf[mucut] * tau_sample_exposure[s_id],
            w_oscbf[ecut] * tau_sample_exposure[s_id],
            w_oscbf[taucut] * tau_sample_exposure[s_id],
            weight[taunccut] * Astrf(pnu[taunccut], gamma=g) * tau_sample_exposure[s_id],
            weight[taucut] * Astrf(pnu[taucut], gamma=g) * tau_sample_exposure[s_id],
        ]
        labels = [
            r"Atm. NC",
            r"Atm. $\nu_{\mu}$ CC",
            r"Atm. $\nu_{e}$ CC",
            r"Atm. $\nu_{\tau}$ CC",
            r"Astro. $\nu_{\tau}$ NC",
            r"Astro. $\nu_{\tau}$ CC",
        ]
        plt.hist(
            data,
            bins=20,
            # color="red",
            weights=w,
            label=labels,
            stacked=True
        )

        plt.title(f"{s_name}")
        plt.xlabel(r"$\cos~\theta^{zen}_{\nu}$")
        plt.ylabel("Events")
        plt.legend()
        plt.grid(True)
        plt.savefig("figs/cosztrue_" + str(s_id) + "_gamma=" + str(g) + ".png")
        # plt.show()
        plt.close()
