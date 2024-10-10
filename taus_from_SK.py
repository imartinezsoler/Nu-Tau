import h5py
import numpy as np
import matplotlib.pyplot as plt
from fluxes import Astrf
import tauNN
import os
plt.style.use(os.environ["PYNU"] + "/../utils/plot.mplstyle")


""" SK/HK part """


class HK:
    def __init__(self):
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
        mr_4_ebins = np.array([0.1, 1.33, 2.5118864315095797, 5.011872336272725, 100.0])
        mr_6_ebins = np.array(
            [0.1, 2.5118864315095797, 10.0, 20.0, 50.0, 100.0, 200.0]
        )
        mr_7_ebins = np.array(
            [0.1, 2.5118864315095797, 5.011872336272725, 10.0, 17.0, 30.0, 50.0, 75.0, 125.0, 250.0]
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
        z10bins_up = np.array(
            [-1, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0]
        )
        z1bins = np.array([-1.0, 1.0])
        z2bins = np.array([-1.0, 0.0, 1.0])

        self.EnergyBins = {
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
            # 10: mr_3_ebins,
            # 11: mr_3_ebins,
            # 12: mr_4_ebins,
            # 13: mg_4_ebins,
            10: mr_6_ebins,
            11: mr_6_ebins,
            12: mr_7_ebins,
            13: mr_7_ebins,
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
        self.CTBins = {
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
            # 10: z10bins,
            # 11: z10bins,
            # 12: z10bins,
            # 13: z10bins,
            10: z2bins,
            11: z2bins,
            12: z2bins,
            13: z2bins,
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

        self.newCTBins = {
            0: z2bins,
            1: z1bins,
            2: z1bins,
            3: z2bins,
            4: z2bins,
            5: z1bins,
            6: z1bins,
            7: z2bins,
            8: z2bins,
            9: z2bins,
            2: z2bins,
            11: z2bins,
            12: z2bins,
            13: z2bins,
            14: z2bins,
            15: z2bins,
            16: z10bins_up,
            17: z10bins_up,
            18: z10bins_up,
            19: z2bins,
            20: z2bins,
            21: z2bins,
            22: z2bins,
            23: z2bins,
            24: z2bins,
            25: z2bins,
            26: z2bins,
            27: z2bins,
            28: z2bins,
        }
        self.all_sample_names = {
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
        self.tau_sample_names = {
            7: "sk1-3_fc_multigev_1ring_nuelike",
            8: "sk1-3_fc_multigev_1ring_nuebarlike",
            10: "sk1-5_fc_multigev_multiring_nuelike",
            11: "sk1-5_fc_multigev_multiring_nuebarlike",
            13: "sk1-5_fc_multigev_multiring_other",
            14: "sk1-5_pc_stop",
            15: "sk1-5_pc_thru",
            9: "sk1-3_fc_multigev_1ring_mulike",
            12: "sk1-5_fc_multigev_multiring_mulike",
            16: "sk1-5_upmu_stop",
            17: "sk1-5_upmu_thru_nonshowering",
            18: "sk1-5_upmu_thru_showering",
        }
        self.hk_tau_sample_names = {
            7: r"HK FC multigev 1ring $\nu_{e}$-like",
            8: r"HK FC multigev 1ring $\overline{\nu_{e}}$-like",
            9: r"HK FC multigev 1ring $\mu$-like",
            10: r"HK FC multigev multiring $\nu_{e}$-like",
            11: r"HK FC multigev multiring $\overline{\nu_{e}}$-like",
            12: r"HK FC multigev multiring $\mu$-like",
            13: "HK FC multigev multiring other",
            14: "HK PC stop",
            15: "HK PC thru",
            16: "HK PC thru",
            17: "HK UPMºU",
            18: "HK PC thru",
        }

        # Exposure in units of HK years, i.e. 188 kton
        sk_days = 6500
        sk123_days = 2795
        sk45_days = sk_days - sk123_days
        sk_expfv = 27.2  # kton
        hk_scale = 6.9  # taking into account expanded FV in SK but not in HK
        how_much_skmc = 50
        hk_years = 10
        self.tau_sample_exposure = {
            7: hk_years * hk_scale * 365.25 / sk123_days / how_much_skmc,
            8: hk_years * hk_scale * 365.25 / sk123_days / how_much_skmc,
            9: hk_years * hk_scale * 365.25 / sk123_days / how_much_skmc,
            10: hk_years * hk_scale * 365.25 / sk_days / how_much_skmc,
            11: hk_years * hk_scale * 365.25 / sk_days / how_much_skmc,
            12: hk_years * hk_scale * 365.25 / sk_days / how_much_skmc,
            13: hk_years * hk_scale * 365.25 / sk_days / how_much_skmc,
            14: hk_years * hk_scale * 365.25 / sk_days / how_much_skmc,
            15: hk_years * hk_scale * 365.25 / sk_days / how_much_skmc,
            16: hk_years * hk_scale * 365.25 / sk_days / how_much_skmc,
            17: hk_years * hk_scale * 365.25 / sk_days / how_much_skmc,
            18: hk_years * hk_scale * 365.25 / sk_days / how_much_skmc,
        }

        with h5py.File(
            "/home/investigator/Pheno/SK-DC/tune_mc/final_SuperK_fullMC_3_2.h5"
        ) as hf:
            self.pnu = np.asarray(hf["pnu"])
            self.ipnu = np.asarray(hf["ipnu"])
            self.ereco = np.asarray(hf["evis"])
            self.nuPDG = np.asarray(hf["ipnu"])
            self.weight = np.asarray(hf["tune_weights"]) * np.asarray(hf["inv_flux"])
            self.sample = np.asarray(hf["itype"])
            self.w_oscbf = np.asarray(hf["tune_weights"]) * np.asarray(hf["w_no"])
            self.cosz_reco = np.asarray(hf["recodirZ"])
            self.mode = np.asarray(hf["mode"])
            self.cosz_true = np.asarray(hf["dirnuZ"])
            self.current = np.array(hf["current"][()]).astype(str)
            self.taunn = np.zeros_like(self.ereco)


        """ remove samples irrelevant for the analysis """
        sample_cut = (self.sample == 7) | (self.sample == 8) | (self.sample == 10) | (self.sample == 11) | (self.sample == 13) | (self.sample == 14) | (self.sample == 15) | (self.sample == 16) | (self.sample == 17) | (self.sample == 18) | (self.sample == 9) | (self.sample == 12)
        self.pnu = self.pnu[sample_cut]
        self.ipnu = self.ipnu[sample_cut]
        self.ereco = self.ereco[sample_cut]
        self.mode = self.mode[sample_cut]
        self.nuPDG = self.nuPDG[sample_cut]
        self.weight = self.weight[sample_cut]
        self.sample = self.sample[sample_cut]
        self.w_oscbf = self.w_oscbf[sample_cut]
        self.cosz_reco = self.cosz_reco[sample_cut]
        self.cosz_true = self.cosz_true[sample_cut]
        self.current = self.current[sample_cut]
        self.taunn = self.taunn[sample_cut]

        self.entries = np.sum(sample_cut)

        mre = (self.sample == 10) | (self.sample == 11) | (self.sample == 13)
        tau = (np.abs(self.ipnu) == 16) & (self.current == "CC")
        notau = np.logical_not(tau)
        cut = mre & notau
        self.taunn[cut] = tauNN.get_bkg(np.sum(cut))
        cut = mre & tau
        self.taunn[cut] = tauNN.get_sig(np.sum(cut))

        self.ereco2 = self.ereco
        cut = mre & (self.ereco>10)
        self.ereco2[cut] = self.continuous_ereco(self.pnu[cut], "multiring", 10)
        self.ereco = self.ereco2

    def continuous_ereco(self, pnu, sample, bin_low):
        """ any """
        if "multiring" in sample: #e-like
            cut = np.full(pnu.size, True)
            ereco = np.zeros_like(pnu)
            rng = np.random.default_rng()
            for __ in range(20):
                prob = np.array([0.00035965, 0.00032244, 0.00042165, 0.00029764, 0.00038445, 0.00037205, 0.00037205, 0.00034724, 0.00057047, 0.00052087, 0.00054567, 0.00065728, 0.00064488, 0.00066968, 0.0007689 , 0.00068209, 0.00068209, 0.0007813, 0.0008061 , 0.00085571, 0.00089291, 0.00109134, 0.00138898, 0.00117815, 0.00128976, 0.00140138, 0.00147579, 0.00140138, 0.00136417, 0.00176102, 0.00133937, 0.0015998 , 0.00163701, 0.00210827, 0.00181063, 0.00168661, 0.00164941, 0.00200905, 0.00225708, 0.00207106, 0.00213307, 0.00214547, 0.00245551, 0.0024059 , 0.0024183 , 0.00218268, 0.00249271, 0.00265393, 0.00298878, 0.00369567, 0.00673405, 0.01309605, 0.02119427, 0.03009859, 0.03590252, 0.04241334, 0.04700192, 0.04789483, 0.04896137, 0.04801885, 0.0468159 , 0.0459974 , 0.04449681, 0.04158244, 0.0396478 , 0.03597693, 0.03230607, 0.03158678, 0.02772989, 0.02567123, 0.02361258, 0.0206734, 0.01808148, 0.01696534, 0.01566317, 0.01338129, 0.01231475, 0.01088857, 0.00992125, 0.00851987, 0.00782539, 0.00705649, 0.00659763, 0.00582873, 0.00512185, 0.00436535, 0.00451417, 0.0039313 , 0.00373287, 0.00370807, 0.0031748 , 0.00257953, 0.00244311, 0.00218268, 0.00173622, 0.00138898, 0.00104173, 0.00073169, 0.00060768, 0.00023563])
                prob = prob/np.sum(prob)
                pull_e = np.array([-0.9986436, -0.97879125, -0.9589389, -0.93908655, -0.91923419, -0.89938184 -0.87952949, -0.85967714, -0.83982479, -0.81997244, -0.80012009, -0.78026774, -0.76041539, -0.74056304, -0.72071069, -0.70085834, -0.68100599, -0.66115364, -0.64130129, -0.62144893, -0.60159658, -0.58174423, -0.56189188, -0.54203953, -0.52218718, -0.50233483, -0.48248248, -0.46263013, -0.44277778, -0.42292543, -0.40307308, -0.38322073, -0.36336838, -0.34351603, -0.32366367, -0.30381132, -0.28395897, -0.26410662, -0.24425427, -0.22440192, -0.20454957, -0.18469722, -0.16484487, -0.14499252, -0.12514017, -0.10528782, -0.08543547, -0.06558312, -0.04573077, -0.02587842, -0.00602606, 0.01382629, 0.03367864, 0.05353099,  0.07338334,  0.09323569,  0.11308804,  0.13294039,  0.15279274,  0.17264509,  0.19249744,  0.21234979,  0.23220214,  0.25205449,  0.27190684,  0.2917592,  0.31161155,  0.3314639,   0.35131625,  0.3711686,   0.39102095,  0.4108733,  0.43072565,  0.450578,    0.47043035,  0.4902827,   0.51013505,  0.5299874,  0.54983975,  0.5696921,   0.58954446,  0.60939681,  0.62924916,  0.64910151,  0.66895386,  0.68880621,  0.70865856,  0.72851091,  0.74836326,  0.76821561,  0.78806796,  0.80792031,  0.82777266,  0.84762501,  0.86747736,  0.88732971,  0.90718207,  0.92703442,  0.94688677,  0.96673912,  0.98659147])
                pe = rng.choice(pull_e, np.sum(cut), p=prob)
                ereco[cut] = pnu[cut] * np.abs(1-pe)
                # print(ereco)
                if np.all(ereco>=bin_low):
                    #print("done")
                    return ereco
                else:
                    cut = ereco<bin_low
                    #print(f"not done, {np.sum(cut)}")
            ereco[ereco<bin_low] = bin_low
            #plt.hist((ereco), bins=self.EnergyBins[10])
            #plt.show()
            return ereco
        elif "pc" in sample: #dufour
            pass

    def tilt(self, x):
        pass
    def norm(self, x):
        return x * np.ones(self.entries)
    def tau_xsec(self, x):
        dw = np.ones(self.entries)
        cut = (np.abs(self.ipnu) == 16) & (self.current == "CC")
        dw[cut] *= x
        return dw
    def dis(self, x):
        dw = np.ones(self.entries)
        cut = (np.abs(self.mode) > 24) & (self.current == "CC")
        dw[cut] *= x
        return dw
    def nc_cc(self, x):
        dw = np.ones(self.entries)
        cut = (np.abs(self.mode) > 24) & (self.current == "CC")
        anticut = np.logical_not(cut)
        dw[cut] *= x
        dw[anticut] *= 1/x
        return dw
    def mr_pid(self, x):
        dw = np.ones(self.entries)
        cut = (self.sample == 10) | (self.sample == 11) | (self.sample == 13)
        dw[cut] *= x
        return dw
    def mr_other(self, x):
        dw = np.ones(self.entries)
        cut = (self.sample == 13)
        anticut = (self.sample == 10) | (self.sample == 11)
        dw[cut] *= x
        dw[anticut] *= 1/x
        return dw
    def nu_nubar(self, x):
        dw = np.ones(self.entries)
        cut = (self.sample == 10)
        anticut = (self.sample == 11)
        dw[cut] *= x
        dw[anticut] *= 1/x
        return dw
    def stop_thru(self, x):
        dw = np.ones(self.entries)
        cut = (self.sample == 14)
        anticut = (self.sample == 15)
        dw[cut] *= x
        dw[anticut] *= 1/x
        return dw

    def binned(self, norm, gamma, fraction, tauNN, bins, w_all=None, w_atm=None):
        if (w_all is None) and (w_atm is None):
            w_all = np.ones(self.entries)
            w_atm = np.ones(self.entries)

        binned = np.array([])

        if not tauNN:
            if bins == "default":
                for s_id, s_name in self.hk_tau_sample_names.items():
                    cut = self.sample == s_id
                    taucut = cut & (np.abs(self.ipnu) == 16) & (self.current == "CC")
                    ecut = cut & (np.abs(self.ipnu) == 12) & (self.current == "CC")
                    mucut = cut & (np.abs(self.ipnu) == 14) & (self.current == "CC")
                    taunccut = cut & (np.abs(self.ipnu) == 16) & (self.current != "CC")
                    nccut = cut & (self.current != "CC")

                    dweights = np.zeros(self.entries)
                    dweights[nccut] += (
                        w_atm[nccut] * w_all[nccut] * self.w_oscbf[nccut] * self.tau_sample_exposure[s_id]
                    )
                    dweights[mucut] += (
                        w_atm[mucut] * w_all[mucut] * self.w_oscbf[mucut] * self.tau_sample_exposure[s_id]
                    )
                    dweights[ecut] += (
                        w_atm[ecut] * w_all[ecut] * self.w_oscbf[ecut] * self.tau_sample_exposure[s_id]
                    )
                    dweights[taucut] += (
                        w_atm[taucut] * w_all[taucut] * self.w_oscbf[taucut] * self.tau_sample_exposure[s_id]
                    )
                    dweights[taunccut] += (
                        w_all[taunccut] * self.weight[taunccut]
                        * Astrf(self.pnu[taunccut], norm=norm, gamma=gamma)
                        * self.tau_sample_exposure[s_id]
                    )
                    dweights[taucut] += (
                        w_all[taucut] * self.weight[taucut]
                        * Astrf(self.pnu[taucut], norm=norm, gamma=gamma)
                        * self.tau_sample_exposure[s_id]
                    )

                    values, __, __ = np.histogram2d(
                        self.ereco[cut],
                        self.cosz_reco[cut],
                        bins=(self.EnergyBins[s_id], self.CTBins[s_id]),
                        weights=dweights[cut],
                    )
                    #print(s_name)
                    #print(values)
                    #print("==============================")
                    binned = np.append(binned, values.flatten())
        if tauNN:
            if bins == "default":
                for s_id, s_name in self.hk_tau_sample_names.items():
                    if "multiring" in s_name:
                        cuts = [
                            (self.sample == s_id) & (self.taunn > 0.5),
                            (self.sample == s_id) & (self.taunn < 0.5),
                        ]
                    else:
                        cuts = [(self.sample == s_id)]
                    for cut in cuts:
                        taucut = (
                            cut & (np.abs(self.ipnu) == 16) & (self.current == "CC")
                        )
                        ecut = cut & (np.abs(self.ipnu) == 12) & (self.current == "CC")
                        mucut = cut & (np.abs(self.ipnu) == 14) & (self.current == "CC")
                        taunccut = (
                            cut & (np.abs(self.ipnu) == 16) & (self.current != "CC")
                        )
                        enccut = cut & (np.abs(self.ipnu) == 12) & (self.current != "CC")
                        munccut = cut & (np.abs(self.ipnu) == 14) & (self.current != "CC")
                        nccut = cut & (self.current != "CC")

                        dweights = np.zeros(self.entries)
                        dweights[nccut] += (
                            w_atm[nccut] * w_all[nccut] * self.w_oscbf[nccut] * self.tau_sample_exposure[s_id]
                        )
                        dweights[mucut] += (
                            w_atm[mucut] * w_all[mucut] * self.w_oscbf[mucut] * self.tau_sample_exposure[s_id]
                        )
                        dweights[ecut] += (
                            w_atm[ecut] * w_all[ecut] * self.w_oscbf[ecut] * self.tau_sample_exposure[s_id]
                        )
                        dweights[taucut] += (
                            w_atm[taucut] * w_all[taucut] * self.w_oscbf[taucut] * self.tau_sample_exposure[s_id]
                        )
                        
                        dweights[taunccut] += (
                            w_all[taunccut] * self.weight[taunccut]
                            * Astrf(self.pnu[taunccut], norm=norm, gamma=gamma, fraction=fraction[2])
                            * self.tau_sample_exposure[s_id]
                        )
                        dweights[taucut] += (
                            w_all[taucut] * self.weight[taucut]
                            * Astrf(self.pnu[taucut], norm=norm, gamma=gamma, fraction=fraction[2])
                            * self.tau_sample_exposure[s_id]
                        )
                        dweights[enccut] += (
                            w_all[enccut] * self.weight[enccut]
                            * Astrf(self.pnu[enccut], norm=norm, gamma=gamma, fraction=fraction[0])
                            * self.tau_sample_exposure[s_id]
                        )
                        dweights[ecut] += (
                            w_all[ecut] * self.weight[ecut]
                            * Astrf(self.pnu[ecut], norm=norm, gamma=gamma, fraction=fraction[0])
                            * self.tau_sample_exposure[s_id]
                        )
                        dweights[munccut] += (
                            w_all[munccut] * self.weight[munccut]
                            * Astrf(self.pnu[munccut], norm=norm, gamma=gamma, fraction=fraction[1])
                            * self.tau_sample_exposure[s_id]
                        )
                        dweights[mucut] += (
                            w_all[mucut] * self.weight[mucut]
                            * Astrf(self.pnu[mucut], norm=norm, gamma=gamma, fraction=fraction[1])
                            * self.tau_sample_exposure[s_id]
                        )

                        values, __, __ = np.histogram2d(
                            self.ereco[cut],
                            self.cosz_reco[cut],
                            bins=(self.EnergyBins[s_id], self.CTBins[s_id]),
                            weights=dweights[cut],
                        )
                        binned = np.append(binned, values.flatten())

        return binned



if __name__ == "__main__":
    """ Prompt flux """
    # Ignored for now
    # prompt_flux = pd.read_csv('/home/investigator/Pheno/AstroTau/Nu-Tau/NuTauSK/Nu_Tau_prompt_H3a_SIBYLL23C_pr.txt', sep='\s+', engine='python')
    hk = HK()
    gammas = [-2.4, -2.8, -3.0]
    gammas = [2.6]
    for g in gammas:
        """ Plot Ereco histograms """
        for s_id, s_name in hk.hk_tau_sample_names.items():
            if "multiring" in s_name:
                cut = (hk.sample == s_id) & (hk.taunn>0.5)
            else:
                cut = hk.sample == s_id
            taucut = cut & (np.abs(hk.ipnu) == 16) & (hk.current == "CC")
            ecut = cut & (np.abs(hk.ipnu) == 12) & (hk.current == "CC")
            mucut = cut & (np.abs(hk.ipnu) == 14) & (hk.current == "CC")
            taunccut = cut & (np.abs(hk.ipnu) == 16) & (hk.current != "CC")
            nccut = cut & (hk.current != "CC")
            plt.figure()
            data = [
                hk.ereco[nccut],
                hk.ereco[mucut],
                hk.ereco[ecut],
                hk.ereco[taucut],
                hk.ereco[taunccut],
                hk.ereco[taucut],
            ]
            w = [
                hk.w_oscbf[nccut] * hk.tau_sample_exposure[s_id],
                hk.w_oscbf[mucut] * hk.tau_sample_exposure[s_id],
                hk.w_oscbf[ecut] * hk.tau_sample_exposure[s_id],
                hk.w_oscbf[taucut] * hk.tau_sample_exposure[s_id],
                hk.weight[taunccut] * Astrf(hk.pnu[taunccut], gamma=g) * hk.tau_sample_exposure[s_id],
                hk.weight[taucut] * Astrf(hk.pnu[taucut], gamma=g) * hk.tau_sample_exposure[s_id],
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
                bins=hk.EnergyBins[s_id],
                # color="red",
                weights=w,
                label=labels,
                stacked=True
            )

            plt.title(f"{s_name}")
            plt.xlabel(r"$E_{reco} / GeV$")
            plt.ylabel("Events")
            plt.yscale("log")
            plt.legend()
            plt.grid(True)
            plt.savefig("figs/log_ereco_" + str(s_id) + "_gamma=" + str(g) + ".png")
            # plt.show()
            plt.close()

        """ Plot Etrue histograms """
        for s_id, s_name in hk.hk_tau_sample_names.items():
            cut = hk.sample == s_id
            taucut = cut & (np.abs(hk.ipnu) == 16) & (hk.current == "CC")
            ecut = cut & (np.abs(hk.ipnu) == 12) & (hk.current == "CC")
            mucut = cut & (np.abs(hk.ipnu) == 14) & (hk.current == "CC")
            taunccut = cut & (np.abs(hk.ipnu) == 16) & (hk.current != "CC")
            nccut = cut & (hk.current != "CC")
            plt.figure()
            data = [
                hk.pnu[nccut],
                hk.pnu[mucut],
                hk.pnu[ecut],
                hk.pnu[taucut],
                hk.pnu[taunccut],
                hk.pnu[taucut],
            ]
            w = [
                hk.w_oscbf[nccut] * hk.tau_sample_exposure[s_id],
                hk.w_oscbf[mucut] * hk.tau_sample_exposure[s_id],
                hk.w_oscbf[ecut] * hk.tau_sample_exposure[s_id],
                hk.w_oscbf[taucut] * hk.tau_sample_exposure[s_id],
                hk.weight[taunccut] * Astrf(hk.pnu[taunccut], gamma=g) * hk.tau_sample_exposure[s_id],
                hk.weight[taucut] * Astrf(hk.pnu[taucut], gamma=g) * hk.tau_sample_exposure[s_id],
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
        for s_id, s_name in hk.hk_tau_sample_names.items():
            cut = hk.sample == s_id
            taucut = cut & (np.abs(hk.ipnu) == 16) & (hk.current == "CC")
            ecut = cut & (np.abs(hk.ipnu) == 12) & (hk.current == "CC")
            mucut = cut & (np.abs(hk.ipnu) == 14) & (hk.current == "CC")
            taunccut = cut & (np.abs(hk.ipnu) == 16) & (hk.current != "CC")
            nccut = cut & (hk.current != "CC")
            plt.figure()
            data = [
                hk.cosz_reco[nccut],
                hk.cosz_reco[mucut],
                hk.cosz_reco[ecut],
                hk.cosz_reco[taucut],
                hk.cosz_reco[taunccut],
                hk.cosz_reco[taucut],
            ]
            w = [
                hk.w_oscbf[nccut] * hk.tau_sample_exposure[s_id],
                hk.w_oscbf[mucut] * hk.tau_sample_exposure[s_id],
                hk.w_oscbf[ecut] * hk.tau_sample_exposure[s_id],
                hk.w_oscbf[taucut] * hk.tau_sample_exposure[s_id],
                hk.weight[taunccut] * Astrf(hk.pnu[taunccut], gamma=g) * hk.tau_sample_exposure[s_id],
                hk.weight[taucut] * Astrf(hk.pnu[taucut], gamma=g) * hk.tau_sample_exposure[s_id],
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
                bins=hk.CTBins[s_id],
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
        for s_id, s_name in hk.hk_tau_sample_names.items():
            cut = hk.sample == s_id
            taucut = cut & (np.abs(hk.ipnu) == 16) & (hk.current == "CC")
            ecut = cut & (np.abs(hk.ipnu) == 12) & (hk.current == "CC")
            mucut = cut & (np.abs(hk.ipnu) == 14) & (hk.current == "CC")
            taunccut = cut & (np.abs(hk.ipnu) == 16) & (hk.current != "CC")
            nccut = cut & (hk.current != "CC")
            plt.figure()
            data = [
                hk.cosz_true[nccut],
                hk.cosz_true[mucut],
                hk.cosz_true[ecut],
                hk.cosz_true[taucut],
                hk.cosz_true[taunccut],
                hk.cosz_true[taucut],
            ]
            w = [
                hk.w_oscbf[nccut] * hk.tau_sample_exposure[s_id],
                hk.w_oscbf[mucut] * hk.tau_sample_exposure[s_id],
                hk.w_oscbf[ecut] * hk.tau_sample_exposure[s_id],
                hk.w_oscbf[taucut] * hk.tau_sample_exposure[s_id],
                hk.weight[taunccut] * Astrf(hk.pnu[taunccut], gamma=g) * hk.tau_sample_exposure[s_id],
                hk.weight[taucut] * Astrf(hk.pnu[taucut], gamma=g) * hk.tau_sample_exposure[s_id],
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
