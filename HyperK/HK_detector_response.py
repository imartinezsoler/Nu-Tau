import pandas as pd
import numpy as np


class HKDetector:
    def __init__(self, input_file):
        """Input file should contain:
        - flavor
        - energy
        - direction (?)
        - weight (flux x cross_section x prob_osc)
        - interaction type (just CC/NC)
        """

        self.data = pd.read_cvs(input_file)

        """ Load digitized response functions for HK """
        self.load_HK()

        """ True event categories """
        self.categories = ["nueCC", "nuebarCC", "numuCC", "numubarCC", "nutauCC", "NC"]

        """ Reconstructed samples """
        self.samples = [
            "fc_multigev_multiring_nuelike",
            "fc_multigev_multiring_nuebarlike",
            "fc_multigev_multiring_other",
        ]
        self.sample_prefix = "sk1-5_fc_"

    def load_HK(self, input_file="HK_energy_response.csv"):
        self.hk_energy = pd.read_cvs(input_file)

    def assign_sample(self):
        pass

    def energy_reco(self):
        pass

    def zenith_reco(self):
        pass
