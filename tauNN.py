import pandas as pd
import numpy as np
from scipy.interpolate import PchipInterpolator
import matplotlib.pyplot as plt


def get_bkg(size):
	file = pd.read_csv("TauNN_bkg.csv", sep=", ")
	taunn = file["taunn"]
	prob = file["prob"]
	pdf = PchipInterpolator(taunn, prob)
	cdf = pdf.antiderivative()
	dummy_x = np.linspace(min(taunn), max(taunn), 1000)
	inv_cdf = PchipInterpolator(cdf(dummy_x), dummy_x)
	y = np.random.uniform(0, 1, size=size)
	mc = inv_cdf(y)
	return mc

def get_sig(size):
	file = pd.read_csv("TauNN_sig.csv", sep=", ")
	taunn = file["taunn"]
	prob = file["prob"]
	pdf = PchipInterpolator(taunn, prob)
	cdf = pdf.antiderivative()
	dummy_x = np.linspace(min(taunn), max(taunn), 1000)
	inv_cdf = PchipInterpolator(cdf(dummy_x), dummy_x)
	y = np.random.uniform(0, 1, size=size)
	mc = inv_cdf(y)
	return mc

