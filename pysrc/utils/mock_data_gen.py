import matplotlib.pyplot as plt
import numpy as np

import sys
from pathlib import Path
headDir = Path(__file__).parents[1].resolve()
sys.path.append(str(headDir))

from utils.random_number_gen import generator

def lorentz(w, A, gamma, w0):
    return A * gamma / np.pi / ((w - w0)**2 + (gamma)**2)

def mock_data_gen_1_spec(x, maxNumPeaks, seed=None, showData=False):
    gen = generator(seed)
    numPeaks = gen.integers(low=1, high=maxNumPeaks, size=1)[0]
    y = gen.normal(size=x.size)
    AArr = gen.uniform(low=1, high=100, size=numPeaks)
    w0Arr = gen.uniform(low=550, high=950, size=numPeaks)
    gammaArr = gen.uniform(low=0.1, high=1, size=numPeaks)
    for A, w0, gamma in zip(AArr, w0Arr, gammaArr):
        y += lorentz(x, A, gamma, w0)
    if showData:
        plt.plot(x, y)
        plt.show()
    return y