import matplotlib.pyplot as plt # type: ignore
import numpy as np
import typing

import sys
from pathlib import Path
headDir = Path(__file__).parents[1].resolve()
sys.path.append(str(headDir))

from utils.random_number_gen import RNGenerator

number = typing.Union[int, float]
numberOrArray = typing.Union[number, np.ndarray]
intOrNone = typing.Union[int, None]

def lorentz(w: numberOrArray, A: number, gamma: number, w0: number) -> numberOrArray:
    """Lorentzian function:

    .. math::
        \dfrac{A}{\gamma\pi}\dfrac{1}{(\omega - \omega_0)^2 + \gamma^2}

    Parameters
    ----------
    w: float/int
        main input
    A: float/int
        integrated Power
    gamma: float/int
        width
    w0: float/int
        mean of the distribution

    Returns
    -------
    float:
        Value of the lorentz function f(w; A, gamma, w0) at w
    """
    return A * gamma / np.pi / ((w - w0)**2 + (gamma)**2)

def mock_data_gen_1_spec(x: np.ndarray, maxNumPeaks: int, seed: intOrNone=None, showData: bool=False) -> np.ndarray:
    """Generates a mock spectrum with multiple peaks.

    Parameters
    ----------
    x: np.ndarray
        Array of wavelengths/energies (input data)
    maxNumPeaks: int
        max number of peaks for the spectrum
    seed: int/None:
        whether to use a seed for the pseudo-random number generator
    showData: bool
        if True shows a plot of the mock spectrum, otherwise nothing is shown

    Returns
    -------
    np.ndarray:
        output data of the spectrum (intensities)
    """
    gen = RNGenerator(seed)
    numPeaks: int = gen.integers(low=1, high=maxNumPeaks, size=1)[0]
    y: np.ndarray = gen.normal(size=x.size)
    AArr: np.ndarray = gen.uniform(low=1, high=100, size=numPeaks)
    w0Arr: np.ndarray = gen.uniform(low=550, high=950, size=numPeaks)
    gammaArr: np.ndarray = gen.uniform(low=0.1, high=1, size=numPeaks)
    A: number
    w0: number
    gamma: number
    for A, w0, gamma in zip(AArr, w0Arr, gammaArr):
        y += lorentz(x, A, gamma, w0)
    if showData:
        plt.plot(x, y)
        plt.show()
    return y