import numpy as np
import scipy.optimize as optimize
import matplotlib.pyplot as plt

import sys
from pathlib import Path
headDir = Path(__file__).parents[1].resolve()
sys.path.append(str(headDir))
from utils.random_number_gen import generator
import utils.misc as misc
from setup.config_logging import LoggingConfig
loggerObj = LoggingConfig()
logger = loggerObj.init_logger(__name__)

class Bootstrap:
    def __init__(self, inputData, outputData, fitFunc, parameter=0, pGuess=None, paramBounds=None, weights=None, iterGuess=False, gen=None):
        assert isinstance(inputData, np.ndarray)
        assert isinstance(outputData, np.ndarray)
        assert outputData.size == inputData.size
        if gen == None:
            self.gen = generator()
        else:
            self.gen = gen
        self.outputData = outputData
        self.inputData = inputData
        self.N = inputData.size
        self.func = fitFunc
        self.parameter = parameter
        self.pGuess = pGuess
        self.weights = weights
        if weights is not None:
            assert len(weights) == outputData.size
        self.paramBounds = paramBounds
        if self.paramBounds is None:
            self.paramBounds = -np.inf, np.inf
        assert isinstance(self.paramBounds, tuple)
        assert len(self.paramBounds) == 2
        assert isinstance(iterGuess, bool)
        self.iterGuess = iterGuess
        self.parameterOriginal = np.nan
        self.parameterErrorMeanBiasCorr = np.nan
        self.parameterMean = np.nan
        self.parameterErrorMean = np.nan

    def gen_bootstrap_samples(self, numSamples, lenSamples):
        if lenSamples > self.N:
            lenSamples = self.N
            logger.warning(f"length of bootstrap samples ({lenSamples}) is larger then the length of the data ({self.N}). Setting the length to the data size.")
        self.numSamples = numSamples
        self.lenSamples = lenSamples
        N = self.numSamples * self.lenSamples
        indices = self.gen.integers(low=0, high=self.N, size=N)
        self.inSamples = self.inputData[indices]
        self.outSamples = self.outputData[indices]

    def statistical_error(self):
        try:
            self.p, _ = optimize.curve_fit(self.func, self.inputData, self.outputData, p0=self.pGuess, bounds=self.paramBounds, sigma=self.weights)
        except RuntimeError:
            logger.error("Initial fitting did not work in bootstrap. Aborting.")
            return False
        else:
            self.parameterArr = np.empty(self.numSamples + 1)
            self.parameterArr[0] = self.p[self.parameter]
            if self.iterGuess:
                self.pGuess = self.p
            for i in range(self.numSamples):
                try:
                    p, _ = optimize.curve_fit(self.func, self.inSamples[i : i + self.lenSamples], self.outSamples[i : i + self.lenSamples], p0=self.pGuess, bounds=self.paramBounds, sigma=self.weights)
                except RuntimeError:
                    continue
                else:
                    self.parameterArr[i + 1] = p[self.parameter]
            self.parameterArr = self.parameterArr[~np.isnan(self.parameterArr)]
            self.parameterOriginal = self.parameterArr[0]
            self.parameterMean = np.mean(self.parameterArr)
            self.parameterErrorMean = np.std(self.parameterArr, ddof=1)
            self.parameterErrorMeanBiasCorr = np.std(self.parameterArr[1:], ddof=0) + abs(np.mean(self.parameterArr[1:]) - self.parameterArr[0])
            return True

    def plot_histo(self):
        maxP = np.amax(self.parameterArr)
        minP = np.amin(self.parameterArr)
        if maxP == minP:
            logger.error("Fitting did not work. All parameters are identical.")
            pass
        else:
            numberOfBins = misc.histo_number_of_bins(self.parameterArr)
            plt.hist(self.parameterArr, numberOfBins)
            plt.show()

    def run(self, numSamples, lenSamples, plotHisto=True):
        self.gen_bootstrap_samples(numSamples=numSamples, lenSamples=lenSamples)
        checkFitFlag = self.statistical_error()
        if checkFitFlag:
            if plotHisto:
                self.plot_histo()
        else:
            self.parameterEstimate = np.nan
            self.parameterError = np.nan

    @property
    def results(self):
        return self.parameterOriginal, self.parameterErrorMeanBiasCorr, self.parameterMean, self.parameterErrorMean

if __name__ == "__main__":
    import numpy.random as random

    def gaussian(x, mu, sigma):
        return np.exp( - (x - mu)**2 / 2 / sigma**2) / np.sqrt(2*np.pi*sigma**2)

    def noise(x, scale):
        return scale*random.random(x.size) - scale/2

    def linear(x, a, b):
        return a*x + b

    mu = 10
    sigma = 3
    a = 2
    b = 1
    xArr = np.linspace(0, 10, 100)
    # yArr = gaussian(xArr, mu, sigma) + 0.01*random.normal(0, 0.5, xArr.size)
    yArr = linear(xArr, a, b) + 1*random.normal(0, 0.5, xArr.size)
    plt.plot(xArr, yArr, "b.")
    plt.show()

    test = Bootstrap(xArr, yArr, linear, parameter=0)
    test.gen_bootstrap_samples(10000, 100)
    test.statistical_error()
    test.plot_histo()