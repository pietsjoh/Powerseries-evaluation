"""Contains a class that makes it possible to estimate the error of
a fit parameter using the bootstrap method.

Running this script provides an example.
"""
import numpy as np
import scipy.optimize as optimize # type: ignore
import matplotlib.pyplot as plt # type: ignore
import typing

import sys
from pathlib import Path
headDir: Path = Path(__file__).parents[1].resolve()
sys.path.append(str(headDir))

from utils.random_number_gen import RNGenerator
import utils.misc as misc
from setup.config_logging import LoggingConfig
loggerObj: LoggingConfig = LoggingConfig()
logger = loggerObj.init_logger(__name__)

number = typing.Union[float, int, np.number]
tupleOrNone = typing.Union[tuple, None]
arrayOrNone = typing.Union[np.ndarray, None]
arrayLikeOrNone = typing.Union[np.ndarray, list, tuple, None]
intOrNone = typing.Union[int, None]

class Bootstrap:
    def __init__(self, inputData: np.ndarray, outputData: np.ndarray, fitFunc: typing.Callable, parameter: int=0,
                pGuess: arrayLikeOrNone=None, paramBounds: tupleOrNone=None, weights: arrayOrNone=None,
                iterGuess: bool=False, seed: intOrNone=None) -> None:
        """Can calculate the uncertainty for 1 fit parameter using the bootstrap method.

        expected setup: outputData = fitFunc(inputData, \*args, \*\*kwargs)
        The parameters (pGuess, paramBounds and weights) are (when not None) directly passed
        into scipy.optimize.curve_fit as (p0, paramBounds and sigma).

        Parameters
        ----------
        inputData: np.ndarray

        outputData: np.ndarray

        fitFunc: Callable

        parameter: int, default=0
            index of the parameter of the function for which the bootstrap method should be applied
            as the first argument of the function needs to correspond to the inputData
            (scipy.optimize.curve_fit requirement), 0 means the argument at position 1
            Example: f(x, a, b); parameter=0 -> bootstrap for a will be performed

        pGuess: tuple/list/np.ndarray/None, default=None
            Initial parameter guesses for the fit

        paramBounds: tuple/None, default=None
            Bounds for the fit parameters, used for every fit

        weights: np.ndarray/None, default=None
            weights for the fitting process

        iterGuess: bool, default=False
            if True, then the results of the initial fit are used as initial guesses for the rest of the bootstrap method
            otherwise, the parameter pGuess is used for all fits

        seed: int/None, default=None
            seed of the random-number-generator
            Provide a seed to achieve reproducible results for the bootstrap process.
        """
        assert isinstance(inputData, np.ndarray)
        assert isinstance(outputData, np.ndarray)
        assert outputData.size == inputData.size

        self.gen: RNGenerator = RNGenerator(seed)
        self.outputData: np.ndarray = outputData
        self.inputData: np.ndarray = inputData
        self.N: int = inputData.size
        self.func: typing.Callable = fitFunc
        self.parameter: int = parameter
        self.pGuess: arrayLikeOrNone = pGuess
        self.weights: arrayOrNone = weights
        if weights is not None:
            assert len(weights) == outputData.size
        self.paramBounds: tupleOrNone = paramBounds
        if self.paramBounds is None:
            self.paramBounds = -np.inf, np.inf
        assert isinstance(self.paramBounds, tuple)
        assert len(self.paramBounds) == 2
        assert isinstance(iterGuess, bool)
        self.iterGuess: bool = iterGuess
        self.parameterOriginal: number = np.nan
        self.parameterErrorMeanBiasCorr: number = np.nan
        self.parameterMean: number = np.nan
        self.parameterErrorMean: number = np.nan

    def gen_bootstrap_samples(self, numSamples: int, lenSamples: int) -> None:
        """Generates samples of (inputData, outputData).

        Parameters
        ----------
        numSamples: int
            number of Samples that shall be generated

        lenSamples: int
            number of data points which shall be used for each sample
        """
        if lenSamples > self.N:
            lenSamples = self.N
            logger.warning(f"length of bootstrap samples ({lenSamples}) is larger then the length of the data ({self.N}). Setting the length to the data size.")
        self.numSamples: int = numSamples
        self.lenSamples: int = lenSamples
        N: int = self.numSamples * self.lenSamples
        indices: np.ndarray = self.gen.integers(low=0, high=self.N, size=N)
        self.inSamples: np.ndarray = self.inputData[indices]
        self.outSamples: np.ndarray = self.outputData[indices]

    def statistical_error(self) -> bool:
        """Performs the fitting of the bootstrap procedure.
        Calculates the value of the original fit parameter,
        the mean of the bootstrap distribution,
        the error of the mean of the bootstrap distribution
        and the error of the original fit parameter (standard deviation + deviation from mean).
        This calculation uses the generated samples.

        Returns
        -------
        bool:
            False, when the fit of the original data did not work
            True, otherwise
        """
        self.p: np.ndarray
        try:
            self.p, _ = optimize.curve_fit(self.func, self.inputData, self.outputData, p0=self.pGuess, bounds=self.paramBounds, sigma=self.weights)
        except RuntimeError:
            logger.error("Initial fitting did not work in bootstrap. Aborting.")
            return False
        else:
            self.parameterArr: np.ndarray = np.empty(self.numSamples + 1)
            self.parameterArr[0] = self.p[self.parameter]
            if self.iterGuess:
                self.pGuess = self.p
            for i in range(self.numSamples):
                p: np.ndarray
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

    def plot_histo(self) -> None:
        """Plots a histogram of the bootstrap distribution.
        Passes when there is only 1 distinct value in the bootstrap result array.
        """
        maxP: number = np.amax(self.parameterArr)
        minP: number = np.amin(self.parameterArr)
        if maxP == minP:
            logger.error("Fitting did not work. All parameters are identical.")
            pass
        else:
            numberOfBins: int = misc.histo_number_of_bins(self.parameterArr)
            plt.hist(self.parameterArr, numberOfBins)
            plt.show()

    def run(self, numSamples: int, lenSamples: int, plotHisto: bool=True) -> None:
        """Main method.
        Performs the bootstrap method by running self.gen_bootstrap_samples() and self.statistical_error().
        Generates the samples and calculates the uncertainties.
        Can also plot the bootstrap distribution using self.plot_histo().

        Parameters
        ----------
        numSamples: int
            number of Samples that shall be generated, input of self.gen_bootstrap_samples()

        lenSamples: int
            number of data points which shall be used for each sample, input of self.gen_bootstrap_samples()

        plotHisto: bool
            if True, plots bootstrap distribution using self.plot_histo()
            if False, no plot is shown
        """
        self.gen_bootstrap_samples(numSamples=numSamples, lenSamples=lenSamples)
        checkFitFlag: bool = self.statistical_error()
        if checkFitFlag:
            if plotHisto:
                self.plot_histo()
        else:
            self.parameterEstimate = np.nan
            self.parameterError = np.nan

    @property
    def results(self):
        """Tuple of bootstrap results.

        Contains the value of the original fit parameter,
        the error of the original fit parameter (standard deviation + deviation from mean).
        the mean of the bootstrap distribution,
        and the error of the mean of the bootstrap distribution
        """
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