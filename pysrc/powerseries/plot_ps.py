"""Contains a class that can visualize the powerseries.
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optimize
from matplotlib.collections import PolyCollection
import matplotlib.ticker as mticker
import pandas as pd
from configparser import ConfigParser
import inspect

def log_tick_formatter(val, pos=None):
    return "{:.3}".format(10**val)

import sys
from pathlib import Path
headDir = Path(__file__).resolve().parents[2]
srcDirPath = (headDir / "pysrc").resolve()
sys.path.append(str(srcDirPath))

from powerseries.eval_ps import EvalPowerSeries
from setup.config_logging import LoggingConfig
from utils import misc
from utils.bootstrap import Bootstrap

loggerObj: LoggingConfig = LoggingConfig()
logger = loggerObj.init_logger(__name__)

hbar: float = 6.582119514 * 10**(-16)
"""Reduced Planck constant in eV*s.
Taken from Particle Physics Booklet (2018, particle data group).
"""

class PlotPowerSeries:
    """Visualization of the powerseries.

    Can plot the original spectra individually and as waterfall plot.
    Moreover, the results of the powerseries evaluation can be visualized.

    Q-factor, mode energy, linewidth and outputpower vs inputpower.

    Furthermore, the fitting of the input-output characteristic is performed here,
    which leads to an estimation of the beta-factor.

    The __init__ method requires a list of Powerseries object (EvalPowerSeries).
    On these objects the get_power_dependent_data() method has to have been called before.
    """
    ## change according to experimental value
    resolutionLimit = 20e-6
    """Resolution limit of the linewidth in eV.
    The value is determined by and obtained from the experimental setup (especially by the grating).
    """
    tauSP, QEstimate, n0EstimatePaper = 10**(-9), 1.5 * 10**4, 3000
    """Estimate of parameters to estimate the xiHat parameter in the fitting of the in-out-curve.
    These estimates were taken from 'Optical pumping of quantum dot micropillar lasers' paper.

    L. Andreoli, X. Porte, T. Heuser, J. Große, B. Moeglen-Paget, L. Furfaro, S. Reitzenstein, and D. Brunner,
    "Optical pumping of quantum dot micropillar lasers," Opt. Express 29, 9084-9097 (2021)
    """
    n0Min, n0Max = 2000, 4000
    QMin, QMax = 10000, 20000
    outputPath = (headDir / "output").resolve()
    """Path to the output directory
    """
    def __init__(self, powerSeriesList):
        assert isinstance(powerSeriesList, (list, np.ndarray))
        assert len(powerSeriesList) != 0
        self.inputPower = np.array([])
        self.outputPowerArr = np.array([])
        self.linewidthArr = np.array([])
        self.modeWavelengthArr = np.array([])
        self.QFactorArr = np.array([])
        self.uncOutputPowerArr = np.array([])
        self.uncLinewidthArr = np.array([])
        self.uncModeWavelengthArr = np.array([])
        self.uncQFactorArr = np.array([])
        self.lenInputPower = 0
        self.powerSeriesList = powerSeriesList
        self.numPowerSeries = len(powerSeriesList)
        self.lenInputPowerArr = np.array([], dtype="int")
        self.inputPowerComplete = np.array([])
        self.minInputPower = np.inf
        self.maxInputPower = 0
        self.minInputPowerIdx = None
        self.maxInputPowerIdx = None
        for i, powerSeries in enumerate(self.powerSeriesList):
            assert isinstance(powerSeries, EvalPowerSeries)
            assert hasattr(powerSeries, "outputPowerArr")
            assert hasattr(powerSeries, "uncOutputPowerArr")
            assert hasattr(powerSeries, "inputPowerPlotArr")
            try:
                minInputPowerCurrent = np.amin(powerSeries.inputPowerPlotArr)
                maxInputPowerCurrent = np.amax(powerSeries.inputPowerPlotArr)
            except ValueError:
                logger.warning(f"{powerSeries.fileName} has no data points. Excluding it for the plots.")
            else:
                if minInputPowerCurrent < self.minInputPower:
                    self.minInputPower = minInputPowerCurrent
                    self.minInputPowerIdx = i
                if maxInputPowerCurrent > self.maxInputPower:
                    self.maxInputPower = maxInputPowerCurrent
                    self.maxInputPowerIdx = i
                self.inputPower = np.concatenate((self.inputPower, powerSeries.inputPowerPlotArr))
                self.outputPowerArr = np.concatenate((self.outputPowerArr, powerSeries.outputPowerArr))
                self.linewidthArr = np.concatenate((self.linewidthArr, powerSeries.linewidthArr))
                self.modeWavelengthArr = np.concatenate((self.modeWavelengthArr, powerSeries.modeWavelengthArr))
                self.QFactorArr = np.concatenate((self.QFactorArr, powerSeries.QFactorArr))
                self.lenInputPower += powerSeries.lenInputPowerPlot
                self.lenInputPowerArr = np.concatenate((self.lenInputPowerArr, np.array([powerSeries.lenInputPower], dtype="int")))
                self.inputPowerComplete = np.concatenate((self.inputPowerComplete, powerSeries.inputPower))
                self.uncOutputPowerArr = np.concatenate((self.uncOutputPowerArr, powerSeries.uncOutputPowerArr))
                self.uncLinewidthArr = np.concatenate((self.uncLinewidthArr, powerSeries.uncLinewidthArr))
                self.uncModeWavelengthArr = np.concatenate((self.uncModeWavelengthArr, powerSeries.uncModeWavelengthArr))
                self.uncQFactorArr = np.concatenate((self.uncQFactorArr, powerSeries.uncQFactorArr))
        self.lenInputPowerArrCumulative = np.cumsum(self.lenInputPowerArr)
        upperSmaxInputPower = self.lenInputPowerArrCumulative[self.maxInputPowerIdx] - 1
        upperSminInputPower = upperSmaxInputPower - self.lenInputPowerArr[self.maxInputPowerIdx] + 1
        self.maxInputPowerRange = upperSminInputPower, upperSmaxInputPower
        lowerSmaxInputPower = self.lenInputPowerArrCumulative[self.minInputPowerIdx] - 1
        lowerSminInputPower = lowerSmaxInputPower - self.lenInputPowerArr[self.minInputPowerIdx] + 1
        self.minInputPowerRange = lowerSminInputPower, lowerSmaxInputPower
## This function call is used to get the n0 estimate
        _ = self.constant_lines_inout()
        logger.debug(f"cavityDecayRateEstimate: {self.cavityDecayRateEstimate}")
        logger.debug(f"xiHatEstimate; n0 estimated: {self.xiHatEstimateWithn0}")
        logger.debug(f"xiHatEstimate; n0 not estimated: {self.xiHatEstimateWithoutn0}")
        logger.debug(f"lenInputPowerArr: {self.lenInputPowerArr}")
        logger.debug(f"lenInputPowerArrCumulative: {self.lenInputPowerArrCumulative}")
        logger.debug(f"maxInputPower/Idx: {self.maxInputPower}/{self.maxInputPowerIdx}")
        logger.debug(f"minInputPower/Idx: {self.minInputPower}/{self.minInputPowerIdx}")
        logger.debug(f"minInputPowerRange: {self.minInputPowerRange}")
        logger.debug(f"maxInputPowerRange: {self.maxInputPowerRange}")
        logger.debug("PlotPowerSeries object initialized.")
        self.read_powerseries_ini_file()
        if self.saveData:
            self.save_powerseries_data("powerseries.csv")
            self.save_settings("settings.csv")

    @property
    def xiHatMin(self):
        """Minimum estimate for xiHat
        """
        return self.n0Min * hbar * self.QMin / self.tauSP / self.modeWavelength[0]

    @property
    def xiHatMax(self):
        """Maximum estimate for xiHat
        """
        return self.n0Max * hbar * self.QMax / self.tauSP / self.modeWavelength[0]

    @property
    def xiHatEstimateWithn0(self):
        """Estimate for xiHat with the estimate for n0 using the linear tails of the
        in-out-characteristic
        """
        return self.n0Estimate / self.cavityDecayRateEstimate / self.tauSP

    @property
    def xiHatEstimateWithoutn0(self):
        """Estimate for xiHat with estimate for n0 from the paper
        """
        return self.n0EstimatePaper / self.cavityDecayRateEstimate / self.tauSP

    @property
    def modeWavelength(self):
        """Estimate of the mode energy using the mean value of the mode energy
        of the 5 lowest input powers
        """
        lowerSInputPowers = self.inputPower[self.minInputPowerRange[0] : self.minInputPowerRange[1] + 1]
        lowerSModeEnergys = self.modeWavelengthArr[self.minInputPowerRange[0] : self.minInputPowerRange[1] + 1]
        if lowerSInputPowers[0] < lowerSInputPowers[-1]:
            if len(lowerSModeEnergys) >= 5:
                return np.mean(lowerSModeEnergys[:5]), np.std(lowerSModeEnergys[:5], ddof=1) / np.sqrt(5)
            else:
                return np.mean(lowerSModeEnergys), np.std(lowerSModeEnergys, ddof=1) / np.sqrt(len(lowerSModeEnergys))
        else:
            if len(lowerSModeEnergys) >= 5:
                return np.mean(lowerSModeEnergys[-5:]), np.std(lowerSModeEnergys[-5:], ddof=1) / np.sqrt(5)
            else:
                return np.mean(lowerSModeEnergys), np.std(lowerSModeEnergys, ddof=1) / np.sqrt(len(lowerSModeEnergys))

    @property
    def cavityDecayRateEstimate(self):
        """Estimate of the cavity decay rate, used to estimate xihat
        """
        return self.modeWavelength[0] / hbar / self.QEstimate

    def read_powerseries_ini_file(self):
        """Reads and handles the values from the config/powerseries.ini file.
        Only takes the plot_ps.py section into account
        """
        logger.debug("Calling read_powerseries_ini_file()")
        configIniPath = (headDir / "config" / "powerseries.ini").resolve()
        config = ConfigParser()
        config.read(str(configIniPath))
        self.saveData = LoggingConfig.check_true_false(config["plot_ps.py"]["save data"].replace(" ", ""))
        self.useWeights = LoggingConfig.check_true_false(config["plot_ps.py"]["use weights"].replace(" ", ""))
        self.minimizeError = config["plot_ps.py"]["minimize error"].replace(" ", "")
        self.useParamBounds = LoggingConfig.check_true_false(config["plot_ps.py"]["use parameter bounds"].replace(" ", ""))
        self.initialParamGuess = config["plot_ps.py"]["initial parameter guess"].replace(" ", "")
        self.useBootstrap = LoggingConfig.check_true_false(config["plot_ps.py"]["use bootstrap"].replace(" ", ""))
        self.plotBootstrap = LoggingConfig.check_true_false(config["plot_ps.py"]["plot bootstrap histo"].replace(" ", ""))
        self.iterGuessBootstrap = LoggingConfig.check_true_false(config["plot_ps.py"]["use iterative guess"].replace(" ", ""))
        self.numBootstrapSamples = misc.int_decode(config["plot_ps.py"]["number of bootstrap samples"].replace(" ", ""))
        self.lenBootstrapSamples = misc.float_decode(config["plot_ps.py"]["length of bootstrap samples"].replace(" ", ""))

    @property
    def initialParamGuess(self):
        """Guess for the initial parameters for the beta fit
        """
        return self._initialParamGuess

    @initialParamGuess.setter
    def initialParamGuess(self, value):
        logger.debug(f"Setting initialParamGuess to {value}")
        assert isinstance(value, str)
        if value.lower() == "none":
            self._initialParamGuess = None
        else:
            p0StrList = value.split(",")
            self._initialParamGuess = list(map(misc.float_decode, p0StrList))
            if None in self._initialParamGuess:
                self._initialParamGuess = None

    @property
    def numBootstrapSamples(self):
        """The number of bootstrap samples that shall be generated for the
        estimation of the uncertainty of the beta-factor
        """
        return self._numBootstrapSamples

    @numBootstrapSamples.setter
    def numBootstrapSamples(self, value):
        logger.debug(f"Setting numBootstrapSamples to {value}.")
        self._numBootstrapSamples = value
        if not np.issubdtype(type(value), np.integer):
            logger.error(f"TypeError: invalid argument for numBootstrapSamples ({value}). Setting numBootstrapSamples to 1000.")
            self._numBootstrapSamples = 1000
        if value <= 0:
            logger.warning(f"ValueError: numBootstrapSamples ({value}) is less or equal to 0. Setting numBootstrapSamples to 1000.")
            self._numBootstrapSamples = 1000

    @property
    def lenBootstrapSamples(self):
        """The length of the bootstrap samples that shall be generated for the
        estimation of the uncertainty of the beta-factor

        Example
        -------
        If the sample has 100 data points. Then the length can be between 0 and 100.
        If the length is 50, then each bootstrap sample has 50 data points.
        """
        return round(self._lenBootstrapSamples * self.lenInputPower)

    @lenBootstrapSamples.setter
    def lenBootstrapSamples(self, value):
        logger.debug(f"Setting lenBootstrapSamples to {value}.")
        self._lenBootstrapSamples = value
        if not ( np.issubdtype(type(value), np.integer) or np.issubdtype(type(value), np.floating) ):
            logger.error(f"TypeError: invalid argument for lenBootstrapSamples ({value}). Setting lenBootstrapSamples to 1.")
            self._lenBootstrapSamples = 1
        if not ( 0 < value <= 1):
            logger.warning(f"ValueError: lenBootstrapSamples ({value}) is out of bounce (0, 1]. Setting lenBootstrapSamples to 1.")
            self._lenBootstrapSamples = 1

    @property
    def minimizeError(self):
        """Switches whether the relative or the absolute error is minimized in the beta-fit
        """
        return self._minimizeError

    @minimizeError.setter
    def minimizeError(self, value):
        logger.debug(f"Setting minimizeError to {value}")
        if value.lower() in ["absolute", "abs"]:
            self._minimizeError = 'abs'
        elif value.lower() in ["relative", "rel"]:
            self._minimizeError = 'rel'
        else:
            logger.error(f"{value} is an invalid argument for minimzeError. Only 'relative' and 'absolute' are accepted. Setting it to relative")
            self._minimizeError = 'rel'

    def save_settings(self, fileName: str) -> None:
        """Handles how the settings of the Powerseries-Evaluation are saved. (into output/filename when saveData is enabled)
        """
        IndexList: list = ["fitmodel", "exclude", "minInputPowerRange", "maxInputPowerRange", "fit range scale", "output scale", "initRange",
            "minEnergy", "maxEnergy", "maxInitRange", "background", "constantPeakWidth", "intCoverage"]
        dictData: dict = dict()
        for powerseries in self.powerSeriesList:
            if powerseries.initRange is None:
                minInitRangeEnergy, maxInitRangeEnergy, maxInitRange = None, None, None
            else:
                minInitRangeEnergy, maxInitRangeEnergy = powerseries.minInitRangeEnergy, powerseries.maxInitRangeEnergy
                maxInitRange = powerseries.maxInitRange

            if powerseries.exclude is []:
                minInputPowerRange, maxInputPowerRange = None, None
            else:
                minInputPowerRange, maxInputPowerRange = powerseries.minInputPowerRange, powerseries.maxInputPowerRange
            dictData[powerseries.fileName] = [powerseries.fitModel.name, powerseries.exclude, minInputPowerRange, maxInputPowerRange,
                powerseries.fitRangeScale, powerseries.powerScale, powerseries.initRange, minInitRangeEnergy, maxInitRangeEnergy, maxInitRange,
                powerseries.backgroundFitMode, powerseries.constantPeakWidth, powerseries.intCoverage]

        df: pd.DataFrame = pd.DataFrame(dictData, index=IndexList)
        filePath: Path = (self.outputPath / fileName).resolve()
        df.to_csv(filePath, sep="\t", index=True)

    def save_fit_data(self, fileName: str) -> None:
        """Handles how the results of the fit are saved. (into output/filename when saveData is enabled)
        """
        dictData: dict = {"xiMin" : self.xiHatMin, "xiMax" : self.xiHatMax, "xiEstimateFit" : self.xiHatEstimateWithoutn0,
        "fitParamsBeta" : self.fitParams, "uncFitParamsBeta" : self.uncFitParams,
        "beta" : self.beta, "uncBetaFit" : self.uncBetaFit, "threshold" : self.thresholdInput,
        "Q-factor" : self.QFactorThreshold, "unc Q-factor" : self.uncQFactorThreshold,
        "modeEnergy" : self.modeWavelength[0], "uncModeEnergy" : self.modeWavelength[1]}
        if self.useBootstrap:
            dictData["uncBetaBootstrap"] = self.uncBetaBootstrap
            dictData["bootstrap seed"] = self.bootstrapSeed
        df: pd.DataFrame = pd.DataFrame(dictData)
        filePath: Path = (self.outputPath / fileName).resolve()
        df.to_csv(filePath, sep="\t", index=False)

    def save_powerseries_data(self, fileName: str) -> None:
        """Handles how the original powerseries data is saved. (into output/filename when saveData is enabled)
        """
        dictData: dict = {"in" : self.inputPower, "out" : self.outputPowerArr, "uncOut" : self.uncOutputPowerArr,
        "lw" : self.linewidthArr, "uncLw" : self.uncLinewidthArr,
        "modeEnergy" : self.modeWavelengthArr, "uncModeEnergy" : self.uncModeWavelengthArr,
        "Q-factor" : self.QFactorArr, "uncQ-factor" : self.uncQFactorArr}
        df: pd.DataFrame = pd.DataFrame(dictData)
        filePath: Path = (self.outputPath / fileName).resolve()
        df.to_csv(filePath, sep="\t", index=False)

    def access_single_spectrum(self, idx):
        """Access an individual spectrum of the powerseries
        """
        logger.debug("Calling access_single_spectrum()")
        logger.debug(f"lenInputPower = {np.sum(self.lenInputPowerArr)}, lenInputPowerArr = {self.lenInputPowerArr}")
        assert np.issubdtype(type(idx), np.integer)
        for i, ps in enumerate(self.powerSeriesList):
            if i == 0:
                start = 0
                end = ps.lenInputPower
            else:
                start = int(np.sum(self.lenInputPowerArr[ : i]))
                end = start + ps.lenInputPower
            logger.debug(f"i: [{i}], start: [{start}], end: [{end}]")
            if start <= idx < end:
                idxPlot = idx - start
                logger.debug(f"idxPlot: [{idxPlot}]")
                return ps.access_single_spectrum(idxPlot)
        else:
            logger.error(f"index [{idx}] is out of range [0, {int(np.sum(self.lenInputPowerArr) - 1)}]")
            return None

    def plot_single_spectrum(self, idx):
        """Plots an individual spectrum of the powerseries
        """
        Fit = self.access_single_spectrum(idx)
        try:
            Fit.plot_original_data()
        except AttributeError:
            logger.error("Could not plot original data, because access_single_spectrum() returned NoneType -> idx out of range")

    def plot_multiple_spectra(self, numPlots=5):
        """Shows a waterfall-plot of the spectra from the powerseries.
        """
        logger.debug("Calling plot_multiple_spectra()")
        assert np.issubdtype(type(numPlots), np.integer)
        assert 1 <= numPlots <= int(np.sum(self.lenInputPowerArr))
        logger.debug(f"lenInputPowerArr = {int(np.sum(self.lenInputPowerArr))}, numPlots = {numPlots}")
        idxDistance = int(np.floor(np.sum(self.lenInputPowerArr) / numPlots))
        logger.debug(f"idxDistance = {idxDistance}")
        idxList = list(range(0, self.lenInputPower, idxDistance))[ : numPlots]
        logger.debug(f"idxList = {idxList}")
        verts = []
        maxHeight = 0
        minEList = []
        maxEList = []
        inputPowerList = []
        for idx in idxList:
            Fit = self.access_single_spectrum(idx)
            try:
                maxHeight = max(maxHeight, max(Fit.originalIntensity))
                wavelengths = Fit.wavelengths
                wavelengthDistance = np.abs(wavelengths[1] - wavelengths[0])
                padLow = Fit.minWavelength - wavelengthDistance
                padHigh = Fit.maxWavelength + wavelengthDistance
                if wavelengths[-1] > wavelengths[0]:
                    wavelengthsPad = np.pad(wavelengths, (1, 1), "constant", constant_values=(padLow, padHigh))
                elif wavelengths[-1] < wavelengths[0]:
                    wavelengthsPad = np.pad(wavelengths, (1, 1), "constant", constant_values=(padHigh, padLow))
                else:
                    logger.error(f"Unexpected behaviour of wavelengths array (only 1 value, or all values the same)")
                    raise ValueError
                verts.append(list(zip(wavelengthsPad, np.pad(Fit.originalIntensity, (1, 1), "constant", constant_values=(0, 0)))))
            except AttributeError:
                logger.error("Could not plot original data, because access_single_spectrum() returned NoneType -> idx out of range")
            except ValueError:
                logger.error(f"Unexpected behaviour of wavelengths array (only 1 value, or all values the same)")
            else:
                minEList.append(padLow)
                maxEList.append(padHigh)
                inputPowerList.append(self.inputPowerComplete[idx])

        logger.debug(f"minEList: {minEList}")
        logger.debug(f"maxEList: {maxEList}")
        logger.debug(f"inputPowerList = {inputPowerList}")
        inputPowerListLog = np.log10(inputPowerList)
        logger.debug(f"inputPowerList = {inputPowerListLog}")
        assert len(set(minEList)) == 1
        assert len(set(maxEList)) == 1

        if len(verts) != 0:
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            poly = PolyCollection(verts, facecolor='white')
            poly.set_edgecolor('black')
            poly.set_alpha(1)
            ax.add_collection3d(poly, zs=inputPowerListLog, zdir='y')
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
            ylimDistance = ( max(inputPowerListLog) - min(inputPowerListLog) ) / 10 / numPlots
            ax.set_ylim3d(min(inputPowerListLog) - ylimDistance, max(inputPowerListLog) + ylimDistance)
            ax.set_zlim3d(0, maxHeight)
            ax.set_xlim3d(minEList[0], maxEList[0])
            ax.set_ylabel("input power [mW]")
            ax.set_xlabel("wavelength [eV]")
            ax.set_zlabel("intensity [a.u.]")
            plt.tight_layout()
            plt.show()

    def constant_lines_inout(self):
        """Calculates constant lines for the in-out-plot.
        This helps to check whether the tails are linear.
        """
        minInputPower = np.argmin(self.inputPower)
        maxInputPower = np.argmax(self.inputPower)
        lowSlope = self.outputPowerArr[minInputPower] / self.inputPower[minInputPower]
        highSlope = self.outputPowerArr[maxInputPower] / self.inputPower[maxInputPower]
        logger.debug(f"lowSlope : {lowSlope}")
        logger.debug(f"highSlope : {highSlope}")
        lowLinearFunc = lambda x : lowSlope * x
        highLinearFunc = lambda x : highSlope * x
        # lowerSInputPower = self.inputPower[self.minInputPowerRange[0] : self.minInputPowerRange[1] + 1]
        # upperSInputPower = self.inputPower[self.maxInputPowerRange[0] : self.maxInputPowerRange[1] + 1]
        lowY = lowLinearFunc(self.inputPower)
        highY = highLinearFunc(self.inputPower)
        self.n0Estimate = lowSlope / highSlope
        """This estimate is only valid for beta << 1.
        The approach is taken from:

        L. Andreoli, X. Porte, T. Heuser, J. Große, B. Moeglen-Paget, L. Furfaro, S. Reitzenstein, and D. Brunner,
        "Optical pumping of quantum dot micropillar lasers," Opt. Express 29, 9084-9097 (2021)
        """
        logger.debug(f"n0Estimate: {self.n0Estimate}")
        return self.inputPower, lowY, self.inputPower, highY

    def decorator_in_out_fit(in_out_func):
        """Decorator for a fitfunction of the in-out-curve.

        Handles the available configurations from the .ini file
        and performs the fit using them.
        """
        def wrapper(self):
            assert hasattr(in_out_func, "__call__")
            funcArgs = inspect.getfullargspec(in_out_func)[0]
            assert funcArgs[1] == "beta"
            assert funcArgs[2] == "p"
            self.in_out_curve = in_out_func
            numArgs = len(funcArgs) - 1
            if self.initialParamGuess is None:
                listOfOnes: list = [1 for _ in range(numArgs)]
                self.initialParamGuess = ",".join(str(i) for i in listOfOnes)
            else:
                assert numArgs == len(self.initialParamGuess)
            if self.useParamBounds:
                lowerBounds = [0] * (numArgs)
                higherBounds = [1] + [np.inf]*(numArgs - 1)
                paramBounds = (lowerBounds, higherBounds)
            else:
                paramBounds = -np.inf, np.inf
            logger.debug(f"paramBounds: {paramBounds}")
            if self.minimizeError == "abs":
                if self.useWeights:
                    if self.useBootstrap:
                        bootstrap = Bootstrap(self.outputPowerArr, self.inputPower, self.in_out_curve, parameter=0,
                                iterGuess=self.iterGuessBootstrap, paramBounds=paramBounds, weights=self.uncOutputPowerArr,
                                pGuess=self.initialParamGuess)
                        bootstrap.run(numSamples=self.numBootstrapSamples, lenSamples=self.lenBootstrapSamples,
                                plotHisto=self.plotBootstrap)
                    p, cov = optimize.curve_fit(self.in_out_curve, self.outputPowerArr, self.inputPower, bounds=paramBounds,
                            sigma=self.uncOutputPowerArr, p0=self.initialParamGuess)
                else:
                    if self.useBootstrap:
                        bootstrap = Bootstrap(self.outputPowerArr, self.inputPower, self.in_out_curve, parameter=0,
                                    iterGuess=self.iterGuessBootstrap, paramBounds=paramBounds, pGuess=self.initialParamGuess)
                        bootstrap.run(numSamples=self.numBootstrapSamples, lenSamples=self.lenBootstrapSamples,
                                    plotHisto=self.plotBootstrap)
                    p, cov = optimize.curve_fit(self.in_out_curve, self.outputPowerArr, self.inputPower, bounds=paramBounds,
                                p0=self.initialParamGuess)
            elif self.minimizeError == "rel":
                self.log_in_out_curve = lambda x, beta, p, *args : np.log(self.in_out_curve(x, beta, p, *args))
                if self.useWeights:
                    if self.useBootstrap:
                        bootstrap = Bootstrap(self.outputPowerArr, np.log(self.inputPower), self.log_in_out_curve, parameter=0,
                                    iterGuess=self.iterGuessBootstrap, paramBounds=paramBounds, weights=self.uncOutputPowerArr,
                                    pGuess=self.initialParamGuess)
                        bootstrap.run(numSamples=self.numBootstrapSamples, lenSamples=self.lenBootstrapSamples,
                                    plotHisto=self.plotBootstrap)
                    p, cov = optimize.curve_fit(self.log_in_out_curve, self.outputPowerArr, np.log(self.inputPower),
                                bounds=paramBounds, sigma=self.uncOutputPowerArr, p0=self.initialParamGuess)
                else:
                    if self.useBootstrap:
                        bootstrap = Bootstrap(self.outputPowerArr, np.log(self.inputPower), self.log_in_out_curve, parameter=0,
                                    iterGuess=self.iterGuessBootstrap, paramBounds=paramBounds, pGuess=self.initialParamGuess)
                        bootstrap.run(numSamples=self.numBootstrapSamples, lenSamples=self.lenBootstrapSamples,
                                    plotHisto=self.plotBootstrap)
                    p, cov = optimize.curve_fit(self.log_in_out_curve, self.outputPowerArr, np.log(self.inputPower),
                                bounds=paramBounds, p0=self.initialParamGuess)
            else:
                raise NotImplementedError("Only relative and absolute modes are implemented")

            logger.debug(f"xiHatEstimate; n0 not estimated: {self.xiHatEstimateWithoutn0:.5f}")
            np.set_printoptions(precision=5, suppress=True)
            self.fitParams = p
            self.uncFitParams = np.sqrt(np.diag(cov))
            logger.info(f"fit parameters (beta, p, A, xiHat):               {self.fitParams}")
            logger.info(f"unc fit params (beta, p, A, xiHat):               {self.uncFitParams}")
            np.set_printoptions(precision=8, suppress=False)
            self.beta = p[0]
            self.uncBetaFit = np.sqrt(cov[0, 0])
            self.thresholdOutput = 1 / p[1]
            self.uncThresholdOutput = np.sqrt(cov[1, 1]) / p[1]
            self.thresholdInput = self.in_out_curve(self.thresholdOutput, *p)
            if self.thresholdInput < self.minInputPower:
                logger.warning(f"""The estimated threshold [{self.thresholdInput} mW] is smaller than the lowest measured input power [{self.minInputPower} mW].
Hence, the Q-factor (taken at the inputpower which is closest to the threshold) will be overestimated.""")
            thresholdIdx = np.argmin(np.abs(self.inputPower - self.thresholdInput))
            self.QFactorThreshold = self.QFactorArr[thresholdIdx]
            self.uncQFactorThreshold = self.uncQFactorArr[thresholdIdx]
            self.QEstimate = self.QFactorThreshold
            # logger.info(f"beta factor (1 fit cov unc): {self.beta:.5f} \u00B1 {self.uncBeta:.5f}")
            logger.debug(f"threshold output: {misc.round_value(self.thresholdOutput, self.uncThresholdOutput)}")
            logger.debug(f"New value for xiHatEstimate (used Q-factor): {self.xiHatEstimateWithoutn0:.5f}")
            logger.info(f"threshold input:                                  {self.thresholdInput:.5f}")
            logger.info(f"mode energy:                                      {misc.round_value(self.modeWavelength[0], self.modeWavelength[1])}")
            logger.info(f"Q-factor at threshold:                            {misc.round_value(self.QFactorThreshold, self.uncQFactorThreshold)}")
            if self.useBootstrap:
                self.uncBetaBootstrap = bootstrap.results[1]
                self.bootstrapSeed = bootstrap.seed
                logger.debug(f"bootstrap seed: {self.bootstrapSeed}")
                logger.info(f"bootstrap beta (original, error bias corrected):  {misc.round_value(bootstrap.results[0], bootstrap.results[1])}")
                logger.info(f"bootstrap beta (mean, error mean):                {misc.round_value(bootstrap.results[2], bootstrap.results[3])}")
            if self.saveData:
                self.save_fit_data("beta_fit.csv")
        return wrapper

## in all models x corresponds to the mean intracavity photon number <n>
## <n> is linear proportional to the output power (maybe not here because of arbitrary units)
## removing tauFrac and making the beta dependence of A and xi explicit
## reduces the uncertainty of beta by a factor of 100
## future plan: remove xiHat as fit parameter
    @decorator_in_out_fit
    def beta_factor_2(x, beta, p, A, xiHat):
        """Main in-out-curve used in the analysis (used by Ching-Wen, Arsenty)
        """
        return (A / beta) * ( p*x / (p*x + 1) * (1 + xiHat*beta) * (1 + beta*p*x) - (beta**2)*xiHat*p*x)

    @decorator_in_out_fit
    def in_out_curve2(x, beta, p, a, xi, tauFrac):
        """Deprecated
        In-out curve original with tauFrac and xi instead of xiHat
        """
        return a * ( p*x / (p*x + 1) * (1 + xi) * (1 + beta*p*x + tauFrac) - beta*xi*p*x)

    @decorator_in_out_fit
    def in_out_curve1(x, beta, p, A, xi):
        """model used by DOI 1.3315869
        """
        return A * p * x * (1 + 2*xi + 2*beta*(p*x - xi)) / (1 + 2*p*x)

## has to be combined with in-out fit
    @staticmethod
    def lw_fit(x, a, xi, p):
        """Deprecated
        fit of the linewidth, produced not very good results."""
        return a * (1 + 2*xi) / (1 + 2*p*x)

    ## Disable the use of the fit
    def plot_linewidth(self, block=True):
        """Plots the linewidth vs inputpower, also tries to fit the data.
        """
        try:
            p, _ = optimize.curve_fit(self.lw_fit, self.outputPowerArr, self.linewidthArr, sigma=self.uncLinewidthArr)
        except RuntimeError:
            logger.error("Linewidth fit did not work.")
        else:
            lwPlotArr = self.lw_fit(self.outputPowerArr, *p)
            plt.plot(self.inputPower, lwPlotArr, color="orange")
        finally:
            plt.errorbar(self.inputPower, self.linewidthArr, yerr=self.uncLinewidthArr, capsize=3, fmt=".")
            plt.yscale("log")
            plt.xscale("log")
            plt.xlabel("input power [mW]")
            plt.ylabel("linewidth [eV]")
            plt.tight_layout()
            if block:
                plt.show()
            else:
                plt.show(block=False)
                plt.pause(1)
                plt.close("all")

    def plot_outputPower(self, block=True):
        """Plots the in-out-curve with the beta-fit already calculated.
        """
        if hasattr(self, "beta"):
            outputPlotArr = np.logspace(np.log10(np.amin(self.outputPowerArr)), np.log10(np.amax(self.outputPowerArr)), 100)
            inputPlotArr = self.in_out_curve(outputPlotArr, *self.fitParams)
            plt.plot(inputPlotArr, outputPlotArr, color="orange")
        plt.errorbar(self.inputPower, self.outputPowerArr, yerr=self.uncOutputPowerArr, capsize=3, fmt=".")
        lowX, lowY, highX, highY = self.constant_lines_inout()
        plt.plot(lowX, lowY, color="orange")
        plt.plot(highX, highY, color="orange")
        plt.xlabel("input power [mW]")
        plt.yscale("log")
        plt.xscale("log")
        plt.ylabel("Intensity [a. u.]")
        plt.tight_layout()
        if block:
            plt.show()
        else:
            plt.show(block=False)
            plt.pause(1)
            plt.close("all")

    def plot_mode_wavelength(self, block=True):
        """Plots the mode energy vs inputpower
        """
        plt.errorbar(self.inputPower, self.modeWavelengthArr, yerr=self.uncModeWavelengthArr, capsize=3, fmt=".")
        plt.xscale("log")
        plt.xlabel("input power [mW]")
        plt.ylabel("wavelength [eV]")
        plt.tight_layout()
        if block:
            plt.show()
        else:
            plt.show(block=False)
            plt.pause(1)
            plt.close("all")

    def plot_QFactor(self, block=True):
        """Plots the Q-factor vs inputpower
        """
        plt.errorbar(self.inputPower, self.QFactorArr, yerr=self.uncQFactorArr, capsize=3, fmt=".")
        plt.xscale("log")
        plt.xlabel("input power [mW]")
        plt.ylabel("Q-factor")
        plt.tight_layout()
        if block:
            plt.show()
        else:
            plt.show(block=False)
            plt.pause(1)
            plt.close("all")

    def plot_lw_s(self, block=True):
        """Plots the in-out-characteristic and the linewidth vs inputpower in the same plot
        """
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.errorbar(self.inputPower, self.outputPowerArr, yerr=self.uncOutputPowerArr, capsize=3, fmt="b.")
        if hasattr(self, "beta"):
            outputPlotArr = np.logspace(np.log10(np.amin(self.outputPowerArr)), np.log10(np.amax(self.outputPowerArr)), 100)
            inputPlotArr = self.in_out_curve(outputPlotArr, *self.fitParams)
            ax1.plot(inputPlotArr, outputPlotArr, color="orange")
        lowX, lowY, highX, highY = self.constant_lines_inout()
        ax1.plot(lowX, lowY, color="orange")
        ax1.plot(highX, highY, color="orange")
        ax1.set_yscale("log")
        ax1.set_xscale("log")
        ax1.set_xlabel("input power [mW]")
        ax1.set_ylabel("Intensity [a. u.]")
        ax2.errorbar(self.inputPower, self.linewidthArr, yerr=self.uncLinewidthArr, capsize=3, fmt="g.")
        ax2.plot(self.inputPower, np.ones(self.lenInputPower)*self.resolutionLimit, color="gray", linestyle="--")
        ax2.set_yscale("log")
        ax2.set_ylabel("Linewidth [eV]")
        plt.tight_layout()
        if block:
            plt.show()
        else:
            plt.show(block=False)
            plt.pause(1)
            plt.close("all")



if __name__ == "__main__":
    ## just testing
    head = (Path(__file__).parents[2]).resolve()
    fileName = "data\\20210303\\NP7509_Ni_4µm_20K_Powerserie_1-01s_deteOD0_fine3_WithoutLensAllSpectra.dat"
    fileName = fileName.replace("\\", "/")
    filePath = (head / fileName).resolve()
    test = EvalPowerSeries(filePath)
    test.get_power_dependent_data()
    test2 = PlotPowerSeries([test])
    test2.plot_outputPower()