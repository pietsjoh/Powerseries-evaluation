import numpy as np
import matplotlib.pyplot as plt # type: ignore
import scipy.signal as signal # type: ignore
import scipy.optimize as optimize # type: ignore
import scipy.ndimage as ndi # type: ignore
import copy
import scipy.interpolate as interpolate # type: ignore
from functools import partial
import typing

import sys
from pathlib import Path
headDirPath: Path = Path(__file__).resolve().parents[1]
sys.path.append(str(headDirPath))
from setup.config_logging import LoggingConfig

loggerObj: LoggingConfig = LoggingConfig()
logger = loggerObj.init_logger(__name__)

tupleIntOrNone = typing.Union[tuple[int, int], None]
number = typing.Union[int, float, np.number]

class PeakFitSuper:
    smoothGauss: number = 7
    smoothSpline: number = 0

    def __init__(self, wavelengths: np.ndarray, intensity: np.ndarray, initRange: tupleIntOrNone=None, intCoverage: number=1,
                fitRangeScale: number=5, constantPeakWidth: int=50, backgroundFitMode: str="spline") -> None:
        assert isinstance(wavelengths, np.ndarray)
        assert isinstance(intensity, np.ndarray)
        assert np.issubdtype(type(intCoverage), np.integer) or np.issubdtype(type(intCoverage), np.floating)
        assert (0 <= intCoverage <= 1)
        assert np.issubdtype(type(fitRangeScale), np.integer) or np.issubdtype(type(fitRangeScale), np.floating)
        assert np.issubdtype(type(constantPeakWidth), np.integer)
        assert (0 <= constantPeakWidth <= (wavelengths.size / 2))
        assert backgroundFitMode in ["spline", "constant", "local_all", "local_left", "local_right", "offset", "none", "disable"]

        self.wavelengths: np.ndarray = wavelengths
        self.minWavelength: number = np.amin(self.wavelengths)
        self.maxWavelength: number = np.amax(self.wavelengths)
        self.data: np.ndarray = intensity
        self.originalData: np.ndarray = copy.deepcopy(self.data)
        self.fitRangeScale: number = fitRangeScale
        self.intCoverage: number = intCoverage
        self.constantPeakWidth: int = constantPeakWidth
        self.initialRange: tupleIntOrNone = initRange
        self.backgroundFitMode: str = backgroundFitMode

        self.muFit: number
        self.fwhmFit: number
        self.integratedPeak: number
        self.uncintegratedPeak: number
        self.uncFwhmFit: number
        self.uncMuFit: number
        logger.debug("PeakFitSuper object initialized.")

    def run(self) -> None:
        logger.debug("Calling PeakFitSuper.run()")
        foundPeakFlag: bool
        convergenceFitFlag: bool
        try:
            self.get_peak(initialRange=self.initialRange, peakWidth=self.constantPeakWidth)
        except RuntimeError:
            foundPeakFlag = False
            logger.error("RuntimeError: get_peak() could not find a peak.")
            plt.title("no peak could be found")
            self.plot_original_data(block=True)
        else:
            foundPeakFlag = True
            if self.backgroundFitMode == "spline":
                self.remove_background_smoothspline(smoothSpline=self.smoothSpline, smoothGauss=self.smoothGauss, peakWidth=self.constantPeakWidth)
            elif self.backgroundFitMode == "constant":
                self.data = self.originalData - np.median(self.originalData)
            elif self.backgroundFitMode == "local_all":
                self.remove_background_local_mean(peakWidth=self.constantPeakWidth, mode="all")
            elif self.backgroundFitMode == "local_left":
                self.remove_background_local_mean(peakWidth=self.constantPeakWidth, mode="left")
            elif self.backgroundFitMode == "local_right":
                self.remove_background_local_mean(peakWidth=self.constantPeakWidth, mode="right")
            if self.backgroundFitMode in ["spline", "constant", "local_all", "local_left", "local_right"]:
                try:
                    self.get_peak(initialRange=self.initialRange, peakWidth=self.constantPeakWidth)
                except ValueError:
                    foundPeakFlag = False
                    logger.error(f"ValueError: estimate of peak height ({self.peakHeightEstimate}) is below 0.")
                    plt.title("peak height estimate below 0")
                    self.plot_original_data(block=True)
                except RuntimeError:
                    foundPeakFlag = False
                    logger.error("RuntimeError: get_peak() could not find a peak.")
                    plt.title("no peak could be found")
                    self.plot_original_data(block=True)
            if foundPeakFlag:
                try:
                    self.fit_peak(intCoverage=self.intCoverage, fitRangeScale=self.fitRangeScale)
                except RuntimeError:
                    convergenceFitFlag = False
                    logger.error("RuntimeError: fit_peak() could not fit the peak.")
                    plt.title("the peak could not be fitted")
                    self.plot_original_data(block=True)
                else:
                    convergenceFitFlag = True
                    self.set_fwhm()
        if not foundPeakFlag or not convergenceFitFlag:
            self.muFit = np.nan
            self.fwhmFit = np.nan
            self.integratedPeak = np.nan
            self.uncintegratedPeak = np.nan
            self.uncFwhmFit = np.nan
            self.uncMuFit = np.nan
        logger.debug("""Results:
                    mu = {} \u00B1 {}
                    fwhm = {} \u00B1 {}
                    integrated Intensity = {} \u00B1 {}""".format(self.muFit, self.uncMuFit,
                    self.fwhmFit, self.uncFwhmFit,
                    self.integratedPeak, self.uncintegratedPeak))

    def get_peak(self, initialRange: tupleIntOrNone=None, peakWidth: int=50) -> None:
        ## setting an initial range, where the algorithm looks for the peak
        ## looking for all peaks
        logger.debug("Calling PeakFitSuper.get_peak()")
        assert isinstance(initialRange, (tuple, type(None)))
        peaks: np.ndarray
        peaks, _ = signal.find_peaks(self.data)
        if isinstance(initialRange, tuple):
            assert len(initialRange) == 2
            assert np.issubdtype(type(initialRange[0]), np.integer)
            assert np.issubdtype(type(initialRange[1]), np.integer)
            minIdx: int = min(initialRange)
            maxIdx: int = max(initialRange)
            logger.debug(f"Detected initial range: [{minIdx}, {maxIdx}]")
            peakList: list = []
            for peak in peaks:
                if (minIdx <= peak) and (maxIdx >= peak):
                    peakList.append(peak)
            peaks = np.array(peakList)
        # logger.debug(f"All found peaks: {peaks}")
        prominences: np.ndarray = signal.peak_prominences(self.data, peaks, wlen=2*peakWidth)[0]
        if len(prominences) == 0:
            raise RuntimeError("No peak could be found")
        self.peak: int = peaks[np.argmax(prominences)]
        self.peakHeightEstimate: number = self.data[self.peak]
        if self.peakHeightEstimate <= 0:
            raise ValueError(f"The estimated height of peak ({self.peakHeightEstimate}) is below 0")
        self.fwhmEstimate: number = signal.peak_widths(self.data, np.array([self.peak]), rel_height=0.5, wlen=2*peakWidth)[0][0]
        logger.debug(f"Found peak at: {self.wavelengths[self.peak]} nm/eV, index: [{self.peak}]")
        logger.debug(f"Estimate of FWHM: {np.abs(self.wavelengths[1] - self.wavelengths[0])*self.fwhmEstimate} nm/eV, index length: [{self.fwhmEstimate}]")
        logger.debug(f"Estimate of the Height: {self.peakHeightEstimate} a.u.")

    def remove_background_smoothspline(self, smoothSpline: number=0, smoothGauss: number=7, peakWidth: int=50) -> None:
        logger.debug(f"Calling remove_background_smoothspline(); smoothspline: {smoothSpline}, smoothGauss: {smoothGauss}, peakWidth: {peakWidth}")
        try:
            dataBeforePeak: np.ndarray = self.data[: self.peak - peakWidth - round(self.fwhmEstimate)] # type: ignore
            dataAfterPeak: np.ndarray = self.data[self.peak + peakWidth + round(self.fwhmEstimate) + 1 :] # type: ignore
        except IndexError:
            logger.error("""Data range for peak exclusion exceeded while trying to use spline background removal.
Either lower constantPeakWidth or use a different background fitting method.""")
        else:
            dataBeforePeak = ndi.gaussian_filter1d(dataBeforePeak, sigma=smoothGauss)
            dataAfterPeak = ndi.gaussian_filter1d(dataAfterPeak, sigma=smoothGauss)
            dataWithoutPeak: np.ndarray = np.hstack((dataBeforePeak, dataAfterPeak))
            wavelengthsBeforePeak: np.ndarray = self.wavelengths[: self.peak - peakWidth - round(self.fwhmEstimate)]  # type: ignore
            wavelengthsAfterPeak: np.ndarray = self.wavelengths[self.peak + peakWidth + round(self.fwhmEstimate) + 1 :]  # type: ignore
            wavelengthsWithoutPeak: np.ndarray = np.hstack((wavelengthsBeforePeak, wavelengthsAfterPeak))
            logger.debug(f"Exclusion range smooth spline: [{min(wavelengthsBeforePeak)}, {max(wavelengthsAfterPeak)}]")
            self.spline: interpolate.UnivariateSpline = interpolate.UnivariateSpline(wavelengthsWithoutPeak, dataWithoutPeak)
            self.spline.set_smoothing_factor(smoothSpline)
            self.data = self.data - self.spline(self.wavelengths)

    def remove_background_local_mean(self, peakWidth: int=50, peakWidthScale: number=3, mode: str="all") -> None:
        logger.debug(f"Calling remove_background_local_mean(); peakWidth: {peakWidth}, peakWidthScale: {peakWidthScale}")
        assert mode in ["left", "right", "all"]
        try:
            dataBeforePeak: np.ndarray = self.data[self.peak - peakWidthScale*peakWidth - round(self.fwhmEstimate) : self.peak - peakWidth - round(self.fwhmEstimate) + 1]  # type: ignore
            dataAfterPeak: np.ndarray = self.data[self.peak + peakWidth + round(self.fwhmEstimate) : self.peak + peakWidthScale*peakWidth + round(self.fwhmEstimate) + 1]  # type: ignore
            logger.debug(f"indices before peak: [{self.peak - peakWidthScale*peakWidth - round(self.fwhmEstimate)}, {self.peak - peakWidth - round(self.fwhmEstimate)}]")  # type: ignore
            logger.debug(f"wavelength before peak: [{self.wavelengths[self.peak - peakWidthScale*peakWidth - round(self.fwhmEstimate)]}, {self.wavelengths[self.peak - peakWidth - round(self.fwhmEstimate)]}]")  # type: ignore
            logger.debug(f"wavelength after peak: [{self.wavelengths[self.peak + peakWidth + round(self.fwhmEstimate)]}, {self.wavelengths[self.peak + peakWidthScale*peakWidth + round(self.fwhmEstimate)]}]")  # type: ignore
        except IndexError:
            logger.error("""Data range for peak exclusion exceeded while trying to use local background removal.
Either lower constantPeakWidth or use a different background fitting method. Now the 'none/disable' method will be used for only this spectrum.""")
        else:
            dataAroundPeak: np.ndarray = np.concatenate((dataBeforePeak, dataAfterPeak))
            if mode == "all":
                self.data = self.originalData - np.median(dataAroundPeak)
            elif mode == "left":
                self.data = self.originalData - np.median(dataBeforePeak)
            elif mode == "right":
                self.data = self.originalData - np.median(dataAfterPeak)

    def fit_peak(self, intCoverage: number=1, fitRangeScale: number=2) -> None:
        logger.debug("Calling PeakFitSuper.fit_peak()")
        assert hasattr(self, "peak")
        fitRange: int = round(fitRangeScale*self.fwhmEstimate)  # type: ignore
        try:
            fitWavelengths: np.ndarray = self.wavelengths[self.peak - fitRange : self.peak + fitRange + 1]
        except IndexError:
            logger.error("Fitrange exceeds dataset. Lower fit range scale to enable fitting.")
        else:
            logger.debug(f"Selected fit range ({len(fitWavelengths)} points): [{min(fitWavelengths)}, {max(fitWavelengths)}] eV")
            if len(fitWavelengths) <= 3:
                logger.warning(f"Less than 3 points are used for the fitting. Aborting fitting proccss.")
                raise RuntimeError("Not enough points for fitting.")
            fitData: np.ndarray = self.data[self.peak - fitRange : self.peak + fitRange + 1]
            self.set_p0()
            self.p: np.ndarray
            self.cov: np.ndarray
            if self.backgroundFitMode == "offset":
                self.p0.append(np.median(self.data))  # type: ignore
                lowerBounds: list
                higherBounds: list
                lowerBounds, higherBounds = self.paramBounds  # type: ignore
                lowerBounds.append(0)
                higherBounds.append(np.inf)
                paramBounds: tuple[list, list] = (lowerBounds, higherBounds)
                self.p, self.cov = optimize.curve_fit(self, fitWavelengths, fitData, p0=self.p0, bounds=paramBounds)  # type: ignore
            else:
                f: typing.Callable = partial(self, offset=0)
                self.p, self.cov = optimize.curve_fit(f, fitWavelengths, fitData, p0=self.p0, bounds=self.paramBounds)  # type: ignore
                self.p = np.append(self.p, 0)
            logger.debug(f"Fit initial guesses: {self.p0}")  # type: ignore
            logger.debug(f"Fit results: {self.p}")
            self.muFit = self.p[1]
            self.integratedPeak = self.p[0] * intCoverage
            self.uncMuFit = np.sqrt(self.cov[1, 1])
            self.uncintegratedPeak = intCoverage * np.sqrt(self.cov[0, 0])

    def plot_original_data(self, block: bool=True) -> None:
        plt.plot(self.wavelengths, self.originalData)
        plt.xlabel("Energy [eV]")
        plt.ylabel("Intensity [a. u.]")
        plt.tight_layout()
        if block:
            plt.show()
        else:
            plt.show(block=False)
            plt.pause(1)
            plt.close("all")

    def plot_interpolation(self, block: bool=True) -> None:
        assert hasattr(self, "spline")
        fig, axs = plt.subplots(nrows=1, ncols=2)
        ax1, ax2 = axs
        ax1.plot(self.wavelengths, self.originalData)
        ax1.plot(self.wavelengths, self.spline(self.wavelengths))
        ax2.plot(self.wavelengths, self.data)
        if block:
            plt.show()
        else:
            plt.show(block=False)
            plt.pause(1)
            plt.close("all")

    def plot_fit(self, block: bool=True) -> None:
        assert hasattr(self, "p")
        wavelengthsPlotArray: np.ndarray = np.linspace(self.wavelengths[0], self.wavelengths[-1], 1000)
        outputPlotArray: np.ndarray = self(wavelengthsPlotArray, *self.p)
        fig, axs = plt.subplots(nrows=1, ncols=2)
        ax1, ax2 = axs
        ax1.plot(self.wavelengths, self.originalData)
        if hasattr(self, "spline"):
            ax1.plot(self.wavelengths, self.spline(self.wavelengths))
        ax2.plot(self.wavelengths, self.data)
        ax2.plot(wavelengthsPlotArray, outputPlotArray)
        if block:
            plt.show()
        else:
            plt.show(block=False)
            plt.pause(1)
            plt.close("all")

    def plot_fitRange_without_fit(self) -> None:
        assert hasattr(self, "peak")
        assert hasattr(self, "fwhmEstimate")
        fig, axs = plt.subplots(nrows=1, ncols=2)
        ax1, ax2 = axs
        ax1.plot(self.wavelengths, self.originalData)
        if hasattr(self, "spline"):
            ax1.plot(self.wavelengths, self.spline(self.wavelengths))
        ax2.plot(self.wavelengths, self.data)
        fitRange: int = round(self.fitRangeScale * self.fwhmEstimate) # type: ignore
        leftBoundary: number = self.wavelengths[self.peak - fitRange]
        rightBoundary: number = self.wavelengths[self.peak + fitRange]
        ax2.axvline(x=leftBoundary, color="black")
        ax2.axvline(x=rightBoundary, color="black")
        ax1.set_title("Original data")
        ax2.set_title("Fitting range")
        ax2.plot(self.wavelengths[self.peak], self.data[self.peak], "+", color="black")
        plt.show()

    def plot_fitRange_with_fit(self) -> None:
        assert hasattr(self, "peak")
        assert hasattr(self, "fwhmEstimate")
        assert hasattr(self, "p")
        wavelengthsPlotArray: np.ndarray = np.linspace(self.wavelengths[0], self.wavelengths[-1], 1000)
        outputPlotArray: np.ndarray = self(wavelengthsPlotArray, *self.p)
        fig, axs = plt.subplots(nrows=1, ncols=2)
        ax1, ax2 = axs
        ax1.plot(self.wavelengths, self.originalData)
        if hasattr(self, "spline"):
            ax1.plot(self.wavelengths, self.spline(self.wavelengths))
        ax2.plot(self.wavelengths, self.data)
        fitRange: int = round(self.fitRangeScale * self.fwhmEstimate) # type: ignore
        leftBoundary: number = self.wavelengths[self.peak - fitRange]
        rightBoundary: number = self.wavelengths[self.peak + fitRange]
        ax1.axvline(x=leftBoundary, color="black")
        ax1.axvline(x=rightBoundary, color="black")
        ax2.axvline(x=leftBoundary, color="black")
        ax2.axvline(x=rightBoundary, color="black")
        ax1.set_title("Fitting range + original data")
        ax2.set_title("Fitting range + fitted data")
        ax2.plot(wavelengthsPlotArray, outputPlotArray)
        ax1.plot(self.wavelengths[self.peak], self.originalData[self.peak], "+", color="black")
        ax2.plot(self.muFit, self(self.muFit, *self.p), "x", color="black")
        plt.show()

    def plot_initRange_without_fit(self) -> None:
        assert hasattr(self, "peak")
        if isinstance(self.initialRange, tuple):
            assert len(self.initialRange) == 2
            idx1: int
            idx2: int
            idx1, idx2 = self.initialRange
            boundary1: number = self.wavelengths[idx1]
            boundary2: number = self.wavelengths[idx2]
            fig, axs = plt.subplots(nrows=1, ncols=2)
            ax1, ax2 = axs
            ax1.plot(self.wavelengths, self.originalData)
            ax2.plot(self.wavelengths, self.originalData)
            ax2.axvline(x=boundary1, color="black")
            ax2.axvline(x=boundary2, color="black")
            ax2.plot(self.wavelengths[self.peak], self.originalData[self.peak], "+", color="black")
            fig.suptitle("initial range and peak visualized")
            plt.show()
        else:
            fig, ax = plt.subplots(nrows=1, ncols=1)
            ax.plot(self.wavelengths, self.originalData)
            ax.plot(self.wavelengths[self.peak], self.originalData[self.peak], "+", color="black")
            fig.suptitle("No initial Range selected")
            plt.show()

    def plot_initRange_with_fit(self) -> None:
        assert hasattr(self, "peak")
        assert hasattr(self, "p")
        wavelengthsPlotArray: np.ndarray = np.linspace(self.wavelengths[0], self.wavelengths[-1], 1000)
        outputPlotArray: np.ndarray = self(wavelengthsPlotArray, *self.p)
        if isinstance(self.initialRange, tuple):
            assert len(self.initialRange) == 2
            idx1: int
            idx2: int
            idx1, idx2 = self.initialRange
            boundary1: number = self.wavelengths[idx1]
            boundary2: number = self.wavelengths[idx2]
            fig, axs = plt.subplots(nrows=1, ncols=2)
            ax1, ax2 = axs
            ax1.plot(self.wavelengths, self.originalData)
            ax2.plot(self.wavelengths, self.originalData)
            ax1.axvline(x=boundary1, color="black")
            ax1.axvline(x=boundary2, color="black")
            ax2.axvline(x=boundary1, color="black")
            ax2.axvline(x=boundary2, color="black")
            ax2.plot(wavelengthsPlotArray, outputPlotArray)
            ax1.plot(self.wavelengths[self.peak], self.originalData[self.peak], "+", color="black")
            ax2.plot(self.muFit, self(self.muFit, *self.p), "x", color="black")
            fig.suptitle("initial range and peak visualized")
            plt.show()
        else:
            fig, ax = plt.subplots(nrows=1, ncols=1)
            ax.plot(self.wavelengths, self.originalData)
            ax.plot(wavelengthsPlotArray, outputPlotArray)
            ax.plot(self.wavelengths[self.peak], self.originalData[self.peak], "+", color="black")
            fig.suptitle("No initial Range selected")
            plt.show()

    def plot_fwhm(self) -> None:
        assert hasattr(self, "peak")
        assert hasattr(self, "p")
        wavelengthsPlotArray: np.ndarray = np.linspace(self.wavelengths[0], self.wavelengths[-1], 1000)
        outputPlotArray: np.ndarray = self(wavelengthsPlotArray, *self.p)
        fig, axs = plt.subplots(nrows=1, ncols=3)
        ax1, ax2, ax3 = axs
        ax1.plot(self.wavelengths, self.data)
        ax2.plot(self.wavelengths, self.data)
        fwhmEstimate1: number = self.wavelengths[self.peak - round(self.fwhmEstimate/2)] # type: ignore
        fwhmEstimate2: number = self.wavelengths[self.peak + round(self.fwhmEstimate/2)] # type: ignore
        ax1.axvline(x=fwhmEstimate1, color="black", linestyle="--")
        ax1.axvline(x=fwhmEstimate2, color="black", linestyle="--")
        fwhm1: number = self.muFit - self.fwhmFit/2
        fwhm2: number = self.muFit + self.fwhmFit/2
        ax2.axvline(x=fwhm1, color="black")
        ax2.axvline(x=fwhm2, color="black")
        ax1.plot(wavelengthsPlotArray, outputPlotArray)
        ax2.plot(wavelengthsPlotArray, outputPlotArray)
        ax1.plot(self.wavelengths[self.peak], self.data[self.peak], "+", color="black")
        ax2.plot(self.muFit, self(self.muFit, *self.p), "x", color="black")
        ax1.set_title("fwhm estimate")
        ax2.set_title("fwhm fit")
        ax3.plot(self.wavelengths, self.data)
        ax3.axvline(x=fwhmEstimate1, color="black", linestyle="--", label="Estimate")
        ax3.axvline(x=fwhmEstimate2, color="black", linestyle="--")
        ax3.axvline(x=fwhm1, color="black")
        ax3.axvline(x=fwhm2, color="black")
        ax3.plot(wavelengthsPlotArray, outputPlotArray)
        ax3.plot(self.wavelengths[self.peak], self.data[self.peak], "+", color="black")
        ax3.plot(self.muFit, self(self.muFit, *self.p), "x", color="black")
        plt.show()

    def plot_fwhmEstimate(self) -> None:
        assert hasattr(self, "peak")
        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.plot(self.wavelengths, self.data)
        fwhmEstimate1: number = self.wavelengths[self.peak - round(self.fwhmEstimate/2)] # type: ignore
        fwhmEstimate2: number = self.wavelengths[self.peak + round(self.fwhmEstimate/2)] # type: ignore
        ax.axvline(x=fwhmEstimate1, color="black")
        ax.axvline(x=fwhmEstimate2, color="black")
        ax.plot(self.wavelengths[self.peak], self.data[self.peak], "+", color="black")
        ax.set_title("fwhm estimate")
        plt.show()

    def plot_exclusionRange(self) -> None:
        assert hasattr(self, "spline")
        fig, axs = plt.subplots(nrows=1, ncols=2)
        ax1, ax2 = axs
        ax1.plot(self.wavelengths, self.originalData)
        ax1.plot(self.wavelengths, self.spline(self.wavelengths))
        ax2.plot(self.wavelengths, self.data)
        leftBoundary: number = self.wavelengths[self.peak - self.constantPeakWidth - round(self.fwhmEstimate)] # type: ignore
        rightBoundary: number = self.wavelengths[self.peak + self.constantPeakWidth + round(self.fwhmEstimate)] # type: ignore
        ax1.axvline(x=leftBoundary, color="black")
        ax1.axvline(x=rightBoundary, color="black")
        fig.suptitle("Peak exclusion range visualized")
        plt.show()

    @classmethod
    def __init_subclass__(cls, *args, **kwargs) -> None:
        super().__init_subclass__(*args, **kwargs) # type: ignore
        assert hasattr(cls, "set_p0")
        assert hasattr(cls, "set_fwhm")
        assert hasattr(cls, "__call__")
        assert hasattr(cls, "name")

    def __call__(self, *args):
        pass

    def set_fwhm(self):
        pass

    def set_p0(self):
        pass

    @property
    def outputParameters(self):
        return self.muFit, self.fwhmFit, self.integratedPeak

    @property
    def uncertaintyOutputParameters(self):
        return self.uncMuFit, self.uncFwhmFit, self.uncintegratedPeak