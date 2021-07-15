"""Contains the base class that performs the fitting of a single peak.
"""
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
    """
    Abstract Base Class that finds a peak in a spectrum and fits the peak.

    The class supports the use of an initial range, where the peak should lie.
    Moreover, the fitrange can be adapted aswell as the handling of the background.
    These parameters are explained in the __init__ method section.

    Also contains different plots to visualize the configuration and the examined spectrum.
    In the config/debugging.ini file it can be selected which plots should be shown as snapshots.

    Use derived classes to define the fit model. The derived classes have to have a name attribute
    (name of the model). Moreover, the __call__ method should be the fit-function. That function should
    satisfy the following functional relationship: :math:`y = f(x, A, center, *args)`.
    Here *x* corresponds to the input data, and *y* to the output data.
    It should be possible to use an numpy ndarray as an input for x producing a numpy ndarray output
    of equal length.

    The first parameter of the function should correspond to the integral over all
    wavelengths (:math:`\int_{-\infty}^{+\infty}f(x, A, center, *args)dx = A`).
    The second parameter should correspond to the central wavelength.

    Furthermore, the method set_p0() is required for the derived classes. This method should provide
    guesses for the fit parameters based on the estimates of get_peak() (fwhm, height and position estimates).

    The derived class must also have a method called set_fwhm(). This method transforms the fit parameters to
    a FWHM (for example for the Gaussian fitmodel it calculates the FWHM from sigma). Moreover, this method
    should also calculate the uncertainty of the FWHM.

    Finally, the derived class has to have an paramBounds attribute, setting the boundaries for the fit parameters.

    The following attributes can be passed into the __init__ method:

    mandatory: [wavelengths, intensity]

    keyword args: [initRange, intCoverage, fitRangeScale, constantPeakWidth, backgroundFitMode]

    For more information about these attributes look into their corresponding
    section. The default values are used in the __init__ method.

    Attributes
    ----------
    wavelengths: np.ndarray, set by __init__
        Wavelengths of the spectrum (usually as energies in units of [eV])

    intensity: np.ndarray, set by __init__
        The intensity of each Wavlength (count rate of detector)

    initRange: tuple/None, set by __init__, default=None
        When set to None, no initial range is used
        Otherwise the program looks only for peaks in between [min(initRange), max(initRange)]

    intCoverage: int/float, set by __init__, default=1
        The outputpower corresponds to the integrated peak. This parameter can change this integration range.
        This parameter should lie in the range [0, 1]. The outputpower will always be 0, when set to 0.
        When set to 1 the integration is performed over all wavelengths (-inf, inf), even for negative wavelengths.
        However, the wavelengths near the peak contribute mostly to the integral as the peak models should vanish fast
        as the wavelength goes to +/- inf.

    fitRangeScale: int/float, set by __init__, default=1
        Determines range of data points that are used for the fitting process. It scales the estimated FWHM of the peak.
        Then the data near the estimated peak location is used for the fitting

        .. math::
            fitRange = round(fitRangeScale\cdot FWHMEstimate)

        .. math::
            fitData = data[peak - fitRange: peak + fitRange + 1]

    constantPeakWidth: int, set by __init__, default=50
        Extra number of data points that should be added to the FWHMEstimate when estimating the peak width.
        This is used for some background extraction methods.
        Namely, smooth_spline and all local methods. For smooth_spline it is used to vary the peak exclusion range.
        The area [peak +/- (fwhmEstimate + constantPeakWidth)] around the peak that should not be smoothed.

        For the local methods, it provides the peak exclusion range.
        Meaning that in the area [peak +/- (fwhmEstimate + constantPeakWidth)] no data points are used to calculate
        the local mean. Furthermore, the other boundary is calculated by scaling constantPeakWidth.
        The scaling parameter is currently set to 3.
        In the future this may be altered to gain better control of the local mean method.
        Therefore a different parameter may be introduced.

        Moreover, this parameter defines the window length for prominence and width calculation of the peak.
        For more information on this look up the documentation for self.get_peak().

    backgroundFitMode: str, set by __init__, default="local_left"
        Selects the way the background of the data is handled. My recommendation is to use one of the local methods.
        These seems to produce the most consistent results for the S-shapes.

        When the background is lower on the left side of the peak than on the right side of the peak,
        then I tend to use local_left instead of local_all. This ended up producing better results for me.

        Usually, I use local_left for everything, because if one of my data sets is skewed than the left side
        was pretty much always lower than the right side. If the data set is not skewed it also does not make
        visible (in the S-shape) difference whether local_all, local_right or local_left is used.

        local_all:
            Subtracts the local mean from the data set and then performs the fitting.
            In this particular method, the mean is calculated for both sides of the peak.
            The area close to the peak is excluded in this calculation. The size of this area can be varied
            by changing the constantPeakWidth parameter.
            On the "right" side the area used for the mean calculation looks like this.

            .. math::
                localData = [peak + fwhmEstimate + constantPeakWidth :

            .. math::
                peak + fwhmEstimate + 3\cdot constantPeakWidth]

            The left side looks similar (- instead of +). The scaling parameter 3 is currently hard coded into the routine
            and cannot be varied in runtime.

        local_left:
            Same as local_all, but only the mean of the "right" side (+) is used.

        local_right:
            Same as local_all, but only the mean of the "left" side (-) is used.

        spline:
            Uses spline interpolation to flatten the background, the area around the peak is excluded.
            The size of the area can be varied by changing constantPeakWidth.

            This is the heaviest transformation (most expensive and biggest change on the data set).
            The form of the S-shape (especially the tails) is dependent on the peak exclusion range.

        constant:
            Subtracts the median of the complete data set (intensities) from the data set.

        offset:
            The background is not changed, but an offset is used in the fitting process (f(x, \*args) + b)

        none/disable:
            The background is not changed and no offset is used

    minWavelength: float, set by __init__

    maxWavelength: float, set by __init__

    muFit: float, set by fit_peak()
        mean wavelength obtained from the fit

    fwhmFit: float, set by fit_peak()
        fwhm of the examined peak, obtained from the fit

    integratedPeak: float, set by fit_peak()
        integration of peak intensity curve fit over all wavelengths, used as output power

    uncintegratedPeak: float, set by fit_peak()
        uncertainty of integrated peak, obtained from the fit

    uncFwhmFit: float, set by fit_peak()
        uncertainty of fwhmFit, obtained from the fit

    uncMuFit: float, set by fit_peak()
        uncertainty of muFit, obtained from the fit

    peak: int, set by get_peak()
        index of the estimated peak position (corresponding wavelength: wavelengths[peak])
        used as initial guess for the fit

    peakHeightEstimate: float, set by get_peak()
        Estimate of the absolute peak height
        used for the initial guess for the fit

    fwhmEstimate: float, set by get_peak()
        Estimate of the fwhm of the peak using scipy.signal.peak_widths()
        used as initial guess for the fit

    p: np.ndarray, set by fit_peak()
        Return value of the fit parameters from scipy.optimize.curve_fit()

    cov: np.ndarray, set by fit_peak()
        Return value of the covariance of the fit parameters from scipy.optimize.curve_fit()

    spline: scipy.interpolate.UnivariateSpline, set by remove_background_spline()
        spline object, used to visualize the spline fit
    """
    smoothGauss: number = 7
    """float/int: default sigma for gaussian filter used in self.remove_background_smoothspline()
    """
    smoothSpline: number = 0
    """float/int: default smoothing factor for the spline used in self.remove_background_smoothspline()
    """

    def __init__(self, wavelengths: np.ndarray, intensity: np.ndarray, initRange: tupleIntOrNone=None,
                intCoverage: number=1, fitRangeScale: number=5, constantPeakWidth: int=50,
                backgroundFitMode: str="local_left") -> None:
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
        self.intensity: np.ndarray = intensity
        self.originalIntensity: np.ndarray = copy.deepcopy(self.intensity)
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
        """
        Main method, performs the peak finding, background handling and fitting using
        the remove_background methods, self.get_peak() and self.fit_peak().

        Should handle all possible errors (peak not found, peak could not be fitted).
        In these cases the original data should be shown and the results should equal np.nan.

        Upon finishing successful, the output properties are set (output power, fwhm, mode wavelength).
        """
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
                self.intensity = self.originalIntensity - np.median(self.originalIntensity)
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
                    self.set_fwhm() # type: ignore
                    assert hasattr(self, "fwhmFit"), "set_fwhm() method did not set fhwmFit"
                    assert hasattr(self, "uncFwhmFit"), "set_fwhm() method did not set uncFwhmFit"
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
        """
        Finds the most prominant peak using scipy.signal.find_peaks()
        and sorting the results based on their prominence (calculated by scipy.signal.peak_prominences()).
        Prominance means the height of the peak
        relative to the lowest point inside a window around the peak.
        Moreover, the peak height and width are estimated.
        The peak height corresponds to the intensity value
        of the peak position. The width is estimated using scipy.signal.peak_widths().
        The wavelength/energy range in which the peak should lie can be selected using
        the initialRange parameter.

        The estimations of the peak position, height and width are used as
        initial parameters for the fitting process.

        In run the parameters are set to their corresponding class attributes
        (initialRange=self.initRange; peakWidth=self.constantPeakWidth)

        Parameters
        ----------
        initialRange: tuple/None, default=None
            When set to None, no initial range is used
            Otherwise the program looks only for peaks in between [min(initRange), max(initRange)]

        peakWidth: int, default=50
            Determines the window length (wlen=2\*peakWidth) of scipy.signal.peak_prominences() and scipy.signal.peak_widths().
            Thereby the range of data points for both methods is limited. Currently, this parameter is hard coded and cannot be
            changed in runtime.
        """
        ## setting an initial range, where the algorithm looks for the peak
        ## looking for all peaks
        logger.debug("Calling PeakFitSuper.get_peak()")
        assert isinstance(initialRange, (tuple, type(None)))
        peaks: np.ndarray
        peaks, _ = signal.find_peaks(self.intensity)
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
        prominences: np.ndarray = signal.peak_prominences(self.intensity, peaks, wlen=2*peakWidth)[0]
        if len(prominences) == 0:
            raise RuntimeError("No peak could be found")
        self.peak: int = peaks[np.argmax(prominences)]
        self.peakHeightEstimate: number = self.intensity[self.peak]
        if self.peakHeightEstimate <= 0:
            raise ValueError(f"The estimated height of peak ({self.peakHeightEstimate}) is below 0")
        self.fwhmEstimate: number = signal.peak_widths(self.intensity, np.array([self.peak]), rel_height=0.5, wlen=2*peakWidth)[0][0]
        logger.debug(f"Found peak at: {self.wavelengths[self.peak]} nm/eV, index: [{self.peak}]")
        logger.debug(f"Estimate of FWHM: {np.abs(self.wavelengths[1] - self.wavelengths[0])*self.fwhmEstimate} nm/eV, index length: [{self.fwhmEstimate}]")
        logger.debug(f"Estimate of the Height: {self.peakHeightEstimate} a.u.")

    def remove_background_smoothspline(self, smoothSpline: number=0, smoothGauss: number=7, peakWidth: int=50) -> None:
        """
        Fits the background by first smoothing the data set using a gaussian filter (scipy.ndimage.gaussian_filter1d())
        and then performing a spline interpolation (scipy.interpolate.UnivariateSpline()). In this process the area around the peak is excluded.
        Personally, I achieved the best results by only using the gaussian filter and not the smoothing spline factor.

        In run these parameters are set to their corresponding class attributes
        (peakWidth=self.constantPeakWidth; smoothSpline=self.smoothSpline, smoothGauss=self.smoothGauss

        Parameters
        ----------
        smoothGauss: int/float, default=7
            sigma for the gaussian filter, (scipy.ndimage.gaussian_filter1d()).
            Larger values for sigma smoothes the data set more.

        smoothSpline: int/float, default=0
            smoothing factor for the Spline (spl.set_smoothing_factor(smoothSpline)).
            Smoothes the spline interpolation itself.

        peakWidth: int, default=50
            Defines the peak exclusion range.
            The area [peak +/- (fwhmEstimate + constantPeakWidth)] around the peak that should not be smoothed.

        """
        logger.debug(f"Calling remove_background_smoothspline(); smoothspline: {smoothSpline}, smoothGauss: {smoothGauss}, peakWidth: {peakWidth}")
        try:
            ## exclude peak area
            dataBeforePeak: np.ndarray = self.intensity[: self.peak - peakWidth - round(self.fwhmEstimate)] # type: ignore
            dataAfterPeak: np.ndarray = self.intensity[self.peak + peakWidth + round(self.fwhmEstimate) + 1 :] # type: ignore
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
            self.intensity = self.intensity - self.spline(self.wavelengths)

    def remove_background_local_mean(self, peakWidth: int=50, peakWidthScale: number=3, mode: str="all") -> None:
        """Removes the background by subtracting the local mean from the data set.

        Parameters
        ----------
        peakWidth: int, default=50
            Defines the boundaries of the subset that is used to calculate the local mean.
            The area near the peak is excluded.

            .. math::
                    localDataRight = [peak + fwhmEstimate + constantPeakWidth :

            .. math::
                    peak + fwhmEstimate + 3\cdot constantPeakWidth]

            .. math::
                    localDataLeft = [peak - fwhmEstimate - constantPeakWidth :

            .. math::
                    peak - fwhmEstimate - 3\cdot constantPeakWidth]

            In run() this parameter is set to self.constantPeakWidth.
        peakWidthScale: int, default=3
            Scales peakWidth to define the outer boundary. Currently, this parameter is hard coded and cannot be changed in runtime.

        mode: str, default="all"
            Selects which method should be used ("all", "local_left" or "local_right").
            The differences between these modes are explained in the options section.

            local_all:
                Here the mean is calculated for both sides of the peak.
                The area close to the peak is excluded in this calculation.

            local_left:
                Same as local_all, but only the mean of the "right" side (+) is used.

            local_right:
                Same as local_all, but only the mean of the "left" side (-) is used.
        """
        logger.debug(f"Calling remove_background_local_mean(); peakWidth: {peakWidth}, peakWidthScale: {peakWidthScale}")
        assert mode in ["left", "right", "all"]
        try:
            dataBeforePeak: np.ndarray = self.intensity[self.peak - peakWidthScale*peakWidth - round(self.fwhmEstimate) : self.peak - peakWidth - round(self.fwhmEstimate) + 1]  # type: ignore
            dataAfterPeak: np.ndarray = self.intensity[self.peak + peakWidth + round(self.fwhmEstimate) : self.peak + peakWidthScale*peakWidth + round(self.fwhmEstimate) + 1]  # type: ignore
            logger.debug(f"indices before peak: [{self.peak - peakWidthScale*peakWidth - round(self.fwhmEstimate)}, {self.peak - peakWidth - round(self.fwhmEstimate)}]")  # type: ignore
            logger.debug(f"wavelength before peak: [{self.wavelengths[self.peak - peakWidthScale*peakWidth - round(self.fwhmEstimate)]}, {self.wavelengths[self.peak - peakWidth - round(self.fwhmEstimate)]}]")  # type: ignore
            logger.debug(f"wavelength after peak: [{self.wavelengths[self.peak + peakWidth + round(self.fwhmEstimate)]}, {self.wavelengths[self.peak + peakWidthScale*peakWidth + round(self.fwhmEstimate)]}]")  # type: ignore
        except IndexError:
            logger.error("""Data range for peak exclusion exceeded while trying to use local background removal.
Either lower constantPeakWidth or use a different background fitting method. Now the 'none/disable' method will be used for only this spectrum.""")
        else:
            dataAroundPeak: np.ndarray = np.concatenate((dataBeforePeak, dataAfterPeak))
            if mode == "all":
                self.intensity = self.originalIntensity - np.median(dataAroundPeak)
            elif mode == "left":
                self.intensity = self.originalIntensity - np.median(dataBeforePeak)
            elif mode == "right":
                self.intensity = self.originalIntensity - np.median(dataAfterPeak)

    def fit_peak(self, intCoverage: number=1, fitRangeScale: number=1) -> None:
        """
        Fits the selected peak using the fitmodel of the derived class.
        Calulates the final parameters (mode wavelengh, integrated intensity and FWHM).

        The range of the fit and the range of the integration for the intensity can be varied.
        In run() the attributes are used as parameters
        (intCoverage=self.intCoverage, fitRangeScale=self.fitRangeScale)

        Parameters
        ----------
        intCoverage: float/int, default=1
            This parameter should lie in the range [0, 1].
            The resulting fit function is integrated to calculate the integrated intensity,
            which is used as the output power. This parameter has the following role.

            .. math::
                output Power = intCoverage\cdot integrated intensity

            If this parameter is 1, then the integration is performed for the complete range
            of wavelengths (-inf, +inf). For 0, it should always return 0.

        fitRangeScale: int/float, default=1
            For the fit not the complete data set is used. Only a small part around the peak should be used
            to get the best results for the fit. This parameter scales the estimated FWHM to get the fitRange.

            .. math::
                fitRange = round(fitRangeScale\cdot FWHMEstimate)

            .. math::
                fitData = data[peak - fitRange: peak + fitRange + 1]
        """
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
            fitData: np.ndarray = self.intensity[self.peak - fitRange : self.peak + fitRange + 1]
            self.set_p0() # type: ignore
            self.p: np.ndarray
            self.cov: np.ndarray
            if self.backgroundFitMode == "offset":
                self.p0.append(np.median(self.intensity))  # type: ignore
                lowerBounds: list
                higherBounds: list
                lowerBounds, higherBounds = self.paramBounds  # type: ignore
                lowerBounds.append(0)
                higherBounds.append(np.inf)
                paramBounds: tuple[list, list] = (lowerBounds, higherBounds)
                self.p, self.cov = optimize.curve_fit(self, fitWavelengths, fitData, p0=self.p0, bounds=paramBounds)  # type: ignore
            else:
                f: typing.Callable = partial(self, offset=0) # type: ignore
                self.p, self.cov = optimize.curve_fit(f, fitWavelengths, fitData, p0=self.p0, bounds=self.paramBounds)  # type: ignore
                self.p = np.append(self.p, 0)
            logger.debug(f"Fit initial guesses: {self.p0}")  # type: ignore
            logger.debug(f"Fit results: {self.p}")
            self.muFit = self.p[1]
            self.integratedPeak = self.p[0] * intCoverage
            self.uncMuFit = np.sqrt(self.cov[1, 1])
            self.uncintegratedPeak = intCoverage * np.sqrt(self.cov[0, 0])

    def plot_original_data(self, block: bool=True) -> None:
        """Plots the original data (x: energy, y:intensity)

        Parameters
        ----------
        block: bool, default=True
            if False, the image will be shut close after 1 sec
            otherwise it will stay on the screen until manually closed
        """
        plt.plot(self.wavelengths, self.originalIntensity)
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
        """Plots the original data with the spline fit side to side
        with the resulting smoothed data set where the spline fit has been subtracted.

        Parameters
        ----------
        block: bool, default=True
            if False, the image will be shut close after 1 sec
            otherwise it will stay on the screen until manually closed
        """
        assert hasattr(self, "spline")
        fig, axs = plt.subplots(nrows=1, ncols=2)
        ax1, ax2 = axs
        ax1.plot(self.wavelengths, self.originalIntensity)
        ax1.plot(self.wavelengths, self.spline(self.wavelengths))
        ax2.plot(self.wavelengths, self.intensity)
        if block:
            plt.show()
        else:
            plt.show(block=False)
            plt.pause(1)
            plt.close("all")

    def plot_fit(self, block: bool=True) -> None:
        """Plots the fit of the peak side by side with the original data.

        Parameters
        ----------
        block: bool, default=True
            if False, the image will be shut close after 1 sec
            otherwise it will stay on the screen until manually closed
        """
        assert hasattr(self, "p")
        wavelengthsPlotArray: np.ndarray = np.linspace(self.wavelengths[0], self.wavelengths[-1], 1000)
        outputPlotArray: np.ndarray = self(wavelengthsPlotArray, *self.p) # type: ignore
        fig, axs = plt.subplots(nrows=1, ncols=2)
        ax1, ax2 = axs
        ax1.plot(self.wavelengths, self.originalIntensity)
        if hasattr(self, "spline"):
            ax1.plot(self.wavelengths, self.spline(self.wavelengths))
        ax2.plot(self.wavelengths, self.intensity)
        ax2.plot(wavelengthsPlotArray, outputPlotArray)
        if block:
            plt.show()
        else:
            plt.show(block=False)
            plt.pause(1)
            plt.close("all")

    def plot_fitRange_without_fit(self) -> None:
        """Visualizes the fitRange side by side with the original data.
        To show this only get_peak() and the background handling need to be performed but no fitting is necessary.
        Because of that, this image can be shown even when the fitting did not work out.

        Parameters
        ----------
        block: bool, default=True
            if False, the image will be shut close after 1 sec
            otherwise it will stay on the screen until manually closed
        """
        assert hasattr(self, "peak")
        assert hasattr(self, "fwhmEstimate")
        fig, axs = plt.subplots(nrows=1, ncols=2)
        ax1, ax2 = axs
        ax1.plot(self.wavelengths, self.originalIntensity)
        if hasattr(self, "spline"):
            ax1.plot(self.wavelengths, self.spline(self.wavelengths))
        ax2.plot(self.wavelengths, self.intensity)
        fitRange: int = round(self.fitRangeScale * self.fwhmEstimate) # type: ignore
        leftBoundary: number = self.wavelengths[self.peak - fitRange]
        rightBoundary: number = self.wavelengths[self.peak + fitRange]
        ax2.axvline(x=leftBoundary, color="black")
        ax2.axvline(x=rightBoundary, color="black")
        ax1.set_title("Original data")
        ax2.set_title("Fitting range")
        ax2.plot(self.wavelengths[self.peak], self.intensity[self.peak], "+", color="black")
        plt.show()

    def plot_fitRange_with_fit(self) -> None:
        """Visualizes the fitRange side by side with the original data.
        Here the fit is also shown.

        Parameters
        ----------
        block: bool, default=True
            if False, the image will be shut close after 1 sec
            otherwise it will stay on the screen until manually closed
        """
        assert hasattr(self, "peak")
        assert hasattr(self, "fwhmEstimate")
        assert hasattr(self, "p")
        wavelengthsPlotArray: np.ndarray = np.linspace(self.wavelengths[0], self.wavelengths[-1], 1000)
        outputPlotArray: np.ndarray = self(wavelengthsPlotArray, *self.p) # type: ignore
        fig, axs = plt.subplots(nrows=1, ncols=2)
        ax1, ax2 = axs
        ax1.plot(self.wavelengths, self.originalIntensity)
        if hasattr(self, "spline"):
            ax1.plot(self.wavelengths, self.spline(self.wavelengths))
        ax2.plot(self.wavelengths, self.intensity)
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
        ax1.plot(self.wavelengths[self.peak], self.originalIntensity[self.peak], "+", color="black")
        ax2.plot(self.muFit, self(self.muFit, *self.p), "x", color="black") # type: ignore
        plt.show()

    def plot_initRange_without_fit(self) -> None:
        """Visualizes the initial range side by side with the original data.
        To show this only get_peak() and the background handling need to be performed but no fitting is necessary.
        Because of that, this image can be shown even when the fitting did not work out.

        Parameters
        ----------
        block: bool, default=True
            if False, the image will be shut close after 1 sec
            otherwise it will stay on the screen until manually closed
        """
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
            ax1.plot(self.wavelengths, self.originalIntensity)
            ax2.plot(self.wavelengths, self.originalIntensity)
            ax2.axvline(x=boundary1, color="black")
            ax2.axvline(x=boundary2, color="black")
            ax2.plot(self.wavelengths[self.peak], self.originalIntensity[self.peak], "+", color="black")
            fig.suptitle("initial range and peak visualized")
            plt.show()
        else:
            fig, ax = plt.subplots(nrows=1, ncols=1)
            ax.plot(self.wavelengths, self.originalIntensity)
            ax.plot(self.wavelengths[self.peak], self.originalIntensity[self.peak], "+", color="black")
            fig.suptitle("No initial Range selected")
            plt.show()

    def plot_initRange_with_fit(self) -> None:
        """Visualizes the initial range side by side with the original data.
        Here the fit is also shown.

        Parameters
        ----------
        block: bool, default=True
            if False, the image will be shut close after 1 sec
            otherwise it will stay on the screen until manually closed
        """
        assert hasattr(self, "peak")
        assert hasattr(self, "p")
        wavelengthsPlotArray: np.ndarray = np.linspace(self.wavelengths[0], self.wavelengths[-1], 1000)
        outputPlotArray: np.ndarray = self(wavelengthsPlotArray, *self.p) # type: ignore
        if isinstance(self.initialRange, tuple):
            assert len(self.initialRange) == 2
            idx1: int
            idx2: int
            idx1, idx2 = self.initialRange
            boundary1: number = self.wavelengths[idx1]
            boundary2: number = self.wavelengths[idx2]
            fig, axs = plt.subplots(nrows=1, ncols=2)
            ax1, ax2 = axs
            ax1.plot(self.wavelengths, self.originalIntensity)
            ax2.plot(self.wavelengths, self.originalIntensity)
            ax1.axvline(x=boundary1, color="black")
            ax1.axvline(x=boundary2, color="black")
            ax2.axvline(x=boundary1, color="black")
            ax2.axvline(x=boundary2, color="black")
            ax2.plot(wavelengthsPlotArray, outputPlotArray)
            ax1.plot(self.wavelengths[self.peak], self.originalIntensity[self.peak], "+", color="black")
            ax2.plot(self.muFit, self(self.muFit, *self.p), "x", color="black") # type: ignore
            fig.suptitle("initial range and peak visualized")
            plt.show()
        else:
            fig, ax = plt.subplots(nrows=1, ncols=1)
            ax.plot(self.wavelengths, self.originalIntensity)
            ax.plot(wavelengthsPlotArray, outputPlotArray)
            ax.plot(self.wavelengths[self.peak], self.originalIntensity[self.peak], "+", color="black")
            fig.suptitle("No initial Range selected")
            plt.show()

    def plot_fwhm(self) -> None:
        """Plots the FWHM, which was calculated in the fitting process
        and also the estimated FWHM from self.get_peak().
        """
        assert hasattr(self, "peak")
        assert hasattr(self, "p")
        wavelengthsPlotArray: np.ndarray = np.linspace(self.wavelengths[0], self.wavelengths[-1], 1000)
        outputPlotArray: np.ndarray = self(wavelengthsPlotArray, *self.p) # type: ignore
        fig, axs = plt.subplots(nrows=1, ncols=3)
        ax1, ax2, ax3 = axs
        ax1.plot(self.wavelengths, self.intensity)
        ax2.plot(self.wavelengths, self.intensity)
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
        ax1.plot(self.wavelengths[self.peak], self.intensity[self.peak], "+", color="black")
        ax2.plot(self.muFit, self(self.muFit, *self.p), "x", color="black") # type: ignore
        ax1.set_title("fwhm estimate")
        ax2.set_title("fwhm fit")
        ax3.plot(self.wavelengths, self.intensity)
        ax3.axvline(x=fwhmEstimate1, color="black", linestyle="--", label="Estimate")
        ax3.axvline(x=fwhmEstimate2, color="black", linestyle="--")
        ax3.axvline(x=fwhm1, color="black")
        ax3.axvline(x=fwhm2, color="black")
        ax3.plot(wavelengthsPlotArray, outputPlotArray)
        ax3.plot(self.wavelengths[self.peak], self.intensity[self.peak], "+", color="black")
        ax3.plot(self.muFit, self(self.muFit, *self.p), "x", color="black") # type: ignore
        plt.show()

    def plot_fwhmEstimate(self) -> None:
        """Plots only the FWHM estimate
        """
        assert hasattr(self, "peak")
        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.plot(self.wavelengths, self.intensity)
        fwhmEstimate1: number = self.wavelengths[self.peak - round(self.fwhmEstimate/2)] # type: ignore
        fwhmEstimate2: number = self.wavelengths[self.peak + round(self.fwhmEstimate/2)] # type: ignore
        ax.axvline(x=fwhmEstimate1, color="black")
        ax.axvline(x=fwhmEstimate2, color="black")
        ax.plot(self.wavelengths[self.peak], self.intensity[self.peak], "+", color="black")
        ax.set_title("fwhm estimate")
        plt.show()

    def plot_exclusionRange(self) -> None:
        """Visualizes the peak exclusion range used for the spline
        background fitting.
        """
        assert hasattr(self, "spline")
        fig, axs = plt.subplots(nrows=1, ncols=2)
        ax1, ax2 = axs
        ax1.plot(self.wavelengths, self.originalIntensity)
        ax1.plot(self.wavelengths, self.spline(self.wavelengths))
        ax2.plot(self.wavelengths, self.intensity)
        leftBoundary: number = self.wavelengths[self.peak - self.constantPeakWidth - round(self.fwhmEstimate)] # type: ignore
        rightBoundary: number = self.wavelengths[self.peak + self.constantPeakWidth + round(self.fwhmEstimate)] # type: ignore
        ax1.axvline(x=leftBoundary, color="black")
        ax1.axvline(x=rightBoundary, color="black")
        fig.suptitle("Peak exclusion range visualized")
        plt.show()

    @classmethod
    def __init_subclass__(cls, *args, **kwargs) -> None:
        """Makes sure that derived classes satisfy the model dependent required methods,
        which cannot be defined in the abstract base class.

        The derived classes have to have a name attribute, a paramBounds attribute, a __call__ method,
        a set_p0() method and a set_fwhm() method.
        """
        super().__init_subclass__(*args, **kwargs) # type: ignore
        assert hasattr(cls, "set_p0")
        assert hasattr(cls, "set_fwhm")
        assert hasattr(cls, "__call__")
        assert hasattr(cls, "name")
        assert hasattr(cls, "paramBounds")

    @property
    def outputParameters(self):
        """Results of the fitting process. Mode wavelength, FWHM and integrated intensity.
        Are set to np.nan when fitting or peak finding did not work.
        """
        return self.muFit, self.fwhmFit, self.integratedPeak

    @property
    def uncertaintyOutputParameters(self):
        """Uncertainty of the fit parameters (outputParameters)
        """
        return self.uncMuFit, self.uncFwhmFit, self.uncintegratedPeak