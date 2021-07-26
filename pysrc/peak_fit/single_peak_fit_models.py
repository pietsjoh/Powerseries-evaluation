"""Contains the fitmodels derived from peak_fit.single_peak_fit_base.PeakFitSuper.
Currently, Gauss, Lorentz, Voigt and Pseudo-Voigt models are available.
"""
import numpy as np
import scipy.special as special # type: ignore
import typing

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from peak_fit.single_peak_fit_base import PeakFitSuper

tupleIntOrNone = typing.Union[tuple[int, int], None]
number = typing.Union[int, float, np.number]
arrayOrNumber = typing.Union[np.ndarray, number]

##############################################################################################################################################################

class GaussianPeakFit(PeakFitSuper):
    """
    Gaussian fitmodel, for more information about the fitting process and the attributes/methods of this class
    take a look at the base class peak_fit.single_peak_fit_base.PeakFitSuper.
    """
    name: str = "Gauss"

    def __init__(self, wavelengths: np.ndarray, spec: np.ndarray, initRange: tupleIntOrNone=None, intCoverage: number=1,
    fitRangeScale: number=1, constantPeakWidth: int=50, backgroundFitMode: str="constant") -> None:

        super().__init__(wavelengths, spec, initRange=initRange, intCoverage=intCoverage, fitRangeScale=fitRangeScale,
constantPeakWidth=constantPeakWidth, backgroundFitMode=backgroundFitMode)

    @staticmethod
    def __call__(x: arrayOrNumber, A: number, mu: number, sigma: number, offset: number) -> arrayOrNumber:
        """Returns the result of the following functional form:

        .. math::
            f(x; A, \mu, \sigma, B) = A\cdot\dfrac{1}{\sqrt{2\pi\sigma^2}}\cdot e^{-\dfrac{(x - \mu)^2}{2\sigma^2}} + B

        For the fitting process the offset is only used when background fit mode is set to "offset" in the
        config/powerseries.ini file.
        """
        return A*np.exp(-(x - mu)**2 / 2 / sigma**2) / np.sqrt(2*np.pi) / sigma + offset

    def set_p0(self) -> None:
        """Sets the initial guesses for the fit parameters based on the results of get_peak().

        \mu:
            wavelength at the estimated peak position

        \sigma:
            :math:`\dfrac{fwhmEstimate}{2\sqrt{2\cdot\log(2)}}`

        A:
            :math:`peakHeightEstimate\cdot\sqrt{2\pi\sigma^2}`
        """
        muEstimate: number = self.wavelengths[self.peak]
        sigmaEstimate: number = self.fwhmEstimate / 2 / np.sqrt(2*np.log(2)) * np.abs(self.wavelengths[1] - self.wavelengths[0])
        AEstimate: number = self.peakHeightEstimate*np.sqrt(2*np.pi)*sigmaEstimate
        self.p0: list[number] = [AEstimate, muEstimate, sigmaEstimate]

    def set_fwhm(self) -> None:
        """Calculates the FWHM based on the fit results. Here the FWHM is calculated from the sigma parameter.
        Moreover, the uncertainty of the FWHM is calculated.

        .. math::
            FWHM = 2\cdot\sigma\cdot\sqrt{2\cdot\log(2)}
        """
        assert hasattr(self, "p")
        assert hasattr(self, "cov")
        sigmaFit: number = np.abs(self.p[2])
        self.fwhmFit: number = 2*sigmaFit*np.sqrt(2*np.log(2))
        self.uncFwhmFit: number = 2*np.sqrt(2*np.log(2))*np.sqrt(self.cov[2, 2])

    @property
    def paramBounds(self) -> tuple[list[number], list[number]]:
        """Sets bounds for the fit parameters. A and sigma are bound by (0, :math:`\infty`).
        mu is bound the (min, max) of the provided wavelengths.
        """
        lowerBounds: list[number] = [0, self.minWavelength, 0]
        upperBounds: list[number] = [np.inf, self.maxWavelength, np.inf]
        return (lowerBounds, upperBounds)

##############################################################################################################################################################

class LorentzPeakFit(PeakFitSuper):
    """
    Lorentzian fitmodel, for more information about the fitting process and the attributes/methods of this class
    take a look at the base class peak_fit.single_peak_fit_base.PeakFitSuper.
    """
    name: str = "Lorentz"

    def __init__(self, wavelengths: np.ndarray, spec: np.ndarray, initRange: tupleIntOrNone=None, intCoverage: number=1,
    fitRangeScale: number=1, constantPeakWidth: int=50, backgroundFitMode: str="constant") -> None:

        super().__init__(wavelengths, spec, initRange=initRange, intCoverage=intCoverage, fitRangeScale=fitRangeScale,
constantPeakWidth=constantPeakWidth, backgroundFitMode=backgroundFitMode)

    @staticmethod
    def __call__(w: arrayOrNumber, A: number, w0: number, gamma: number, offset: number) -> arrayOrNumber:
        """Returns the result of the following functional form:

        .. math::
            f(\omega; A, \omega_0, \gamma, B) = A\cdot\dfrac{\gamma}{\pi}\cdot\dfrac{1}{(\omega - \omega_0)^2 + \gamma^2} + B

        For the fitting process the offset (B) is only used when background fit mode is set to "offset" in the
        config/powerseries.ini file.
        """
        return A * gamma / np.pi / ((w - w0)**2 + (gamma)**2) + offset

    def set_p0(self) -> None:
        """Sets the initial guesses for the fit parameters based on the results of get_peak().

        \mu:
            wavelength at the estimated peak position

        \gamma:
            :math:`\dfrac{fwhmEstimate}{2}`

        A:
            :math:`peakHeightEstimate\cdot\pi\cdot\gamma`
        """
        muEstimate: number = self.wavelengths[self.peak]
        gammaEstimate: number = self.fwhmEstimate * np.abs(self.wavelengths[1] - self.wavelengths[0]) / 2
        AEstimate: number = self.peakHeightEstimate * np.pi * gammaEstimate
# gammaEstimate = fwhm*np.sqrt(1 - (fwhm / 2 / w0Estimate)**2)
        self.p0: list[number] = [AEstimate, muEstimate, gammaEstimate]

    def set_fwhm(self) -> None:
        """Calculates the FWHM based on the fit results. Here the FWHM is calculated from the sigma parameter.
        Moreover, the uncertainty of the FWHM is calculated.

        .. math::
            FWHM = 2\cdot\gamma}
        """
        assert hasattr(self, "p")
        assert hasattr(self, "cov")
        self.fwhmFit: number = 2*np.abs(self.p[2])
        self.uncFwhmFit: number = 2 * np.sqrt(self.cov[2, 2])

    @property
    def paramBounds(self) -> tuple[list[number], list[number]]:
        """Sets bounds for the fit parameters. A and \gamma are bound by (0, :math:`\infty`).
        \mu is bound the (min, max) of the provided wavelengths.
        """
        lowerBounds: list[number] = [0, self.minWavelength, 0]
        higherBounds: list[number] = [np.inf, self.maxWavelength, np.inf]
        return (lowerBounds, higherBounds)
##############################################################################################################################################################

class VoigtPeakFit(PeakFitSuper):
    """
    Voigt fitmodel, for more information about the fitting process and the attributes/methods of this class
    take a look at the base class peak_fit.single_peak_fit_base.PeakFitSuper.
    """
    name: str = "Voigt"

    def __init__(self, wavelengths: np.ndarray, spec: np.ndarray, initRange: tupleIntOrNone=None, intCoverage: number=1,
    fitRangeScale: number=1, constantPeakWidth: int=50, backgroundFitMode: str="constant") -> None:

        super().__init__(wavelengths, spec, initRange=initRange, intCoverage=intCoverage, fitRangeScale=fitRangeScale,
constantPeakWidth=constantPeakWidth, backgroundFitMode=backgroundFitMode)

    @staticmethod
    def __call__(x: arrayOrNumber, A: number, mu: number, sigma: number, gamma: number, offset: number) -> arrayOrNumber:
        """Returns the result of the following functional form:

        .. math::
            z = \dfrac{x + i\gamma - \mu}{\sigma\sqrt{2}}

        .. math::
            f(x; A, \mu, \sigma, \gamma, B) = A\cdot \dfrac{\operatorname{Re}[w(z)]}{\sigma\sqrt{2\pi}} + B

        Here :math:`w(z)` is the Faddeeva function.

        For the fitting process the offset (B) is only used when background fit mode is set to "offset" in the
        config/powerseries.ini file.
        """
        z = (x + 1j*gamma - mu) / sigma / np.sqrt(2)
        return A*special.wofz(z).real / sigma / np.sqrt(2*np.pi) + offset

    def set_p0(self) -> None:
        r"""Sets the initial guesses for the fit parameters based on the results of get_peak().

        \mu:
            wavelength at the estimated peak position

        \gamma:
            :math:`\dfrac{fwhmEstimate}{2}`

        \sigma:
            :math:`\dfrac{fwhmEstimate}{2\sqrt{2\cdot\log(2)}}`

        A:
            :math:`peakHeightEstimate\cdot \dfrac{\sqrt{2\pi\sigma^2}}{erfc\biggl( \dfrac{\gamma}{\sigma\sqrt{2}}\biggr) \cdot e^{\dfrac{\gamma^2}{2\sigma^2}}}`
        """
        muEstimate: number = self.wavelengths[self.peak]
        fwhm: number = self.fwhmEstimate * np.abs(self.wavelengths[1] - self.wavelengths[0])
        sigmaEstimate: number = fwhm / 2 / np.sqrt(2*np.log(2))
        gammaEstimate: number = fwhm / 2
        AEstimate: number = self.peakHeightEstimate * (1 / special.erfc(gammaEstimate / sigmaEstimate / np.sqrt(2)) 
                    * np.exp(gammaEstimate**2 / 2 / sigmaEstimate**2)) * np.sqrt(2*np.pi)*sigmaEstimate
## gammaEstimate = fwhm*np.sqrt(1 - (fwhm / 2 / muEstimate)**2)
        self.p0: list[number] = [AEstimate, muEstimate, sigmaEstimate, gammaEstimate]

    def set_fwhm(self) -> None:
## uncGauss, uncLorentz are correlated -> use covariance matrix
        """Calculates the FWHM based on the fit results. Here the FWHM is calculated from the sigma and the gamma parameter.
        Moreover, the uncertainty of the FWHM is calculated. However, the covariance of the uncertainties is taken into account.
        The formula is taken from wikipedia (`voigt wiki <https://en.wikipedia.org/wiki/Voigt_profile>`_)

        .. math::
            FWHM = 0.5346\cdot fwhm_{L} + \sqrt{0.2166\cdot fwhm_{L}^2 + fwhm_{G}^2}

        Here the following definitions are used.

        .. math::
            fwhm_{L} = 2\gamma

        .. math::
            fwhm_{G} = 2\sigma\sqrt{2\cdot\log(2)}
        """
        assert hasattr(self, "p")
        fwhmGauss: number = 2*np.abs(self.p[2])*np.sqrt(2*np.log(2))
        uncFwhmGauss: number = 2*np.sqrt(2*np.log(2))*np.sqrt(self.cov[2, 2])
        fwhmLorentz: number = 2*np.abs(self.p[3])
        uncFwhmLorentz: number = 2*np.sqrt(self.cov[3, 3])
        self.fwhmFit: number = 0.5346*fwhmLorentz + np.sqrt(0.2166*fwhmLorentz**2 + fwhmGauss**2)
        self.uncFwhmFit: number = np.sqrt(((0.5346 + 0.2166*fwhmLorentz / np.sqrt(0.2166*fwhmLorentz**2 + fwhmGauss**2)) * uncFwhmLorentz)**2
        + (fwhmGauss*uncFwhmGauss / np.sqrt(0.2166*fwhmLorentz**2 + fwhmGauss**2))**2)

    @property
    def paramBounds(self) -> tuple[list[number], list[number]]:
        """Sets bounds for the fit parameters. A, \sigma and \gamma are bound by (0, :math:`\infty`).
        \mu is bound the (min, max) of the provided wavelengths.
        """
        lowerBounds: list[number] = [0, self.minWavelength, 0, 0]
        upperBounds: list[number] = [np.inf, self.maxWavelength, np.inf, np.inf]
        return (lowerBounds, upperBounds)

##############################################################################################################################################################

class PseudoVoigtPeakFit(PeakFitSuper):
    """
    --Deprecated--

    Pseudo-Voigt fitmodel, for more information about the fitting process and the attributes/methods of this class
    take a look at the base class peak_fit.single_peak_fit_base.PeakFitSuper.

    As the uncertainty of the fwhm is not calculated currently, this model cannot be used in combination with
    powerseries.eval_ps.EvalPowerSeries. Hence, it has been deprecated.
    """
    name: str = "Pseudo Voigt"

    def __init__(self, wavelengths: np.ndarray, spec: np.ndarray, initRange: tupleIntOrNone=None, intCoverage: number=1,
    fitRangeScale: number=1, constantPeakWidth: int=50, backgroundFitMode: str="constant") -> None:

        super().__init__(wavelengths, spec, initRange=initRange, intCoverage=intCoverage, fitRangeScale=fitRangeScale,
constantPeakWidth=constantPeakWidth, backgroundFitMode=backgroundFitMode)

    @staticmethod
    def __call__(x: arrayOrNumber, A: number, mu: number, sigma: number, alpha: number, offset: number) -> arrayOrNumber:
        sigmaGauss = sigma / np.sqrt(2* np.log(2))
        return (1 - alpha)*A * np.exp(-(x - mu)**2 / 2 / sigmaGauss**2) / sigmaGauss / np.sqrt(2*np.pi) + alpha*A*sigma / np.pi / ((x - mu)**2 + sigma**2) + offset

    def set_p0(self) -> None:
        muEstimate: number = self.wavelengths[self.peak]
        fwhm: number = self.fwhmEstimate * np.abs(self.wavelengths[1] - self.wavelengths[0])
        sigmaEstimate: number = fwhm / 2
        sigmaGaussEstimate: number = sigmaEstimate / np.sqrt(2*np.log(2))
        alphaEstimate: number = 0.5
        AEstimate: number = self.peakHeightEstimate / ((1 - alphaEstimate) / sigmaGaussEstimate / np.sqrt(2*np.pi) + alphaEstimate / np.pi / sigmaEstimate )
# gammaEstimate = fwhm*np.sqrt(1 - (fwhm / 2 / w0Estimate)**2)
        self.p0: list[number] = [AEstimate, muEstimate, sigmaEstimate, alphaEstimate]

    def set_fwhm(self) -> None:
## uncGauss, uncLorentz are correlated -> use covariance matrix
        assert hasattr(self, "p")
        fwhmLorentz: number = 2*np.abs(self.p[2])
        uncLorentz: number = 2*np.sqrt(self.cov[2, 2])
        fwhmGauss: number = fwhmLorentz / np.sqrt(2*np.log(2))
        self.fwhmFit: number = (fwhmGauss**5 + 2.69269*fwhmGauss**4*fwhmLorentz + 2.42843*fwhmGauss**3*fwhmLorentz**2
                       + 4.47163*fwhmGauss**2*fwhmLorentz**3 + 0.07842*fwhmGauss*fwhmLorentz**4 + fwhmLorentz**5)**(1/5)

    @property
    def paramBounds(self) -> tuple[list[number], list[number]]:
        lowerBounds: list[number] = [0, self.minWavelength, 0, 0]
        upperBounds: list[number] = [np.inf, self.maxWavelength, np.inf, 1]
        return (lowerBounds, upperBounds)

if __name__ == "__main__":
    ## just testing
    pass
    # import os, sys
    # from pathlib import Path
    # import pandas as pd

    # sys.path.append(str(Path(__file__).parents[2]))
    # rowsToSkip = [1, 3]
    # dataDirPath = (Path(__file__).parents[2] / "data").resolve()
    # fineDataDirPath = (Path(__file__).parents[2] / "fine_data").resolve()
    # date = "13022021"
    # fileName = "NP7509_Ni_3µm_157-70K_Powerserie_05-001s_deteOD05AllSpectra.dat"
    # dataPath = (dataDirPath / date / fileName).resolve()
    # data = pd.read_csv(dataPath, sep="\t", skiprows=lambda x: x in rowsToSkip)
    # wavelengths = data["Energy"].to_numpy()[1:][::-1]
    # data.drop(["Energy", "Wavelength"], axis=1, inplace=True)
    # inputPower = data.loc[0].to_numpy()*9
    # data.drop([0], axis=0, inplace=True)
    # if (wavelengths[-1] - wavelengths[0]) <= 20:
    #     if os.path.exists(str((fineDataDirPath / fileName).resolve())):
    #         os.remove(str((fineDataDirPath / fileName).resolve()))
    #     console.cp(str(dataPath), str(fineDataDirPath))
    # print(wavelengths[-1] - wavelengths[0], len(wavelengths))
    # for dir in os.listdir(str(fineDataDirPath)):
    #     dirPath = (fineDataDirPath / dir).resolve()
    #     for file in os.listdir(str(dirPath)):
    #         print(file)
    #         dataPath = (dirPath / file ).resolve()
    #         data = pd.read_csv(dataPath, sep="\t", skiprows=lambda x: x in rowsToSkip)
    #         wavelengths = data["Wavelength"].to_numpy()[1:]
    #         data.drop(["Energy", "Wavelength"], axis=1, inplace=True)
    #         data.drop([0], axis=0, inplace=True)
    #         for spec in data.columns[0::5]:
    #             test = LorentzPeakFit(wavelengths, data[spec].to_numpy(), plot=True, polyOrder=3)
    #     else:
    #         continue

    # listFiles = os.listdir(str(fineDataDirPath))
    # for file in listFiles:
    #     if "AllSpectra.dat" in file and not "fail" in file:
        #     print(file)
        #     dataPath = (fineDataDirPath / file ).resolve()
        #     data = pd.read_csv(dataPath, sep="\t", skiprows=lambda x: x in rowsToSkip)
        #     wavelengths = data["Wavelength"].to_numpy()[1:]
        #     data.drop(["Energy", "Wavelength"], axis=1, inplace=True)
        #     data.drop([0], axis=0, inplace=True)
        #     for spec in data.columns[0::5]:
        #         test = LorentzPeakFit(wavelengths, data[spec].to_numpy(), plot=True, polyOrder=3)
        # else:
        #     continue


    # for spec in data.columns:
    #     print(spec)
    #     test = GaussianPeakFit(wavelengths, data[spec], plot=True, useLMFit=False, autoGuess=False)

    # for i, spec in zip(list(range(len(data.columns)))[1::14], data.columns[1::14]):
    #     # print(inputPower[i])
    #     test = LorentzPeakFit(wavelengths, data[spec].to_numpy()[::-1])
    #     test.plot_fit()

    # arr = np.arange(500, 900, 1)
    # for i, spec in enumerate(data.columns):
    #     test = PseudoVoigtPeakFit(wavelengths, data[spec].to_numpy(), plot=True, polyOrder=3, initRange=None)
        # print(test.outputParameters)

    # AList = []
    # for i, spec in enumerate(data.columns):
    #     print(spec)
    #     test = VoigtPeakFit(wavelengths, data[spec], True, specNumber=i + 1)
    #     print(test.specNumber)
    #     AList.append(test.p[0])
    # BList = AList[:10]
    # def linear(x, a, b, c):
    #     return a*x**2 + b*x + c
    
    # x1 = np.arange(1, 18, 1)
    # x = np.arange(1, 11, 1)
    # print(x)
    # p, _ = optimize.curve_fit(linear, x, BList)

    # print(p)
    # plt.plot(x1, AList, "b.")
    # plt.plot(x, BList, "g.")
    # plt.plot(x, linear(x, *p))
    # plt.show()


    # LorentzPeakFit(wavelengths, data["Spec 2"], True)
    # GaussianPeakFit(wavelengths, data["Spec 1"], True)
    # VoigtPeakFit(wavelengths, data["Spec 1"], True)
