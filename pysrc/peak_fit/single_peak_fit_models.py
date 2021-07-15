import numpy as np
import scipy.special as special

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from peak_fit.single_peak_fit_base import PeakFitSuper

##############################################################################################################################################################

class GaussianPeakFit(PeakFitSuper):
    name = "Gauss"

    def __init__(self, wavelengths, spec, initRange=None, intCoverage=1, fitRangeScale=1,
constantPeakWidth=50, backgroundFitMode="constant"):

        super().__init__(wavelengths, spec, initRange=initRange, intCoverage=intCoverage, fitRangeScale=fitRangeScale,
constantPeakWidth=constantPeakWidth, backgroundFitMode=backgroundFitMode)

    @staticmethod
    def __call__(x, A, mu, sigma, offset):
        return A*np.exp(-(x - mu)**2 / 2 / sigma**2) / np.sqrt(2*np.pi) / sigma + offset

    def set_p0(self):
        muEstimate = self.wavelengths[self.peak]
        sigmaEstimate = self.fwhmEstimate / 2 / np.sqrt(2*np.log(2)) * np.abs(self.wavelengths[1] - self.wavelengths[0])
        AEstimate = self.peakHeightEstimate*np.sqrt(2*np.pi)*sigmaEstimate
        self.p0 = [AEstimate, muEstimate, sigmaEstimate]

    def set_fwhm(self):
        assert hasattr(self, "p")
        assert hasattr(self, "cov")
        sigmaFit = np.abs(self.p[2])
        self.fwhmFit = 2*sigmaFit*np.sqrt(2*np.log(2))
        self.uncFwhmFit = 2*np.sqrt(2*np.log(2))*np.sqrt(self.cov[2, 2])

    @property
    def paramBounds(self):
        lowerBounds = [0, 0, 0]
        upperBounds = [np.inf, np.inf, np.inf]
        return (lowerBounds, upperBounds)

##############################################################################################################################################################

class LorentzPeakFit(PeakFitSuper):
    name = "Lorentz"

    def __init__(self, wavelengths, spec, initRange=None, intCoverage=1, fitRangeScale=1,
constantPeakWidth=50, backgroundFitMode="constant"):

        super().__init__(wavelengths, spec, initRange=initRange, intCoverage=intCoverage, fitRangeScale=fitRangeScale,
constantPeakWidth=constantPeakWidth, backgroundFitMode=backgroundFitMode)

    @staticmethod
    def __call__(w, A, w0, gamma, offset):
        return A * gamma / np.pi / ((w - w0)**2 + (gamma)**2) + offset

    def set_p0(self):
        muEstimate = self.wavelengths[self.peak]
        sigmaEstimate = self.fwhmEstimate * np.abs(self.wavelengths[1] - self.wavelengths[0]) / 2
        AEstimate = self.peakHeightEstimate * np.pi * sigmaEstimate
# gammaEstimate = fwhm*np.sqrt(1 - (fwhm / 2 / w0Estimate)**2)
        self.p0 = [AEstimate, muEstimate, sigmaEstimate]

    def set_fwhm(self):
        assert hasattr(self, "p")
        assert hasattr(self, "cov")
        self.fwhmFit = 2*np.abs(self.p[2])
        self.uncFwhmFit = 2 * np.sqrt(self.cov[2, 2])

    @property
    def paramBounds(self):
        lowerBounds = [0, self.minWavelength, 0]
        higherBounds = [np.inf, self.maxWavelength, np.inf]
        return (lowerBounds, higherBounds)
##############################################################################################################################################################

class VoigtPeakFit(PeakFitSuper):
    name = "Voigt"

    def __init__(self, wavelengths, spec, initRange=None, intCoverage=1, fitRangeScale=1,
constantPeakWidth=50, backgroundFitMode="constant"):

        super().__init__(wavelengths, spec, initRange=initRange, intCoverage=intCoverage, fitRangeScale=fitRangeScale,
constantPeakWidth=constantPeakWidth, backgroundFitMode=backgroundFitMode)

    @staticmethod
    def __call__(x, A, mu, sigma, gamma, offset):
        z = (x + 1j*gamma - mu) / sigma / np.sqrt(2)
        return A*special.wofz(z).real / sigma / np.sqrt(2*np.pi) + offset

    def set_p0(self):
        muEstimate = self.wavelengths[self.peak]
        fwhm = self.fwhmEstimate * np.abs(self.wavelengths[1] - self.wavelengths[0])
        sigmaEstimate = fwhm / 2 / np.sqrt(2*np.log(2))
        gammaEstimate = fwhm / 2
        AEstimate = self.peakHeightEstimate * (1 / special.erfc(gammaEstimate / sigmaEstimate / np.sqrt(2)) 
                    * np.exp(gammaEstimate**2 / 2 / sigmaEstimate**2)) * np.sqrt(2*np.pi)*sigmaEstimate
## gammaEstimate = fwhm*np.sqrt(1 - (fwhm / 2 / muEstimate)**2)
        self.p0 = [AEstimate, muEstimate, sigmaEstimate, gammaEstimate]

    def set_fwhm(self):
## uncGauss, uncLorentz are correlated -> use covariance matrix
        assert hasattr(self, "p")
        fwhmGauss = 2*np.abs(self.p[2])*np.sqrt(2*np.log(2))
        uncFwhmGauss = 2*np.sqrt(2*np.log(2))*np.sqrt(self.cov[2, 2])
        fwhmLorentz = 2*np.abs(self.p[3])
        uncFwhmLorentz = 2*np.sqrt(self.cov[3, 3])
        self.fwhmFit = 0.5346*fwhmLorentz + np.sqrt(0.2166*fwhmLorentz**2 + fwhmGauss**2)
        self.uncFwhmFit = np.sqrt(((0.5346 + 0.2166*fwhmLorentz / np.sqrt(0.2166*fwhmLorentz**2 + fwhmGauss**2)) * uncFwhmLorentz)**2
        + (fwhmGauss*uncFwhmGauss / np.sqrt(0.2166*fwhmLorentz**2 + fwhmGauss**2))**2)

    @property
    def paramBounds(self):
        lowerBounds = [0, 0, 0, 0]
        upperBounds = [np.inf, np.inf, np.inf, np.inf]
        return (lowerBounds, upperBounds)

##############################################################################################################################################################

class PseudoVoigtPeakFit(PeakFitSuper):
    name = "Pseudo Voigt"

    def __init__(self, wavelengths, spec, initRange=None, intCoverage=1, fitRangeScale=1,
constantPeakWidth=50, backgroundFitMode="constant"):

        super().__init__(wavelengths, spec, initRange=initRange, intCoverage=intCoverage, fitRangeScale=fitRangeScale,
constantPeakWidth=constantPeakWidth, backgroundFitMode=backgroundFitMode)

    @staticmethod
    def __call__(x, A, mu, sigma, alpha, offset):
        sigmaGauss = sigma / np.sqrt(2* np.log(2))
        return (1 - alpha)*A * np.exp(-(x - mu)**2 / 2 / sigmaGauss**2) / sigmaGauss / np.sqrt(2*np.pi) + alpha*A*sigma / np.pi / ((x - mu)**2 + sigma**2) + offset

    def set_p0(self):
        muEstimate = self.wavelengths[self.peak]
        fwhm = self.fwhmEstimate * np.abs(self.wavelengths[1] - self.wavelengths[0])
        sigmaEstimate = fwhm / 2
        sigmaGaussEstimate = sigmaEstimate / np.sqrt(2*np.log(2))
        alphaEstimate = 0.5
        AEstimate = self.peakHeightEstimate / ((1 - alphaEstimate) / sigmaGaussEstimate / np.sqrt(2*np.pi) + alphaEstimate / np.pi / sigmaEstimate )
# gammaEstimate = fwhm*np.sqrt(1 - (fwhm / 2 / w0Estimate)**2)
        self.p0 = [AEstimate, muEstimate, sigmaEstimate, alphaEstimate]

    def set_fwhm(self):
## uncGauss, uncLorentz are correlated -> use covariance matrix
        assert hasattr(self, "p")
        fwhmLorentz = 2*np.abs(self.p[2])
        uncLorentz = 2*np.sqrt(self.cov[2, 2])
        fwhmGauss = fwhmLorentz / np.sqrt(2*np.log(2))
        self.fwhmFit = (fwhmGauss**5 + 2.69269*fwhmGauss**4*fwhmLorentz + 2.42843*fwhmGauss**3*fwhmLorentz**2
                       + 4.47163*fwhmGauss**2*fwhmLorentz**3 + 0.07842*fwhmGauss*fwhmLorentz**4 + fwhmLorentz**5)**(1/5)

    @property
    def paramBounds(self):
        lowerBounds = [0, 0, 0, 0]
        upperBounds = [np.inf, np.inf, np.inf, 1]
        return (lowerBounds, upperBounds)







if __name__ == "__main__":
    import os, sys
    from pathlib import Path
    import pandas as pd

    sys.path.append(str(Path(__file__).parents[2]))
    rowsToSkip = [1, 3]
    dataDirPath = (Path(__file__).parents[2] / "data").resolve()
    fineDataDirPath = (Path(__file__).parents[2] / "fine_data").resolve()
    date = "13022021"
    fileName = "NP7509_Ni_3µm_157-70K_Powerserie_05-001s_deteOD05AllSpectra.dat"
    dataPath = (dataDirPath / date / fileName).resolve()
    data = pd.read_csv(dataPath, sep="\t", skiprows=lambda x: x in rowsToSkip)
    wavelengths = data["Energy"].to_numpy()[1:][::-1]
    data.drop(["Energy", "Wavelength"], axis=1, inplace=True)
    inputPower = data.loc[0].to_numpy()*9
    data.drop([0], axis=0, inplace=True)
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

    for i, spec in zip(list(range(len(data.columns)))[1::14], data.columns[1::14]):
        # print(inputPower[i])
        test = LorentzPeakFit(wavelengths, data[spec].to_numpy()[::-1])
        test.plot_fit()

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
