import os, sys
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.signal as signal
import matplotlib.pyplot as plt
import pathlib

HeadDir = Path(__file__).resolve().parents[2]
srcDirPath = (HeadDir / "pysrc").resolve()
sys.path.append(str(srcDirPath))

from data_tools.data_formats import DataQlab2
from setup.config_logging import LoggingConfig

loggerObj = LoggingConfig()
logger = loggerObj.init_logger(__name__)

class FSRselector:
    constantPeakWidth = 15

    def __init__(self, filePath, savePath="default", numberOfPeaks=3,
                initialPowerIndex=10, initialMode="single", initialPowerScale=10):
        if savePath != "default":
            assert isinstance(savePath, pathlib.PurePath)
        strFilePath = str(filePath)
        splitLinuxFilePath = strFilePath.split("/")
        splitWindowsFilePath = strFilePath.split("\\")
        if len(splitLinuxFilePath) == 1:
            self.fileName = splitWindowsFilePath[-1]
        elif len(splitWindowsFilePath) == 1:
            self.fileName = splitLinuxFilePath[-1]
        else:
            logger.error("ValueError: The file name could not be extracted.")
            raise ValueError("The file name could not be extracted.")
        self.savePath = savePath
        self.data = DataQlab2(filePath)
        self.wavelengths = self.data.wavelengths
        self.diameter = self.data.diameter
        self.numberOfPeaks = numberOfPeaks
        self.powerIndex = initialPowerIndex
        self.powerScale = initialPowerScale
        self.fsrList = []
        self.set_spectrum_and_power()
        self.get_peaks()
        self.set_mode(initialMode)
        self.run()

    def get_peaks(self):
        peaks, _ = signal.find_peaks(self.spectrum, distance=2*self.constantPeakWidth, rel_height=0.5)
        prominences = signal.peak_prominences(self.spectrum, peaks, wlen=self.constantPeakWidth)[0]
        if prominences.size < self.numberOfPeaks:
            logger.warning("""ValueError: The number of peaks {} exceeds the number of found peaks {}.
Using the max possible value.""".format(self.numberOfPeaks, prominences.size))
            argsortProminences = np.argsort(prominences)
        else:
            argsortProminences = np.argsort(prominences)[- self.numberOfPeaks : ]
        self.peaks = np.sort(peaks[argsortProminences])

    @property
    def powerIndex(self):
        return self._powerIndex

    @powerIndex.setter
    def powerIndex(self, value):
        if (0 <= value <= self.data.lenInputPower - 1):
            self._powerIndex = value
        else:
            logger.warning("""ValueError: {} is not in the valid range: [0, {}].
The powerIndex will not be changed.
The powerIndex is currently set to {}.""".format(value, self.data.lenInputPower - 1, self._powerIndex))

    def set_spectrum_and_power(self):
        self.power = self.data.inputPower[self.powerIndex]
        self.spectrum = self.data[self.powerIndex]

    def _mode_wrapper(func):
        def wrapper(self):
            DeltaLetter = "\u0394"
            lambdaLetter = "\u03BB"
            print()
            print("/"*50)
            print()
            print("current configuration:")
            print("-"*50)
            print(f"file:               {self.fileName}")
            print(f"diameter:           {self.diameter}")
            print(f"mode:               {self.modeName}")
            print(f"number of peaks:    {self.numberOfPeaks}")
            print(f"power [mW]:         {self.power}")
            print(f"power index:        {self.powerIndex}")
            print(f"power scale:        {self.powerScale}")
            print()
            print("/"*50)
            print()
            print("current selected FSR: ")
            print("-"*50)
            roundedFSRList = [round(fsr, 2) for fsr in self.fsrList]
            print(roundedFSRList)
            print()
            print("/"*50)
            print()
            print("peaks:")
            print("-"*50)
            for peak in self.peaks:
                print(round(self.wavelengths[peak], 2))
                print()
            print("/"*50)
            print()
            print("differences between peaks")
            print()
            print(f"[]  |   {lambdaLetter} 1   |   {lambdaLetter} 2   |   {DeltaLetter}{lambdaLetter}")
            print("-"*50)
            func(self)
            plt.plot(self.wavelengths, self.spectrum, label="data")
            plt.plot(self.wavelengths[self.peaks], self.spectrum[self.peaks], "x", label="peaks")
            plt.legend()
            plt.show()
        return wrapper

    @_mode_wrapper
    def mode_all(self):
        counter = 0
        self.deltaLambdaList = []
        for i, peak in enumerate(self.peaks):
            j = 1
            wl1 = self.wavelengths[peak]
            while j < len(self.peaks) - i:
                wl2 = self.wavelengths[self.peaks[j + i]]
                deltaLambda = abs(wl1 - wl2)
                self.deltaLambdaList.append(deltaLambda)
                print(f"[{counter}]   {round(wl1, 2)}     {round(wl2, 2)}     {round(deltaLambda, 2)}")
                print()
                j += 1
                counter += 1

    @_mode_wrapper
    def mode_single(self):
        self.deltaLambdaList = []
        for i, peak in enumerate(self.peaks[:-1]):
            wl1 = self.wavelengths[peak]
            wl2 = self.wavelengths[self.peaks[i + 1]]
            deltaLambda = abs(wl1 - wl2)
            self.deltaLambdaList.append(deltaLambda)
            print(f"[{i}]   {round(wl1, 2)}     {round(wl2, 2)}     {round(deltaLambda, 2)}")
            print()

    @_mode_wrapper
    def mode_double(self):
        counter = 0
        self.deltaLambdaList = []
        for i, peak in enumerate(self.peaks[:-2]):
            wl1 = self.wavelengths[peak]
            for j in range(1, 3, 1):
                wl2 = self.wavelengths[self.peaks[j + i]]
                deltaLambda = abs(wl1 - wl2)
                self.deltaLambdaList.append(deltaLambda)
                print(f"[{counter}]   {round(wl1, 2)}     {round(wl2, 2)}     {round(deltaLambda, 2)}")
                print()
                counter += 1
        wl1 = self.wavelengths[self.peaks[-2]]
        wl2 = self.wavelengths[self.peaks[-1]]
        deltaLambda = abs(wl1 - wl2)
        self.deltaLambdaList.append(deltaLambda)
        print(f"[{counter}]   {round(wl1, 2)}     {round(wl2, 2)}     {round(deltaLambda, 2)}")
        print()

    def set_mode(self, mode):
        if mode == "single":
            self.modeName = "single"
            self._mode = self.mode_single
        elif mode == "double":
            self.modeName = "double"
            self._mode = self.mode_double
        elif mode == "all":
            self.modeName = "all"
            self._mode = self.mode_all
        else:
            logger.error(f"ValueError: mode {mode} not implemented. Setting mode to single")
            print("Implemented modes: single, double, all")
            self.modeName = "single"
            self._mode = self.mode_single

    @property
    def powerScale(self):
        return self._powerScale

    @powerScale.setter
    def powerScale(self, value):
        if (1 <= value <= self.data.lenInputPower - 1):
            self._powerScale = value
        else:
            logger.warning("""ValueError: {} is not in the valid range: [1, {}].
The value for powerScale will not be changed.
Currently, powerScale is set to {}.""".format(value, self.data.lenInputPower - 1, self._powerScale))

    def change_power(self, mode="+"):
        if mode == "+":
            if self.powerIndex + self.powerScale >= self.data.lenInputPower:
                logger.warning(f"ValueError: index is out of bounce, maximum is selected [{self.data.lenInputPower - 1}]")
                self.powerIndex = self.data.lenInputPower - 1
            else:
                self.powerIndex = self.powerIndex + self.powerScale
        else:
            if self.powerIndex - self.powerScale < 0:
                logger.warning(f"ValueError: index is out of bounce, minimum is selected [{0}]")
                self.powerIndex = 0
            else:
                self.powerIndex = self.powerIndex - self.powerScale
        self.set_spectrum_and_power()
        self.get_peaks()

    def select_fsr(self, number):
        if (0 <= number < len(self.deltaLambdaList)):
            fsr = self.deltaLambdaList[number]
            self.fsrList.append(fsr)
        else:
            logger.warning(f"ValueError: index {number} is out of range [0, {len(self.deltaLambdaList) - 1}]")

    def save_fsr(self):
        if self.savePath == "default":
            headPath = Path(__file__).parents[1]
            fileName = "fsr_test.csv"
            self.savePath = ( headPath / "output" / fileName).resolve()
        self.savePath.touch(exist_ok=True)
        if len(self.fsrList) == 0:
            logger.warning("ValueError: no fsr selected, nothing will be saved")
        diameterFSRlist = [self.diameter, *self.fsrList]
        if os.path.getsize(str(self.savePath)) == 0:
            df = pd.DataFrame({self.fileName: diameterFSRlist})
        else:
            df = pd.read_csv(self.savePath, sep="\t")
            lenDf = len(df.index)
            if lenDf > len(diameterFSRlist):
                NaNlist = [np.nan for _ in range(lenDf - len(diameterFSRlist))]
                diameterFSRlist = [*diameterFSRlist, *NaNlist]
                print(diameterFSRlist)
            elif len(diameterFSRlist) > lenDf:
                NaNlist = [np.nan for _ in range(len(diameterFSRlist) - lenDf)]
                tmpDf = pd.DataFrame(columns=df.columns)
                for col in df.columns:
                    tmpDf[col] = NaNlist
                df = pd.concat([df, tmpDf])

            if self.fileName in df.columns:
                print(f"{self.savePath} already contains data from this file.")
                write = input("What to you want to do? Overwrite data, append data or do nothing?       [o/a/n]: ")
                if write == "o":
                    df[self.fileName] = diameterFSRlist
                elif write == "a":
                    if df[self.fileName].isnull().values.any():
                        startNaN = df.loc[pd.isna(df[self.fileName]), :].index[0]
                    else:
                        startNaN = lenDf
                    for i, fsr in enumerate(self.fsrList):
                        df.loc[startNaN + i, self.fileName] = fsr
                elif write == "n":
                    pass
                else:
                    logger.warning("""ValueError: '{}' is not a valid input.
                                        'o', 'a' or 'n' are the only valid inputs.""".format(write))
            else:
                df[self.fileName] = diameterFSRlist
        df.to_csv(self.savePath, sep="\t", header=True, index=False, na_rep="NaN")

    @staticmethod
    def help():
        print()
        print("q: FSRselector")
        print()
        print("exit: end program (especially loops of FSRselector)")
        print()
        print("just hit Enter: run program")
        print()
        print("+: change power index by +powerscale (default: powerscale=10)")
        print()
        print("-: change power index by -powerscale (default: powerscale=10)")
        print()
        print("set p scale: changes powerscale, used when changing power with '+' or '-'")
        print()
        print("set p idx: set the power index manually")
        print()
        print("switch mode: switches between modes 'single', 'double' and 'all'")
        print()
        print("#p: changes the number of peaks; syntax: #p 10 or #p +/-")
        print()
        print("save: save selected fsr list to the provided file path")
        print()
        print("clear fsr: deletes fsr list")
        print()
        print("any integer: selects the index to add to the fsr list, requires program to run atleast once; syntax: 3")
        print()



    def input_decoder(self):
        print()
        print()
        print("/"*100)
        print()
        print()
        case = input("enter instruction (type help for list of instructions): ")
        if case == "q":
            return 0
        elif case == "exit":
            return -1
        elif case == "":
            self._mode()
            return 1
        elif case == "+":
            self.change_power("+")
            self._mode()
            return 1
        elif case == "-":
            self.change_power("-")
            self._mode()
            return 1
        elif case == "set p scale":
            strPowerScale = input("enter power scale: ")
            try:
                int(strPowerScale)
            except ValueError:
                logger.error(f"ValueError: {strPowerScale} is an invalid argument.")
            else:
                self.powerScale = int(strPowerScale)
            finally:
                return 1
        elif case == "set p idx":
            strPowerIndex = input("enter power index: ")
            try:
                int(strPowerIndex)
            except ValueError:
                logger.error(f"ValueError: {strPowerIndex} is an invalid argument.")
            else:
                self.powerIndex = int(strPowerIndex)
                self.set_spectrum_and_power()
                self._mode()
            finally:
                return 1
        elif case == "switch mode":
            mode = input("enter mode: ")
            self.set_mode(mode)
            self._mode()
            return 1
        elif "#p " in case:
            numberIndex = case.find("#p ") + len("#p ")
            truncatedCase = case[numberIndex : ]
            try:
                self.numberOfPeaks = int(truncatedCase)
            except ValueError:
                if truncatedCase == "+":
                    self.numberOfPeaks += 1
                elif truncatedCase == "-":
                    self.numberOfPeaks -= 1
                else:
                    logger.error("ValueError: invalid format encountered for #p")
                    print("valid format: #p 10")
            else:
                self.get_peaks()
                self._mode()
            finally:
                return 1
        elif case == "save":
            self.save_fsr()
            return 1
        elif case == "clear fsr":
            self.fsrList = []
            return 1
        elif case == "help":
            self.help()
            return 1
        else:
            try:
                int(case)
            except ValueError:
                logger.error("ValueError: Not implemented")
            else:
                indexFSR = int(case)
                self.select_fsr(indexFSR)
            finally:
                return 1

    def run(self):
        j = 1
        while j == 1:
            j = self.input_decoder()
        if j < 0:
            exit()







if __name__ == "__main__":

    diameter = 4.5
    strDiameter = str(diameter).replace(".", "-")
    fileName = "20210301_NP7509_Ni_4-5µm_20K_Powerserie_1-001s_deteOD3AllSpectra.dat"
    dataPath = (HeadDir / "sorted_data" / strDiameter / "full_spectra" ).resolve()

    # test = FSRselector(dataPath)

    files = dataPath.glob("*")
    for file in files:
        test = FSRselector(file)
    # constantPeakWidth = 10

    # def get_peaks(data, numberOfPeaks=3):
    #     peaks, _ = signal.find_peaks(data, distance=2*constantPeakWidth, rel_height=0.5)
    #     prominences = signal.peak_prominences(data, peaks, wlen=constantPeakWidth)[0]
    #     argsortProminences = np.argsort(prominences)[- (numberOfPeaks + 1) : ]
    #     return peaks[argsortProminences]

    # def choose_power(data, number=0):
    #     spec = data.columns[number]
    #     return data[spec].to_numpy()

    # def show_possible_fsr(peaks, wavelengths, intensity):
    #     DeltaLetter = "\u0394"
    #     lambdaLetter = "\u03BB"
    #     print()
    #     print(f"[]  |   {lambdaLetter} 1   |   {lambdaLetter} 2   |   {DeltaLetter}{lambdaLetter}")
    #     print("-"*35)
    #     counter = 0
    #     for i, peak in enumerate(peaks):
    #         j = 1
    #         while j < len(peaks) - i:
    #             wl1 = wavelengths[peak]
    #             wl2 = wavelengths[peaks[j + i]]
    #             print(f"[{counter}]   {round(wl1, 2)}     {round(wl2, 2)}     {round(abs(wl1 - wl2), 2)}")
    #             print()
    #             j += 1
    #             counter += 1
    #     plt.plot(wavelengths, intensity)
    #     plt.show()

    # def input_selector(data, wavelengths, number=0, powerAdd=10):
    #     if not (0 <= number <= len(data)):
    #         print("number not in range of data.")
    #         return 1
    #     case = input("instruction: ")

    #     if case == ("q" or "n"):
    #         return 0
    #     elif case == "exit":
    #         return -1
    #     elif case == "":
    #         intensity = choose_power(data, number=number)
    #         peaks = get_peaks(intensity)
    #         show_possible_fsr(peaks, wavelengths, intensity)
    #         return 1

    #     elif case == "+":
    #         number += powerAdd
    #         if not (0 <= number <= len(data)):
    #             print("number not in range of data.")
    #             return 1
    #         intensity = choose_power(data, number=number)
    #         peaks = get_peaks(intensity)
    #         show_possible_fsr(peaks, wavelengths, intensity)
    #         return 1
    #     elif case == "-":
    #         number -= powerAdd
    #         if not (0 <= number <= len(data)):
    #             print("number not in range of data.")
    # ## set automatically to 0 or highest number in this case
    #             return 1
    #         intensity = choose_power(data, number=number)
    #         peaks = get_peaks(intensity)
    #         show_possible_fsr(peaks, wavelengths, intensity)
    #         return 1

    #     else:
    #         print("not implemented")
    #         return 1

    # ## implement methods: set power, switch mode (single, double, all), save distances, change power add number

    # for file in files:
    #     dataObj = Data(file)
    #     data = dataObj.data
    #     wavelengths = dataObj.wavelengths
    #     option = 1
    #     while option == 1:
    #         option = input_selector(data, wavelengths)
    #     if option == 0:
    #         continue
    #     else:
    #         break
