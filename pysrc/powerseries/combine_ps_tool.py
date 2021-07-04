import sys
from pathlib import Path
import matplotlib.pyplot as plt
from configparser import ConfigParser

headDir = Path(__file__).resolve().parents[2]
srcDirPath = (headDir / "pysrc").resolve()
sys.path.append(str(srcDirPath))

from powerseries.ps_tool import PowerSeriesTool
from powerseries.plot_ps import PlotPowerSeries
from setup.config_logging import LoggingConfig
import utils.misc as misc

loggerObj = LoggingConfig()
logger = loggerObj.init_logger(__name__)

class CombinePowerSeriesTool:
    fileDict = {}

    def __init__(self):
        self.ps = None
        logger.debug("CombinePowerSeriesTool object initialized.")
        self.read_powerseries_ini_file()

    def read_powerseries_ini_file(self):
        logger.debug("Calling read_powerseries_ini_file()")
        configIniPath = (headDir / "config" / "powerseries.ini").resolve()
        config = ConfigParser()
        config.read(str(configIniPath))
        self.diameter = misc.diameter_decode(config["combine_ps_tool.py"]["diameter"].replace(" ", ""), returnStr=True)
        self.fineSpectraDirName = config["combine_ps_tool.py"]["fine spectra dir name"].replace(" ", "")
        self.sortedDataDirName = config["combine_ps_tool.py"]["sorted data dir name"].replace(" ", "")

    def set_diameter(self):
        logger.debug("Calling set_diameter()")
        diameterStr = input("diameter: ")
        logger.debug(f"User input for set_diameter(): {diameterStr}")
        self.diameter = misc.diameter_decode(diameterStr, returnStr=True)

    def add_file(self):
        logger.debug("Calling add_file()")
        if self.diameter == None:
            logger.warning("AttributeError: No diameter selected. Select one now.")
            self.set_diameter()
        diameterPath = (headDir / self.sortedDataDirName / self.diameter / self.fineSpectraDirName).resolve()
        logger.debug(f"diameter path {diameterPath}")
        if not diameterPath.exists():
            logger.critical(f"Directory path does not exist {str(diameterPath)}")
            return 0
        diameterFiles = list(diameterPath.glob("*"))
        for i, file in enumerate(diameterFiles):
            fileName = file.name
            print(f"[{i}]   {fileName}")
            print()

        fileIdxStr = input("select file to add: ")
        logger.debug(f"User input for add_file(), select file to add: {fileIdxStr}")
        fileIdx = misc.int_decode(fileIdxStr)
        if fileIdx == None:
            return 0
        if fileIdx >= len(diameterFiles):
            logger.warning(f"ValueError: The selected index [{fileIdx}] exceeds the max value [{len(diameterFiles)}]")
            return 0

        filePath = diameterFiles[fileIdx]
        logger.debug(f"Selected file path {filePath}")
        fileName = filePath.name
        self.fileDict[fileName] = PowerSeriesTool(filePath)

    def del_file(self):
        logger.debug("Calling del_file()")
        if len(self.fileDict.keys()) == 0:
            logger.warning("There is no file that can be deleted.")
            return 0
        for i, file in enumerate(self.fileDict.keys()):
            print(f"[{i}]   {file}")
            print()
        fileIdxStr = input("select file to delete: ")
        logger.debug(f"User input for del_file(), select file to delete: {fileIdxStr}")
        if fileIdxStr == "all":
            self.fileDict.clear()
            return 0
        else:
            fileIdx = misc.int_decode(fileIdxStr)
        if fileIdx == None:
            return 0
        if fileIdx >= len(self.fileDict.keys()):
            logger.warning(f"ValueError: The selected index [{fileIdx}] exceeds the max value [{len(self.fileDict.keys()) - 1}].")
            return 0

        fileName = list(self.fileDict.keys())[fileIdx]
        del self.fileDict[fileName]

    def run_powerseries(self):
        logger.debug("Calling run_powerseries()")
        for i, file in enumerate(self.fileDict.keys()):
            print(f"[{i}]   {file}")
            print()
        fileIdxStr = input("select file to run powerseries: ")
        logger.debug(f"User input for run_powerseries(), select file to run powerseries: {fileIdxStr}")
        if fileIdxStr == "all":
            for file in self.fileDict.keys():
                self.fileDict[file].run()
        else:
            fileIdx = misc.int_decode(fileIdxStr)
            if fileIdx == None:
                return 0
            if fileIdx >= len(self.fileDict.keys()):
                logger.warning(f"ValueError: The selected index [{fileIdx}] exceeds the max value [{len(self.fileDict.keys())}].")
                return 0
            fileName = list(self.fileDict.keys())[fileIdx]
            self.fileDict[fileName].run()

    def init_plot(self):
        logger.debug("Calling init_plot()")
        try:
            self.plots = PlotPowerSeries(list(self.fileDict.values()))
            return True
        except AssertionError:
            logger.error("AttributeError: Cannot initialize 'PlotPowerSeries'. Run powerseries first ('run ps')")
            return False

    def input_plot_selector(self):
        plotStr = input("""plot [S+lw (lws), power(p), linewidth(lw), QFactor(q), modeEnergy(m),
single spectrum (ss), multiple spectra (ms)]: """)
        logger.debug(f"User input for input_plot_selector(): {plotStr}")
        if plotStr.upper() in ["POWER", "P"]:
            self.plots.plot_outputPower()
        elif plotStr.upper() in ["S", "S+LW", "LWS", ""]:
            self.plots.plot_lw_s()
        elif plotStr.upper() in ["LINEWIDTH", "LW"]:
            self.plots.plot_linewidth()
        elif plotStr.upper() in ["QFACTOR", "Q"]:
            self.plots.plot_QFactor()
        elif plotStr.upper() in ["MODEENERGY", "M"]:
            self.plots.plot_mode_wavelength()
        elif plotStr.upper() in ["SINGLE SPECTRUM", "SS"]:
            idxStr = input("Enter index: ")
            logger.debug(f"User input for idxStr: {idxStr}")
            idx = misc.int_decode(idxStr)
            try:
                self.plots.plot_single_spectrum(idx)
            except AssertionError:
                logger.error(f"TypeError: [{idxStr}] is not a valid input.")
        elif plotStr.upper() in ["MULTIPLE SPECTRA", "MS"]:
            numPlotsStr = input("number of plots: ")
            logger.debug(f"User input for numPlotsStr: {numPlotsStr}")
            numPlots = misc.int_decode(numPlotsStr)
            try:
                self.plots.plot_multiple_spectra(numPlots)
            except AssertionError:
                logger.error(f"Invalid input (numbers in the range [1:{self.plots.lenInputPower}] are accepted).")
        else:
            logger.error(f"ValueError: {plotStr} is not a valid input.")

    def scale_outputPower(self):
        logger.debug("Calling scale_outputPower()")
        # if len(list(self.fileDict.keys())) < 2:
        #     logger.warning("ValueError: Less than 2 files selected. No need to rescale data. Aborting.")
        #     return 0
        # elif len(list(self.fileDict.keys())) == 2:
        #     scaleStr = input("scale: ")
        #     logger.debug(f"User input for scale_outputPower(), scale: {scaleStr}")
        #     if scaleStr == "q":
        #         return 0
        #     scale = misc.float_decode(scaleStr)
        #     if scale == None:
        #         return 0
        #     fileName = list(self.fileDict.keys())[0]
        #     self.fileDict[fileName].powerScale = scale
        #     return 1
        # else:
        for i, file in enumerate(self.fileDict.keys()):
            print(f"[{i}]   {file}")
            print()
        selectFileStr = input("select file for scaling: ")
        logger.debug(f"User input for selectFileStr, fileIdx: {selectFileStr}")
        if selectFileStr == "q":
            return 0
        fileIdx = misc.int_decode(selectFileStr)
        if fileIdx == None:
            return 0
        elif fileIdx >= len(self.fileDict.keys()):
            logger.warning(f"ValueError: The selected index [{fileIdx}] exceeds the max value [{len(self.fileDict.keys())}].")
            return 0
        fileName = list(self.fileDict.keys())[fileIdx]
        scaleStr = input("scale: ")
        logger.debug(f"User input for scale_outputPower(), scale: {scaleStr}")
        if scaleStr == "q":
            return 0
        scale = misc.float_decode(scaleStr)
        if scale == None:
            return 0
        self.fileDict[fileName].powerScale = scale
        return 1

    def plot_combine(self):
        if len(self.fileDict.keys()) == 0:
            logger.error("No files selected yet, use add for that.")
            return 0
        else:
            for i, file in enumerate(self.fileDict.keys()):
                try:
                    plt.plot(self.fileDict[file].inputPowerPlotArr, self.fileDict[file].outputPowerArr, ".", label=str(i))
                except AttributeError:
                    logger.error("AttributeError: Run powerseries first")
                    plt.close()
                    return 0
            else:
                plt.yscale("log")
                plt.xscale("log")
                plt.legend()
                plt.show()
                return 1

    def scaling_routine(self):
        flag = 1
        while flag == 1:
            flagScalingPossible = self.plot_combine()
            if flagScalingPossible == 1:
                flag = self.scale_outputPower()
            else:
                flag = 0

    def config(self):
        print()
        print("/"*100)
        print()
        print(f"diameter:   {self.diameter}")
        print()
        if len(list(self.fileDict.keys())) == 0:
            print("{}")
        for file in self.fileDict.keys():
            psTmp = self.fileDict[file]
            print(f"file name:      {file}")
            print(f"power scale:    {psTmp._powerScale}")
            if hasattr(psTmp, "outputPowerArr"):
                runCompFlag = u'\u2713'
            else:
                runCompFlag = u"\u2717"
            print(f"run completed:  {runCompFlag}")
            print()
            print("-"*100)
            print()
        print()
        print("/"*100)
        print()

    def input_decoder(self):
        print()
        case = input("enter instruction (type help for more information): ")
        logger.debug(f"User input for input_decoder(): {case}")
        print()
        if case == "q":
            confirmationExit = input("Do you really want to exit the tool? [y/n]: ")
            logger.debug(f"User input for confirmation exit: {confirmationExit}")
            if confirmationExit.lower().strip() == "y":
                return 0
            elif confirmationExit.lower().strip() == "n":
                return 1
            else:
                logger.error(f"Invalid input {confirmationExit} only 'y' and 'n' are accepted. Not closing the application.")
                return 1
        elif case == "exit":
            exit()
        elif case == "set diameter":
            self.set_diameter()
            return 1
        elif case == "add":
             self.add_file()
             return 1
        elif case == "del":
            self.del_file()
            return 1
        elif case == "config":
            self.config()
            return 1
        elif case == "scale":
            self.scaling_routine()
            return 1
        elif case == "run":
            self.run_powerseries()
            return 1
        elif case == "plot":
            if self.init_plot():
                self.input_plot_selector()
            return 1
        elif case == "beta":
            if self.init_plot():
                try:
                    self.plots.beta_factor_2()
                except RuntimeError:
                    logger.error("Beta fit did not work.")
                else:
                    print()
                    self.input_plot_selector()
            return 1
        else:
            logger.error(f"ValueError: {case} is not implemented. Type help for more information.")
            return 1

    def run(self):
        print()
        print("-"*100)
        print("Running CombinePowerSeriesTool")
        print("-"*100)
        print()
        self.config()
        j = self.input_decoder()
        while j == 1:
            j = self.input_decoder()


if __name__ == "__main__":
    test = CombinePowerSeriesTool()
    test.run()