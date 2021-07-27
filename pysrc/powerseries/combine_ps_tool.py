import sys
from pathlib import Path
import matplotlib.pyplot as plt
from configparser import ConfigParser
import typing

headDir = Path(__file__).resolve().parents[2]
srcDirPath = (headDir / "pysrc").resolve()
sys.path.append(str(srcDirPath))

from powerseries.ps_tool import PowerSeriesTool
from powerseries.plot_ps import PlotPowerSeries
from setup.config_logging import LoggingConfig
import utils.misc as misc

loggerObj: LoggingConfig = LoggingConfig()
logger = loggerObj.init_logger(__name__)

class CombinePowerSeriesTool:
    fileDict: dict = {}
    _addFileModeList: list[str] = ["data", "sorted_data"]
    _dataDirPath: Path = (headDir / "data").resolve()

    def __init__(self):
        self.ps = None
        logger.debug("CombinePowerSeriesTool object initialized.")
        self.read_data_format_ini_file()

    def read_data_format_ini_file(self):
        logger.debug("Calling read_powerseries_ini_file()")
        configDataFormatIniPath: Path = (headDir / "config" / "data_format.ini").resolve()
        configPowerseriesIniPath: Path = (headDir / "config" / "powerseries.ini").resolve()
        config: ConfigParser = ConfigParser()
        config.read([str(configDataFormatIniPath), str(configPowerseriesIniPath)], encoding="UTF-8")
        self.useAttribute: bool = LoggingConfig.check_true_false(
            config["data format"]["use attribute"])
        self.distinguishFullFineSpectra: bool = LoggingConfig.check_true_false(
            config["data format"]["distinguish full fine spectra"].replace(" ", ""))
        self.addFileMode: str = config["combine_ps_tool.py"]["add file mode"].replace(" ", "")
        if self.useAttribute:
            self.attrName: str = config["data format"]["attribute name"].replace(" ", "")
            self._sortedDataDirName: str = f"sorted_data_{self.attrName}"
            self._possibleAttrList: typing.Union[list[str], None] = config["data format"]["attribute possibilities"].replace(" ", "").split(",")
            if len(self._possibleAttrList) == 1 and self._possibleAttrList[0].upper() == "NONE":
                self._possibleAttrList = None
            self.defaultAttribute: str = config["combine_ps_tool.py"]["default attribute"].replace(" ", "")
            try:
                self.attribute: str = self.defaultAttribute
            except AssertionError:
                logger.critical("The initial value read from the data_format.ini file is invalid. Aborting.")
                sys.exit()
        elif not self.useAttribute and self.distinguishFullFineSpectra:
            try:
                self.sortedDataPath: Path = (headDir / "sorted_data" / "fine_spectra").resolve()
            except AssertionError:
                logger.critical(f"""Path to the sorted data [sorted_data/fine_spectra] does not exist. Aborting.""")
                sys.exit()
        elif not self.useAttribute and not self.distinguishFullFineSpectra:
            if not self.addFileMode == "data":
                logger.warning(f"""Sorting by attribute and full/fine is disabled.
Hence, addFileMode [{self.addFileMode}] is expected to be data. It will be set to data now.""")
                self.addFileMode = "data"

    @property
    def sortedDataPath(self) -> Path:
        return self._sortedDataPath

    @sortedDataPath.setter
    def sortedDataPath(self, value: Path):
        assert isinstance(value, Path)
        assert value.exists()
        self._sortedDataPath: Path = value

    @property
    def attribute(self) -> str:
        return self._attribute

    @attribute.setter
    def attribute(self, value):
        logger.debug(f"Setting attribute to {value}.")
        if self.addFileMode == "data":
            pass
        else:
            if self._possibleAttrList is not None:
                assert value in self._possibleAttrList
            self._attribute: str = value
            if self.distinguishFullFineSpectra:
                self.sortedDataPath: Path = (headDir / self._sortedDataDirName / self.attribute / "fine_spectra").resolve()
            else:
                self.sortedDataPath: Path = (headDir / self._sortedDataDirName / self.attribute).resolve()

    @property
    def addFileMode(self):
        return self._addFileMode

    @addFileMode.setter
    def addFileMode(self, value):
        logger.debug(f"Setting addFileMode to {value}.")
        if value in self._addFileModeList:
            self._addFileMode = value
        else:
            logger.error(f"{value} is an invalid argument for addFileMode (only diameter and data are accepted). Using data mode.")
            self._addFileMode = "data"

    def set_attribute(self):
        logger.debug("Calling set_attribute()")
        if self.addFileMode == "data":
            print("This command is only available when addFileMode is set to attribute.")
            return 0
        else:
            attrInput = input(f"{self.attrName}: ")
            logger.debug(f"User input for set_attribute(): {attrInput}")
            try:
                self.attribute = attrInput
            except AssertionError:
                logger.error(f"Invalid input [{attrInput}] for {self.attrName}. Keeping the current value.")
                return 0

    def add_file(self, dataDirPath):
        logger.debug(f"Calling add_file() with dataDirPath = {str(dataDirPath)}")
        assert isinstance(dataDirPath, Path)
        if not dataDirPath.exists():
            logger.critical(f"Directory path does not exist {str(dataDirPath)}")
            return 0
        fileList = list(dataDirPath.rglob("*AllSpectra.dat"))
        for i, file in enumerate(fileList):
            fileName = file.name
            print(f"[{i}]   {fileName}")
            print()

        fileIdxStr = input("select file to add: ")
        logger.debug(f"User input for add_file_data(), select file to add: {fileIdxStr}")
        fileIdx = misc.int_decode(fileIdxStr)
        if fileIdx == None:
            return 0
        if fileIdx >= len(fileList):
            logger.warning(f"ValueError: The selected index [{fileIdx}] exceeds the max value [{len(fileList)}]")
            return 0

        filePath = fileList[fileIdx]
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
        for i, file in enumerate(self.fileDict.keys()):
            print(f"[{i}]   (scale: {self.fileDict[file].powerScale})   {file}")
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
        if not self.addFileMode == "data":
            print(f"{self.attrName}:   {self.attribute}")
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
            sys.exit()
        elif not self.addFileMode == "data" and case == f"set {self.attrName}":
            self.set_attribute()
            return 1
        elif case == "add":
            if self.addFileMode == "data":
                self.add_file(self._dataDirPath)
            else:
                self.add_file(self._sortedDataPath)
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
    main = CombinePowerSeriesTool()
    main.run()