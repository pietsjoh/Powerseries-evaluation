"""Contains a class that can stitch different powerseries together.
Running this script initializes and runs this CombinePowerSeriesTool class.
This the main script to execute when doing a powerseries evaluation.
"""
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
    """Loads powerseries data and runs them using PowerSeriesTool. Can stitch multiple
    powerseries together.

    Upon initialization the data format file is read.
    """
    fileDict: dict = {}
    _addFileModeList: list[str] = ["data", "sorted_data"]
    _dataDirPath: Path = (headDir / "data").resolve()

    def __init__(self):
        self.ps = None
        logger.debug("CombinePowerSeriesTool object initialized.")
        self.read_data_format_ini_file()

    def read_data_format_ini_file(self):
        """Reads the config/data_format.ini file.

        This is used to decide whether to use the data inside the data or
        the sorted_data directory.
        """
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
            if len(self._possibleAttrList) == 1 and self._possibleAttrList[0].lower() == "none":
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
        """Path to the sorted_data directory
        """
        return self._sortedDataPath

    @sortedDataPath.setter
    def sortedDataPath(self, value: Path):
        assert isinstance(value, Path)
        assert value.exists()
        self._sortedDataPath: Path = value

    @property
    def attribute(self) -> str:
        """Value of the attribute, not set when addFileMode is data.
        Upon setting sortedDataPath is also set.
        """
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
        """Can either be data or sorted_data.
        The respective data in the folder can then be evaluated.
        """
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
        """Takes and handles values for the attribute.

        Example
        -------
        Let's say the attribute name is diameter. This method shall
        be used to change the value of the attribute in runtime.
        (from 2 to 3)
        """
        logger.debug("Calling set_attribute()")
        if self.addFileMode == "data":
            print("This command is only available when addFileMode is set to attribute.")
            return 0
        else:
            attrInput = input(f"{self.attrName}: ").replace(" ", "")
            logger.debug(f"User input for set_attribute(): {attrInput}")
            try:
                self.attribute = attrInput
            except AssertionError:
                logger.error(f"Invalid input [{attrInput}] for {self.attrName}. Keeping the current value.")
                return 0

    @misc.input_loop
    def add_file(self, dataDirPath):
        """Takes and handles user input to select a file.

        Upon calling a list of all appropiate (see data_format.ini) files
        are displayed. Enter the number on the left to add the desired file.

        The file is saved in a dictionary. The content of the file
        is never altered.
        One can run the powerseries then.
        The performed actions of PowerSeriesTool are saved in runtime.
        (setting initRange for example)
        Upon exiting the program or calling del_file() these saved settings are lost.
        This method works independent of addFileMode.
        """
        logger.debug(f"Calling add_file() with dataDirPath = {str(dataDirPath)}")
        assert isinstance(dataDirPath, Path)
        if not dataDirPath.exists():
            logger.critical(f"Directory path does not exist {str(dataDirPath)}")
            sys.exit()
        fileList = list(dataDirPath.rglob("*AllSpectra.dat"))
        for i, file in enumerate(fileList):
            fileName = file.name
            print(f"[{i}]   {fileName}")
            print()

        fileIdxStr = input("select file to add: ").replace(" ", "")
        logger.debug(f"User input for add_file_data(), select file to add: {fileIdxStr}")
        if fileIdxStr == "q":
            return 0
        elif fileIdxStr == "exit":
            sys.exit()
        fileIdx = misc.int_decode(fileIdxStr)
        if fileIdx == None:
            return 1
        if fileIdx >= len(fileList):
            logger.warning(f"ValueError: The selected index [{fileIdx}] exceeds the max value [{len(fileList)}]")
            return 1

        filePath = fileList[fileIdx]
        logger.debug(f"Selected file path {filePath}")
        fileName = filePath.name
        self.fileDict[fileName] = PowerSeriesTool(filePath)
        return 1

    @misc.input_loop
    def del_file(self):
        """Takes and handles user input to delete file.

        Upon calling a list of all files saved in the dictionary
        are displayed. Enter the number on the left to delete the desired file from
        the dictionary. Entering all will delete all files available.

        The file is not deleted in the data/sorted_data directory.
        It is only deleted from the dictionary. There,
        the files which are run are saved.
        Deleting a file there deletes all changed settings of PowerSeriesTool.
        (for example initRange)
        """
        logger.debug("Calling del_file()")
        if len(self.fileDict.keys()) == 0:
            logger.warning("There is no file that can be deleted.")
            return 0
        for i, file in enumerate(self.fileDict.keys()):
            print(f"[{i}]   {file}")
            print()
        fileIdxStr = input("select file to delete: ").replace(" ", "")
        logger.debug(f"User input for del_file(), select file to delete: {fileIdxStr}")
        if fileIdxStr == "all":
            self.fileDict.clear()
            return 0
        elif fileIdxStr == "q":
            return 0
        elif fileIdxStr == "exit":
            sys.exit()
        else:
            fileIdx = misc.int_decode(fileIdxStr)
        if fileIdx == None:
            return 1
        if fileIdx >= len(self.fileDict.keys()):
            logger.warning(f"ValueError: The selected index [{fileIdx}] exceeds the max value [{len(self.fileDict.keys()) - 1}].")
            return 1

        fileName = list(self.fileDict.keys())[fileIdx]
        del self.fileDict[fileName]
        return 1

    @misc.input_loop
    def run_powerseries(self):
        """Takes and handles user input to select which powerseries shall be run.

        Upon calling a list of all files saved in the dictionary are displayed.
        Enter the number on the left to run the desired powerseries. Entering all
        will run all available powerseries one behind the other.
        """
        logger.debug("Calling run_powerseries()")
        for i, file in enumerate(self.fileDict.keys()):
            print(f"[{i}]   {file}")
            print()
        fileIdxStr = input("select file to run powerseries: ").replace(" ", "")
        logger.debug(f"User input for run_powerseries(), select file to run powerseries: {fileIdxStr}")
        if fileIdxStr == "all":
            for file in self.fileDict.keys():
                self.fileDict[file].run()
            return 0
        elif fileIdxStr == "q":
            return 0
        elif fileIdxStr == "exit":
            sys.exit()
        else:
            fileIdx = misc.int_decode(fileIdxStr)
            if fileIdx == None:
                return 1
            if fileIdx >= len(self.fileDict.keys()):
                logger.warning(f"ValueError: The selected index [{fileIdx}] exceeds the max value [{len(self.fileDict.keys())}].")
                return 1
            fileName = list(self.fileDict.keys())[fileIdx]
            self.fileDict[fileName].run()
            return 0

    def init_plot(self):
        """Initializes PlotPowerSeries object to create plots.
        Also checks whether this can be done
        (the commannd run has to be called atleast once).
        """
        logger.debug("Calling init_plot()")
        try:
            self.plots = PlotPowerSeries(list(self.fileDict.values()))
            return True
        except AssertionError:
            logger.error("AttributeError: Cannot initialize 'PlotPowerSeries'. Run powerseries first ('run ps')")
            return False

    @misc.input_loop
    def input_plot_selector(self):
        """Takes user input to select which plot shall be shown.

        linewidth and outputpower vs inputpower:        s+lw, lws, "Enter"
        outputpower vs inputpower:                      power, p
        linewidth vs inputpower:                        linewidth, lw
        Q-factor vs inputpower:                         qfactor, q
        mode energy vs inputpower:                      modeenergy, m
        single spectrum (intensity vs energy):          single spectrum, ss
        multiple spectra (intensity vs energy):         multiple spectra, ms

        For single spectrum the index has to be entered. It shows the intensity
        measured vs energy for inputpower.
        For multiple spectra the number of inputpowers to use has to be entered.
        Here a waterfall plot of the measured intensity vs energy is shown for
        different inputpowers.
        """
        plotStr = input("""plot [S+lw (lws), power(p), linewidth(lw), QFactor(qf), modeEnergy(m),
single spectrum (ss), multiple spectra (ms)]: """).lower().replace(" ", "")
        logger.debug(f"User input for input_plot_selector(): {plotStr}")
        if plotStr in ["power", "p"]:
            self.plots.plot_outputPower()
            return 0
        elif plotStr in ["s", "s+lw", "lws", ""]:
            self.plots.plot_lw_s()
            return 0
        elif plotStr in ["linewidth", "lw"]:
            self.plots.plot_linewidth()
            return 0
        elif plotStr in ["qfactor", "qf"]:
            self.plots.plot_QFactor()
            return 0
        elif plotStr in ["modeenergy", "m"]:
            self.plots.plot_mode_wavelength()
            return 0
        elif plotStr in ["singlespectrum", "ss"]:
            idxStr = input("Enter index: ").replace(" ", "")
            logger.debug(f"User input for idxStr: {idxStr}")
            idx = misc.int_decode(idxStr)
            try:
                self.plots.plot_single_spectrum(idx)
            except AssertionError:
                logger.error(f"ValueError: [{idxStr}] is not a valid input.")
                return 1
            else:
                return 0
        elif plotStr in ["multiplespectra", "ms"]:
            numPlotsStr = input("number of plots: ").replace(" ", "")
            logger.debug(f"User input for numPlotsStr: {numPlotsStr}")
            numPlots = misc.int_decode(numPlotsStr)
            try:
                self.plots.plot_multiple_spectra(numPlots)
            except AssertionError:
                logger.error(f"Invalid input (numbers in the range [1:{self.plots.lenInputPower}] are accepted).")
                return 1
            else:
                return 0
        elif plotStr == "q":
            return 0
        elif plotStr == "exit":
            sys.exit()
        else:
            logger.error(f"ValueError: {plotStr} is not a valid input.")
            return 1

    @misc.input_loop
    def scale_outputPower(self):
        """Takes and handles user input to select and scale the outputpower of
        the files saved in the dictionary.
        """
        logger.debug("Calling scale_outputPower()")
        flag = self.plot_combine()
        if not flag:
            return 0
        for i, file in enumerate(self.fileDict.keys()):
            print(f"[{i}]   (scale: {self.fileDict[file].powerScale})   {file}")
            print()
        selectFileStr = input("select file for scaling: ").replace(" ", "")
        logger.debug(f"User input for selectFileStr, fileIdx: {selectFileStr}")
        if selectFileStr == "q":
            return 0
        elif selectFileStr == "exit":
            sys.exit()
        fileIdx = misc.int_decode(selectFileStr)
        if fileIdx == None:
            return 1
        elif fileIdx >= len(self.fileDict.keys()):
            logger.warning(f"ValueError: The selected index [{fileIdx}] exceeds the max value [{len(self.fileDict.keys())}].")
            return 1
        fileName = list(self.fileDict.keys())[fileIdx]
        scaleStr = input("scale: ").replace(" ", "")
        logger.debug(f"User input for scale_outputPower(), scale: {scaleStr}")
        if scaleStr == "q":
            return 0
        elif scaleStr == "exit":
            sys.exit()
        scale = misc.float_decode(scaleStr)
        if scale == None:
            return 1
        self.fileDict[fileName].powerScale = scale
        return 1

    def plot_combine(self):
        """Plots outputpower vs inputpower for all files in the dictionary.
        """
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

    def config(self):
        """Prints out basic information about the selected files in the dictionary.
        """
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

    @misc.input_loop
    def input_decoder(self):
        """Function that takes user input and decides what to do.
        """
        print()
        case = input("enter instruction (type help for more information): ")
        logger.debug(f"User input for input_decoder(): {case}")
        print()
        if case == "q":
            confirmationExit = input("Do you really want to exit the tool? [y/n]: ").lower().replace(" ", "")
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
            self.scale_outputPower()
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
        elif case == "help":
            self.input_help()
            return 1
        else:
            logger.error(f"ValueError: {case} is not implemented. Type help for more information.")
            return 1

    @misc.input_loop
    def input_help(self):
        """Takes and handles input for the help command.
        """
        helpType = input("""What kind of information are you looking for?

        - commands of CombinePowerSeriesTool:   enter combine

        - commands of PowerSeriesTool:          enter commands

        - powerseries program parameters:       enter powerseries

        - peak fit program parameters:          enter peak fit

        - available plots:                      enter plots

        - everything: enter all
        """).lower().replace(" ", "")
        logger.debug(f"User inout for input_help(): {helpType}")
        inputList = ["combine", "commands", "powerseries", "peakfit", "plots", "all", "q", "exit"]
        if not helpType in inputList:
            logger.error(f"{helpType} is an invalid input.")
            return 1
        elif helpType == "q":
            return 0
        elif helpType == "exit":
            sys.exit()
        elif helpType == "combine":
            self.help_combine_ps()
            return 0
        elif helpType == "commands":
            PowerSeriesTool.help_commands()
            return 0
        elif helpType == "powerseries":
            PowerSeriesTool.help_powerseries()
            return 0
        elif helpType == "peakfit":
            PowerSeriesTool.help_peak_fit()
            return 0
        elif helpType == "plots":
            PowerSeriesTool.help_plots()
            return 0
        elif helpType == "all":
            self.help_combine_ps()
            PowerSeriesTool.help_commands()
            PowerSeriesTool.help_peak_fit()
            PowerSeriesTool.help_powerseries()
            PowerSeriesTool.help_plots()
            return 0

    @staticmethod
    def help_combine_ps():
        """Prints out information about the commands.

        This information can also be found in the documentation of CombinePowerSeries.
        """
        pass

    def run(self):
        """Main Method, runs the input decoder until an exit is called.
        """
        print()
        print("-"*100)
        print("Running CombinePowerSeriesTool")
        print("-"*100)
        print()
        self.config()
        self.input_decoder()



if __name__ == "__main__":
    main = CombinePowerSeriesTool()
    main.run()