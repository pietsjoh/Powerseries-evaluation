"""Contains a console based application wrapper
around eval_ps.py
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from powerseries.eval_ps import EvalPowerSeries
from powerseries.plot_ps import PlotPowerSeries
from setup.config_logging import LoggingConfig
import utils.misc as misc

loggerObj = LoggingConfig()
logger = loggerObj.init_logger(__name__)

class PowerSeriesTool(EvalPowerSeries):
    """Enables adjustments of parameters for the powerseries evaluation using console input.

    Child of EvalPowerSeries. Fulfills the same purpose aswell, but one can change the
    program parameters in runtime with console input.

    In the __init__ method a filepath to the powerseries data has to be provided.
    Then the base class, EvalPowerSeries, is instantiated with the data object from
    the file. The data model of the config/data_format.ini file is used.
    """
    minInitRangeEnergy = None
    maxInitRangeEnergy = None
    minInputPowerRange = None
    maxInputPowerRange = None

    def __init__(self, filePath):
        self.read_data_format_ini_file()
        DataObj = self.dataModel(filePath)
        super().__init__(DataObj)
        logger.debug("""PowerSeriesTool initialized.
            filepath: {}""".format(filePath))

    @misc.input_loop
    def input_decoder(self):
        """Function that takes the user input and decides what to do.
        """
        print()
        case = input("enter instruction (type help for more information): ").lower().replace(" ", "")
        logger.debug(f"User input for input_decoder(): {case}")
        print()
        if case == "q":
            return 0
        elif case == "exit":
            sys.exit()
        elif case == "config":
            self.config()
            return 1
        elif case == "setbackground" or "bg":
            self.input_backgroundFitMode()
            return 1
        elif case == "setinitrange" or "ir":
            minInitRangeEnergyStr, maxInitRangeEnergyStr, maxInitRangeStr = self.input_initial_range()
            self.check_input_initial_range(minInitRangeEnergyStr, maxInitRangeEnergyStr, maxInitRangeStr)
            return 1
        elif case == "setfitmodel" or "fm":
            fitModelStr = self.input_fitmodel()
            self.check_input_fitmodel(fitModelStr)
            return 1
        elif case == "setsnapshots" or "ss":
            self.input_snapshots()
            return 1
        elif case == "setpeakwidth" or "pw":
            self.input_constantPeakWidth()
            return 1
        elif case == "setexclude" or "ex":
            minInputPowerRangeStr, maxInputPowerRangeStr = self.input_exclude()
            self.check_input_exclude(minInputPowerRangeStr, maxInputPowerRangeStr)
            return 1
        elif case == "setfitrange" or "fr":
            self.input_fitRangeScale()
            return 1
        elif case == "run" or case == "":
            self.get_power_dependent_data()
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
        elif case == "plot":
            if self.init_plot():
                self.input_plot_selector()
            return 1
        elif case == "help":
            self.input_help()
            return 1
        else:
            logger.error(f"ValueError: {case} is not a valid input (type help for more information).")
            return 1

    @misc.input_loop
    def input_help(self):
        """Takes and handles input for the help command.
        """
        helpType = input("""What kind of information are you looking for?

        - commands of PowerSeriesTool:          enter commands

        - powerseries program parameters:       enter powerseries

        - peak fit program parameters:          enter peak fit

        - available plots:                      enter plots

        - everything: enter all
        """).lower().replace(" ", "")
        logger.debug(f"User inout for input_help(): {helpType}")
        inputList = ["commands", "powerseries", "peakfit", "plots", "all", "q", "exit"]
        if not helpType in inputList:
            logger.error(f"{helpType} is an invalid input.")
            return 0
        elif helpType == "q":
            return 1
        elif helpType == "exit":
            sys.exit()
        elif helpType == "commands":
            self.help_commands()
            return 1
        elif helpType == "powerseries":
            self.help_powerseries()
            return 1
        elif helpType == "peakfit":
            self.help_peak_fit()
            return 1
        elif helpType == "plots":
            self.help_plots()
            return 1
        elif helpType == "all":
            self.help_commands()
            self.help_peak_fit()
            self.help_powerseries()
            self.help_plots()
            return 1

    @staticmethod
    def help_commands():
        """Prints out information about the available commands.

        This information can also be found in the documentation of this class.
        """
        pass

    @staticmethod
    def help_powerseries():
        """Prints out information about the powerseries program parameters.

        This information can also be found in the powerseries/eval_ps.py documentation.
        """
        pass

    @staticmethod
    def help_peak_fit():
        """Prints out information about the peak fitting program parameters.

        This information can also be found in the peak_fit/single_peak_fit_base.py documentation"""
        pass

    @staticmethod
    def help_plots():
        """Prints out information about the available plots.

        This information can also be found in the powerseries/plot_ps.py documentation."""
        pass

    def input_fitRangeScale(self):
        """Takes and handles user input for fitRangeScale
        For more information on this parameter take a look at
        single_peak_fit_base.py.
        """
        fitRangeScaleStr = input("fit range scale: ").replace(" ", "")
        logger.debug(f"User input for input_fitRangeScale(): {fitRangeScaleStr}")
        self.fitRangeScale = misc.float_decode(fitRangeScaleStr)

    def input_constantPeakWidth(self):
        """Takes and handles user input for constantPeakWidth.
        For more information on this parameter take a look at
        single_peak_fit_base.py.
        """
        constantpeakWidthStr = input("constant peak width: ").replace(" ", "")
        logger.debug(f"User input for input_constantPeakWidth(): {constantpeakWidthStr}")
        self.constantPeakWidth = misc.int_decode(constantpeakWidthStr)

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
                logger.error(f"Invalid input (numbers in the range [1:{self.lenInputPower}] are accepted).")
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

    def init_plot(self):
        """Initializes PlotPowerSeries object to create plots.
        Also checks whether this can be done
        (the commannd run has to be called atleast once).
        """
        logger.debug("Calling init_plot()")
        try:
            self.plots = PlotPowerSeries([self])
            return True
        except AssertionError:
            logger.error("AttributeError: Cannot initialize 'PlotPowerSeries'. Use the command 'run' first.")
            return False

    def input_snapshots(self):
        """Takes and handles user input for snapshots.
        For more information on this parameter take a look at
        eval_ps.py.
        """
        snapshotsStr = input("number of snapshots: ").replace(" ", "")
        logger.debug(f"User input for input_snapshots(): {snapshotsStr}")
        self.snapshots = misc.int_decode(snapshotsStr)

    def input_backgroundFitMode(self):
        """Takes and handles user input for backgroundFitMode.
        For more information on this parameter take a look at
        single_peak_fit_base.py.
        """
        backgroundFitModeStr = input("background fit mode: ").lower().replace(" ", "")
        logger.debug(f"User input for input_backgroundFitMode(): {backgroundFitModeStr}")
        self.backgroundFitMode = backgroundFitModeStr

    @staticmethod
    def input_fitmodel():
        """Takes and handles user input for fitModel.
        For more information on this parameter take a look at
        single_peak_fit_models.py.
        """
        fitModelStr = input("fitmodel (gauss, lorentz, voigt, pseudovoigt): ").lower().replace(" ", "")
        logger.debug(f"User input for input_fitModel(): {fitModelStr}")
        return fitModelStr

    @staticmethod
    def input_initial_range():
        """Takes and handles user input for initialRange.
        For more information on this parameter take a look at
        single_peak_fit_base.py.
        """
        minInitRangeEnergyStr = input("min energy: ").replace(" ", "")
        logger.debug(f"User input for input_initial_range(), min energy: {minInitRangeEnergyStr}")
        maxInitRangeEnergyStr = input("max energy: ").replace(" ", "")
        logger.debug(f"User input for input_initial_range(), max energy: {maxInitRangeEnergyStr}")
        maxInitRangeStr = input("max index to use init range: ").replace(" ", "")
        logger.debug(f"User input for input_initial_range(), maxInitRange: {maxInitRangeStr}")
        return minInitRangeEnergyStr, maxInitRangeEnergyStr, maxInitRangeStr

    @staticmethod
    def input_exclude():
        """Takes and handles user input for exclude.
        For more information on this parameter take a look at
        single_peak_fit_base.py.
        """
        minInputPowerRangeStr = input("min input power: ").replace(" ", "")
        logger.debug(f"User input for input_exclude(), min input power: {minInputPowerRangeStr}")
        maxInputPowerRangeStr = input("max input power: ").replace(" ", "")
        logger.debug(f"User input for input_exclude(), max input power: {maxInputPowerRangeStr}")
        return minInputPowerRangeStr, maxInputPowerRangeStr

    def config(self):
        """Prints gathered information about the current evaluation of the powerseries.
        """
        print()
        print("/"*100)
        print()
        print(f"file name:                      {self.fileName}")
        print()
        print(f"number of energies:             {self.energies.size}")
        print(f"number of input powers:         {self.lenInputPower}")
        print()
        print(f"energy range:                   {self.energies[0], self.energies[-1]} eV")
        print(f"wavelength range:               {self.wavelengths[0], self.wavelengths[-1]} nm")
        print(f"input power range:              {self.minInputPower, self.maxInputPower} mW")
        print()
        print(f"excluded points:                {self.exclude}")
        print(f"min input power (exclude):      {self.minInputPowerRange}")
        print(f"max input power (exclude):      {self.maxInputPowerRange}")
        print()
        print(f"initial range:                  {self.initRange}")
        print(f"min E (initial range):          {self.minInitRangeEnergy} eV")
        print(f"max E (initial range):          {self.maxInitRangeEnergy} eV")
        print(f"max index to use init range:    {self.maxInitRange}")
        print()
        print(f"background fit mode:            {self.backgroundFitMode}")
        print(f"number of snapshots:            {self.snapshots}")
        print(f"fit range scale:                {self.fitRangeScale}")
        print(f"constantPeakWidth:              {self.constantPeakWidth}")
        print(f"integration coverage:           {self.intCoverage}")
        print(f"fit model:                      {self.fitModel.name}")
        print()
        print("/"*100)
        print()

    def run(self):
        """Main Method, runs the input decoder until an exit is called.
        """
        print()
        print("-"*100)
        print("Running PowerSeriesTool")
        print("-"*100)
        print()
        self.config()
        j = self.input_decoder()

if __name__ == "__main__":
    ## just testing
    head = (Path(__file__).parents[2]).resolve()
    fileName = "data\\20210303\\NP7509_Ni_4µm_20K_Powerserie_1-01s_deteOD0_fine3_WithoutLensAllSpectra.dat"
    fileName = fileName.replace("\\", "/")
    filePath = (head / fileName).resolve()
    test = PowerSeriesTool(filePath)
    test.initialRange = (786, 687)
    test.maxInitRange = 5
    test.run()
