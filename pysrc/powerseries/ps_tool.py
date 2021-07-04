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
    minInitRangeEnergy = None
    maxInitRangeEnergy = None
    minInputPowerRange = None
    maxInputPowerRange = None

    def __init__(self, filePath, diameterIndicator="np7509_ni_"):
        super().__init__(filePath, diameterIndicator=diameterIndicator)
        logger.debug("""PowerSeriesTool initialized.
            filepath: {}""".format(filePath))

    def input_decoder(self):
        print()
        case = input("enter instruction (type help for more information): ")
        logger.debug(f"User input for input_decoder(): {case}")
        print()
        if case == "q":
            return 0
        elif case == "exit":
            exit()
        elif case == "config":
            self.config()
            return 1
        elif case == "bg":
            self.input_backgroundFitMode()
            return 1
        elif case == "set init range":
            minInitRangeEnergyStr, maxInitRangeEnergyStr, maxInitRangeStr = self.input_initial_range()
            self.check_input_initial_range(minInitRangeEnergyStr, maxInitRangeEnergyStr, maxInitRangeStr)
            return 1
        elif case == "set fitmodel":
            fitModelStr = self.input_fitmodel()
            self.check_input_fitmodel(fitModelStr)
            return 1
        elif case == "set snapshots":
            self.input_snapshots()
            return 1
        elif case == "set peak width":
            self.input_constantPeakWidth()
            return 1
        elif case == "set exclude":
            minInputPowerRangeStr, maxInputPowerRangeStr = self.input_exclude()
            self.check_input_exclude(minInputPowerRangeStr, maxInputPowerRangeStr)
            return 1
        elif case == "set fit range":
            self.input_fitRangeScale()
            return 1
        elif case == "run":
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
        else:
            logger.error(f"ValueError: {case} is not a valid input (type help for more information).")
            return 1

    def input_fitRangeScale(self):
        fitRangeScaleStr = input("fit range scale: ")
        logger.debug(f"User input for input_fitRangeScale(): {fitRangeScaleStr}")
        self.fitRangeScale = misc.float_decode(fitRangeScaleStr)

    def input_constantPeakWidth(self):
        constantpeakWidthStr = input("constant peak width: ")
        logger.debug(f"User input for input_constantPeakWidth(): {constantpeakWidthStr}")
        self.constantPeakWidth = misc.int_decode(constantpeakWidthStr)

    def input_plot_selector(self):
        plotStr = input("""plot [S+lw (lws), power(p), linewidth(lw), QFactor(q), modeEnergy(m),
single spectrum (ss), multiple spectra (ms)]: """)
        logger.debug(f"User input for input_plot_selector(): {plotStr}")
        if plotStr.upper() in ["POWER", "P"]:
            self.plots.plot_outputPower()
        elif plotStr.upper() in ["S+lw", "LWS", ""]:
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
                logger.error(f"Invalid input (numbers in the range [1:{self.lenInputPower}] are accepted).")
        else:
            logger.error(f"ValueError: {plotStr} is not a valid input.")

    def init_plot(self):
        logger.debug("Calling init_plot()")
        try:
            self.plots = PlotPowerSeries([self])
            return True
        except AssertionError:
            logger.error("AttributeError: Cannot initialize 'PlotPowerSeries'. Use the command 'run' first.")
            return False

    def input_snapshots(self):
        snapshotsStr = input("number of snapshots: ")
        logger.debug(f"User input for input_snapshots(): {snapshotsStr}")
        self.snapshots = misc.int_decode(snapshotsStr)

    def input_backgroundFitMode(self):
        backgroundFitModeStr = input("background fit mode: ")
        logger.debug(f"User input for input_backgroundFitMode(): {backgroundFitModeStr}")
        self.backgroundFitMode = backgroundFitModeStr

    @staticmethod
    def input_fitmodel():
        fitModelStr = input("fitmodel (gauss, lorentz, voigt, pseudovoigt): ")
        logger.debug(f"User input for input_fitModel(): {fitModelStr}")
        return fitModelStr

    @staticmethod
    def input_initial_range():
        minInitRangeEnergyStr = input("min energy: ")
        logger.debug(f"User input for input_initial_range(), min energy: {minInitRangeEnergyStr}")
        maxInitRangeEnergyStr = input("max energy: ")
        logger.debug(f"User input for input_initial_range(), max energy: {maxInitRangeEnergyStr}")
        maxInitRangeStr = input("max index to use init range: ")
        logger.debug(f"User input for input_initial_range(), maxInitRange: {maxInitRangeStr}")
        return minInitRangeEnergyStr, maxInitRangeEnergyStr, maxInitRangeStr

    @staticmethod
    def input_exclude():
        minInputPowerRangeStr = input("min input power: ")
        logger.debug(f"User input for input_exclude(), min input power: {minInputPowerRangeStr}")
        maxInputPowerRangeStr = input("max input power: ")
        logger.debug(f"User input for input_exclude(), max input power: {maxInputPowerRangeStr}")
        return minInputPowerRangeStr, maxInputPowerRangeStr

    def config(self):
        print()
        print("/"*100)
        print()
        print(f"file name:                      {self.fileName}")
        print(f"diameter:                       {self.diameter} µm")
        print(f"temperature:                    {self.temperature} K")
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
        print()
        print("-"*100)
        print("Running PowerSeriesTool")
        print("-"*100)
        print()
        self.config()
        j = self.input_decoder()
        while j == 1:
            j = self.input_decoder()

if __name__ == "__main__":
    head = (Path(__file__).parents[2]).resolve()
    fileName = "data\\20210303\\NP7509_Ni_4µm_20K_Powerserie_1-01s_deteOD0_fine3_WithoutLensAllSpectra.dat"
    fileName = fileName.replace("\\", "/")
    filePath = (head / fileName).resolve()
    test = PowerSeriesTool(filePath)
    test.initialRange = (786, 687)
    test.maxInitRange = 5
    test.run()
