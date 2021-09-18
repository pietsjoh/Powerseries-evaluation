"""Contains a class that evaluates a powerseries. The individual spectra are analyzed using the peak_fit module.
"""
import numpy as np
from configparser import ConfigParser
import typing
import sys
from pathlib import Path
headDir: Path = Path(__file__).resolve().parents[2]
srcDirPath: Path = (headDir / "pysrc").resolve()
sys.path.append(str(srcDirPath))

from peak_fit.single_peak_fit_models import GaussianPeakFit, LorentzPeakFit, VoigtPeakFit, PseudoVoigtPeakFit
from data_tools.data_formats import DataQlab2
from peak_fit.single_peak_fit_base import PeakFitSuper
from setup.config_logging import LoggingConfig
import utils.misc as misc

loggerObj: LoggingConfig = LoggingConfig()
logger = loggerObj.init_logger(__name__)

tupleIntOrNone = typing.Union[tuple[int, int], None]
number = typing.Union[int, float, np.number]
numberOrNone = typing.Optional[number]
floatOrNone = typing.Union[float, None]
intOrNone = typing.Union[int, None]
listOrNone = typing.Union[list, None]

class EvalPowerSeries:
    """Evaluates a powerseries extracting outputpower, linewidth, Q-factor and wavelength for each input power.

    The main method get_power_dependent_data() performs the evalutation of the powerseries.
    Apart from that the rest of the class consists of setter/getter methods which handle adjustable parameters for
    the powerseries evaluation.

    The file ps_tool.py contains a wrapper for this class to set the parameters using console input.

    A Data object is required for the __init__ method.
    Currently, only DataQlab2 objects are accepted.
    If you want to use a different data format, then create a class in the data_tools/data_formats.py file
    which is similar to DataQlab2.
    Moreover, this class should be added to the imports (line 13) and into the _dataModelList.
    Look at the check_DataObj_attributes() method to check whether your class satisfies the required attributes.

    In the __init__ method the required attributes of the DataObj are saved as attributes. Furthermore, the .ini files
    powerseries.ini and data_format.ini are read.
    """
    _fitModelList: dict = {"LORENTZ": LorentzPeakFit, "GAUSS": GaussianPeakFit, "VOIGT": VoigtPeakFit, "PSEUDOVOIGT": PseudoVoigtPeakFit}
    _dataModelList: dict = {"QLAB2" : DataQlab2}
    _snapshots: intOrNone = None
    _exclude: list = []
    _maxInitRange: int = 0
    _initRange: tupleIntOrNone = None

    def __init__(self, DataObj) -> None:
        try:
            self.check_DataObj_attributes(DataObj)
            assert hasattr(DataObj, "name")
            self.dataModel = DataObj.name
        except AssertionError:
            logger.exception("DataObj does not satisfy the requirements. Aborting.")
            sys.exit()
        else:
            self.data = DataObj
            self.wavelengths: np.ndarray = self.data.wavelengths
            self.energies: np.ndarray = self.data.energies
            self.inputPower: np.ndarray = self.data.inputPower
            self.lenInputPower: int = self.data.lenInputPower
            self.maxEnergy: number = self.data.maxEnergy
            self.minEnergy: number = self.data.minEnergy
            self.minInputPower: number = self.data.minInputPower
            self.maxInputPower: number = self.data.maxInputPower
            self.fileName: str = self.data.fileName
            logger.debug("""EvalPowerSeries object initialized.
                file name: {}""".format(DataObj.fileName))
            self.read_powerseries_ini_file()
            self.read_data_format_ini_file()

    @property
    def dataModel(self):
        """Format of the data.
        """
        return self._dataModel

    @dataModel.setter
    def dataModel(self, value: str) -> None:
        logger.debug(f"Setting dataModel to {value}.")
        if not value.upper() in self._dataModelList.keys():
            logger.error(f"{value} is not a valid data model (subclass of DataSuper). Aborting.")
            raise AssertionError("Invalid data model.")
        else:
            self._dataModel = self._dataModelList[value.upper()]

    @staticmethod
    def check_DataObj_attributes(DataObj) -> None:
        """Checks whether the provided DataObj has the required attributes.

        Raises
        ------
        AssertionError:
            When an attribute is missing or of an invalid datatype.
        """
        assert hasattr(DataObj, "__getitem__")
        assert hasattr(DataObj, "energies")
        assert hasattr(DataObj, "wavelengths")
        assert hasattr(DataObj, "inputPower")
        assert hasattr(DataObj, "maxEnergy")
        assert hasattr(DataObj, "minEnergy")
        assert hasattr(DataObj, "minInputPower")
        assert hasattr(DataObj, "maxInputPower")
        assert hasattr(DataObj, "lenInputPower")
        assert hasattr(DataObj, "fileName")
        assert isinstance(DataObj.energies, np.ndarray)
        assert isinstance(DataObj.inputPower, np.ndarray)
        assert isinstance(DataObj.wavelengths, np.ndarray)
        assert np.issubdtype(type(DataObj.lenInputPower), np.integer)
        assert isinstance(DataObj.fileName, str)
        assert (np.issubdtype(type(DataObj.minEnergy), np.integer) or np.issubdtype(type(DataObj.minEnergy), np.floating))
        assert (np.issubdtype(type(DataObj.maxEnergy), np.integer) or np.issubdtype(type(DataObj.maxEnergy), np.floating))
        assert (np.issubdtype(type(DataObj.minInputPower), np.integer) or np.issubdtype(type(DataObj.minInputPower), np.floating))
        assert (np.issubdtype(type(DataObj.maxInputPower), np.integer) or np.issubdtype(type(DataObj.maxInputPower), np.floating))
        assert (DataObj.energies >= 0).all()
        assert (DataObj.inputPower >= 0).all()
        assert (DataObj.wavelengths >= 0).all()
        assert all(DataObj.inputPower[i] <= DataObj.inputPower[i + 1] for i in range(len(DataObj.inputPower) - 1))

    def read_powerseries_ini_file(self) -> None:
        """Reads the config/powerseries.ini file and uses the settings from the eval_ps.py section.
        These attributes are used as initial parameters for the powerseries evaluation.
        """
        logger.debug("Calling read_powerseries_ini_file()")
        configIniPath: Path = (headDir / "config" / "powerseries.ini").resolve()
        config: ConfigParser = ConfigParser()
        config.read(str(configIniPath), encoding="UTF-8")
        self.snapshots: intOrNone = misc.int_decode(config["eval_ps.py"]["snapshots"].replace(" ", ""))
        self.fitRangeScale: floatOrNone = misc.float_decode(config["eval_ps.py"]["fit range scale"].replace(" ", ""))
        self.intCoverage: floatOrNone = misc.float_decode(config["eval_ps.py"]["integration coverage"].replace(" ", ""))
        self.constantPeakWidth: intOrNone = misc.int_decode(config["eval_ps.py"]["constant peak width"].replace(" ", ""))
        self.powerScale: floatOrNone = misc.float_decode(config["eval_ps.py"]["power scale"].replace(" ", ""))
        fitModel: str = config["eval_ps.py"]["fit model"].replace(" ", "")
        self.check_input_fitmodel(fitModel)
        minInitRangeEnergyStr: str = config["eval_ps.py"]["min energy"].replace(" ", "")
        maxInitRangeEnergyStr: str = config["eval_ps.py"]["max energy"].replace(" ", "")
        maxInitRangeStr: str = config["eval_ps.py"]["max init range"].replace(" ", "")
        useInitRange: bool
        self.initRange: tupleIntOrNone
        try:
            useInitRange = LoggingConfig.check_true_false(config["eval_ps.py"]["use init range"].replace(" ", ""))
        except ValueError:
            logger.error("ValueError: useInitRange could not be extracted from the .ini file. Setting it to False.")
            useInitRange = False
        finally:
            if useInitRange:
                self.check_input_initial_range(minInitRangeEnergyStr, maxInitRangeEnergyStr, maxInitRangeStr)
            else:
                self.initRange = None
        minInputPowerRangeStr: str = config["eval_ps.py"]["min inputpower"].replace(" ", "")
        maxInputPowerRangeStr: str = config["eval_ps.py"]["max inputpower"].replace(" ", "")
        useExclude: bool
        self.exclude: list[int]
        try:
            useExclude = LoggingConfig.check_true_false(config["eval_ps.py"]["use exclude"].replace(" ", ""))
        except ValueError:
            logger.error("ValueError: useExclude could not be extracted from the .ini file. Setting it to False.")
            useExclude = False
        finally:
            if useExclude:
                self.check_input_exclude(minInputPowerRangeStr, maxInputPowerRangeStr)
            else:
                self.exclude = []
        self.backgroundFitMode: str = config["eval_ps.py"]["background fit mode"].replace(" ", "")

    def read_data_format_ini_file(self) -> None:
        """Reads the config/data_format.ini file. Only reads the data model.
        """
        logger.debug("Calling read_powerseries_ini_file()")
        configIniPath: Path = (headDir / "config" / "data_format.ini").resolve()
        config: ConfigParser = ConfigParser()
        config.read(str(configIniPath), encoding="UTF-8")
        try:
            self.dataModel = config["data format"]["data model"].replace(" ", "")
        except AssertionError:
            raise TypeError("Config file value for data model is invalid.")

    def energy_to_idx(self, energy: number) -> int:
        """If the provided energy lies within the range [minEnergy, maxEnergy] then
        the index of the energy array closest to the input energy is returned.
        Otherwise the maxEnergy and the minEnergy respectively are returned with a warning message.

        Returns
        -------
        int:
            Index of the element of the energy array closest to the input.
        """
        logger.debug(f"Calling energy_to_idx(), energy: {energy}")
        if (energy < self.minEnergy):
            logger.warning("""ValueError: The selected energy {} is out of bounce. Possible range: [{}, {}].
Selecting the minimum value.""".format(energy, self.minEnergy, self.maxEnergy))
        elif (energy > self.maxEnergy):
            logger.warning("""ValueError: The selected energy {} is out of bounce. Possible range: [{}, {}].
Selecting the maximum value.""".format(energy, self.minEnergy, self.maxEnergy))
        idx: int = int(np.argmin(np.abs(self.energies - energy)))
        logger.debug(f"energy_to_idx(), Index: {idx}")
        return idx

    def inputPower_to_idx(self, inputPower: number) -> int:
        """If the provided inputPower lies within the range [minInputPower, maxInputPower] then
        the index of the inputPower array closest to the input inputPower is returned.
        Otherwise the maxInputPower and the minInputPower respectively are returned with a warning message.

        Returns
        -------
        int:
            Index of the element of the inputPower array closest to the input.
        """
        logger.debug(f"Calling inputPower_to_idx(), inputPower: {inputPower}")
        idx: int
        if (inputPower < self.minInputPower):
            logger.warning("""ValueError: The selected input power {} is out of bounce. Possible range: [{}, {}].
Selecting the minimum value.""".format(inputPower, self.minInputPower, self.maxInputPower))
            idx = 0
            return idx
        elif (inputPower > self.maxInputPower):
            logger.warning("""ValueError: The selected input power {} is out of bounce. Possible range: [{}, {}].
Selecting the maximum value.""".format(inputPower, self.minInputPower, self.maxInputPower))
            idx = self.lenInputPower
            return idx
        idx = int(np.argmin(np.abs(self.inputPower - inputPower)))
        logger.debug(f"inputPower_to_idx(), index: {idx}")
        return idx

    @property
    def backgroundFitMode(self) -> str:
        """Sets the way the background of spectra is handled. For more information look into peak_fit/single_peak_fit_base.py.
        """
        return self._backgroundFitMode

    @backgroundFitMode.setter
    def backgroundFitMode(self, value: str) -> None:
        logger.debug(f"Setting backgroundFitMode to {value}.")
        assert isinstance(value, str)
        val: str = value.lower().replace(" ", "")
        self._backgroundFitMode: str
        if val in ["spline", "constant", "local_all", "local_left", "local_right", "none", "offset", "disable"]:
            self._backgroundFitMode = val
        else:
            logger.error(f"{value} is not an implemented background fit mode. Using none.")
            self._backgroundFitMode = "none"

    @property
    def snapshots(self) -> int:
        """Defines how many spectra are shown when evaluating a powerseries.

        Example
        -------
        If snapshots is set to 10, then every 10th spectra is shown starting at 0.
        """
        return self._snapshots

    @snapshots.setter
    def snapshots(self, value: int) -> None:
        logger.debug(f"Setting snapshots to {value}.")
        self._snapshots: int = value
        if self._snapshots is None:
            self._snapshots = 0
        if not isinstance(self._snapshots, (int, np.integer)):
            logger.error(f"TypeError: invalid argument for snapshot ({self._snapshots}). Setting snapshots to 0.")
            self._snapshots = 0
        if self._snapshots < 0:
            self._snapshots = 0
        logger.debug(f"snapshots has been set to {self._snapshots}")

    def check_input_initial_range(self, minInitRangeEnergyStr: str, maxInitRangeEnergyStr: str,
        maxInitRangeStr: str) -> None:
        """Transforms energy values for the initial range into index values, which are
        required for peak_fit/single_peak_fit_base.py.
        """
        logger.debug("Calling check_input_initial_range()")
        self.minInitRangeEnergy: floatOrNone = misc.float_decode(minInitRangeEnergyStr)
        self.maxInitRangeEnergy: floatOrNone = misc.float_decode(maxInitRangeEnergyStr)
        self.maxInitRange: intOrNone = misc.int_decode(maxInitRangeStr)
        if self.minInitRangeEnergy is None:
            self.minInitRangeEnergy = self.minEnergy
            logger.error(f"Invalid input for min energy. Using the min possible value {self.minInitRangeEnergy}")
        if self.maxInitRangeEnergy is None:
            self.maxInitRangeEnergy = self.maxEnergy
            logger.error(f"Invalid input for max energy. Using the max possible value {self.maxInitRangeEnergy}")
        minIdx: int = self.energy_to_idx(self.minInitRangeEnergy)
        maxIdx: int = self.energy_to_idx(self.maxInitRangeEnergy)
        logger.debug(f"input_initial_range(), Initial range: [{minIdx}, {maxIdx}]")
        self.initRange: tuple[int, int] = minIdx, maxIdx

    @property
    def initRange(self) -> tupleIntOrNone:
        """When set to None, no initial range is used
        Otherwise the program looks only for peaks in between [min(initRange), max(initRange)]
        """
        return self._initRange

    @initRange.setter
    def initRange(self, value: tupleIntOrNone) -> None:
        logger.debug(f"Setting initRange to {value}.")
        self._initRange: tupleIntOrNone = value
        if isinstance(self._initRange, tuple):
            if len(self._initRange) == 2:
                idx1: int
                idx2: int
                idx1, idx2 = self._initRange
                if idx1 != idx2:
                    idxList: list[int, int] = [idx1, idx2]
                    i: int
                    idx: int
                    for i, idx in enumerate(idxList):
                        if not np.issubdtype(type(idx), np.integer):
                            logger.error(f"TypeError: Index [{idx}] is not an integer. Setting initialRange to None.")
                            self._initRange = None
                            break
                        if idx < 0:
                            logger.warning(f"ValueError: Index [{idx}] is smaller than 0. Setting {idx} to 0.")
                            idxList[i] = 0
                        if idx >= self.data[0].size:
                            logger.warning(f"ValueError: Index [{idx}] is larger than the max value {self.data[0].size - 1}. Setting {idx} to the max value.")
                            idxList[i] = self.data[0].size - 1
                else:
                    logger.error(f"ValueError: {self._initRange} has identical elements (-> no range). Setting initialRange to None.")
            else:
                logger.error(f"TypeError: {self._initRange} has more than 2 elements. Setting initialRange to None.")
        elif self._initRange is None:
            pass
        else:
            logger.error(f"TypeError: {self._initRange} is not a tuple. Setting initialRange to None.")
            self._initRange = None
        logger.debug(f"initRange has been set to {self._initRange}")

    @property
    def maxInitRange(self) -> int:
        """Max index where initial range shall be used.
        """
        return self._maxInitRange

    @maxInitRange.setter
    def maxInitRange(self, value: intOrNone) -> None:
        logger.debug(f"Setting maxInitRange to {value}.")
        if value is None:
            self._maxInitRange = 0
        elif not np.issubdtype(type(value), np.integer):
            logger.error(f"TypeError: {self._maxInitRange} is not a valid input. Setting maxInitRange to 0.")
            self._maxInitRange = 0
        else:
            self._maxInitRange: int = value
        if self._maxInitRange < 0:
            logger.warning(f"ValueError: {self._maxInitRange} is smaller than 0. Setting maxInitRange to 0.")
            self._maxInitRange = 0
        elif self._maxInitRange >= self.lenInputPower:
            logger.warning("""ValueError: maxInitRange ({}) exceeds the number of data sets
({}) (different input powers).
Setting maxInitRange to max possible value.""".format(self._maxInitRange, self.lenInputPower))
            self._maxInitRange = self.lenInputPower
        logger.debug(f"maxInitRange has been set to {self._maxInitRange}")

    @property
    def fitModel(self):
        """The fit model that shall be used. Check peak_fit/single_peak_fit_models.py for
        more information about the available fit models.

        When creating a new fit model do not forget to add it to the imports and to _fitModelList.
        """
        return self._fitModel

    @fitModel.setter
    def fitModel(self, value) -> None:
        logger.debug(f"Setting fitModel to {value}.")
        self._fitModel = value
        if not issubclass(self._fitModel, PeakFitSuper):
            logger.error(f"ValueError: {self._fitModel} is not a subclass of PeakFitSuper. The Lorentzian model will be used.")
            self._fitModel = LorentzPeakFit
        logger.debug(f"fitModel has been set to {self._fitModel}")

    def check_input_fitmodel(self, value: str) -> None:
        """Tries to transform a string to a fit model.
        """
        logger.debug("Calling check_input_fitmodel()")
        if value.upper() in self._fitModelList.keys():
            self.fitModel = self._fitModelList[value.upper()]
        else:
            logger.error(f"ValueError: {value} is not a valid model (gauss, lorentz, voigt and pseudovoigt are implemented). Using Lorentz now.")
            self.fitModel = LorentzPeakFit
        logger.debug(f"fitModel has been set to {value}")

    @property
    def exclude(self) -> list[number]:
        """List of indices of the inputpower that shall be excluded.
        """
        return self._exclude

    @exclude.setter
    def exclude(self, value: typing.Union[list[number], None]) -> None:
        logger.debug(f"Setting exclude to {value}.")
        self._exclude: list[number]
        if value is None:
            self._exclude = []
        if isinstance(value, (list, np.ndarray)):
            self._exclude = value
        else:
            logger.error(f"TypeError: {self.exclude} is not valid input. Excluding no points.")
            self._exclude = []
        logger.debug(f"exclude has been set to {self._exclude}")

    def check_input_exclude(self, minInputPowerRangeStr: str, maxInputPowerRangeStr: str) -> None:
        """Checks the user input for exclude. Transforms input powers to indices. Using this method
        only tails can be cut, but one cannot exclude data points in the middle.
        """
        logger.debug("Calling check_input_exclude()")
        self.minInputPowerRange: floatOrNone = misc.float_decode(minInputPowerRangeStr)
        self.maxInputPowerRange: floatOrNone = misc.float_decode(maxInputPowerRangeStr)
        if self.minInputPowerRange is None:
            self.minInputPowerRange = self.minInputPower
            logger.error(f"Invalid input for min input power. Using the min possible value {self.minInputPowerRange}")
        if self.maxInputPowerRange is None:
            self.maxInputPowerRange = self.maxInputPower
            logger.error(f"Invalid input for max input power. Using the max possible value {self.maxInputPowerRange}")
        minIdx: int = self.inputPower_to_idx(self.minInputPowerRange)
        maxIdx: int = self.inputPower_to_idx(self.maxInputPowerRange)
        logger.debug(f"input_exclude(), Exclude: [minIdx: {minIdx}, maxIdx: {maxIdx}]")
        self.exclude: list[number] = list(range(0, minIdx)) + list(range(maxIdx, self.lenInputPower))

    @property
    def fitRangeScale(self) -> number:
        """For more information look into peak_fit/single_peak_fit_base.py
        """
        return self._fitRangeScale

    @fitRangeScale.setter
    def fitRangeScale(self, value: number) -> None:
        logger.debug(f"Setting fitRangeScale to {value}.")
        self._fitRangeScale: number = value
        if not ( np.issubdtype(type(value), np.integer) or np.issubdtype(type(value), np.floating) ):
            logger.error(f"TypeError: {value} is not a valid argument. Setting fitRangeScale to 1.")
            self._fitRangeScale = 1
        if not (0 <= self._fitRangeScale):
            logger.warning(f"ValueError: {value} is out of bounce [0, inf). Setting fitRangeScale to 1.")
            self._fitRangeScale = 1
        logger.debug(f"fitRangeScale has been set to {self._fitRangeScale}")

    @property
    def intCoverage(self) -> number:
        """For more information look into peak_fit/single_peak_fit_base.py
        """
        return self._intCoverage

    @intCoverage.setter
    def intCoverage(self, value: number) -> None:
        logger.debug(f"Setting intCoverage to {value}.")
        self._intCoverage: number = value
        if not ( np.issubdtype(type(value), np.integer) or np.issubdtype(type(value), np.floating) ):
            logger.error(f"TypeError: {value} is not a valid argument. Setting intCoverage to 1.")
            self._intCoverage = 1
        if not (0 <= self._intCoverage <= 1):
            logger.warning(f"ValueError: {value} is out of bounce [0, 1]. Setting intCoverage to 1.")
            self._intCoverage = 1
        logger.debug(f"intCoverage has been set to {self._intCoverage}")

    @property
    def constantPeakWidth(self) -> int:
        """For more information look into peak_fit/single_peak_fit_base.py
        """
        return self._constantPeakWidth

    @constantPeakWidth.setter
    def constantPeakWidth(self, value: int) -> None:
        logger.debug(f"Setting constantPeakWidth to {value}.")
        self._constantPeakWidth: int = value
        if not ( np.issubdtype(type(value), np.integer) ):
            logger.error(f"TypeError: {value} is not a valid argument. Setting constantPeakWidth to 50.")
            self._constantPeakWidth = 50
        if not (0 <= self._constantPeakWidth <= self.energies.size / 2):
            logger.warning(f"ValueError: {value} is out of bounce [0, {self.energies.size / 2}]. Setting constantPeakWidth to 50.")
            self._constantPeakWidth = 50
        logger.debug(f"constantPeakWidth has been set to {self._constantPeakWidth}")

    @property
    def outputPowerArr(self) -> np.ndarray:
        """Outputpower array scaled by powerScale.
        """
        return self.powerScale * self._outputPowerArr

    @property
    def uncOutputPowerArr(self) -> np.ndarray:
        """Uncertainty outputpower array scaled by powerScale.
        """
        return self.powerScale * self._uncOutputPowerArr

    @property
    def powerScale(self) -> number:
        """Scales the outputpower by multiplying the array with this number.

        This is used to stitch multiple data sets with different OD-filters together.
        """
        return self._powerScale

    @powerScale.setter
    def powerScale(self, value: number) -> None:
        logger.debug(f"Setting powerScale to {value}.")
        self._powerScale: number = value
        if not ( np.issubdtype(type(value), np.integer) or np.issubdtype(type(value), np.floating) ):
            logger.error(f"TypeError: {value} is not a valid argument. Setting powerScale to 1.")
            self._powerScale = 1
        if self._powerScale <= 0:
            logger.warning(f"ValueError: {self._powerScale} is out of bounce. Setting powerScale to 1.")
            self._powerScale = 1
        logger.debug(f"powerScale has been set to {self._powerScale}")

    @staticmethod
    def select_debugging_plots(Fit):
        """When snapshots are enabled, then this method is used to plot the snapshots.

        The settings in the config/debugging.ini file are applied.
        Hence, the fwhm/initial range or the fit range are shown if selected.
        """
        if hasattr(Fit, "p"):
            if loggerObj._debugFitRange:
                Fit.plot_fitRange_with_fit()
            if loggerObj._debuginitialRange:
                Fit.plot_initRange_with_fit()
            if loggerObj._debugFWHM:
                Fit.plot_fwhm()
            if not ( loggerObj._debugFitRange or loggerObj._debuginitialRange or loggerObj._debugFWHM ):
                Fit.plot_fit(block=True)
        else:
            if loggerObj._debugFitRange:
                Fit.plot_fitRange_without_fit()
            if loggerObj._debuginitialRange:
                Fit.plot_initRange_without_fit()
            if loggerObj._debugFWHM:
                Fit.plot_fwhmEstimate()

    def get_power_dependent_data(self):
        """The main method, performs the evaluation of the powerseries.

        First, the config/debugging.ini file is read.
        Then peak_fit with the specified fit model is used to fit every spectra
        except those on the exclude list.
        The peak_fit is instantiated with all the adjustable parameters.

        The outputpower, linewidth, mode energy and Q-factor and their respectable uncertainties 
        are saved in an array.

        In the process snapshots are shown.
        """
        logger.debug("Calling get_power_dependent_data")
        loggerObj.read_debugging_ini_file()
        if self.snapshots > 0:
            snap = self.snapshots
        else:
            snap = self.lenInputPower + 1
        self.linewidthArr = np.empty(self.lenInputPower)
        self._outputPowerArr = np.empty(self.lenInputPower)
        self.modeWavelengthArr = np.empty(self.lenInputPower)
        self.QFactorArr = np.empty(self.lenInputPower)
        self.uncLinewidthArr = np.empty(self.lenInputPower)
        self.uncModeWavelengthArr = np.empty(self.lenInputPower)
        self._uncOutputPowerArr = np.empty(self.lenInputPower)
        self.uncQFactorArr = np.empty(self.lenInputPower)
        for i in range(self.lenInputPower):
            if i in self.exclude:
                logger.debug(f"Excluding entry [{i}], input power: {round(self.inputPower[i], 3)} mW")
                self.modeWavelengthArr[i] = np.nan
                self._outputPowerArr[i] = np.nan
                self.linewidthArr[i] = np.nan
                self.QFactorArr[i] = np.nan
                self.uncLinewidthArr[i] = np.nan
                self.uncModeWavelengthArr[i] = np.nan
                self._uncOutputPowerArr[i] = np.nan
                self.uncQFactorArr[i] = np.nan
            else:
                logger.debug(f"Fitting entry [{i}], input power {round(self.inputPower[i], 3)} mW")
                logger.debug(f"i = {i}, snap = {snap}")
                if i <= self.maxInitRange:
                    logger.debug(f"maxInitRange = {self.maxInitRange}")
                    Fit = self._fitModel(self.energies, self.data[i], intCoverage=self.intCoverage,
initRange=self.initRange, fitRangeScale=self.fitRangeScale, constantPeakWidth=self.constantPeakWidth,
backgroundFitMode=self.backgroundFitMode)
                else:
                    Fit = self._fitModel(self.energies, self.data[i], intCoverage=self.intCoverage,
initRange=None, fitRangeScale=self.fitRangeScale, constantPeakWidth=self.constantPeakWidth,
backgroundFitMode=self.backgroundFitMode)
                Fit.run()
                if i % snap == 0 and snap <= self.lenInputPower:
                    logger.info(f"[{i}] input power: {round(self.inputPower[i], 3)} mW")
                    try:
                        logger.info(f"mode energy: {self.energies[Fit.peak]} eV")
                    except AttributeError:
                        pass
                    self.select_debugging_plots(Fit)
                modeWavelength, linewidth, outputPower = Fit.outputParameters
                uncModeWavelength, uncLinewidth, uncOutputPower = Fit.uncertaintyOutputParameters
                QFactor = modeWavelength / linewidth
                uncQFactor = np.sqrt((uncModeWavelength / linewidth)**2 
                + (uncLinewidth * modeWavelength / linewidth**2)**2)

                logger.debug("""Results:
    mu = {} \u00B1 {}
    fwhm = {} \u00B1 {}
    integrated Intensity = {} \u00B1 {}
    Q-factor = {} \u00B1 {}""".format(modeWavelength, uncModeWavelength,
                                    linewidth, uncLinewidth,
                                    outputPower, uncOutputPower,
                                    QFactor, uncQFactor))

                self.modeWavelengthArr[i] = modeWavelength
                self._outputPowerArr[i] = outputPower
                self.linewidthArr[i] = linewidth
                self.QFactorArr[i] = QFactor
                self.uncQFactorArr[i] = uncQFactor
                self.uncLinewidthArr[i] = uncLinewidth
                self.uncModeWavelengthArr[i] = uncModeWavelength
                self._uncOutputPowerArr[i] = uncOutputPower
        ## remove nan values
        indicesNotNaN = np.argwhere(~np.isnan(self._outputPowerArr))
        self.linewidthArr = self.linewidthArr[indicesNotNaN][:, 0]
        self._outputPowerArr = self._outputPowerArr[indicesNotNaN][:, 0]
        self.modeWavelengthArr = self.modeWavelengthArr[indicesNotNaN][:, 0]
        self.QFactorArr = self.QFactorArr[indicesNotNaN][:, 0]
        self._uncOutputPowerArr = self._uncOutputPowerArr[indicesNotNaN][:, 0]
        self.uncLinewidthArr = self.uncLinewidthArr[indicesNotNaN][:, 0]
        self.uncModeWavelengthArr = self.uncModeWavelengthArr[indicesNotNaN][:, 0]
        self.uncQFactorArr = self.uncQFactorArr[indicesNotNaN][:, 0]
        self.inputPowerPlotArr = self.inputPower[indicesNotNaN][:, 0]
        self.lenInputPowerPlot = self.inputPowerPlotArr.size

    def access_single_spectrum(self, idx):
        """Accesses a single spectrum out of the powerseries data set.

        This is used in plot_ps.py to plot one spectrum
        Returns
        -------
        PeakFitSuper object
        """
        logger.debug("Calling access_single_spectrum()")
        assert 0 <= idx < self.lenInputPower
        return self._fitModel(self.energies, self.data[idx], intCoverage=self.intCoverage, initRange=self.initRange,
fitRangeScale=self.fitRangeScale, constantPeakWidth=self.constantPeakWidth, backgroundFitMode=self.backgroundFitMode)



if __name__ == "__main__":
    ## just testing
    pass
    # dataDirPath = (Path(__file__).parents[1] / "data").resolve()
    # head = (Path(__file__).parents[2]).resolve()
    # fileName = "data\\20210303\\NP7509_Ni_4µm_20K_Powerserie_1-01s_deteOD0_fine3_WithoutLensAllSpectra.dat"
    # fileName = fileName.replace("\\", "/")
    # filePath = (head / fileName).resolve()
    # test = EvalPowerSeries(filePath)
    # test.initRange = 500, 700
    # test.get_power_dependent_data()
    # print(test.modeWavelength)
    # test.plot_lw_s()
    # test.plot_outputPower()
    # test.plot_linewidth()
    # test.plot_mode_wavelength()
    # test.plot_QFactor()

    # for dir in os.listdir(str(fineDataDirPath)):
    #     dirPath = (fineDataDirPath / dir).resolve()
    #     for file in os.listdir(str(dirPath)):
    #         print(file)
    #         dataPath = (dirPath / file ).resolve()
    #         test = EvalPowerSeries(file, dataPath, LorentzPeakFit)
    #         test.plot_outputPower()
    #         test.plot_linewidth()