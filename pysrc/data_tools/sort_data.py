"""Contains the class SortData, which can be used to sort measurement data.
"""
import sys
from pathlib import Path
import shutil
from configparser import ConfigParser
import typing

headDirPath: Path = Path(__file__).resolve().parents[2]
srcDirPath: Path = (headDirPath / "pysrc").resolve()
sys.path.append(str(srcDirPath))
from data_tools.filename_analysis import FileNameReader
from setup.config_logging import LoggingConfig
from data_tools.data_formats import DataQlab2

loggerObj: LoggingConfig = LoggingConfig()
logger = loggerObj.init_logger(__name__)

number = typing.Union[float, int]
floatOrNone = typing.Union[float, None]
class SortData:
    """Sorts the data files in the data directory.

    The data can be sorted by an attribute in the filename (for example: diameter, temperature, ...).
    Moreover, the class can sort the data based on the wavelength range (full/fine).
    So one can easily differentiate between different gratings which are used.
    Currently, one can only differentiate between 2 different gratings.

    Attributes
    ----------
    _dataModelDict: dict, class variable
        Contains the available data models (currently only Qlab2)

    _dataDirPath: pathlib.Path, class variable
        path to the data directory

    originalPathsGen: Generator, set by __init__
        Generator of all *AllSpectra.dat files in the data directory.

    distinguishFullFineSpectra: bool, set by read_data_format_ini_file
        when set to True, then the files will be sorted based on their wavelength range
        into fine_spectra/full_spectra folders

    borderFullFineWavelength: float, set by read_data_format_ini_file
        border wavelength between full and fine spectra in nm.
        If the wavelength range of the file is larger than this border then it is sorted into the full_spectra folder

    useAttribute: bool, set by read_data_format_ini_file
        if True then sorting by attribute is enabled

    attrName: str, set by read_data_format_ini_file
        name of the attribute after that shall be sorted, used for naming the top level directory

    sortedDataDirPath: str, set by read_data_format_ini_file
        top level directory path to the sorted data

    possibleAttrList: List/None, set by read_data_format_ini_file
        List of possible values for the attribute
        If set to None, then any value is accepted

    AttrReader: data_tools.filename_analysis.FileNameReader, set by read_data_format_ini_file
        FileNameReader object that is used to extract the information about the attribute from the filenames

    Raises
    ------
    NotImplementedError:
        When the datamodel is not QLAB2
    """
    _dataModelDict: dict = {"QLAB2" : DataQlab2}
    _dataDirPath: Path = (headDirPath / "data").resolve()

    def __init__(self) -> None:
        logger.debug("Initializing SortData object.")
        self.read_data_format_ini_file()
        if self.dataModel.name == "QLAB2":
            self.originalPathsGen: typing.Generator[Path, None, None] = self._dataDirPath.rglob("*AllSpectra.dat")
        else:
            raise NotImplementedError

    def read_data_format_ini_file(self) -> None:
        """Reads the config/data_format.ini file and saves the information in the form of attributes.

        Raises
        ------
        TypeError:
            invalid input for datamodel
        """
        logger.debug("Calling read_data_format_ini_file()")
        configIniPath: Path = (headDirPath / "config" / "data_format.ini").resolve()
        config: ConfigParser = ConfigParser()
        config.read(str(configIniPath), encoding="UTF-8")
        self.distinguishFullFineSpectra: bool = LoggingConfig.check_true_false(
            config["data format"]["distinguish full fine spectra"].replace(" ", ""))
        if self.distinguishFullFineSpectra:
            try:
                self.borderFullFineWavelength: float = float(config["data format"]["full fine border"].replace(" ", ""))
            except ValueError:
                logger.error("Invalid argument for full fine border in the .ini file (no float).")
                exit()
        self.useAttribute: bool = LoggingConfig.check_true_false(
            config["data format"]["use attribute"].replace(" ", ""))
        self.sortedDataDirPath: Path
        if self.useAttribute:
            self.attrName: str = config["data format"]["attribute name"].replace(" ", "")
            sortedDataDirName: str = f"sorted_data_{self.attrName}"
            self.sortedDataDirPath = (headDirPath / sortedDataDirName).resolve()
            indicator: str = config["data format"]["indicator"].replace(" ", "")
            splitter: str = config["data format"]["splitter"].replace(" ", "")
            indicatorAtStart: bool = LoggingConfig.check_true_false(
                config["data format"]["indicator at start"])
            self.possibleAttrList: typing.Union[list[str], None] = config["data format"]["attribute possibilities"].replace(" ", "").split(",")
            self.AttrReader: FileNameReader = FileNameReader(name=self.attrName, indicator=indicator,
                splitter=splitter, indicatorAtStart=indicatorAtStart)
            if len(self.possibleAttrList) == 1 and self.possibleAttrList[0].upper() == "NONE":
                self.possibleAttrList = None
        elif not self.useAttribute and self.distinguishFullFineSpectra:
            self.sortedDataDirPath = (headDirPath / "sorted_data").resolve()
        else:
            logger.error("According to the .ini file, no sorting shall be done. Aborting.")
            exit()
        try:
            self.dataModel = config["data format"]["data model"].replace(" ", "")
        except AssertionError:
            raise TypeError("Config file value for data model is invalid.")

    @property
    def dataModel(self):
        """The data format that shall be used for loading the data.

        In this class this is only used to extract the wavelength range from the measurement data.
        When setting this attribute, it is checked whether the value is a key in _dataModelDict.
        """
        return self._dataModel

    @dataModel.setter
    def dataModel(self, value: str):
        logger.debug(f"Setting dataModel to {value}.")
        if not value.upper() in self._dataModelDict.keys():
            logger.error(f"{value} is not a valid data model (subclass of DataSuper). Aborting.")
            raise AssertionError("Invalid data model.")
        else:
            self._dataModel = self._dataModelDict[value.upper()]

    def sort_data_attribute(self) -> None:
        """Sorts the data according to an attribute.

        Raises
        ------
        AssertionError:
            when self.useAttribute is not True
        """
        logger.debug("Calling sort_data_attribute()")
        assert self.useAttribute
        if not self.sortedDataDirPath.exists():
            self.sortedDataDirPath.mkdir()

        filePath: Path
        for filePath in self.originalPathsGen:
            fileName: str = filePath.name
            fileDir: str = filePath.parts[-2]
            newFileName: str = f"{fileDir}_{fileName}"

            logger.debug(f"""Information about the to be copied file:
    fileName: {fileName}
    fileDir: {fileDir}
    filePath: {filePath}""")
            if "tesst" in fileName or "fail" in fileName:
                continue
            try:
                attr: str = self.AttrReader(fileName)
            except AssertionError:
                logger.error(f"{filePath} could not be read. It will not copied into the sorted_data dir.")
                continue
            else:
                if self.possibleAttrList is not None:
                    if attr not in self.possibleAttrList:
                        logger.error(f"""Extracted attribute {attr} is not a part of the specified list
'attribute possiblities' in the data_format.ini file.""")
                        continue

                attrPath: Path = (self.sortedDataDirPath / attr).resolve()
                if not attrPath.exists():
                    attrPath.mkdir()
                newFilePath: Path = (attrPath / newFileName).resolve()

                logger.debug(f"""Information about the new file location:
    read attribute: {attr}
    attribute path: {attrPath}
    newFileName: {newFileName}
    newFilePath: {newFilePath}""")

                if newFilePath.exists():
                    logger.warning(f"File at end location [{newFilePath}] already exists. Not copying the file [{fileName}]")
                    continue
                shutil.copy2(filePath, newFilePath)

    def sort_data_fine(self, dirPath: Path) -> None:
        """Sorts the files inside dirPath into full / fine directories.

        Parameters
        ----------
        dirPath: pathlib.Path
            Path to the directory which shall be sorted.

        Raises
        ------
        AssertionError:
            when dirPath does not exist or is of invalid datatype
            when self.distinguishFullFineSpectra is not True
        """
        logger.debug("Calling sort_data_fine")
        assert self.distinguishFullFineSpectra
        assert isinstance(dirPath, Path)
        assert dirPath.exists()
        pathsGen: typing.Generator[Path, None, None] = dirPath.rglob("*AllSpectra.dat")
        fineSpectraPath: Path = (dirPath / "fine_spectra").resolve()
        fullSpectraPath: Path = (dirPath / "full_spectra").resolve()
        filePath: Path
        for filePath in pathsGen:
            data = self.dataModel(filePath)
            wavelengthRange: number = max(data.wavelengths) - min(data.wavelengths)
            newFilePath: Path
            if wavelengthRange <= self.borderFullFineWavelength:
                if not fineSpectraPath.exists():
                    fineSpectraPath.mkdir()
                newFilePath = (fineSpectraPath / filePath.name).resolve()
                shutil.move(filePath, newFilePath)
            else:
                if not fullSpectraPath.exists():
                    fullSpectraPath.mkdir()
                newFilePath = (fullSpectraPath / filePath.name).resolve()
                shutil.move(filePath, newFilePath)

    def run(self) -> None:
        """Main method, combines sort_data_attribute() and sort_data_fine().

        This method covers every combination (whether to sort by attribute or full/fine).
        """
        if self.useAttribute:
            self.sort_data_attribute()
            if self.distinguishFullFineSpectra:
                dirPath: Path
                dirPathGen: typing.Generator[Path, None, None] = self.sortedDataDirPath.glob("*")
                for dirPath in dirPathGen:
                    self.sort_data_fine(dirPath)
        else:
            if self.distinguishFullFineSpectra:
                filePath: Path
                for filePath in self.originalPathsGen:
                    if not self.sortedDataDirPath.exists():
                        self.sortedDataDirPath.mkdir()
                    fileName: str = filePath.name
                    fileDir: str = filePath.parts[-2]
                    newFileName: str = f"{fileDir}_{fileName}"
                    newFilePath: Path = (self.sortedDataDirPath / newFileName).resolve()
                    if newFilePath.exists():
                        logger.warning(f"File at end location [{newFilePath}] already exists. Not copying the file [{fileName}]")
                        continue
                    shutil.copy2(filePath, newFilePath)
                self.sort_data_fine(self.sortedDataDirPath)
            else:
                logger.error("According to the .ini file, no sorting shall be done. Aborting.")
                exit()

def main():
    """Uses the SortData class to sort the data. Is called upon running the script.
    """
    runSortData: SortData = SortData()
    runSortData.run()


if __name__ == "__main__":
    main()