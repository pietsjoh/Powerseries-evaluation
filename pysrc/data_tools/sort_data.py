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
import utils.misc as misc

loggerObj: LoggingConfig = LoggingConfig()
logger = loggerObj.init_logger(__name__)

number = typing.Union[float, int]
floatOrNone = typing.Union[float, None]
class SortData:
    _dataModelList: dict = {"QLAB2" : DataQlab2}
    dataDirPath: Path = (headDirPath / "data").resolve()

    def __init__(self) -> None:
        logger.debug("Initializing SortData object.")
        self.read_data_format_ini_file()
        if self.dataModel.name == "QLAB2":
            self.originalPathsGen = self.dataDirPath.rglob("*AllSpectra.dat")
        else:
            raise NotImplementedError

    def read_data_format_ini_file(self) -> None:
        logger.debug("Calling read_data_format_ini_file()")
        configIniPath: Path = (headDirPath / "config" / "data_format.ini").resolve()
        config: ConfigParser = ConfigParser()
        config.read(str(configIniPath), encoding="UTF-8")
        self.distinguishFullFineSpectra: bool = LoggingConfig.check_true_false(
            config["data format"]["distinguish full fine spectra"].replace(" ", ""))
        if self.distinguishFullFineSpectra:
            self.borderFullFineWavelength: floatOrNone = misc.float_decode(config["data format"]["full fine border"].replace(" ", ""))
            if self.borderFullFineWavelength is None:
                logger.error("Invalid argument for full fine border in the .ini file.")
                exit()
        self.useAttribute: bool = LoggingConfig.check_true_false(
            config["data format"]["use attribute"].replace(" ", ""))
        if self.useAttribute:
            self.attrName: str = config["data format"]["attribute name"].replace(" ", "")
            sortedDataDirName: str = f"sorted_data_{self.attrName}"
            self.sortedDataDirPath: Path = (headDirPath / sortedDataDirName).resolve()
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
            self.sortedDataDirPath: Path = (headDirPath / "sorted_data").resolve()
        else:
            logger.error("According to the .ini file, no sorting shall be done. Aborting.")
            exit()
        try:
            self.dataModel = config["data format"]["data model"].replace(" ", "")
        except AssertionError:
            raise TypeError("Config file value for data model is invalid.")

    @property
    def dataModel(self):
        return self._dataModel

    @dataModel.setter
    def dataModel(self, value: str):
        logger.debug(f"Setting dataModel to {value}.")
        if not value.upper() in self._dataModelList.keys():
            logger.error(f"{value} is not a valid data model (subclass of DataSuper). Aborting.")
            raise AssertionError("Invalid data model.")
        else:
            self._dataModel = self._dataModelList[value.upper()]

    def sort_data_attribute(self) -> None:
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
        logger.debug("Calling sort_data_fine")
        assert self.distinguishFullFineSpectra
        assert isinstance(dirPath, Path)
        assert dirPath.exists()
        pathsList: list[Path] = dirPath.rglob("*AllSpectra.dat")
        fineSpectraPath: Path = (dirPath / "fine_spectra").resolve()
        fullSpectraPath: Path = (dirPath / "full_spectra").resolve()
        filePath: Path
        for filePath in pathsList:
            data = self.dataModel(filePath)
            wavelengthRange: number = max(data.wavelengths) - min(data.wavelengths)
            if wavelengthRange <= self.borderFullFineWavelength:
                if not fineSpectraPath.exists():
                    fineSpectraPath.mkdir()
                newFilePath: Path = (fineSpectraPath / filePath.name).resolve()
                shutil.move(filePath, newFilePath)
            else:
                if not fullSpectraPath.exists():
                    fullSpectraPath.mkdir()
                newFilePath: Path = (fullSpectraPath / filePath.name).resolve()
                shutil.move(filePath, newFilePath)

    def run(self) -> None:
        if self.useAttribute:
            self.sort_data_attribute()
            if self.distinguishFullFineSpectra:
                dirPath: Path
                dirPathGen = self.sortedDataDirPath.glob("*")
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
    runSortData: SortData = SortData()
    runSortData.run()


if __name__ == "__main__":
    main()