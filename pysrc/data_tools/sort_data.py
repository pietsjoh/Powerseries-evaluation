import sys
from pathlib import Path
import shutil
from configparser import ConfigParser
import typing

headDirPath: Path = Path(__file__).resolve().parents[2]
srcDirPath: Path = (headDirPath / "pysrc").resolve()
sys.path.append(str(srcDirPath))
from utils.filename_analysis import FileNameReader
from setup.config_logging import LoggingConfig
from data_tools.data_formats import DataQlab2

loggerObj: LoggingConfig = LoggingConfig()
logger = loggerObj.init_logger(__name__)

class SortData:
    _dataModelList = {"QLAB2" : DataQlab2}
    dataDirPath: Path = (headDirPath / "data").resolve()

    def __init__(self):
        logger.debug("Initializing SortData object.")
        self.read_data_format_ini_file()
        if self.dataModel.name == "QLAB2":
            self.PathsList: list = list(self.dataDirPath.rglob("*AllSpectra.dat"))
        else:
            raise NotImplementedError

    def read_data_format_ini_file(self):
        logger.debug("Calling read_data_format_ini_file()")
        configIniPath: Path = (headDirPath / "config" / "data_format.ini").resolve()
        config: ConfigParser = ConfigParser()
        config.read(str(configIniPath), encoding="UTF-8")
        self.attrName: str = config["data format"]["attribute name"].replace(" ", "")
        sortedDataDirName: str = f"sorted_data_{self.attrName}"
        self.sortedDataDirPath: Path = (headDirPath / sortedDataDirName).resolve()
        self.distinguishFullFineSpectra: bool = LoggingConfig.check_true_false(
        config["data format"]["distinguish full fine spectra"].replace(" ", ""))
        indicator: str = config["data format"]["indicator"].replace(" ", "")
        splitter: str = config["data format"]["splitter"].replace(" ", "")
        indicatorAtStart: bool = LoggingConfig.check_true_false(
            config["data format"]["indicator at start"])
        self.possibleAttrList: typing.Union[list[str], None] = config["data format"]["attribute possibilities"].replace(" ", "").split(",")
        try:
            self.dataModel = config["data format"]["data model"].replace(" ", "")
        except AssertionError:
            raise TypeError("Config file value for data model is invalid.")
        self.AttrReader: FileNameReader = FileNameReader(name=self.attrName, indicator=indicator,
            splitter=splitter, indicatorAtStart=indicatorAtStart)

        logger.debug(f"""Read attributes from data_format.ini file:
    attrName: {self.attrName}
    sortedDataDirPath: {self.sortedDataDirPath}
    distinguishFullFineSpectra: {self.distinguishFullFineSpectra}
    indicator: {indicator}
    splitter: {splitter}
    indicatorAtStart: {indicatorAtStart}
    possibleAttrList: {self.possibleAttrList}
    dataModel: {self.dataModel.name}""")

        if len(self.possibleAttrList) == 1 and self.possibleAttrList[0].upper() == "NONE":
            self.possibleAttrList = None

    @property
    def dataModel(self):
        return self._dataModel

    @dataModel.setter
    def dataModel(self, value):
        logger.debug(f"Setting dataModel to {value}.")
        if not value.upper() in self._dataModelList.keys():
            logger.error(f"{value} is not a valid data model (subclass of DataSuper). Aborting.")
            raise AssertionError("Invalid data model.")
        else:
            self._dataModel = self._dataModelList[value.upper()]

    def sort_data_attribute(self):
        logger.debug("Calling sort_data_attribute()")
        if not self.sortedDataDirPath.exists():
            self.sortedDataDirPath.mkdir()

        for filePath in self.PathsList:
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
                shutil.copy2(str(filePath), str(newFilePath))
def main():
    runSortData: SortData = SortData()
    runSortData.sort_data_attribute()


if __name__ == "__main__":
    main()

    # diameterIndicator = "NP7509_NI_"
    # fineBorder = 30 ##nm

    # HeadDirPath = Path(__file__).resolve().parents[2]
    # srcDirPath = (HeadDirPath / "pysrc").resolve()
    # sys.path.append(str(srcDirPath))
    # import utils.console_commands as console
    # import utils.misc as misc

    # dataDirPath = (HeadDirPath / "data").resolve()

    # sortDataDirPath = (HeadDirPath / "sorted_data").resolve()

    # spectraPaths = list(dataDirPath.rglob("*AllSpectra.dat"))

    # ## sort QLab-2 data only (Allspectra.dat)
    # ## first: sort data by diameter

    # if not sortDataDirPath.exists():
    #     sortDataDirPath.mkdir()

    # for path in spectraPaths:
    #     strFileName = str(path)
    #     # print(strFileName)
    #     if "tesst" in strFileName or "fail" in strFileName:
    #         continue
    #     strFileNameUpper = strFileName.upper()
    #     fileNameIndex = strFileNameUpper.find(diameterIndicator.upper())
    #     diameterIndex = fileNameIndex + len(diameterIndicator)
    #     truncatedStrFileName = strFileName[diameterIndex : ]
    #     diameter = misc.diameter_decode(truncatedStrFileName, returnStr=True)

    #     diameterDirPath = (sortDataDirPath / diameter).resolve()
    #     if not diameterDirPath.exists():
    #         diameterDirPath.mkdir()
    #     strParentDir = str(path.parent).split("/")[-1]
    #     oldFileName = strFileName[fileNameIndex : ]
    #     # print(oldFileName)
    #     newFileName = strParentDir + "_" + strFileName[fileNameIndex : ]
    #     # print(newFileName)
    #     # print()
    #     newFilePath = (diameterDirPath / newFileName).resolve()
    #     if newFilePath.exists():
    #         continue
    #     else:
    #         console.cp(strFileName, str(diameterDirPath))
    #         console.mv(str((diameterDirPath / oldFileName).resolve()), str(newFilePath))

    # ## second sort data by full_spectra and fine_spectra

    # diameterDirs = os.listdir(str(sortDataDirPath))
    # for diameter in diameterDirs:
    #     diameterPath = (sortDataDirPath / diameter).resolve()
    #     files = os.listdir(str(diameterPath))
    #     fineDataPath = (diameterPath / "fine_spectra").resolve()
    #     fullDataPath = (diameterPath / "full_spectra").resolve()
    #     for file in files:
    #         if file == "fine_spectra" or file == "full_spectra":
    #             continue
    #         fileName = file.split("/")[-1]
    #         print(fileName)
    #         filePath = (diameterPath / fileName).resolve()
    #         rowsToSkip = [1, 3]
    #         data = pd.read_csv(filePath, sep="\t", skiprows=lambda x: x in rowsToSkip)
    #         wavelengths = data["Wavelength"].to_numpy()[1:]
    #         wavelengthRange = abs(wavelengths[-1] - wavelengths[0])
    #         if wavelengthRange <= fineBorder:
    #             if not fineDataPath.exists():
    #                 fineDataPath.mkdir()
    #             console.mv(str(filePath), str((fineDataPath / fileName).resolve()))
    #         else:
    #             if not fullDataPath.exists():
    #                 fullDataPath.mkdir()
    #             console.mv(str(filePath), str((fullDataPath / fileName).resolve()))

