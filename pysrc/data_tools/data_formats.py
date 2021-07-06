import pandas as pd
import numpy as np
import sys
import pathlib

headDirPath = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(headDirPath))

from setup.config_logging import LoggingConfig
import utils.misc as misc

loggerObj = LoggingConfig()
logger = loggerObj.init_logger(__name__)

class DataQlab2:
    rowsToSkip = [1, 3]
    name = "DataQlab2"

    def __init__(self, dataPath, diameterIndicator="np7509_ni_", temperatureIndicator="µm_"):
        assert isinstance(dataPath, pathlib.PurePath)
        assert dataPath.exists()
        try:
            self.data = pd.read_csv(dataPath, sep="\t", skiprows=lambda x: x in self.rowsToSkip)
        except:
            logger.exception(f"Data from {str(dataPath)} could not be read.")
            raise ValueError("Data from file could not be read.")
        else:
            self.wavelengths = self.data["Wavelength"].to_numpy()[1:]
            self.energies = self.data["Energy"].to_numpy()[1:][::-1]
            self.data.drop(["Energy", "Wavelength"], axis=1, inplace=True)
            self.inputPower = self.data.loc[0].to_numpy()*9
            self.lenInputPower = self.inputPower.size
            self.data.drop([0], axis=0, inplace=True)
            self.columns = self.data.columns
            self.diameterIndicator = diameterIndicator
            self.temperatureIndicator = temperatureIndicator
            self.strFilePath = str(dataPath)
            self.maxEnergy = np.amax(self.energies)
            self.minEnergy = np.amin(self.energies)
            self.minInputPower = np.amin(self.inputPower)
            self.maxInputPower = np.amax(self.inputPower)
            self.fileName = dataPath.name
            assert len(self.data.columns) == self.lenInputPower, f"{len(self.data.columns)}, {self.lenInputPower}"
            logger.info(f"Data from {self.fileName} successfully loaded.")

    def __getitem__(self, number):
        return self.data[self.columns[number]].to_numpy()[::-1]

    def __len__(self):
        return self.lenInputPower

    @property
    def diameter(self):
        strFileNameUpper = self.strFilePath.upper()
        fileNameIdx = strFileNameUpper.find(self.diameterIndicator.upper())
        diameterIdx = fileNameIdx + len(self.diameterIndicator)
        truncatedStrFileName = self.strFilePath[diameterIdx : ]
        diameter = misc.diameter_decode(truncatedStrFileName, returnStr=False)
        return diameter

    @property
    def temperature(self):
        strFileNameUpper = self.strFilePath.upper()
        fileNameIdx = strFileNameUpper.find(self.temperatureIndicator.upper())
        tempStartIdx = fileNameIdx + len(self.temperatureIndicator)
        truncatedStrFileName = self.strFilePath[tempStartIdx : ]
        tempStr = truncatedStrFileName.split("K")[0]
        try:
            temperature = int(tempStr)
        except ValueError:
            logger.error(f"ValueError: temperature ({tempStr}) could not be extracted from file name. Setting it to None.")
            temperature = None
        finally:
            return temperature