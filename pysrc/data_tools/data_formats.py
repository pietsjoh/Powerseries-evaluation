import pandas as pd
import numpy as np
import sys
import pathlib

headDirPath = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(headDirPath))

from setup.config_logging import LoggingConfig
from utils.filename_analysis import FileNameReader

loggerObj = LoggingConfig()
logger = loggerObj.init_logger(__name__)

class DataQlab2:
    rowsToSkip = [1, 3]
    name = "QLAB2"

    def __init__(self, dataPath):
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
        DiameterReader = FileNameReader("diameter", "µm", "_", indicatorAtStart=False)
        return DiameterReader(self.fileName)

    @property
    def temperature(self):
        TemperatureReader = FileNameReader("temperature", "K_", "_", indicatorAtStart=False)
        return TemperatureReader(self.fileName)

if __name__ == "__main__":
    testFileName = "NP7509_Ni_4µm_20K_Powerserie_1-01s_deteOD05_fine4AllSpectra.dat"
    testPath = (headDirPath / ".." / "data" / "20210308" / testFileName).resolve()
    print(testPath)
    test = DataQlab2(testPath)
    print(test.diameter)
    print(test.temperature)