import pandas as pd # type: ignore
import numpy as np
import sys
import typing
from pathlib import Path

headDirPath = Path(__file__).resolve().parents[1]
sys.path.append(str(headDirPath))

from setup.config_logging import LoggingConfig
from utils.filename_analysis import FileNameReader

loggerObj: LoggingConfig = LoggingConfig()
logger = loggerObj.init_logger(__name__)

number = typing.Union[int, float, np.number]

class DataQlab2:
    rowsToSkip: list[int] = [1, 3]
    name: str = "QLAB2"

    def __init__(self, dataPath: Path) -> None:
        assert isinstance(dataPath, Path)
        assert dataPath.exists()
        try:
            self.data: pd.DataFrame = pd.read_csv(dataPath, sep="\t", skiprows=lambda x: x in self.rowsToSkip)
        except:
            logger.exception(f"Data from {str(dataPath)} could not be read.")
            raise ValueError("Data from file could not be read.")
        else:
            self.wavelengths: np.ndarray = self.data["Wavelength"].to_numpy()[1:]
            self.energies: np.ndarray = self.data["Energy"].to_numpy()[1:][::-1]
            self.data.drop(["Energy", "Wavelength"], axis=1, inplace=True)
            self.inputPower: np.ndarray = self.data.loc[0].to_numpy()*4
            self.lenInputPower: int = self.inputPower.size
            self.data.drop([0], axis=0, inplace=True)
            self.columns: list = self.data.columns
            self.strFilePath: str = str(dataPath)
            self.maxEnergy: number = np.amax(self.energies)
            self.minEnergy: number = np.amin(self.energies)
            self.minInputPower: number = np.amin(self.inputPower)
            self.maxInputPower: number = np.amax(self.inputPower)
            self.fileName: str = dataPath.name
            assert len(self.data.columns) == self.lenInputPower, f"{len(self.data.columns)}, {self.lenInputPower}"
            logger.info(f"Data from {self.fileName} successfully loaded.")

    def __getitem__(self, idx: int) -> number:
        return self.data[self.columns[idx]].to_numpy()[::-1]

    def __len__(self) -> int:
        return self.lenInputPower

    @property
    def diameter(self) -> str:
        DiameterReader: FileNameReader = FileNameReader("diameter", "µm", "_", indicatorAtStart=False)
        return DiameterReader(self.fileName)

    @property
    def temperature(self) -> str:
        TemperatureReader: FileNameReader = FileNameReader("temperature", "K_", "_", indicatorAtStart=False)
        return TemperatureReader(self.fileName)

if __name__ == "__main__":
    testFileName = "NP7509_Ni_4µm_20K_Powerserie_1-01s_deteOD05_fine4AllSpectra.dat"
    testPath = (headDirPath / ".." / "data" / "20210308" / testFileName).resolve()
    print(testPath)
    test = DataQlab2(testPath)
    print(test.diameter)
    print(test.temperature)