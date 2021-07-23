"""This file contains the data formats of the powerseries.
Currently, only Qlab2 data is implemented.

Any new version of a data format has to have the following attributes.

energies, wavelengths, inputPower, maxEnergy, minEnergy, minInputPower, maxInputPower, lenInputPower, fileName, __getitem__
"""
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
    """
    Class that reads the data from a file created by the Qlab2 setup. (Intensity per wavelength for different input powers)

    Upon initialization the path to the data has to be provided. All attributes of this class are set
    after a successful initialization.

    Attributes
    ----------
    dataPath: pathlib.Path
        path to the to be read data

    data: pd.DataFrame
        dataframe of the complete file that has been read

    wavelengths: np.ndarray
        Contains the wavelengths from the measurement [nm]

    energies: np.ndarray
        Contains the energies from the measurement [eV]

    inputPower: np.ndarray
        Contains the input powers from the measurement [mW].
        The values are multiplied by 4 to account for the beamsplitter.

    lenInputPower: int
        number of different input powers

    columns: list
        list of the names of the columns, used to extract columns from the dataframe

    maxEnergy: float/int

    minEnergy: float/int

    minInputPower: float/int

    maxInputPower: float/int

    fileName: str

    Raises
    ------
    AssertionError:
        When the dataPath does not exist or is of invalid datatype.
        And when the number of input powers differs from the number of intensity columns.

    ValueError:
        When the data from the file could not be read from the file using pandas.read_csv()
    """
    rowsToSkip: list[int] = [1, 3]
    """list[int]: Specific rows that are skipped when reading the data.
    """
    name: str = "QLAB2"
    """str: name of this class, used for identification purposes
    """

    def __init__(self, dataPath: Path) -> None:
        assert isinstance(dataPath, Path)
        assert dataPath.exists()
        self.dataPath = dataPath
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
            self.maxEnergy: number = np.amax(self.energies)
            self.minEnergy: number = np.amin(self.energies)
            self.minInputPower: number = np.amin(self.inputPower)
            self.maxInputPower: number = np.amax(self.inputPower)
            self.fileName: str = self.dataPath.name
            assert len(self.data.columns) == self.lenInputPower, f"{len(self.data.columns)}, {self.lenInputPower}"
            logger.info(f"Data from {self.fileName} successfully loaded.")

    def __getitem__(self, idx: int) -> np.ndarray:
        """Get the intensities for each wavelength for one input power.

        Returns
        -------
        np.ndarray:
            array of intensities
        """
        return self.data[self.columns[idx]].to_numpy()[::-1]

    @property
    def diameter(self) -> str:
        """The diameter of the pillar, extracted from the filename.
        """
        DiameterReader: FileNameReader = FileNameReader("diameter", "µm", "_", indicatorAtStart=False)
        return DiameterReader(self.fileName)

    @property
    def temperature(self) -> str:
        """The temperature of the measurement, extracted from the filename.
        """
        TemperatureReader: FileNameReader = FileNameReader("temperature", "K_", "_", indicatorAtStart=False)
        return TemperatureReader(self.fileName)

if __name__ == "__main__":
    testFileName = "NP7509_Ni_4µm_20K_Powerserie_1-01s_deteOD05_fine4AllSpectra.dat"
    testPath = (headDirPath / ".." / "data" / "20210308" / testFileName).resolve()
    print(testPath)
    test = DataQlab2(testPath)
    print(test.diameter)
    print(test.temperature)