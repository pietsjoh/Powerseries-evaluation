"""This script contains everything to set up the logging configuration as well as the debugging configuration.
Upon execution the config/logging.ini file is read and the setup/config_logging.json file is updated accordingly.
This defines the logging setup. Executed during build process.
"""
import json
from pathlib import Path
from configparser import ConfigParser
import logging
import logging.config
import typing

listOrNone = typing.Union[list, None] # used for type hint

class LoggingConfig:
    """
    Contains the methods to provide the logging and debugging setup.

    This class can be used to read the config/logging.ini file and create a setup/config_logging.json file.
    Moreover, the config/debugging.ini is read.
    Finally, a logger object can be created which should be used to create logging messages.

    Attributes
    ----------
    _headDirPath: Path, class variable
        Path to the head directory containing the pysrc and scripts directory.

    _configLogJsonPath: Path, class variable
        Path to the config_logging.json file. This file contains the logging setup in a format
        that can easily be read by the logging library.

    _logsDirPath: Path, class variable
        Path to the logs directory (log files are stored there)

    _loggingLevelList: list[str], class variable
        list of available logging levels. This used to check whether invalid options are provided in the config file.

    _configFileBase: dict, class variable
        Contains the logging configuration in a format that can easily transformed into a json file using the json library.

    _consoleLevel: str, set by read_logging_ini_file
            logging level of the console

    _enableConsoleLogging: bool, set by read_logging_ini_file
        if False then no logging messages will be printed in the console independent of the console level

    _enabledFileList: list, set by read_logging_ini_file
        list of logging files that should be created.
        When None then no debug files are created
        example: [debug, info, error, warnings, critical] or subsample of this list

    _enableFileLogging: bool, set by read_logging_ini_file
        if False, then no log files are created independent of enabledFileList

    _enableDebugging: bool, set by read_debugging_ini_file
        if False then no debugging plots will be shown independent of all other options

    _debugFitRange: bool, set by read_debugging_ini_file
        if True then the fitting range of the peak will be visualized in the snapshots

    _debuginitialRange: bool, set by read_debugging_ini_file
        if True then the initial range, where the program looks for a peak will be visualized in the snapshots

    _debugFWHM: bool, set by read_debugging_ini_file
        if True then the estimated and the fitted FWHM will be shown in the snapshots
    """
    _headDirPath: Path = Path(__file__).parents[2].resolve()
    _configLogJsonPath: Path = (_headDirPath / "pysrc" / "setup" / "config_logging.json").resolve()
    _logsDirPath: Path = (_headDirPath / "logs").resolve()
    _loggingLevelList: list[str] = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    _configFileBase: dict = {
        "version": 1,
        "disable_existing_loggers": 0,
        "formatters": {
            "simple": {
                "format": "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
            }
        },

        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": "DEBUG",
                "formatter": "simple"
            },

            "debug_file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "DEBUG",
                "formatter": "simple",
                "filename": str((_logsDirPath / "debug.log").resolve()),
                "maxBytes": 1048576,
                "backupCount": 0,
                "encoding": "utf8"
            },

            "info_file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "INFO",
                "formatter": "simple",
                "filename": str((_logsDirPath / "info.log").resolve()),
                "maxBytes": 1048576,
                "backupCount": 0,
                "encoding": "utf8"
            },

            "error_file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "ERROR",
                "formatter": "simple",
                "filename": str((_logsDirPath / "error.log").resolve()),
                "maxBytes": 1048576,
                "backupCount": 0,
                "encoding": "utf8"
            },

            "warning_file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "WARNING",
                "formatter": "simple",
                "filename": str((_logsDirPath / "warnings.log").resolve()),
                "maxBytes": 1048576,
                "backupCount": 0,
                "encoding": "utf8"
            },

            "critical_file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "CRITICAL",
                "formatter": "simple",
                "filename": str((_logsDirPath / "critical.log").resolve()),
                "maxBytes": 1048576,
                "backupCount": 0,
                "encoding": "utf8"
            }
        },

        ## I don't know what this does
        # "loggers": {
        #     "test": {
        #         "level": "INFO",
        #         "handlers": ["console"],
        #         "propagate": 0
        #     }
        # },

        "root": {
            "level": "DEBUG",
            "handlers": ["console", "debug_file", "info_file", "warning_file", "error_file", "critical_file"]
        }
    }

    def __init__(self) -> None:
        pass

    def write_logging_json_file(self) -> None:
        """Writes the information saved in the attributes (set by read_logging_ini_file) to a json file.

        The json file can be found in setup/config_logging.json.
        """
        disabledFileList: list = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        handlersList: list = ["console", "debug_file", "info_file", "warning_file", "error_file", "critical_file"]
        for i in self._enabledFileList:
            disabledFileList.remove(i.upper())

        for i in disabledFileList:
            dictKey: str = i.lower() + "_file"
            del self._configFileBase["handlers"][dictKey]
            handlersList.remove(dictKey)

        self._configFileBase["handlers"]["console"]["level"] = self._consoleLevel.upper()
        if not self._enableConsoleLogging:
            del self._configFileBase["handlers"]["console"]
            handlersList.remove("console")

        self._configFileBase["root"]["handlers"] = handlersList

        with open(str(self._configLogJsonPath), "w", encoding="utf-8") as f:
            json.dump(self._configFileBase, f, ensure_ascii=False, indent=4)

    def read_logging_ini_file(self) -> None:
        """Reads the config/logging.ini file and saves the configurations as attributes.
        (consoleLevel, enableConsoleLogging, enabledFileList, enableFileLogging).

        Raises
        ------
        AssertionError:
            when consoleLevel, enableConsoleLogging or enabledFileList are invalid datatypes
        """
        configIniPath = (self._headDirPath / "config" / "logging.ini").resolve()
        config = ConfigParser()
        config.read(str(configIniPath), encoding="UTF-8")

        self._consoleLevel = config["logging configuration"]["console level"].replace(" ", "")
        self._enableConsoleLogging = self.check_true_false(config["logging configuration"]["enable console logging"].replace(" ", ""))
        self._enabledFileList = config["logging configuration"]["enabled file list"].replace(" ", "").split(",")
        self._enableFileLogging = self.check_true_false(config["logging configuration"]["enable file logging"].replace(" ", ""))

        assert isinstance(self._consoleLevel, str)
        assert self._consoleLevel.upper() in self._loggingLevelList
        if self._enableFileLogging:
            if self._enabledFileList == [""]:
                self._enabledFileList = []
            for i in self._enabledFileList:
                assert i.upper() in self._loggingLevelList
        else:
            self._enabledFileList = []

    def read_debugging_ini_file(self) -> None:
        """Reads the config/debugging.ini file and saves the configurations as attributes.
        (enableDebugging, debugFitRange, debuginitialRange, debugFWHM)

        Raises
        ------
        ValueError:
            when any of the read attributes cannot be transformed to bools using self.check_true_false()
        """
        configIniPath: Path = (self._headDirPath / "config" / "debugging.ini").resolve()
        config: ConfigParser = ConfigParser()
        config.read(str(configIniPath), encoding="UTF-8")

        self._enableDebugging: bool = self.check_true_false(config["debugging configuration"]["enable debugging"].replace(" ", ""))
        self._debugFitRange: bool
        self._debuginitialRange: bool
        self._debugFWHM: bool
        if not self._enableDebugging:
            self._debugFitRange = False
            self._debuginitialRange = False
            self._debugFWHM = False
        else:
            self._debugFitRange = self.check_true_false(config["debugging configuration"]["fit range"].replace(" ", ""))
            self._debuginitialRange = self.check_true_false(config["debugging configuration"]["initial range"].replace(" ", ""))
            self._debugFWHM = self.check_true_false(config["debugging configuration"]["fwhm"].replace(" ", ""))

    @staticmethod
    def check_true_false(value: str) -> bool:
        """This method is used to transform strings into bools.

        The function is independent of uppercase or lowercase inputs.

        Parameters
        ----------
        var: str
            The function whether this parameter is true or false

        Returns
        -------
        bool:
            True if var is "true" or "1", False if var is "false" or "0"

        Raises
        ------
        ValueError
            if var is neither true nor false
        """
        if value.upper() == "TRUE" or value == "1":
            return True
        elif value.upper() == "FALSE" or value == "0":
            return False
        else:
            raise ValueError(f"{value} can only be true or false (check .ini file)")

    def run_logging_config(self) -> None:
        """Updates the setup/config_logging.json file with the information from the config/logging.ini file.
        This is done by calling self.read_logging_ini_file() and self.write_logging_json_file().
        """
        self.read_logging_ini_file()
        self.write_logging_json_file()

## instantiating a logger for files
    def init_logger(self, name: str) -> logging.Logger:
        """Initializes a logger object.

        The logger should be initialized in the files where logging is desired.
        After initializing as logger, for example logger.error() can be used to write logging messages.

        Parameters
        ----------
        name: str
            name of the logging, ideally the file name (__name__) should be used.

        Returns
        -------
        logger: logging.Logger
            logger object that can be used to write logging messages

        Raises
        ------
        AssertionError:
            when the config_logging.json file does not exist
        """
        assert self._configLogJsonPath.exists()
        with open(str(self._configLogJsonPath), "r") as f:
            config = json.load(f)
        logging.config.dictConfig(config)
        logging.getLogger('matplotlib').setLevel(logging.WARNING)
        return logging.getLogger(name)

def main():
    """Updates the logging configuration .json file based on the .ini file.
    """
    runLoggingConfig: LoggingConfig = LoggingConfig()
    runLoggingConfig.run_logging_config()

if __name__ == "__main__":
    main()