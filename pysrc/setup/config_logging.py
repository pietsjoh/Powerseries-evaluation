"""This script contains everything to set up the logging configuration as well as the debugging configuration.
Upon execution the config/logging.ini file is read and the setup/config_logging.json file is updated accordingly.
This defines the logging setup.
"""
import json
from pathlib import Path
from configparser import ConfigParser
import logging
import logging.config

class LoggingConfig:
    """Contains the methods to provide the logging and debugging setup.

    This class can be used to read the config/logging.ini file and create a setup/config_logging.json file.
    Moreover, the config/debugging.ini is read.
    Finally, a logger object can be created which should be used to create logging messages.
    """
    headDirPath = Path(__file__).parents[2].resolve()

    configLogJsonPath = (headDirPath / "pysrc" / "setup" / "config_logging.json").resolve()
    """default path to the config_logging.json file
    """

    logsDirPath = (headDirPath / "logs").resolve()
    """default path to the logs directory
    """

    loggingLevelList = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    """list: list of available logging levels. This used to check whether invalid options are provided in the config file.
    """

    configFileBase = {
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
                "filename": str((logsDirPath / "debug.log").resolve()),
                "maxBytes": 1048576,
                "backupCount": 0,
                "encoding": "utf8"
            },

            "info_file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "INFO",
                "formatter": "simple",
                "filename": str((logsDirPath / "info.log").resolve()),
                "maxBytes": 1048576,
                "backupCount": 0,
                "encoding": "utf8"
            },

            "error_file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "ERROR",
                "formatter": "simple",
                "filename": str((logsDirPath / "error.log").resolve()),
                "maxBytes": 1048576,
                "backupCount": 0,
                "encoding": "utf8"
            },

            "warning_file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "WARNING",
                "formatter": "simple",
                "filename": str((logsDirPath / "warnings.log").resolve()),
                "maxBytes": 1048576,
                "backupCount": 0,
                "encoding": "utf8"
            },

            "critical_file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "CRITICAL",
                "formatter": "simple",
                "filename": str((logsDirPath / "critical.log").resolve()),
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
    """dict: Contains the logging configuration in a format that can easily transformed into a json file.
    """

    def __init__(self):
        """Upon initialization the debugging ini file is read.
        """
        self.read_debugging_ini_file()

    def write_logging_json_file(self, consoleLevel="DEBUG", enableConsoleLogging=True, enabledFileList=None):
        """Writes the information saved in the attributes (from read_logging_ini_file) to a json file.

        The json file can be found in setup/config_logging.json.
        """
        assert isinstance(consoleLevel, str)
        assert consoleLevel.upper() in self.loggingLevelList
        assert isinstance(enableConsoleLogging, bool)
        if enabledFileList is None:
            enabledFileList = []
        disabledFileList = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        handlersList = ["console", "debug_file", "info_file", "warning_file", "error_file", "critical_file"]
        for i in enabledFileList:
            assert isinstance(i, str)
            assert i.upper() in self.loggingLevelList
            disabledFileList.remove(i.upper())

        for i in disabledFileList:
            dictKey = i.lower() + "_file"
            del self.configFileBase["handlers"][dictKey]
            handlersList.remove(dictKey)

        self.configFileBase["handlers"]["console"]["level"] = consoleLevel.upper()
        if not enableConsoleLogging:
            del self.configFileBase["handlers"]["console"]
            handlersList.remove("console")

        self.configFileBase["root"]["handlers"] = handlersList

        with open(str(self.configLogJsonPath), "w", encoding="utf-8") as f:
            json.dump(self.configFileBase, f, ensure_ascii=False, indent=4)

    def read_logging_ini_file(self):
        """Reads the config/logging.ini file and saves the configurations as attributes.

        The following attributes are set by this method:

        Attributes
        ----------
        consoleLevel: str
            logging level of the console

        enableConsoleLogging: bool
            if False then no logging messages will be printed in the console independent of the console level

        enabledFileList: list
            list of logging files that should be created.
            example: [debug, info, error, warnings, critical] or subsample of this list
        """
        configIniPath = (self.headDirPath / "config" / "logging.ini").resolve()
        config = ConfigParser()
        config.read(str(configIniPath))
        self.consoleLevel = config["logging configuration"]["console level"].replace(" ", "")
        self.enableConsoleLogging = self.check_true_false(config["logging configuration"]["enable console logging"].replace(" ", ""))
        self.enabledFileList = config["logging configuration"]["enabled file list"].replace(" ", "").split(",")

    def read_debugging_ini_file(self):
        """Reads the config/debugging.ini file and saves the configurations as attributes.

        The following attributes are set by this method:

        Attributes
        ----------
        enableDebugging: bool
            if False then no debugging plots will be shown independent of all other options

        debugFitRange: bool
            if True then the fitting range of the peak will be visualized in the snapshots

        debuginitialRange: bool
            if True then the initial range, where the program looks for a peak will be visualized in the snapshots

        debugFWHM: bool
            if True then the estimated and the fitted FWHM will be shown in the snapshots
        """
        configIniPath = (self.headDirPath / "config" / "debugging.ini").resolve()
        config = ConfigParser()
        config.read(str(configIniPath))
        self.enableDebugging = self.check_true_false(config["debugging configuration"]["enable debugging"].replace(" ", ""))
        if not self.enableDebugging:
            self.debugFitRange = False
            self.debuginitialRange = False
            self.debugFWHM = False
        else:
            self.debugFitRange = self.check_true_false(config["debugging configuration"]["fit range"].replace(" ", ""))
            self.debuginitialRange = self.check_true_false(config["debugging configuration"]["initial range"].replace(" ", ""))
            self.debugFWHM = self.check_true_false(config["debugging configuration"]["fwhm"].replace(" ", ""))

    @staticmethod
    def check_true_false(var):
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
        if var.upper() == "TRUE" or var == "1":
            var = True
            return var
        elif var.upper() == "FALSE" or var == "0":
            var = False
            return var
        else:
            raise ValueError(f"{var} can only be true or false (check .ini file)")

    def run_logging_config(self):
        """Updates the setup/config_logging.json file with the information from the config/logging.ini file.
        """
        self.read_logging_ini_file()
        self.write_logging_json_file(consoleLevel=self.consoleLevel,
            enableConsoleLogging=self.enableConsoleLogging,
            enabledFileList=self.enabledFileList)

## instantiating a logger for files
    def init_logger(self, name):
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
        """
        assert self.configLogJsonPath.exists()
        with open(str(self.configLogJsonPath), "r") as f:
            config = json.load(f)
        logging.config.dictConfig(config)
        logging.getLogger('matplotlib').setLevel(logging.WARNING)
        logger = logging.getLogger(name)
        return logger

if __name__ == "__main__":
    runLoggingConfig = LoggingConfig()
    runLoggingConfig.run_logging_config()