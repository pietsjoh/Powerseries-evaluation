import json
from pathlib import Path
from configparser import ConfigParser
import logging
import logging.config

class LoggingConfig:
    headDirPath = Path(__file__).parents[2].resolve()

    ## default relative path to the config_logging.json file
    configLogJsonPath = (headDirPath / "pysrc" / "setup" / "config_logging.json").resolve()

    ## default absolute path to the logs directory
    logsDirPath = (headDirPath / "logs").resolve()

    ## available logging levels
    loggingLevelList = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

    ## base configuration of the logging file (logs/config_logging.json)
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
                "backupCount": 3,
                "encoding": "utf8"
            },

            "info_file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "INFO",
                "formatter": "simple",
                "filename": str((logsDirPath / "info.log").resolve()),
                "maxBytes": 1048576,
                "backupCount": 3,
                "encoding": "utf8"
            },

            "error_file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "ERROR",
                "formatter": "simple",
                "filename": str((logsDirPath / "error.log").resolve()),
                "maxBytes": 1048576,
                "backupCount": 3,
                "encoding": "utf8"
            },

            "warning_file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "WARNING",
                "formatter": "simple",
                "filename": str((logsDirPath / "warnings.log").resolve()),
                "maxBytes": 1048576,
                "backupCount": 3,
                "encoding": "utf8"
            },

            "critical_file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "CRITICAL",
                "formatter": "simple",
                "filename": str((logsDirPath / "critical.log").resolve()),
                "maxBytes": 1048576,
                "backupCount": 3,
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

    def __init__(self):
        self.read_debugging_ini_file()

    ## function that writes the desired input to the configuration file (logs/config_logging.json)
    def write_logging_json_file(self, consoleLevel="DEBUG", enableConsoleLogging=True, enabledFileList=None):
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

## reading the config/logging.ini file and writing these configurations to the logs/config_logging.json file
    def read_logging_ini_file(self):
        configIniPath = (self.headDirPath / "config" / "logging.ini").resolve()
        config = ConfigParser()
        config.read(str(configIniPath))
        self.consoleLevel = config["logging configuration"]["console level"].replace(" ", "")
        self.enableConsoleLogging = self.check_true_false(config["logging configuration"]["enable console logging"].replace(" ", ""))
        self.enabledFileList = config["logging configuration"]["enabled file list"].replace(" ", "").split(",")

    def read_debugging_ini_file(self):
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
        if var.upper() == "TRUE" or var == "1":
            var = True
            return var
        elif var.upper() == "FALSE" or var == "0":
            var = False
            return var
        else:
            raise ValueError(f"{var} can only be true or false (check .ini file)")

## updating the json config file
    def run_logging_config(self):
        self.read_logging_ini_file()
        self.write_logging_json_file(consoleLevel=self.consoleLevel,
            enableConsoleLogging=self.enableConsoleLogging,
            enabledFileList=self.enabledFileList)

## instantiating a logger for files
    def init_logger(self, name):
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