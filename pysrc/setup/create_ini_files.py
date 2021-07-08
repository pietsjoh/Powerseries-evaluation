"""Creates default .ini files in the config directory.
(debugging.ini, logging.ini, powerseries.ini).
Executed during build process.
"""
if __name__ == "__main__":
    from configparser import ConfigParser
    from pathlib import Path

    headDir: Path = Path(__file__).parents[2]
    configDir: Path = (headDir / "config").resolve()

    ## create debugging.ini file
    debuggingFilePath: Path = (configDir / "debugging.ini").resolve()
    configDebugging: ConfigParser = ConfigParser(allow_no_value=True)
    configDebugging.add_section("debugging configuration")
    configDebugging.set("debugging configuration", "enable debugging", "false\n")
    configDebugging.set("debugging configuration", "fit range", "false\n")
    configDebugging.set("debugging configuration", "initial range", "false\n")
    configDebugging.set("debugging configuration", "fwhm", "false\n")

    with open(str(debuggingFilePath), "w") as configFile:
        configDebugging.write(configFile)

    ## create logging.ini file
    loggingFilePath: Path = (configDir / "logging.ini").resolve()
    configLogging: ConfigParser = ConfigParser(allow_no_value=True)
    configLogging.add_section("logging configuration")
    configLogging.set("logging configuration", "; to apply these changes, use the run_logging_config.sh script", None)
    configLogging.set("logging configuration", "; available logging level: debug, info, warning, error, critical\n", None)
    configLogging.set("logging configuration", "; console logging level (options: all available logging levels)", None)
    configLogging.set("logging configuration", "; console logging level (options: all available logging levels)", None)
    configLogging.set("logging configuration", "console level", "info\n")
    configLogging.set("logging configuration", "; enable console logging (options: true/false)", None)
    configLogging.set("logging configuration", "enable console logging", "true\n")
    configLogging.set("logging configuration", "; enabled files for logging (options: all available logging levels)", None)
    configLogging.set("logging configuration", "enabled file list", "debug, info, warning")

    with open(str(loggingFilePath), "w") as configFile:
        configLogging.write(configFile)

    ## create powerseries.ini file
    powerseriesFilePath: Path = (configDir / "powerseries.ini").resolve()
    configPowerseries: ConfigParser = ConfigParser(allow_no_value=True)
    configPowerseries.add_section("combine_ps_tool.py")
    configPowerseries.add_section("eval_ps.py")
    configPowerseries.add_section("plot_ps.py")
    configPowerseries.set("combine_ps_tool.py", "; add file mode (data or diameter)", None)
    configPowerseries.set("combine_ps_tool.py", "add file mode", "data\n")
    configPowerseries.set("combine_ps_tool.py", "; default diameter", None)
    configPowerseries.set("combine_ps_tool.py", "diameter", "4\n")
    configPowerseries.set("combine_ps_tool.py", "; default path to the data directory", None)
    configPowerseries.set("combine_ps_tool.py", "sorted data dir name", "sorted_data\n")
    configPowerseries.set("combine_ps_tool.py", "; data directory 'fine'-indicator", None)
    configPowerseries.set("combine_ps_tool.py", "fine spectra dir name", "fine_spectra")
    configPowerseries.set("eval_ps.py", "; the data model that is used (available: Qlab2)", None)
    configPowerseries.set("eval_ps.py", "data model", "Qlab2\n")
    configPowerseries.set("eval_ps.py", "; distance between snapshots", None)
    configPowerseries.set("eval_ps.py", "snapshots", "0\n")
    configPowerseries.set("eval_ps.py", "; background fit modes (available: spline, constant, local_all, local_left, local_right, offset, none/disable", None)
    configPowerseries.set("eval_ps.py", "background fit mode", "local_left\n")
    configPowerseries.set("eval_ps.py", "; initial range for peak finding", None)
    configPowerseries.set("eval_ps.py", "use init range", "false")
    configPowerseries.set("eval_ps.py", "min energy", "0")
    configPowerseries.set("eval_ps.py", "max energy", "0\n")
    configPowerseries.set("eval_ps.py", "; max index to use initial range for", None)
    configPowerseries.set("eval_ps.py", "max init range", "0\n")
    configPowerseries.set("eval_ps.py", "; data points which should be excluded", None)
    configPowerseries.set("eval_ps.py", "use exclude", "false")
    configPowerseries.set("eval_ps.py", "min inputpower", "0")
    configPowerseries.set("eval_ps.py", "max inputpower", "0\n")
    configPowerseries.set("eval_ps.py", "; the model that is used for the fitting procedure (available: lorentz, gauss, pseudo-voigt, voigt)", None)
    configPowerseries.set("eval_ps.py", "fit model", "lorentz\n")
    configPowerseries.set("eval_ps.py", "; determines the range of points (fitRangeScale*Estimated_FWHM) that is used for the fitting", None)
    configPowerseries.set("eval_ps.py", "fit range scale", "1\n")
    configPowerseries.set("eval_ps.py", "; integration range for output power (1 corresponds to (-inf, + inf))", None)
    configPowerseries.set("eval_ps.py", "integration coverage", "1\n")
    configPowerseries.set("eval_ps.py", "; parameter that determines the window length for the peak prominences calculation", None)
    configPowerseries.set("eval_ps.py", "; also defines the peak exclusion range for the spline based background fitting", None)
    configPowerseries.set("eval_ps.py", "; moreover, defines the data range for the local mean background estimation method", None)
    configPowerseries.set("eval_ps.py", "constant peak width", "50\n")
    configPowerseries.set("eval_ps.py", "; is multiplied to the output power, needed to stitch the S-curve together when 2 parts (lower-S, upper-S) are measured", None)
    configPowerseries.set("eval_ps.py", "power scale", "1")
    configPowerseries.set("plot_ps.py", "; whether to save the data from powerseries or not (into output/powerseries.csv)", None)
    configPowerseries.set("plot_ps.py", "save data", "false\n")
    configPowerseries.set("plot_ps.py", "; minimize absolute or relative error in curve fitting the in-out-curve (S-shape for beta factor)", None)
    configPowerseries.set("plot_ps.py", "minimize error", "relative\n")
    configPowerseries.set("plot_ps.py", "; use weights (uncertainty of outputpower from the fitting procedure) for the fitting of the in-out-curve", None)
    configPowerseries.set("plot_ps.py", "use weights", "false\n")
    configPowerseries.set("plot_ps.py", "; use bounds for the fitting parameters of the in-out-curve", None)
    configPowerseries.set("plot_ps.py", "use parameter bounds", "false\n")
    configPowerseries.set("plot_ps.py", "; initial guesses for the fitting parameters of the in-out-curve", None)
    configPowerseries.set("plot_ps.py", "initial parameter guess", "none\n")
    configPowerseries.set("plot_ps.py", "; whether to use bootstrap for the error estimation of the beta factor and configure the length and number of samples.", None)
    configPowerseries.set("plot_ps.py", "use bootstrap", "false")
    configPowerseries.set("plot_ps.py", "number of bootstrap samples", "10000")
    configPowerseries.set("plot_ps.py", "length of bootstrap samples", "1")
    configPowerseries.set("plot_ps.py", "plot bootstrap histo", "true")
    configPowerseries.set("plot_ps.py", "use iterative guess", "false")

    with open(str(powerseriesFilePath), "w") as configFile:
        configPowerseries.write(configFile)