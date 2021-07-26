"""Creates default .ini files in the config directory.
(debugging.ini, logging.ini, powerseries.ini).
Executed during build process.

Attributes
----------
configDir: pathlib.Path
    path to the config directory

debuggingFilePath: pathlib.Path
    path to the debugging.ini file

configDebugging: str
    content of debugging.ini file

loggingFilePath: pathlib.Path
    path to the logging.ini file

configLogging: str
    content of logging.ini file

powerseriesFilePath: pathlib.Path
    path to the powerseries.ini file

configPowerseries: str
    content of powerseries.ini file

dataFormatFilePath: pathlib.Path
    path to the data_format.ini file

configDataFormat: str
    content of data_format.ini file
"""

if __name__ == "__main__":
    from pathlib import Path

    headDir: Path = Path(__file__).parents[2]
    configDir: Path = (headDir / "config").resolve()

    ## create debugging.ini file
    debuggingFilePath: Path = (configDir / "debugging.ini").resolve()
    configDebugging: str = """[debugging configuration]
enable debugging = false

fit range = false

initial range = false

fwhm = false"""

    with open(str(debuggingFilePath), "w", encoding="utf-8") as configFile:
        configFile.write(configDebugging)

    ## create logging.ini file
    loggingFilePath: Path = (configDir / "logging.ini").resolve()
    configLogging: str = """[logging configuration]
; to apply these changes, use the run_logging_config.sh script
; available logging level: debug, info, warning, error, critical

; console logging level (options: all available logging levels)
console level = info

; enable console logging (options: true/false)
enable console logging = true

; enabled files for logging (options: all available logging levels)
enabled file list = debug, info, warning

; enable file logging (options: true/false)
enable file logging = true"""

    with open(str(loggingFilePath), "w", encoding="utf-8") as configFile:
        configFile.write(configLogging)

    ## create powerseries.ini file
    powerseriesFilePath: Path = (configDir / "powerseries.ini").resolve()
    configPowerseries: str = """[combine_ps_tool.py]
; add file mode (data or attribute)
add file mode = data

; default attribute (of attribute name), ignored when add file mode is set to data
default attribute = 4

[eval_ps.py]
; distance between snapshots
snapshots = 0

; background fit modes (available: spline, constant, local_all, local_left, local_right, offset, none/disable
background fit mode = local_left

; initial range for peak finding
use init range = false
min energy = 0
max energy = 0

; max index to use initial range for
max init range = 0

; data points which should be excluded
use exclude = false
min inputpower = 0
max inputpower = 0

; the model that is used for the fitting procedure (available: lorentz, gauss, pseudo-voigt, voigt)
fit model = lorentz

; determines the range of points (fitrangescale*estimated_fwhm) that is used for the fitting
fit range scale = 1

; integration range for output power (1 corresponds to (-inf, + inf))
integration coverage = 1

; parameter that determines the window length for the peak prominences calculation
; also defines the peak exclusion range for the spline based background fitting
; moreover, defines the data range for the local mean background estimation method
constant peak width = 50

; is multiplied to the output power, needed to stitch the s-curve together when 2 parts (lower-s, upper-s) are measured
power scale = 1

[plot_ps.py]
; whether to save the data from powerseries or not (into output/powerseries.csv)
save data = false

; minimize absolute or relative error in curve fitting the in-out-curve (s-shape for beta factor)
minimize error = relative

; use weights (uncertainty of outputpower from the fitting procedure) for the fitting of the in-out-curve
use weights = false

; use bounds for the fitting parameters of the in-out-curve
use parameter bounds = false

; initial guesses for the fitting parameters of the in-out-curve
initial parameter guess = none

; whether to use bootstrap for the error estimation of the beta factor and configure the length and number of samples.
use bootstrap = false
number of bootstrap samples = 10000
length of bootstrap samples = 1
plot bootstrap histo = true
use iterative guess = false"""

    with open(str(powerseriesFilePath), "w", encoding="utf-8") as configFile:
        configFile.write(configPowerseries)

    ## create data_format.ini file
    dataFormatFilePath: Path = (configDir / "data_format.ini").resolve()

    configDataFormat: str = u"""[data format]
; the data model that shall be used (available: qlab2)
data model = Qlab2

; special sequence name from filename (for example: diameter or temperature)
; this will be used to sort the data
attribute name = diameter

; list of available values for the attribute
; when provided, used to check whether the value extracted from the filename does make sense
; set to none if any value should be allowed
attribute possibilities = 1, 1-5, 2, 2-5, 3, 3-5, 4, 4-5, 5, 5-5, 6, 6-5, 7, 7-5, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20

; indicator to find the special sequence (start or end, must not be a part of the desired sequence). the first occurence is taken.
indicator = \u03BCm

; other end of the special sequence, the first occurence on the other side of the indicator is taken
splitter = _

; whether the indicator marks the start or the end of the special sequence
indicator at start = false

; whether to distinguish between full spectra and fine spectra
; used for data sorting
distinguish full fine spectra = true"""

    with open(str(dataFormatFilePath), "w", encoding="utf-8") as configFile:
        configFile.write(configDataFormat)