# To Do

## scripts

### General

- linewidth fit
- always use same boostrap/fitting setup and note it down to reproduce results
- find min beta uncertainty for fixed xi fit -> using this for plotting
- calculate uncertainty for the input power
- write a script that reproduces the beta plot and its numbers using the output directory files
- save all settings (exclude, initRange, fitmodel, fitRangeScale, intCoverage, ...) [x]
- linear approximation for Q-factor? (in between 2 input powers)
- check whether all files have been transfered when calling sort_data [x]
- constant fit range ???
- use loops for add / delete / run powerseries (only exit when q is called) [x]
- Write help functions
- Write description of available commands also into the docstring of input decoder
- rework create_ini_files (not use configparser) [x]
- docs
  - move __init__ doc to base def
  - add all attributes description to base class
  - do not forget Raises sections
- config file:
  - list of fitmodels, list of data formats
  - selected data format
  - temperatureIndicator
  - diameterIndicator
- write a function that prints the results in the right format with rounded uncertainties
- add logging messages for bootstrap (paramBounds, initialParamGuesses, useWeights, useBounds, bootstrap setup)
  - should make it easier to debug which way in the BST is taken
- add script to plot quickevaluated.dat files
- move log setup to src/utils and create log directory in build script [x]
- installation script [x]
  - either windows or linux scripts
  - use python to create these scripts based on user input (windows/linux)
- doc (setup and docstrings)
- seperate README.md, TODO.md, BUGS.md [x]
- enable background fitting and change background fitting mode into 1 variable [x]
- more verbose / add space between messages
- add breakout for all input statements (enter q for 'run ps' for example)
- add logging? [x]
  - add loggers into configuration (not needed) [x]
  - print statements around log warnings???
- write unit tests / performance tests / integration tests?
- integrate debug possibilities (plots with vertical lines) [x]
  - now combine this with logging [x]
- rethink general structure
- table for default values, or not print them out and save them also to the log file. [x]
  - maybe use config file for this [x]
  - also set diameter for CombinePowerSeriesTool [x]
- write unit tests
  - done for PeakFitSuper [x]
  - otherwise not really possible, because library code, data reading and user input
- config file for fsr?
- make input independent of spaces (.replace(" ", "")) and uppercase/lowercase (.lower())

> misc.py

- diameter list in config file
- split into 2 files (decode and statistics)

### FSR

> fsr_plot.py

- fitting the FSR curve

> fsr_selector.py

- fitting the peaks instead of using find_peaks only
- rethinking the use of constantPeakWidth for get_peaks()
- integrate misc.int_decode()
- change like powerseries/ps_tool.py
  - check_input(),...

> fsr_selection.py

- restructure like powerseries/combine_powerseries.py
  - set diameter option, add, del,...

### Data tools

> data_import.py

- add QLab 5 data import
- add indicator string attribute (date)
- change to abstract base class (qlab2, qlab5 inherit) [x]
  - not the prefered solution here
  - instead checker method in EvalPowerSeries
- add config for indicator strings and data directory
- add general indicator method to extract info from file name
  - indicator + name of attribute should be specified
  - output in ps_tool.py config method

> sort_data.py

- restructure more readable
- combine with data class
- more general sorting
- config for strings

### Peak fitting

> single_peak_fit_base.py

- add Raises section for docstrings
- think about the use of paramBounds, maybe do not use it
- rework local background extraction, better control of the range
- peak exclusion may return error (remove_background_local/spline) [x]
  - when exclusion range exceeds data set [x]
- maybe use local not median for constant mode [x]
- change initial fit parameter guesses [x]
  - I think this is not needed [x]
- calibrate background extraction [x]
  - (using no background extraction for now) [x]
- check whether to use Voigt model or not
  - maybe change lorentz model
- double peak fitting
  - automate it???
- multiple peaks exclude in background extraction??? [x]
  - (using no background extraction for now) [x]
- check maxValue for fitRangeScale
- still make it possible to use the background extraction [x]
- add logging for background extraction [x]
  - if it will be used also add peak exclusion range plot to debugging.ini
- list of available background fit models in config
  - use this list for import / check_input_backgroundFitMode()
- rework debugging plots (peak highlight, data vs. originalData)
- detect constant background locally [x]

> single_peak_fit_models.py

- list of available fit models in config
  - use this list for import / check_input_fitModel()
- uncertainty for fwhm for voigt/pseudovoigt
- add param_bounds

### powerseries

> eval_ps.py

- index in console output when running (for shown snapshots) [x]
- maybe also set smoothSpline and smoothGauss depending on calibration [x]
  - (not needed anymore because background extraction excluded) [x]

> ps_tool.py

- add help function
- plot ms / ss before running
- enable/disable background fitting [x]
- exclude points based on input power (or idx?) -> view from plot [x]
  - add option to also exclude individual points
- view individual specs [x]
  - fit and original data
- set parameters:
  - fitRangeScale [x]
  - intCoverage
  - constantPeakWidth [x]
  - Q-Factor (take below threshold)
  - threshold
- add to config:
  - backgroundFitMode [x]
  - maxInitRange [x]
  - fitmodel [x]
  - fitRangeScale [x]
  - constantPeakWidth [x]
  - exclude [x]
  - intCoverage [x]
  - Q factor
  - threshold
  - beta factor
  - Purcell factor???

> plot_ps.py

- waterfall plot 5 spectra [x]
- errorbars from fit parameters [x]
- linear lines in power spectra [x]
- beta fit does not work good in tested samples (error when not in [0, 1]?), maybe gets better with exclude
- maybe add initial parameters for fitting based on experience [x]

> combine_ps_tool.py

- read data directly from the data directory (only AllSpectra.dat, include all subdirs) [x]
  - restructure this
  - more general approach and seperate adding process
- more verbose scale_routine() [x]
- loop for scale and plot s -> until satisfied [x]
- add help function
- switch the order of the selected files to match plot_multiple_spectra() [x]
  - not needed because now y-axis is sorted by input power
- plot ms / ss before running
- seperate add / del file methods (input and actual add)
- del files automatically (but ask) when changing diameter
- handle overlap by excluding data points???
- check if all selected files share the same diameter, temperature and date
- automate scaling
- add to config:
  - lenInputPower, inputPowerRange, energyRange
  - scaling finished
  - exclude
  - Q factor
  - threshold
  - beta factor
  - Purcell factor???

### temperature series

- new class???

## data evaluation

> diameter dependent plots:

- FSR
- threshold
- beta factor
- Q factor
- Purcell factor???
- for each diameter 1 s-shape, where the pillar is lasing
- which diameter show double peak, at which input power, go back to normal peak?

> temperature dependent plots:

- mode energy (expected to increase)
- beta factor (expected to decrease)
- Q factor (expected to decrease)
  - log-log plot linear behaviour
  - stability from linear increase parameter
  - fwhm while lasing vs. temperature
- Purcell factor???
- fwhm at threshold

> pol series:

## Bachelorarbeit

- 5 seiten theorie
- 5 seiten experiment aufbau / proben
- 15 bleiben
  - 5 seiten software
  - 10 auswertung

## präsentation 21.05.21 feedback

- fix fit parameters (xi, A)
- get these from diameter dependent analysis
- threshold: mean photon = 1 -> p*x = 1
- q-factor can take anywhere -> may use threshold (cannot use theoretical definition)

## next steps

- fix xi with the results from the paper
- compare results for different diameters
- then focus on threshold and take q-factor there
- redo the diameter analysis and evaluate the temperature series
