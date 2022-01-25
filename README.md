# Evaluation of WGM spectra

## Requirements

> python3

- tested on versions 3.8.5 and 3.9.1

## Installation / Setup

Both a Windows and a Linux version are available.
For the installation/setup run the respective build script from the scripts directory. This script creates necessary directories, files and creates a virtual environment including all required packages.

After running the build script, copy the to be analyzed data into the empty data directory. Running the software should not change the data, but just to make sure copy it nontheless.

## Workflow

The program should be used with the scripts in the scripts directory. These ensure that the virtual environment will be activated.

- build: use for the first installation or to create anew clean version

- docs: view the documentation in a web browser

- fsr: starts the FSR-Analysis program

- plot_fsr: plots the results from the FSR-Analysis

- powerseries: starts the Powerseries-Evaluation

- sort_data: sorts the data as specified in the config files

- unit_tests: run unit tests (build already does that)

- create_docs: create documentation (build already does this, just used when updating the docs)

- mypy: mypy syntax check (Deprecated)

- performance_tests: run performance tests (Deprecated)

The .ini files in the config directory can be adapted to match one's needs. They mostly contain the default values for the program.

The settings of the [plot_ps.py] section can be changed mid run.
The settings of the other sections cannot be changed once the program runs.
However, the values of the [eval_ps.py] section can be changed using console inputs while
running the powerseries script.

## FSR-Analysis

Start the software using the fsr script from the scripts directory. The aim of this program is to extract the FSR from data sets in save it in a file sorted by diameter. The workflow works as follows.

- load a file

- switch through input powers to find the best spectrum

- the software automatically detects peaks (The number of peaks it displays can be varied)

- select the peak differences (FSR) which shall be saved

- save the data and repeat

After saving the data the plot_fsr script can be used to display the FSR vs diameter data points and fit.

## Powerseries-Evaluation

Start the software using the powerseries script from the scripts directory. The aim of this program is to extract different specific parameters like Q-factor, mode wavelength, beta-factor and threshold from a input-output-powerseries. Firstly, the individual spectra (intensity vs energy) of all input powers are evaluated by fitting one peak. Changeable program parameters include:

- the fitmodel (gauss, lorentz, voigt)

- the range where the program should look for the peak

- different methods to handle the background (spline fit, subtracting mean, extra offset fit parameter)

- number of snapshots shown

- to be excluded data points

- the range where the fit should be done

After that process the resulting Q-factor, mode energy, output power vs input power can be displayed.

Furthermore, a fit of the input-output-characteristic can be performed. Thereby, the beta-factor, Q-factor and the threshold are estimated.

It also possible to stitch multiple data sets together.
