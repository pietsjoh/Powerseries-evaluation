# Evaluation of WGM spectra

## Requirements

> python3

- tested on versions 3.8.5 and 3.9.1

## Installation

There is a Windows and a Linux version available.
The installation should be completed by running either the /scripts/linux/build.sh or the
\scripts\windows\build.bat script.

After running the build script the to be analyzed data should be copied into the empty data directory.

## Workflow

The .ini files in the config directory can be adapted to match one's needs.
They mostly contain the default values for the program.

The settings of the [plot_ps.py] section can be changed mid run.
The settings of the other sections cannot be changed once the program runs.
However, the values of the [eval_ps.py] section can be changed using console inputs while
running the powerseries script.

The bash/batch scripts in the scripts directory should be used to run the program.
