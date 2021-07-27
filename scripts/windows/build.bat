@echo off
set start_dir=%cd%
cd %~dp0
cd ..\..

@REM create virtual environment, activate it and install required packages
if not exist .\env (echo Creating virtual environment && python -m venv .\env)
echo Activating the virtual environment
call .\env\Scripts\activate.bat
echo Upgrading pip
python -m pip install --upgrade pip
echo Installing required packages
pip install -r requirements.txt

@REM if config directory does not exist then create it
if not exist .\config (echo Creating config directory && mkdir .\config && echo Creating config files && python .\pysrc\setup\create_ini_files.py)

@REM if any of the .ini files misses then all are created new
set result=false
if not exist .\config\debugging.ini set result=true
if not exist .\config\logging.ini set result=true
if not exist .\config\powerseries.ini set result=true
if not exist .\config\data_format.ini set result=true
if "%result%"=="true" (echo Creating config files && python .\pysrc\setup\create_ini_files.py)

@REM if logs directory does not exist then create it
if not exist .\logs (echo Creating logs directory && mkdir .\logs)

@REM setup the logging configuration
@echo setting up the logging configuration
python .\pysrc\setup\config_logging.py

@REM setup documentation files
echo Setting up the documentation files
set result=false
if not exist .\docs set result=true
if not exist .\docs\make.bat set result=true
if "%result%"=="true" (echo Setting up docs configuration && mkdir .\docs && cd docs && sphinx-quickstart && cd ..)

if not exist .\docs\build (mkdir .\docs\build)
if exist .\docs\source (@RD /S /Q ".\docs\source")
mkdir .\docs\source
mkdir .\docs\source\_static
mkdir .\docs\source\_templates
python .\pysrc\setup\create_doc_files.py
cd docs
call make clean
call sphinx-apidoc --module-first --force --no-toc -o source ..\pysrc
call make html
cd ..

@REM deactivating the virtual environment
echo Deactivating the virtual environment
call deactivate

@REM run unit tests
@echo run unit tests
call .\scripts\windows\unit_tests.bat 10

@REM if data directory does not exist then create it
if not exist .\data (echo Creating data directory && mkdir .\data)

@REM if output directory does not exist then create it
if not exist .\output (echo Creating output directory && mkdir .\output)

cd %start_dir%
