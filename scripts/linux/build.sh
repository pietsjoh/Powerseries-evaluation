#!/usr/bin/env bash
cd "${0%/*}"
cd ../..
## create virtual environment, activate it and install required packages
[ ! -d "./env" ] && echo "Creating virtual environment" && python3 -m venv env
echo "Activating the virtual environment"
source env/bin/activate
echo "Upgrading pip"
python3 -m pip install --upgrade pip
echo "Installing required packages"
pip3 install -r requirements.txt

## if config directory does not exist then create it
[ ! -d "./config" ] && echo "Creating config directory" && mkdir ./config && echo "Creating config (.ini) files" && python3 pysrc/setup/create_ini_files.py

## if any of the .ini files misses then all are created new
[[ ! -f "./config/logging.ini" || ! -f "./config/debugging.ini" || ! -f "./config/powerseries.ini" ]] && echo "Creating config (.ini) files" && python3 pysrc/setup/create_ini_files.py

## if logs directory does not exist then create it
[ ! -d "./logs" ] && echo "Creating logs directory" && mkdir ./logs

## setup the logging configuration
echo "Setting up the logging configuration" && python3 pysrc/setup/config_logging.py

## run unit tests
echo "Run unit tests" && ./scripts/linux/test.sh 10

## setup documentation files
echo "Setting up the documentation files"
[[ ! -d "./docs" || ! -f "./docs/Makefile" ]] && echo "Setting up docs configuration" && mkdir ./docs && cd docs && sphinx-quickstart && cd ..
[ ! -d "./docs/build" ] && mkdir ./docs/build
[ -d "./docs/source" ] && rm -rf ./docs/source
mkdir ./docs/source
mkdir ./docs/source/_static
mkdir ./docs/source/_templates
python3 pysrc/setup/create_doc_files.py
cd docs
make clean
sphinx-apidoc --module-first --force --no-toc -o source ../pysrc
make html
cd ..

## Deactivating the virtual environment (not necessary though)
echo "Deactivating the virtual environment"
deactivate

## if data directory does not exist then create it
[ ! -d "./data" ] && echo "Creating data directory" && mkdir ./data

## if output directory does not exist then create it
[ ! -d "./output" ] && echo "Creating output directory" && mkdir ./output
