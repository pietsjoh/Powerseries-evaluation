#!/usr/bin/env bash
cd "${0%/*}"
cd ../..
source env/bin/activate
python3 pysrc/FSR_analysis/fsr_selection.py
