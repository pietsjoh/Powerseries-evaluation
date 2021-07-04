#!/usr/bin/env bash
cd "${0%/*}"
cd ../..
source env/bin/activate
python3 pysrc/powerseries/combine_ps_tool.py
