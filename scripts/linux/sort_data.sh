#!/usr/bin/env bash
cd "${0%/*}"
cd ../..
echo ""
echo "sorting the data"
source env/bin/activate
python3 pysrc/data_tools/sort_data.py
echo "finished data sorting"
echo ""
