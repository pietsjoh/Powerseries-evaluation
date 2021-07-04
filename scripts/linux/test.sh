#!/usr/bin/env bash
cd "${0%/*}"
cd ../..
source env/bin/activate
python3 -m pytest tests/unit_tests -xvs --repeat $1
