#! /bin/bash
set -e
rm -Rf stability-tables
mkdir stability-tables
runalyzer -m output.dat make-tables.py
