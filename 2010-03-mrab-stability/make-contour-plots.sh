#! /bin/sh
rm -Rf contour-plots
mkdir contour-plots
python make-contour-plots.py output-hires.dat
