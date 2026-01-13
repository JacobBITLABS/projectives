#!/bin/bash
DIR=$(dirname $0)
FILE=$1
STEM=${FILE%%.*}
$DIR/barchart.py $FILE 25 28 --sorted --output ${STEM}_barchart.pdf
#$DIR/barchart.py $FILE 25 31 --sorted --stacked
$DIR/corr.py $FILE 1 13 --output ${STEM}_corr.pdf
#$DIR/boxplot.py $FILE 1 13 --top-limit 100000
$DIR/boxplot.py $FILE 13 25 --invert --sorted --output ${STEM}_boxplot.pdf
