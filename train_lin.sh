#!/bin/bash
# Trains 55 linear models, one for each cell

for file in ./dataset/*
do
    name=${file: -4}
    python3 main_better_testing.py --globstr-train="dataset/${name}/classification/train.csv" --globstr-val=$name
done