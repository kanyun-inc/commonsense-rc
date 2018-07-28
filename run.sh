#!/bin/bash

# Get conceptnet triples
python3 src/preprocess.py conceptnet

# Preprocess dataset
python3 src/preprocess.py

# Start training
python3 -u src/main.py --gpu 0 > run.log &
