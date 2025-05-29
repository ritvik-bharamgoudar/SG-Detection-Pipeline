# SG Detection Pipeline

This repository contains a modular Python pipeline for detecting the Stria of Gennari (SG) from ultra-high-resolution ex vivo MRI using voxel-wise intensity profiles and machine learning.

## Features
- Uses LAYNII-generated layers and columns
- Extracts custom SG-relevant features per column
- Trains a classifier and generates SG probability maps
- Applies neighbour-based smoothing to improve anatomical plausibility

## Getting Started
See `main_pipeline.py` and `sample_data/` for a working example.

## Requirements
Install Python packages listed in `requirements.txt`.

## Credits
Code and methods developed as part of the EN3100 dissertation project by Ritvik Bharamgoudar.

