# mistic-ablation
Final project for 6.7830, coauthored by JuliaHolz (Julia Holz) and smaffa (Samuel Maffa)

This repository contains all scripts and notebooks used to reproduce some of the results in [Yang et al., 2025](https://doi.org/10.64898/2025.12.11.693759), as well as perform an ablation study on the parameter relating to transcript locations.

# Directory structure

## MisTIC

Contains the source code from the MisTIC package, along with modifications relevant to the spatially ablated model

## running_mistic_scripts

Contains python and SLURM scripts used to run MisTIC jobs on MIT's Engaging cluster.

## src

Contains python notebooks for cell type annotation, synthetic data generation, running ResolVI, and analysis of results.

# Setup

## MisTIC

A complete list of versioned packages is included in `python_3.11_mistic_package_list.txt`

## Setup for RESOLVI:

A complete list of versioned packages is included in `python_3.10_resolvi_package_list.txt`

The environment was created using the following commands:
```
conda create --name <NAME> python=3.10
conda activate <NAME>
conda install jupyter psutil numpy pandas scanpy scvi-tools anndata pytorch seaborn scipy
```
