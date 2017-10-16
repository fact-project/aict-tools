# klaas [![Build Status](https://travis-ci.org/fact-project/classifier-tools.svg?branch=master)](https://travis-ci.org/fact-project/classifier-tools)

Executables to perform machine learning tasks on FACT eventlist data.
Possibly also able to handle input of other experiments if in the same file format.

As of version `0.4.0`, all three tasks of gamma-ray astronomy can be solved using scripts from this package:

* Energy Regression
* Gamma/Hadron Separation
* Reconstruction of origin


# Installation

Then you can install the classifier-tools by:
```
pip install https://github.com/fact-project/classifier-tools/archive/v0.4.0.tar.gz
```

Alternatively you can clone the repo, `cd` into the folder and do and then the usual `pip install .` dance.


# Usage 

For each task, there are two executables, installed to your `PATH`.
Each take `yaml` configuration files and `h5py` style hdf5 files as input.
The models are saved as `pickle` using `joblib` and/or `pmmml`.
 
* `klass_train_<...>`   
  This script is used to train a model on events with known truth
  values for the target variable, usually monte carlo simulations.

* `klass_apply_<...>` 
  This script applies a given model, previously trained with `klaas_train_<...>` and applies it to data, either a test data set or data with unknown truth values for the target variable.

The apply scripts can iterate through the data files in chunks using
the `--chunksize=<N>` option, this can be handy for very large files (> 1 million events). 

## Energy Regression

Energy regression for gamma-rays require a `yaml` configuration file
and simulated gamma-rays in the event list format.

The two scripts to perform energy regression are called

* `klaas_train_energy_regressor`
* `klaas_apply_energy_regressor`

An example configuration can be found in [examples/config_energy.yaml](examples/config_energy.yaml).

To apply a model, use `klaas_apply_energy_regressor`.

## Separation

Binary classification or Separation requires a `yaml` configuration file,
one data file for the signal class and one data file for the background class.

The two scripts to perform separation are called

* `klaas_train_separation_model`
* `klaas_apply_separation_model`.

An example configuration can be found in [examples/config_separator.yaml](examples/config_separator.yaml).


## Reconstruction of gamma-ray origin using the disp method

To estimate the origin of the gamma-rays in camera coordinates, the 
`disp`-method can be used.

Here it is implemented as a two step regression/classification task.
One regression model is trained to estimate `abs(disp)` and a
classification model is trained to estimate `sgn(disp)`.

Training requires simulated diffuse gamma-ray events.

* `klaas_train_disp_regressor`
* `klaas_apply_disp_regressor`

An example configuration can be found in [examples/config_disp.yaml](examples/config_disp.yaml).


# Utility scripts

## Applying straight cuts

For data selection, e.g. to get rid of not well reconstructable events,
it is customary to apply so called pre- or quality cuts before applying machine learning models.

This can be done with `klaas_apply_cuts` and a `yaml` configuration file of the cuts to apply. See [examples/quality_cuts.yaml](examples/quality_cuts.yaml) for an example configuration file.


## Split data into training/test sets

Using `klaas_split_data`, a dataset can be randomly split into sets,
e.g. to split a monte carlo simulation dataset into train and test set.
