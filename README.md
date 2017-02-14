# Klaas

Scripts to classify FACT MC data, do energy regression and save models and stuff.

These are some executables which take some configuration.yaml files as input (see examples folder)
and do classification or regression tasks on them.

# Installation

You will have to install teh sklearn2pmml dependency by hand first.
[https://github.com/jpmml/sklearn2pmml](https://github.com/jpmml/sklearn2pmml)

Clone the repo, `cd` into the folder and do the usual `pip install .` dance.


# Usage 

## Regression

There are two programs, `klaas_train_energy_regressor` and `klaas_apply_energy_regressor`.

To train a model, `klaas_train_energy_regressor` takes a yaml configuration file
(see `examples/config_regressor.yaml`) and a hdf5 file with simulated gamma events. 
The hdf5 file can either be a pandas hdf5 as created by the `erna` gridmap processing
or an `h5py` hdf5 as created by the `erna_gather_fits` program.

To apply a model, use `klaas_apply_energy_regressor`, which supports
only h5py-like hdf5 files. 
It can iterate over the files in chunks, thus supporting very large files.


## Classification

Like for the regression, there are two programs: `klaas_train_separation_model` and `klaas_apply_separation_model`.

To train a model, `klaas_train_separation_model` takes a yaml configuration file
(see `examples/config_regressor.yaml`), a hdf5 file with simulated gamma events and 
a hdf5 file with simulated proton events. 
The hdf5 files can either be pandas hdf5 as created by the `erna` gridmap processing
or `h5py` hdf5 as created by the `erna_gather_fits` program.

To apply a model, use `klaas_apply_separation_model`, which supports
only h5py-like hdf5 files. 
It can iterate over the files in chunks, thus supporting very large files.
