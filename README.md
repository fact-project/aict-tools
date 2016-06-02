# classifier
Scripts to classify FACT MC data and save models and stuff 
These are some executables which take some configuration.yaml files as input (see examples folder) and do classification or regression tasks on them.


# usage 

### regressor

    Usage: train_energy_regressor [OPTIONS] CONFIGURATION_PATH SIGNAL_PATH
                                  PREDICTIONS_PATH MODEL_PATH
    
      Train a classifier on signal and background monte carlo data and write the
      model to MODEL_PATH in pmml or pickle format.
    
      CONFIGURATION_PATH: Path to the config yaml file
    
      SIGNAL_PATH: Path to the signal data
    
      PREDICTIONS_PATH : path to the file where the mc predictions are stored.
    
      MODEL_PATH: Path to save the model to. Allowed extensions are .pkl and
      .pmml. If extension is .pmml, then both pmml and pkl file will be saved
    
    Options:
      --help  Show this message and exit.


### seperator

    Usage: train_separation_model [OPTIONS] CONFIGURATION_PATH SIGNAL_PATH
                                  BACKGROUND_PATH PREDICTIONS_PATH MODEL_PATH
    
      Train a classifier on signal and background monte carlo data and write the
      model to MODEL_PATH in pmml or pickle format.
    
      CONFIGURATION_PATH: Path to the config yaml file
    
      BACKGROUND_PATH: Path to the background data
    
      SIGNAL_PATH: Path to the signal data
    
      PREDICTIONS_PATH : path to the file where the mc predictions are stored.
    
      MODEL_PATH: Path to save the model to. Allowed extensions are .pkl and
      .pmml. If extension is .pmml, then both pmml and pkl file will be saved
    
    Options:
      --help  Show this message and exit.


## Model application

Two scripts are provided to apply the models to real data `apply_regression_model` and `apply_separation_model`.


# installation

The usual `pip install -e .` dance

