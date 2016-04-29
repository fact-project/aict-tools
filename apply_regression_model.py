import pandas as pd
import numpy as np
import click
# from IPython import embed
from sklearn.externals import joblib
from os import path
import json
import yaml

def write_data(df, file_path, hdf_key='table'):
    name, extension =  path.splitext(file_path)
    if extension in ['.hdf', '.hdf5', '.h5']:
        df.to_hdf(file_path, key=hdf_key)
    elif extension == '.json':
        df.to_json(file_path)
    elif extension == '.csv':
        df.to_csv(file_path, index_label='index')
    else:
        print('cannot write tabular data with extension {}'.format(extension))

def read_data(file_path, hdf_key='table'):
    name, extension =  path.splitext(file_path)
    if extension in ['.hdf', '.hdf5', '.h5']:
        return pd.read_hdf(file_path, key=hdf_key)
    if extension == '.json':
        with open(file_path, 'r') as j:
            d = json.load(j)
            return pd.DataFrame(d)

def check_extension(file_path, allowed_extensions= ['.hdf', '.hdf5', '.h5', '.json', '.csv']):
    p, extension = path.splitext(file_path)
    if not extension in allowed_extensions:
        print('Extension {} not allowed here.'.format(extension))

@click.command()
@click.argument('configuration_path', type=click.Path(exists=True, dir_okay=False, file_okay=True) )
@click.argument('data_path', type=click.Path(exists=True, dir_okay=False, file_okay=True) )
@click.argument('model_path', type=click.Path(exists=True, dir_okay=False, file_okay=True) )
@click.argument('output_path', type=click.Path(exists=False, dir_okay=False, file_okay=True) )
def main(configuration_path, data_path, model_path, output_path):
    '''
    Apply loaded model to data. The cuts applied during model training will also be applied here.
    WARNING: currently only taking 1 off position into account.

    CONFIGURATION_PATH: Path to the config yaml file.

    DATA_PATH: path to the FACT data.

    MODEL_PATH: Path to the pickled model.

    OUTPUT_PATH: Path to the data with added prediction columns.
    '''
    check_extension(output_path)

    with open(configuration_path) as f:
        config = yaml.load(f)

    training_variables = config['training_variables']
    query = config['query']

    model = joblib.load(model_path)
    #sklearn needs float32 values. after downcasting some -infs appear somehow. here i drop them.
    print('Loading data')
    df_data = read_data(data_path)
    df_data[training_variables] = df_data[training_variables].astype('float32')
    df_data = df_data.replace([np.inf, -np.inf], np.nan).dropna(how='any')

    if query:
        print('Quering with string: {}'.format(query))
        df_data = df_data.copy().query(query)


    print('After query there are {} events left.'.format(len(df_data)))
    print('Predicting on data...')
    prediction = model.predict(df_data[training_variables])
    df_data['energy_prediction'] = prediction

    print('Writing data')
    write_data(df_data, output_path)



if __name__ == '__main__':
    main()
