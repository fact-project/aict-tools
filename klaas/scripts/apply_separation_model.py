import pandas as pd
import numpy as np
import click
# from IPython import embed
import joblib
from klaas import check_extension, read_data, write_data
import yaml

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
    df_data = read_data(data_path, query=query)
    df_data[training_variables] = df_data[training_variables].astype('float32')
    df_data = df_data.replace([np.inf, -np.inf], np.nan).dropna(how='any')


    # embed()
    print('After dropping nans there are {} events left.'.format(len(df_data)))
    print('Predicting on data...')
    prediction = model.predict_proba(df_data[training_variables])
    df_data['signal_prediction'] = prediction[:,1]
    df_data['signal_theta'] = df_data['Theta']
    df_data['signal_distance'] = df_data['Distance']
    if 'Theta' in training_variables:
        thetas = df_data['Theta'].copy()
        distances = df_data['Distance'].copy()

        print('Predicting off data...')
        for region in [1,2,3,4,5]:
            theta_key = 'Theta_Off_{}'.format(region)
            distance_key = 'Distance_Off_{}'.format(region)
            df_data['Theta'] = df_data[theta_key]
            df_data['Distance'] = df_data[distance_key]
            prediction = model.predict_proba(df_data[training_variables])
            df_data['background_prediction_{}'.format(region)] =  prediction[:,1]


        df_data['Distance'] = distances
        df_data['Theta'] = thetas

    print('Writing data')
    write_data(df_data, output_path)



if __name__ == '__main__':
    main()
