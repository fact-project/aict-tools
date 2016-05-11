import pandas as pd
import numpy as np
import click
# from IPython import embed
from sklearn.externals import joblib
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

    thetas = df_data['Theta']
    distances = df_data['Distance']

    print('Predicting off data...')
    df_data['Theta'] = df_data['Theta_Off_3']
    df_data['Distance'] = df_data['Distance_Off_3']
    df_data['background_prediction'] =  model.predict_proba(df_data[training_variables])[:,1]
    df_data['background_theta']  = df_data['Theta']
    df_data['background_distance']  = df_data['Distance']

    # df_data['background_prediction'] = 0
    # df_data['background_theta'] = np.nan
    # df_data['background_distance'] = np.nan

    # off_columns = zip(['Theta_Off_1', 'Theta_Off_2', 'Theta_Off_3', 'Theta_Off_4', 'Theta_Off_5'],
    #                     ['Distance_Off_1', 'Distance_Off_2', 'Distance_Off_3', 'Distance_Off_4', 'Distance_Off_5'])
    # for key in off_columns:
    #     df_data['Theta'] = df_data[key[0]]
    #     df_data['Distance'] = df_data[key[1]]
    #     prediction_for_background = model.predict_proba(df_data[config.training_variables])[:,0]
    #     # embed()
    #     mask = (prediction_for_background > df_data['background_prediction']).values
    #     df_data['background_prediction'][mask] = prediction_for_background[mask]
    #     df_data['background_theta'][mask]  = df_data['Theta'][mask]
    #     df_data['background_distance'][mask]  = df_data['Distance'][mask]

    df_data['Distance'] = distances
    df_data['Theta'] = thetas
    print('Writing data')
    write_data(df_data, output_path)



if __name__ == '__main__':
    main()
