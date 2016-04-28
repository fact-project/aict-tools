import pandas as pd
import click
from sklearn import cross_validation
# from sklearn import linear_model
from sklearn2pmml import sklearn2pmml
from sklearn import metrics
from tqdm import tqdm
from os import path
import numpy as np
from sklearn_pandas import DataFrameMapper
import yaml
from sklearn.externals import joblib
from sklearn import ensemble



def write_data(df, file_path, hdf_key='table'):
    name, extension =  path.splitext(file_path)
    if extension in ['.hdf', '.hdf5', '.h5']:
        df.to_hdf(file_path, key=hdf_key)
    elif extension == '.json':
        df.to_json(file_path)
    else:
        print('cannot write tabular data with extension {}'.format(extension))

@click.command()
@click.argument('configuration_path', type=click.Path(exists=True, dir_okay=False, file_okay=True))
@click.argument('gamma_path', type=click.Path(exists=True, dir_okay=False, file_okay=True, readable=True) )
def main(configuration_path, gamma_path):
    '''
    Train a RF regressor and write the model to OUT in pmml format.
    '''
    print("Loading data")
    with open(configuration_path) as f:
        config = yaml.load(f)


    #load paths
    prediction_path = config['prediction_path']
    importances_path = config['importances_path']
    model_path = config['model_path']

    sample = config['sample']
    query = config['query']
    num_cross_validations = config['num_cross_validations']
    training_variables = config['training_variables']

    classifier = eval(config['classifier'])

    df = pd.read_hdf(gamma_path, key='table')

    if sample > 0:
        df = df.sample(sample)

    if query:
        print('Quering with string: {}'.format(query))
        df = df.query(query)

    df_train = df[training_variables]
    df_train = df_train.dropna(axis=0, how='any')

    df_target = df['MCorsikaEvtHeader.fTotalEnergy']
    df_target.name = 'true_energy'
    df_target = df_target[df_train.index]
    # embed()
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(df_train, df_target, test_size=0.2)

    print('Starting {} fold cross validation... '.format(num_cross_validations) )
    scores = []
    cv_predictions = []

    cv = cross_validation.KFold(len(y_train), n_folds=num_cross_validations, shuffle=True)
    for fold, (train, test) in tqdm(enumerate(cv)):
        # embed()
        # select data
        cv_x_train, cv_x_test = X_train.values[train], X_train.values[test]
        cv_y_train, cv_y_test = y_train.values[train], y_train.values[test]
        # fit and predict
        # embed()
        classifier.fit(cv_x_train, cv_y_train)
        cv_y_prediciton = classifier.predict(cv_x_test)

        #calcualte r2 score
        scores.append(metrics.r2_score(cv_y_test, cv_y_prediciton))

        cv_predictions.append(pd.DataFrame({'label':cv_y_test, 'label_prediction':cv_y_prediciton, 'cv_fold':fold}))



    predictions_df = pd.concat(cv_predictions,ignore_index=True)

    print('writing predictions from cross validation')
    predictions_df.to_hdf(prediction_path, key='table')

    scores = np.array(scores)
    print("Cross validated R^2 scores: {}".format(scores))
    print("Mean R^2 score from CV: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))


    print("Building new model on complete data set...")
    # rf = ensemble.ExtraTreesRegressor(n_estimators=n_trees,max_features="sqrt", oob_score=True, n_jobs=n_jobs, max_depth=max_depth)
    classifier.fit(X_train, y_train)
    print("Score on complete data set: {}".format(classifier.score(X_test, y_test)))


    print("Saving importances")
    importances = pd.DataFrame(classifier.feature_importances_, index=df_train.columns, columns=['importance'])
    write_data(importances, importances_path)


    p, extension = path.splitext(model_path)
    if (extension == '.pmml'):
        print("Pickling model to {} ...".format(model_path))
        # joblib.dump(rf, mode, compress = 4)
        mapper = DataFrameMapper([
                                (list(df_train.columns), None),
                                ('estimated_energy', None)
                        ])

        # joblib.dump(mapper, out, compress = 4)
        sklearn2pmml(classifier, mapper,  model_path)

        joblib.dump(classifier,p + '.pkl', compress = 4)
    else:
        joblib.dump(classifier, model_path, compress = 4)

if __name__ == '__main__':
    main()
