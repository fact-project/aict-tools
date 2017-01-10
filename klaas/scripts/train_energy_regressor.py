import pandas as pd
import click
from sklearn import cross_validation
from sklearn import metrics
from tqdm import tqdm
import numpy as np
import yaml
from klaas import write_data, pickle_model, read_data


@click.command()
@click.argument('configuration_path', type=click.Path(exists=True, dir_okay=False, file_okay=True))
@click.argument('signal_path', type=click.Path(exists=True, dir_okay=False, file_okay=True))
@click.argument('predictions_path', type=click.Path(exists=False, dir_okay=False, file_okay=True))
@click.argument('model_path', type=click.Path(exists=False, dir_okay=False, file_okay=True))
def main(configuration_path, signal_path, predictions_path, model_path):
    '''
    Train a classifier on signal and background monte carlo data and write the model to MODEL_PATH in pmml or pickle format.

    CONFIGURATION_PATH: Path to the config yaml file

    SIGNAL_PATH: Path to the signal data

    PREDICTIONS_PATH : path to the file where the mc predictions are stored.

    MODEL_PATH: Path to save the model to. Allowed extensions are .pkl and .pmml. If extension is .pmml, then both pmml and pkl file will be saved
    '''

    print("Loading data")
    with open(configuration_path) as f:
        config = yaml.load(f)


    sample = config['sample']
    query = config['query']
    num_cross_validations = config['num_cross_validations']
    training_variables = config['training_variables']

    classifier = eval(config['classifier'])

    df = read_data(file_path=signal_path, sample=sample, query=query)

    df_train = df[training_variables].astype('float32').replace([np.inf, -np.inf], np.nan).dropna(how='any')

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
    write_data(predictions_df, predictions_path)

    scores = np.array(scores)
    print("Cross validated R^2 scores: {}".format(scores))
    print("Mean R^2 score from CV: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))


    print("Building new model on complete data set...")
    # rf = ensemble.ExtraTreesRegressor(n_estimators=n_trees,max_features="sqrt", oob_score=True, n_jobs=n_jobs, max_depth=max_depth)
    classifier.fit(X_train, y_train)
    print("Score on complete data set: {}".format(classifier.score(X_test, y_test)))

    print("Pickling model to {} ...".format(model_path))
    pickle_model(
            classifier=classifier,
            feature_names=list(df_train.columns),
            model_path=model_path,
            label_text = 'estimated_energy',
    )


if __name__ == '__main__':
    main()
