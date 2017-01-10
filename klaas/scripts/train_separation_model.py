import pandas as pd
import click
from sklearn import cross_validation
from tqdm import tqdm
import numpy as np
from sklearn import metrics
import yaml

from ..io import read_data, pickle_model, write_data, check_extension


@click.command()
@click.argument('configuration_path', type=click.Path(exists=True, dir_okay=False, file_okay=True))
@click.argument('signal_path', type=click.Path(exists=True, dir_okay=False, file_okay=True))
@click.argument('background_path', type=click.Path(exists=True, dir_okay=False, file_okay=True))
@click.argument('predictions_path', type=click.Path(exists=False, dir_okay=False, file_okay=True))
@click.argument('model_path', type=click.Path(exists=False, dir_okay=False, file_okay=True))
def main(configuration_path, signal_path, background_path, predictions_path, model_path):
    '''
    Train a classifier on signal and background monte carlo data and write the model
    to MODEL_PATH in pmml or pickle format.

    CONFIGURATION_PATH: Path to the config yaml file

    BACKGROUND_PATH: Path to the background data

    SIGNAL_PATH: Path to the signal data

    PREDICTIONS_PATH : path to the file where the mc predictions are stored.

    MODEL_PATH: Path to save the model to. Allowed extensions are .pkl and .pmml.
    If extension is .pmml, then both pmml and pkl file will be saved
    '''

    with open(configuration_path) as f:
        config = yaml.load(f)

    sample = config['sample']
    query = config['query']
    num_cross_validations = config['num_cross_validations']
    training_variables = config['training_variables']

    classifier = eval(config['classifier'])

    check_extension(predictions_path)
    check_extension(model_path, allowed_extensions=['.pmml', '.pkl'])

    # load configuartion stuff
    df_gamma = read_data(file_path=signal_path, query=query, sample=sample)
    df_proton = read_data(file_path=background_path, query=query, sample=sample)

    df_gamma['label_text'] = 'signal'
    df_gamma['label'] = 1
    df_proton['label_text'] = 'background'
    df_proton['label'] = 0

    df_full = pd.concat([df_proton, df_gamma], ignore_index=True)
    # df_full[training_variables] = df_full[training_variables].astype('float32').replace([np.inf, -np.inf], np.nan).dropna(how='any')
    df_training = df_full[training_variables].astype('float32').replace([np.inf, -np.inf], np.nan).dropna(how='any')
    df_label = df_full['label']
    df_label = df_label[df_training.index]


    num_gammas = len(df_label[df_label==1])
    num_protons = len(df_label[df_label==0])
    print('Training classifier with {} protons and {} gammas'.format(num_protons, num_gammas))

    # save prediction_path for each cv iteration
    cv_predictions = []
    # iterate over test and training sets
    X = df_training.values
    y = df_label.values
    print('Starting {} fold cross validation... '.format(num_cross_validations) )
    cv = cross_validation.StratifiedKFold(y, n_folds=num_cross_validations)

    aucs =  []
    for fold, (train, test) in enumerate(tqdm(cv)):
        # select data
        xtrain, xtest = X[train], X[test]
        ytrain, ytest = y[train], y[test]
        # fit and predict
        classifier.fit(xtrain, ytrain)

        y_probas = classifier.predict_proba(xtest)[:, 1]
        y_prediction = classifier.predict(xtest)
        cv_predictions.append(pd.DataFrame({'label':ytest, 'label_prediction':y_prediction, 'probabilities':y_probas, 'cv_fold':fold}))
        aucs.append(metrics.roc_auc_score(ytest, y_prediction))
        #labels_predictions.append([ytest, y_prediction, y_probas])


    print('Mean AUC ROC : {}'.format(np.array(aucs).mean()))

    predictions_df = pd.concat(cv_predictions,ignore_index=True)
    print('writing predictions from cross validation')
    write_data(predictions_df, predictions_path)

    print("Training model on complete dataset")
    classifier.fit(X,y)

    # print("Saving importances")
    # importances = pd.DataFrame(classifier.feature_importances_, index=df_training.columns, columns=['importance'])
    # write_data(importances, importances_path)

    print("Pickling model to {} ...".format(model_path))
    pickle_model(
        classifier=classifier,
        model_path=model_path,
        label_text='label',
        feature_names=list(df_training.columns)
    )

    # print('Adding data information to pmml...')
    # ET.register_namespace('',"http://www.dmg.org/PMML-4_2")
    # xml_tree = ET.parse('rf.pmml')
    # root = xml_tree.getroot()
    # header = root.findall('{http://www.dmg.org/PMML-4_2}Header')[0]
    # newNode = ET.Element('Description')
    # newNode.text = 'Data was queried with {} and contained {} gammas and {} protons'.format(query, len(df_gamma), len(df_proton))
    # header.append(newNode)
    # xml_tree.write(out,
    #        xml_declaration=True,encoding='utf-8',
    #        method='xml')
    #


if __name__ == '__main__':
    main()
