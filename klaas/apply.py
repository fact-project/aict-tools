import numpy as np
import logging
import pandas as pd

from .preprocessing import convert_to_float32, check_valid_rows

log = logging.getLogger(__name__)


def predict(df, model, features):
    df[features] = convert_to_float32(df[features])
    valid = check_valid_rows(df[features])

    prediction = np.full(len(df), np.nan)
    prediction[valid.values] = model.predict_proba(df.loc[valid, features])

    return prediction


def predict_off_positions(df, model, features, used_source_feautures, n_off=5):
    ''' Predicts using the given model for each off position '''

    stored_vars = {
        var: df[var].copy()
        for var in used_source_feautures
    }

    predictions = pd.DataFrame(index=df.index)
    for region in range(1, n_off + 1):
        log.info('Predicting off position {}'.format(region))

        for var in used_source_feautures:
            df[var] = df[var + '_Off_{}'.format(region)]

        valid = check_valid_rows(df[features])

        prediction = np.full(len(df), np.nan)
        prediction[valid.values] = model.predict_proba(df.loc[valid, features])[:, 1]

        predictions['background_prediction_{}'.format(region)] = prediction

    for var, data in stored_vars.items():
        df[var] = data

    return predictions
