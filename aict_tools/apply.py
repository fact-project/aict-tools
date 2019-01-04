import numpy as np
import logging
from operator import le, lt, eq, ne, ge, gt
import h5py
from tqdm import tqdm

from .preprocessing import convert_to_float32, check_valid_rows
from .feature_generation import feature_generation
from fact.io import h5py_get_n_rows

log = logging.getLogger(__name__)


OPERATORS = {
    '<': lt, 'lt': lt,
    '<=': le, 'le': le,
    '==': eq, 'eq': eq,
    '=': eq,
    '!=': ne, 'ne': ne,
    '>': gt, 'gt': gt,
    '>=': ge, 'ge': ge,
}

text2symbol = {
    'lt': '<',
    'le': '<=',
    'eq': '==',
    'ne': '!=',
    'gt': '>',
    'ge': '>=',
}


def build_query(selection_config):
    queries = []
    for k, (o, v) in selection_config.items():
        o = text2symbol.get(o, o)

        queries.append(
            '{} {} {}'.format(k, o, '"' + v + '"' if isinstance(v, str) else v)
        )

    query = '(' + ') & ('.join(queries) + ')'
    return query


def predict_energy(df, model, log_target=False):
    df_features = convert_to_float32(df)
    valid = check_valid_rows(df_features)

    energy_prediction = np.full(len(df_features), np.nan)
    energy_prediction[valid] = model.predict(df_features.loc[valid].values)

    if log_target:
        energy_prediction[valid] = np.exp(energy_prediction[valid])

    return energy_prediction


def predict_disp(df, abs_model, sign_model):
    df_features = convert_to_float32(df)
    valid = check_valid_rows(df_features)

    disp_abs = abs_model.predict(df_features.loc[valid].values)
    disp_sign = sign_model.predict(df_features.loc[valid].values)

    disp_prediction = np.full(len(df_features), np.nan)
    disp_prediction[valid] = disp_abs * disp_sign

    return disp_prediction


def predict_separator(df, model):
    df_features = convert_to_float32(df)
    valid = check_valid_rows(df_features)

    score = np.full(len(df_features), np.nan)
    score[valid] = model.predict_proba(df_features.loc[valid].values)[:, 1]

    return score


def create_mask_h5py(input_path, selection_config, key='events', start=None, end=None, mode="r"):

    with h5py.File(input_path) as infile:

        n_events = h5py_get_n_rows(input_path, key=key, mode=mode)
        start = start or 0
        end = min(n_events, end) if end else n_events

        n_selected = end - start
        mask = np.ones(n_selected, dtype=bool)

        for name, (operator, value) in selection_config.items():

            before = mask.sum()
            mask = np.logical_and(
                mask, OPERATORS[operator](infile[key][name][start:end], value)
            )
            after = mask.sum()
            log.debug('Cut "{} {} {}" removed {} events'.format(
                name, operator, value, before - after
            ))

    return mask


def apply_cuts_h5py_chunked(
        input_path,
        output_path,
        selection_config,
        key='events',
        chunksize=100000,
        progress=True,
        ):
    '''
    Apply cuts defined in selection config to input_path and write result to
    outputpath. Apply cuts to chunksize events at a time.
    '''

    n_events = h5py_get_n_rows(input_path, key=key, mode="r")
    n_chunks = int(np.ceil(n_events / chunksize))
    log.debug('Using {} chunks of size {}'.format(n_chunks, chunksize))

    with h5py.File(input_path, 'r') as infile, h5py.File(output_path, 'w') as outfile:
        group = outfile.create_group(key)

        for chunk in tqdm(range(n_chunks), disable=not progress, total=n_chunks):
            start = chunk * chunksize
            end = min(n_events, (chunk + 1) * chunksize)

            mask = create_mask_h5py(
                input_path, selection_config, key=key, start=start, end=end
            )

            for name, dataset in infile[key].items():
                if chunk == 0:
                    if dataset.ndim == 1:
                        group.create_dataset(name, data=dataset[start:end][mask], maxshape=(None, ))
                    elif dataset.ndim == 2:
                        group.create_dataset(
                            name, data=dataset[start:end, :][mask, :], maxshape=(None, 2)
                        )
                    else:
                        log.warning('Skipping not 1d or 2d column {}'.format(name))

                else:

                    n_old = group[name].shape[0]
                    n_new = mask.sum()
                    group[name].resize(n_old + n_new, axis=0)

                    if dataset.ndim == 1:
                        group[name][n_old:n_old + n_new] = dataset[start:end][mask]
                    elif dataset.ndim == 2:
                        group[name][n_old:n_old + n_new, :] = dataset[start:end][mask, :]
                    else:
                        log.warning('Skipping not 1d or 2d column {}'.format(name))
