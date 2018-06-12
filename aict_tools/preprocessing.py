import numpy as np
import logging


log = logging.getLogger(__name__)


def euclidean_distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)


def convert_to_float32(df):
    ''' Convert a pandas dataframe to float32, replacing overflows with maxvalues '''

    df = df.astype('float32')
    df.replace(np.inf, np.finfo('float32').max, inplace=True)
    df.replace(-np.inf, np.finfo('float32').min, inplace=True)

    return df


def check_valid_rows(df):
    ''' Check for nans in df, warn if there are any, returns a mask with non-nan rows'''
    valid = np.logical_not(df.isnull().any(axis=1))

    if len(df.loc[valid]) < len(df):
        invalid_columns = df.isnull().any(axis=0)
        log.warning(
            'Data contains not-predictable events.\n'
            'There are nan-values in columns: {}'.format(
                df.columns[invalid_columns]
            )
        )

    return valid


def calc_true_disp(source_x, source_y, cog_x, cog_y, delta):
    true_disp = euclidean_distance(
        source_x, source_y,
        cog_x, cog_y
    )

    true_delta = np.arctan2(
        cog_y - source_y,
        cog_x - source_x,
    )
    true_sign = np.sign(np.abs(delta - true_delta) - np.pi / 2)

    return true_disp, true_sign
