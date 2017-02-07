import numpy as np
import logging


log = logging.getLogger(__name__)


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
                df.columns.loc[invalid_columns]
            )
        )

    return valid
