import numpy as np


def convert_to_float32(df):
    ''' Convert a pandas dataframe to float32, replacing overflows with maxvalues '''

    df = df.astype('float32')
    df.replace(np.inf, np.finfo('float32').max, inplace=True)
    df.replace(-np.inf, np.finfo('float32').min, inplace=True)

    return df
