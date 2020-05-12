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


def calc_true_disp(source_x, source_y, cog_x, cog_y, delta, project_disp=False):
    '''
    Calculate the training variables for the disp regressor

    Parameters
    ----------
    source_x: ndarray
        x coordinate of the source in camera coordinate frame
    source_y: ndarray
        y coordinate of the source in camera coordinate frame
    cog_x: ndarray
        x coordinate of the shower cog in camera coordinate frame
    cog_y: ndarray
        y coordinate of the shower cog in camera coordinate frame
    project_disp: bool
        If true, disp is the projection of the source position onto
        the main shower axis. If False, disp is simply the distance
        of the source to the cog.

    Returns
    -------
    abs_disp: absolute value of disp, either projected or not
    sign_disp: sign of disp
    '''
    delta_x = source_x - cog_x
    delta_y = source_y - cog_y

    # in the projected case,
    # true disp is the coordinate of the source on the long axis
    true_disp = np.cos(delta) * delta_x + np.sin(delta) * delta_y
    sign_disp = np.sign(true_disp)

    if project_disp is False:
        abs_disp = euclidean_distance(
            source_x, source_y,
            cog_x, cog_y
        )
    else:
        abs_disp = np.abs(true_disp)

    return abs_disp, sign_disp
