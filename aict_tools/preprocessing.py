import numpy as np
import logging
import astropy.units as u
from fact.coordinates.utils import horizontal_to_camera as horizontal_to_camera_fact


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


def convert_units(df, model_config):
    '''
    Transforms the columns to the desired units.

    Parameters:
    -----------
    df: pd.DataFrame
        The dataframe containing the values
    model_config: .configuration.DISPConfig
        Config for the DISP model. Contains the
        information about the units of the provided values.

    Returns:
    --------
    df: The DataFrame with converted values.
    '''
    coordinate_units = (
        (model_config.delta_column, model_config.delta_unit, 'rad'),
        (model_config.focal_length_column, model_config.focal_length_unit, 'm'),
        (model_config.source_az_column, model_config.source_az_unit, 'deg'),
        (model_config.source_zd_column, model_config.source_zd_unit, 'deg'),
        (model_config.source_alt_column, model_config.source_alt_unit, 'deg'),
        (model_config.pointing_az_column, model_config.pointing_az_unit, 'deg'),
        (model_config.pointing_zd_column, model_config.pointing_zd_unit, 'deg'),
        (model_config.pointing_alt_column, model_config.pointing_alt_unit, 'deg'),
    )
    for column, unit, expected_unit in coordinate_units:
        if column in df.columns and unit != expected_unit:
            df[column] = u.Quantity(
                df[column].to_numpy(),
                unit,
                copy=False,
            ).to_value(expected_unit)

    return df


def get_alt(df, model_config):
    '''
    Return altitude for source and pointing from the df.
    Transforms from zenith distance to altitude if zd is given
    '''
    if model_config.source_zd_column:
        source_alt = 90 - df[model_config.source_zd_column]
    else:
        source_alt = df[model_config.source_alt_column]

    if model_config.pointing_zd_column:
        pointing_alt = 90 - df[model_config.pointing_zd_column]
    else:
        pointing_alt = df[model_config.pointing_alt_column]

    return source_alt, pointing_alt


def get_zd(df, model_config):
    '''
    Return altitude for source and pointing from the df.
    Transforms from zenith distance to altitude if zd is given
    '''
    if model_config.source_alt_column:
        source_zd = 90 - df[model_config.source_alt_column]
    else:
        source_zd = df[model_config.source_zd_column]

    if model_config.pointing_alt_column:
        pointing_zd = 90 - df[model_config.pointing_alt_column]
    else:
        pointing_zd = df[model_config.pointing_zd_column]

    return source_zd, pointing_zd


def horizontal_to_camera(df, model_config):
    if model_config.coordinate_transformation == 'CTA':
        from .cta_helpers import horizontal_to_camera_cta_simtel

        alt_source, alt_pointing = get_alt(df, model_config)
        source_x, source_y = horizontal_to_camera_cta_simtel(
            az=df[model_config.source_az_column],
            alt=alt_source,
            az_pointing=df[model_config.pointing_az_column],
            alt_pointing=alt_pointing,
            focal_length=df[model_config.focal_length_column],
        )

    elif model_config.coordinate_transformation == 'FACT':
        zd_source, zd_pointing = get_zd(df, model_config)
        source_x, source_y = horizontal_to_camera_fact(
            az=df[model_config.source_az_column],
            zd=zd_source,
            az_pointing=df[model_config.pointing_az_column],
            zd_pointing=zd_pointing,
        )
    else:
        raise ValueError('Unsupported value for coordinate_transformation')

    return source_x, source_y
