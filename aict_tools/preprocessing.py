import numpy as np
import logging
import astropy.units as u
from astropy.table import Table
from fact.coordinates.utils import horizontal_to_camera as horizontal_to_camera_fact
from fact.coordinates.utils import camera_to_horizontal as camera_to_horizontal_fact
from fact.coordinates.utils import camera_to_equatorial as camera_to_equatorial_fact
from .configuration import AICTConfig


log = logging.getLogger(__name__)


def euclidean_distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def convert_to_float32(df):
    """ Convert a pandas dataframe to float32, replacing overflows with maxvalues """

    df = df.astype("float32")
    df.replace(np.inf, np.finfo("float32").max, inplace=True)
    df.replace(-np.inf, np.finfo("float32").min, inplace=True)

    return df


def check_valid_rows(df):
    """ Check for nans in df, warn if there are any, returns a mask with non-nan rows"""
    nans = df.isnull()
    valid = ~nans.any(axis=1)

    nan_counts = nans.sum()
    if (nan_counts > 0).any():
        nan_counts_str = ", ".join(f"{k}: {v}" for k, v in nan_counts.items() if v > 0)
        log.warning("Data contains not-predictable events.")
        log.warning(f"There are nan-values in columns: {nan_counts_str}")

    return valid


def calc_true_disp(source_x, source_y, cog_x, cog_y, delta, project_disp=False):
    """
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
    """
    delta_x = source_x - cog_x
    delta_y = source_y - cog_y

    # in the projected case,
    # true disp is the coordinate of the source on the long axis
    true_disp = np.cos(delta) * delta_x + np.sin(delta) * delta_y
    sign_disp = np.sign(true_disp)

    if project_disp is False:
        abs_disp = euclidean_distance(source_x, source_y, cog_x, cog_y)
    else:
        abs_disp = np.abs(true_disp)

    return abs_disp, sign_disp


def convert_units(df, config):
    """
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
    """
    coordinate_units = (
        (config.delta_column, config.delta_unit, "rad"),
        (config.focal_length_column, config.focal_length_unit, "m"),
        (config.source_az_column, config.source_az_unit, "deg"),
        (config.source_zd_column, config.source_zd_unit, "deg"),
        (config.source_alt_column, config.source_alt_unit, "deg"),
        (config.pointing_az_column, config.pointing_az_unit, "deg"),
        (config.pointing_zd_column, config.pointing_zd_unit, "deg"),
        (config.pointing_alt_column, config.pointing_alt_unit, "deg"),
    )
    for column, unit, expected_unit in coordinate_units:
        if column in df.columns:
            if isinstance(df, Table):
                if df[column].unit:
                    df[column] = df[column].quantity.to_value(expected_unit)
            elif unit != expected_unit:
                df[column] = u.Quantity(
                    df[column].to_numpy(),
                    unit,
                    copy=False,
                ).to_value(expected_unit)
    return df


def get_alt(df, config: AICTConfig):
    """
    Return altitude for source and pointing from the df.
    Transforms from zenith distance to altitude if zd is given
    """
    if config.source_zd_column:
        source_alt = 90 - df[config.source_zd_column]
    else:
        source_alt = df[config.source_alt_column]
    pointing_alt = get_alt_pointing(df, config)
    return source_alt, pointing_alt


def get_zd(df, config):
    """
    Return altitude for source and pointing from the df.
    Transforms from zenith distance to altitude if zd is given
    """
    if config.source_alt_column:
        source_zd = 90 - df[config.source_alt_column]
    else:
        source_zd = df[config.source_zd_column]

    pointing_zd = get_zd_pointing(df, config)
    return source_zd, pointing_zd


def get_alt_pointing(df, config):
    if config.pointing_zd_column:
        pointing_alt = 90 - df[config.pointing_zd_column]
    else:
        pointing_alt = df[config.pointing_alt_column]
    return pointing_alt


def get_zd_pointing(df, config):
    if config.pointing_alt_column:
        pointing_zd = 90 - df[config.pointing_alt_column]
    else:
        pointing_zd = df[config.pointing_zd_column]
    return pointing_zd


def horizontal_to_camera(df, config):
    if config.coordinate_transformation == "CTA":
        from .cta_helpers import horizontal_to_camera as horizontal_to_camera_cta

        alt_source, alt_pointing = get_alt(df, config)
        source_x, source_y = horizontal_to_camera_cta(
            az=df[config.source_az_column],
            alt=alt_source,
            az_pointing=df[config.pointing_az_column],
            alt_pointing=alt_pointing,
            focal_length=df[config.focal_length_column],
        )
    elif config.coordinate_transformation == "FACT":
        zd_source, zd_pointing = get_zd(df, config)
        source_x, source_y = horizontal_to_camera_fact(
            az=df[config.source_az_column],
            zd=zd_source,
            az_pointing=df[config.pointing_az_column],
            zd_pointing=zd_pointing,
        )
    else:
        raise ValueError("Unsupported value for coordinate_transformation")

    return source_x, source_y


def camera_to_horizontal(df, config, source_x, source_y):
    if config.coordinate_transformation == "CTA":
        from .cta_helpers import camera_to_horizontal as camera_to_horizontal_cta

        alt_pointing = get_alt_pointing(df, config)
        source_alt, source_az = camera_to_horizontal_cta(
            x=source_x,
            y=source_y,
            az_pointing=df[config.pointing_az_column],
            alt_pointing=alt_pointing,
            focal_length=df[config.focal_length_column],
        )
    elif config.coordinate_transformation == "FACT":
        zd_pointing = get_zd_pointing(df, config)
        source_zenith, source_az = camera_to_horizontal_fact(
            x=source_x,
            y=source_y,
            az_pointing=df[config.pointing_az_column],
            zd_pointing=zd_pointing,
        )
        source_alt = 90 - source_zenith
    else:
        raise ValueError("Unsupported value for coordinate_transformation")

    return source_alt, source_az


def delta_error(data_df, model_config):
    df = data_df.copy()
    source_x, source_y = horizontal_to_camera(df, model_config)
    true_delta = np.arctan2(
        source_y - df[model_config.cog_y_column],
        source_x - df[model_config.cog_x_column],
    )

    ####    calculate the difference    ####
    delta_diff = true_delta - df[model_config.delta_column]
    ####    fold pi and -pi into the middle    ####
    delta_diff[delta_diff < np.pi / 2] = delta_diff[delta_diff < np.pi / 2] + np.pi
    delta_diff[delta_diff > np.pi / 2] = delta_diff[delta_diff > np.pi / 2] - np.pi

    return delta_diff
