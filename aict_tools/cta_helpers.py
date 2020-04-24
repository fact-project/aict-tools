import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.coordinates import EarthLocation
from astropy.coordinates import AltAz
from astropy.time import Time
import logging
import warnings

from astropy.coordinates import SkyCoord, AltAz
from ctapipe.coordinates import (
    NominalFrame,
    CameraFrame,
    TiltedGroundFrame,
    project_to_ground,
    GroundFrame,
    MissingFrameAttributeWarning
)
from ctapipe.instrument import TelescopeDescription

log = logging.getLogger(__name__)


def horizontal_to_camera_cta_simtel(df):
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=MissingFrameAttributeWarning)

        alt_pointing = u.Quantity(df.pointing_altitude.to_numpy(), u.rad, copy=False)
        az_pointing = u.Quantity(df.pointing_azimuth.to_numpy(), u.rad, copy=False)
        fl = u.Quantity(df.focal_length.to_numpy(), u.m, copy=False)
        mc_alt = u.Quantity(df.mc_alt.to_numpy(), u.deg, copy=False)
        mc_az = u.Quantity(df.mc_az.to_numpy(), u.deg, copy=False)

        altaz = AltAz()
        
        tel_pointing = SkyCoord(
            alt=alt_pointing,
            az=az_pointing,
            frame=altaz,
        )
        camera_frame = CameraFrame(
            focal_length=fl,
            telescope_pointing=tel_pointing,
        )
        
        source_altaz = SkyCoord(
            az=mc_az,
            alt=mc_alt,
            frame=altaz,
        )
        
        cam_coords = source_altaz.transform_to(camera_frame)
        return cam_coords.x.to_value(u.m), cam_coords.y.to_value(u.m)



def camera_to_horizontal_cta_simtel(df, x_key='source_x_prediction', y_key='source_y_prediction'):
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=MissingFrameAttributeWarning)

        alt_pointing = u.Quantity(df.pointing_altitude.to_numpy(), u.rad, copy=False)
        az_pointing = u.Quantity(df.pointing_azimuth.to_numpy(), u.rad, copy=False)
        x = u.Quantity(df[x_key].to_numpy(), u.m, copy=False)
        y = u.Quantity(df[y_key].to_numpy(), u.m, copy=False)
        fl = u.Quantity(df.focal_length.to_numpy(), u.m, copy=False)
        
        altaz = AltAz()

        tel_pointing = SkyCoord(
            alt=alt_pointing,
            az=az_pointing,
            frame=altaz,
        )

        frame = CameraFrame(
            focal_length = fl,
            telescope_pointing=tel_pointing,
        )


        cam_coords = SkyCoord(
                x=x,
                y=y,
                frame=frame,
        )

        source_altaz = cam_coords.transform_to(altaz)

        # rad verwenden? 
        return source_altaz.alt.to_value(u.deg), source_altaz.az.to_value(u.deg)


def camera_to_nominal_cta_simtel(df, x_key='source_x_prediction', y_key='source_y_prediction'):
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=MissingFrameAttributeWarning)

	# this should be taken from the files, but we throw it away in the preprocessing...
        alt_array_pointing = (70 * u.deg)
        az_array_pointing = (180 * u.deg)

        alt_pointing = u.Quantity(df.pointing_altitude.to_numpy(), u.rad, copy=False)
        az_pointing = u.Quantity(df.pointing_azimuth.to_numpy(), u.rad, copy=False)
        x = u.Quantity(df[x_key].to_numpy(), u.m, copy=False)
        y = u.Quantity(df[y_key].to_numpy(), u.m, copy=False)
        fl = u.Quantity(df.focal_length.to_numpy(), u.m, copy=False)
        
        altaz = AltAz()

        array_pointing = SkyCoord(
            alt = alt_array_pointing,
            az = az_array_pointing,
            frame = altaz,
        )

        nom_frame = NominalFrame(origin=array_pointing)

        tel_pointing = SkyCoord(
            alt=alt_pointing,
            az=az_pointing,
            frame=altaz,
        )

        camera_frame = CameraFrame(
            focal_length = fl,
            telescope_pointing=tel_pointing,
        )


        cam_xy = SkyCoord(
                x=x,
                y=y,
                frame=camera_frame,
        )
        
        nominal_xy = cam_xy.transform_to(nom_frame)

        return nominal_xy.delta_alt.to_value(u.deg), nominal_xy.delta_az.to_value(u.deg)


def nominal_to_horizontal_cta_simtel(
	df,
	alt_key='source_alt_prediction_nominal',
	az_key='source_az_prediction_nominal'):
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=MissingFrameAttributeWarning)

	# this should be taken from the files, but we throw it away in the preprocessing...
        alt_array_pointing = (70 * u.deg)
        az_array_pointing = (180 * u.deg)

        d_alt_pred = u.Quantity(df[alt_key].to_numpy(), u.deg, copy=False)
        d_az_pred = u.Quantity(df[az_key].to_numpy(), u.deg, copy=False)
        
        altaz = AltAz()

        array_pointing = SkyCoord(
            alt = alt_array_pointing,
            az = az_array_pointing,
            frame = altaz,
        )

        nom_frame = NominalFrame(origin=array_pointing)
        
        prediction_nominal = SkyCoord(
            delta_az = d_az_pred,
            delta_alt = d_alt_pred,
            frame = nom_frame,
        )

        prediction_horizon = prediction_nominal.transform_to(altaz)
        return prediction_horizon.alt.to_value(u.deg), prediction_horizon.az.to_value(u.deg)
