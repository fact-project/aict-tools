import astropy.units as u
from astropy.coordinates import SkyCoord, AltAz
from ctapipe.coordinates import CameraFrame


def horizontal_to_camera_cta_simtel(zd, az, zd_pointing, az_pointing, focal_length):
    with warnings.catch_warnings():

        altaz = AltAz()
        source_altaz = SkyCoord(
            az=az*u.deg,
            alt=zd*u.deg,
            frame=altaz,
        )
        
        tel_pointing = SkyCoord(
            alt=zd_pointing*u.deg,
            az=az_pointing*u.deg,
            frame=altaz,
        )
        camera_frame = CameraFrame(
            focal_length=focal_length*u.m,
            telescope_pointing=tel_pointing,
        )
        
        cam_coords = source_altaz.transform_to(camera_frame)
        return cam_coords.x.to_value(u.m), cam_coords.y.to_value(u.m)
