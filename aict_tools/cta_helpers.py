import warnings
try:
    import astropy.units as u
    from astropy.coordinates import SkyCoord, AltAz
    from ctapipe.coordinates import CameraFrame
except ImportError:
    raise ImportError('This functionality requires ctapipe to be installed')


def horizontal_to_camera_cta_simtel(zd, az, zd_pointing, az_pointing, focal_length):
    with warnings.catch_warnings():

        altaz = AltAz()
        source_altaz = SkyCoord(
            az=u.Quantity(az, u.deg, copy=False),
            alt=u.Quantity(zd, u.deg, copy=False),
            frame=altaz,
        )

        tel_pointing = SkyCoord(
            alt=u.Quantity(zd_pointing, u.deg, copy=False),
            az=u.Quantity(az_pointing, u.deg, copy=False),
            frame=altaz,
        )
        camera_frame = CameraFrame(
            focal_length=u.Quantity(focal_length, u.m, copy=False),
            telescope_pointing=tel_pointing,
        )
 
        cam_coords = source_altaz.transform_to(camera_frame)
        return cam_coords.x.to_value(u.m), cam_coords.y.to_value(u.m)
