import warnings

try:
    import astropy.units as u
    from astropy.coordinates import SkyCoord, AltAz
    from ctapipe.coordinates import CameraFrame, MissingFrameAttributeWarning
except ImportError:
    raise ImportError("This functionality requires ctapipe to be installed")


def horizontal_to_camera(alt, az, alt_pointing, az_pointing, focal_length):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MissingFrameAttributeWarning)

        altaz = AltAz()
        source_altaz = SkyCoord(
            alt=u.Quantity(alt, u.deg, copy=False),
            az=u.Quantity(az, u.deg, copy=False),
            frame=altaz,
        )

        tel_pointing = SkyCoord(
            alt=u.Quantity(alt_pointing, u.deg, copy=False),
            az=u.Quantity(az_pointing, u.deg, copy=False),
            frame=altaz,
        )
        camera_frame = CameraFrame(
            focal_length=u.Quantity(focal_length, u.m, copy=False),
            telescope_pointing=tel_pointing,
        )

        cam_coordinates = source_altaz.transform_to(camera_frame)
        return cam_coordinates.x.to_value(u.m), cam_coordinates.y.to_value(u.m)


def camera_to_horizontal(x, y, alt_pointing, az_pointing, focal_length):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MissingFrameAttributeWarning)

        altaz = AltAz()
        tel_pointing = SkyCoord(
            alt=u.Quantity(alt_pointing, u.deg, copy=False),
            az=u.Quantity(az_pointing, u.deg, copy=False),
            frame=altaz,
        )
        camera_coordinates = CameraFrame(
            x=u.Quantity(x, u.m, copy=False),
            y=u.Quantity(y, u.m, copy=False),
            focal_length=u.Quantity(focal_length, u.m, copy=False),
            telescope_pointing=tel_pointing,
        )

        horizontal_coordinates = camera_coordinates.transform_to(altaz)
        return horizontal_coordinates.alt.to_value(u.deg), horizontal_coordinates.az.to_value(u.deg)