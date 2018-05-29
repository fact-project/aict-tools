from astropy.coordinates import AltAz, SkyCoord
from astropy import units as u
from fact.coordinates.camera_frame import CameraFrame


@u.quantity_input
def horizontal_to_camera(alt: u.deg, az: u.deg, alt_pointing: u.deg, az_pointing: u.deg, focal_length: u.m):
    '''
    Convert sky coordinates from the equatorial frame to FACT camera
    coordinates.

    Parameters
    ----------
    alt: Quantity number or array-like
        altitude
    az: Quantity number or array-like
        azimuth
    zd_pointing: Quantity number or array-like
        Altitude of the telescope pointing direction
    az_pointing: Quantity number or array-like
        Azimuth of the telescope pointing direction

    Returns
    -------
    x: number or array-like
        x-coordinate in the camera plane.
    y: number or array-like
        y-coordinate in the camera plane.
    '''
    alt_az = AltAz(az=az, alt=alt,)
    alt_az_pointing = AltAz(az=az_pointing, alt=alt_pointing,)

    camera_frame = CameraFrame(
        pointing_direction=alt_az_pointing, focal_length=focal_length
    )
    cam_coordinates = alt_az.transform_to(camera_frame)
    source_x, source_y = cam_coordinates.x.to(u.m), cam_coordinates.y.to(u.m)
    source_x, source_y = -source_y, -source_x
    return source_x, source_y


@u.quantity_input
def camera_to_horizontal(x: u.m, y: u.m, alt_pointing: u.deg, az_pointing: u.deg, focal_length: u.m):
    '''
    Convert FACT camera coordinates to sky coordinates in the equatorial (icrs)
    frame.

    Parameters
    ----------
    x: Quantity number or array-like
        x-coordinate in the camera plane
    y: Quantity number or array-like
        y-coordinate in the camera plane
    alt_pointing: Quantity number or array-like
        Altitude of the telescope pointing direction
    az_pointing: Quantity number or array-like
        Azimuth of the telescope pointing direction
    focal_length: Quantity number or array-like
        The focal length of the telescope

    Returns
    -------
    zd: number or array-like
        Zenith distance in degrees
    az: number or array-like
        Declination in degrees
    '''
    x, y = -y, -x
    alt_az_pointing = AltAz(az=az_pointing, alt=alt_pointing)

    frame = CameraFrame(
        pointing_direction=alt_az_pointing,
        focal_length=focal_length,
    )
    cam_coordinates = SkyCoord(x=x, y=y, frame=frame,)

    # altaz = cam_coordinates.transform_to(AltAz(location=LOCATION))
    altaz = cam_coordinates.transform_to(AltAz())

    return (altaz.alt.deg) * u.deg, (altaz.az.deg) * u.deg
