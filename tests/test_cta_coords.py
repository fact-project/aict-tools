from numpy.testing import assert_allclose


def test_horizontal_to_camera_cta():
    from aict_tools.io import read_data
    from aict_tools.cta_helpers import horizontal_to_camera

    df = read_data("tests/cta_coord_test.hdf", "telescope_events")
    expected_x = df.x
    expected_y = df.y
    transformed_x, transformed_y = horizontal_to_camera(
        alt=90 - df.zd,
        az=df.az,
        alt_pointing=90 - df.zd_pointing,
        az_pointing=df.az_pointing,
        focal_length=df.focal_length,
    )
    assert_allclose(expected_x, transformed_x)
    assert_allclose(expected_y, transformed_y)


def test_camera_to_horizontal():
    from aict_tools.io import read_data
    from aict_tools.cta_helpers import camera_to_horizontal

    df = read_data("tests/cta_coord_test.hdf", "telescope_events")
    expected_alt = df.alt
    expected_az = df.az
    transformed_alt, transformed_az = camera_to_horizontal(
        x=df.x,
        y=df.y,
        alt_pointing=90 - df.zd_pointing,
        az_pointing=df.az_pointing,
        focal_length=df.focal_length,
    )
    assert_allclose(expected_alt, transformed_alt)
    assert_allclose(expected_az, transformed_az)