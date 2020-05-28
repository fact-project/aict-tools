from numpy.testing import assert_allclose


def test_horizontal_to_camera():
    from aict_tools.io import read_data
    from aict_tools.cta_helpers import horizontal_to_camera_cta_simtel

    df = read_data('tests/cta_coord_test.hdf', 'telescope_events')
    expected_x = df.x
    expected_y = df.y
    transformed_x, transformed_y = horizontal_to_camera_cta_simtel(
            alt=90-df.zd,
            az=df.az,
            alt_pointing=90-df.zd_pointing,
            az_pointing=df.az_pointing,
            focal_length=df.focal_length,
    )
    assert_allclose(expected_x, transformed_x)
    assert_allclose(expected_y, transformed_y)
