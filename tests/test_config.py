from pytest import raises


def test_disp():

    from aict_tools.configuration import AICTConfig

    c = AICTConfig.from_yaml('examples/config_source.yaml')

    assert 'cog_x' in c.disp.columns_to_read_apply


def test_energy():

    from aict_tools.configuration import AICTConfig

    c = AICTConfig.from_yaml('examples/config_energy.yaml')

    assert c.energy is not None
    assert c.disp is None
    assert c.separator is None
    assert 'size' in c.energy.columns_to_read_apply
    assert 'corsika_event_header_total_energy' not in c.energy.columns_to_read_apply
    assert 'corsika_event_header_total_energy' in c.energy.columns_to_read_train


def test_separator():

    from aict_tools.configuration import AICTConfig

    c = AICTConfig.from_yaml('examples/config_separator.yaml')

    assert c.energy is None
    assert c.disp is None
    assert c.separator is not None


def test_full():

    from aict_tools.configuration import AICTConfig

    c = AICTConfig.from_yaml('examples/full_config.yaml')

    assert c.energy is not None
    assert c.energy_unit is not None
    assert c.disp is not None
    assert c.separator is not None


def test_source():

    from aict_tools.configuration import AICTConfig

    with raises(ValueError):
        AICTConfig.from_yaml('tests/config_source.yaml')


def test_altitude():

    from aict_tools.configuration import AICTConfig

    zd_config = AICTConfig.from_yaml('examples/config_source.yaml')
    assert 'source_position_zd' in zd_config.disp.columns_to_read_train

    alt_config = AICTConfig.from_yaml('examples/config_source_altitude.yaml')
    assert 'source_position_alt' in alt_config.disp.columns_to_read_train


def test_cta():

    from aict_tools.configuration import AICTConfig

    cta_config = AICTConfig.from_yaml('examples/cta_full_config.yaml')
    assert cta_config.data_format == 'CTA'
    assert cta_config.disp.pointing_alt_column == 'altitude'
    assert cta_config.telescopes is not None
