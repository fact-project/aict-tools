from pytest import raises

def test_disp():

    from klaas.configuration import KlaasConfig

    c = KlaasConfig.from_yaml('examples/config_source.yaml')

    assert 'cog_x' in c.disp.columns_to_read_apply


def test_energy():

    from klaas.configuration import KlaasConfig

    c = KlaasConfig.from_yaml('examples/config_energy.yaml')

    assert c.energy is not None
    assert c.disp is None
    assert c.separator is None
    assert 'size' in c.energy.columns_to_read_apply
    assert 'corsika_event_header_total_energy' not in c.energy.columns_to_read_apply
    assert 'corsika_event_header_total_energy' in c.energy.columns_to_read_train


def test_separator():

    from klaas.configuration import KlaasConfig

    c = KlaasConfig.from_yaml('examples/config_separator.yaml')

    assert c.energy is None
    assert c.disp is None
    assert c.separator is not None


def test_full():

    from klaas.configuration import KlaasConfig

    c = KlaasConfig.from_yaml('examples/full_config.yaml')

    assert c.energy is not None
    assert c.disp is not None
    assert c.separator is not None


def test_source():

    from klaas.configuration import KlaasConfig

    with raises(ValueError):
        KlaasConfig.from_yaml('tests/config_source.yaml')
