import yaml
from sklearn import ensemble
from collections import namedtuple
from .features import find_used_source_features
import numpy as np

FeatureGenerationConfig = namedtuple(
    'FeatureGenerationConfig',
    ['needed_columns', 'features']
)


class AICTConfig:
    __slots__ = (
        'seed',
        'telescope_events_key',
        'array_events_key',
        'array_event_id_column',
        'runs_key',
        'run_id_column',
        'is_array',
        'disp',
        'energy',
        'separator',
        'has_multiple_telescopes',
        'class_name',
    )

    @classmethod
    def from_yaml(cls, configuration_path):
        with open(configuration_path) as f:
            return cls(yaml.load(f))

    def __init__(self, config):
        self.has_multiple_telescopes = config.get('multiple_telescopes', False)
        self.runs_key = config.get('runs_key', 'runs')

        if self.has_multiple_telescopes:
            self.telescope_events_key = config.get('telescope_events_key', 'events')
            self.array_events_key = config.get('array_events_key', 'array_events')

            self.array_event_id_column = config.get(
                'array_event_id_column', 'array_event_id'
            )
            self.run_id_column = config.get('run_id_column', 'run_id')
        else:
            self.telescope_events_key = config.get('telescope_events_key', 'events')
            self.run_id_column = config.get('run_id_column', 'run_id')

            self.array_events_key = None
            self.array_event_id_column = None

        self.seed = config.get('seed', 0)
        np.random.seed(self.seed)
        self.class_name = config.get('class_name', 'gamma')

        self.disp = self.energy = self.separator = None
        if 'disp' in config:
            self.disp = DispConfig(config)

        if 'energy' in config:
            self.energy = EnergyConfig(config)

        if 'separator' in config:
            self.separator = SeparatorConfig(config)


class DispConfig:
    __slots__ = [
        'disp_regressor',
        'sign_classifier',
        'n_cross_validations',
        'n_signal',
        'features',
        'feature_generation',
        'columns_to_read_apply',
        'columns_to_read_train',
        'source_az_column',
        'source_zd_column',
        'pointing_az_column',
        'pointing_zd_column',
        'cog_x_column',
        'cog_y_column',
        'delta_column',
    ]

    def __init__(self, config):
        model_config = config['disp']
        self.disp_regressor = eval(model_config['disp_regressor'])
        self.sign_classifier = eval(model_config['sign_classifier'])

        self.n_signal = model_config.get('n_signal', None)
        k = 'n_cross_validations'
        setattr(self, k, model_config.get(k, config.get(k, 5)))

        self.features = model_config['features'].copy()

        gen_config = model_config.get('feature_generation')
        source_features = find_used_source_features(self.features, gen_config)
        if len(source_features):
            raise ValueError('Source dependent features used: {}'.format(source_features))

        if gen_config:
            self.features.extend(gen_config['features'].keys())
            self.feature_generation = FeatureGenerationConfig(**gen_config)
        else:
            self.feature_generation = None
        self.features.sort()

        self.source_az_column = model_config.get('source_az_column', 'source_position_az')
        self.source_zd_column = model_config.get('source_zd_column', 'source_position_zd')

        self.pointing_az_column = model_config.get('pointing_az_column', 'pointing_position_az')
        self.pointing_zd_column = model_config.get('pointing_zd_column', 'pointing_position_zd')
        self.cog_x_column = model_config.get('cog_x_column', 'cog_x')
        self.cog_y_column = model_config.get('cog_y_column', 'cog_y')
        self.delta_column = model_config.get('delta_column', 'delta')

        cols = {
            self.cog_x_column,
            self.cog_y_column,
            self.delta_column,
        }

        cols.update(model_config['features'])
        if self.feature_generation:
            cols.update(self.feature_generation.needed_columns)
        self.columns_to_read_apply = list(cols)
        cols.update({
            self.pointing_az_column,
            self.pointing_zd_column,
            self.source_az_column,
            self.source_zd_column,
        })
        self.columns_to_read_train = list(cols)


class EnergyConfig:
    __slots__ = [
        'model',
        'n_cross_validations',
        'n_signal',
        'features',
        'feature_generation',
        'columns_to_read_train',
        'columns_to_read_apply',
        'target_column',
        'log_target',
    ]

    def __init__(self, config):
        model_config = config['energy']
        self.model = eval(model_config['regressor'])
        self.features = model_config['features'].copy()

        self.n_signal = model_config.get('n_signal', None)
        k = 'n_cross_validations'
        setattr(self, k, model_config.get(k, config.get(k, 5)))

        self.target_column = model_config.get(
            'target_column', 'corsika_event_header_total_energy'
        )
        self.log_target = model_config.get('log_target', False)

        gen_config = model_config.get('feature_generation')
        source_features = find_used_source_features(self.features, gen_config)
        if len(source_features):
            raise ValueError('Source dependent features used: {}'.format(source_features))
        if gen_config:
            self.features.extend(gen_config['features'].keys())
            self.feature_generation = FeatureGenerationConfig(**gen_config)
        else:
            self.feature_generation = None
        self.features.sort()

        cols = set(model_config['features'])
        if self.feature_generation:
            cols.update(self.feature_generation.needed_columns)
        self.columns_to_read_apply = list(cols)
        cols.add(self.target_column)
        self.columns_to_read_train = list(cols)


class SeparatorConfig:
    __slots__ = [
        'model',
        'n_cross_validations',
        'n_signal',
        'n_background',
        'features',
        'feature_generation',
        'columns_to_read_train',
        'columns_to_read_apply',
        'calibrate_classifier',
    ]

    def __init__(self, config):
        model_config = config['separator']
        self.model = eval(model_config['classifier'])
        self.features = model_config['features'].copy()

        self.n_signal = model_config.get('n_signal', None)
        self.n_background = model_config.get('n_background', None)
        k = 'n_cross_validations'
        setattr(self, k, model_config.get(k, config.get(k, 5)))
        self.calibrate_classifier = model_config.get('calibrate_classifier', False)

        gen_config = model_config.get('feature_generation')
        source_features = find_used_source_features(self.features, gen_config)
        if len(source_features):
            raise ValueError('Source dependent features used: {}'.format(source_features))
        if gen_config:
            self.features.extend(gen_config['features'].keys())
            self.feature_generation = FeatureGenerationConfig(**gen_config)
        else:
            self.feature_generation = None
        self.features.sort()

        cols = set(model_config['features'])
        if self.feature_generation:
            cols.update(self.feature_generation.needed_columns)
        self.columns_to_read_train = list(cols)
        self.columns_to_read_apply = list(cols)
