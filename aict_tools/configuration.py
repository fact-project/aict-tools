import astropy.units as u
from ruamel.yaml import YAML
from collections import namedtuple
import numpy as np
from sklearn.base import is_classifier, is_regressor
import logging

from sklearn import ensemble
from sklearn import linear_model
from sklearn import neighbors
from sklearn import svm
from sklearn import tree
from sklearn import naive_bayes

from .features import find_used_source_features


sklearn_modules = {
    'ensemble': ensemble,
    'linear_model': linear_model,
    'neighbors': neighbors,
    'svm': svm,
    'tree': tree,
    'naive_bayes': naive_bayes,
}


log = logging.getLogger(__name__)
yaml = YAML(typ='safe')


_feature_gen_config = namedtuple(
    'FeatureGenerationConfig',
    ['needed_columns', 'features'],
)


class FeatureGenerationConfig(_feature_gen_config):
    '''
    Stores the needed features and the expressions for the
    feature generation
    '''

    def __new__(cls, needed_columns, features):
        if features is None:
            log.warning('Feature generation config present but no features defined.')
            features = {}
        return super().__new__(cls, needed_columns, features)


def print_models(filter_func=is_classifier):
    for name, module in sklearn_modules.items():
        for cls_name in dir(module):
            cls = getattr(module, cls_name)
            if filter_func(cls):
                logging.info(name + '.' + cls.__name__)


def print_supported_classifiers():
    logging.info('Supported Classifiers:')
    print_models(is_classifier)


def print_supported_regressors():
    logging.info('Supported Regressors:')
    print_models(is_regressor)


def load_regressor(config):
    try:
        return eval(config, {}, sklearn_modules)
    except (NameError, AttributeError):
        log.error('Unsupported Regressor: "' + config + '"')
        print_supported_regressors()
        raise


def load_classifier(config):
    try:
        return eval(config, {}, sklearn_modules)
    except (NameError, AttributeError):
        log.error('Unsupported Regressor: "' + config + '"')
        print_supported_classifiers()
        raise


class AICTConfig:
    __slots__ = (
        'seed',
        'events_key',
        'disp',
        'energy',
        'separator',
        'data_format',
        'telescopes',
        'energy_unit',
        'true_energy_column',
        'size_column',
    )

    @classmethod
    def from_yaml(cls, configuration_path):
        with open(configuration_path) as f:
            return cls(yaml.load(f))

    def __init__(self, config):
        self.data_format = config.get('data_format', "simple")
        self.telescopes = config.get('telescopes', None)
        self.size_column = config.get('size')
        self.energy_unit = u.Unit(config.get('energy_unit', 'GeV'))
        self.seed = config.get('seed', 0)
        np.random.seed(self.seed)

        self.true_energy_column = config.get('true_energy_column')
        self.events_key = config.get('events_key', 'events')
        if self.data_format == "CTA":
            if config.get('events_key'):
                log.warning(
                    'You specified an event key for CTA data.'
                    'We assume the file to be in the official dl1 format'
                    'so this value will be ignored'
                )
        elif self.data_format == 'simple':
            if self.telescopes:
                log.warning(
                    'The telescopes key is currently only available for CTA'
                )
        else:
            raise NotImplementedError(
                'Unsupported data format! Supported: "CTA", "simple"'
            )

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
        'coordinate_transformation',
        'source_az_column',
        'source_az_unit',
        'source_zd_column',
        'source_zd_unit',
        'source_alt_column',
        'source_alt_unit',
        'pointing_az_column',
        'pointing_az_unit',
        'pointing_zd_column',
        'pointing_zd_unit',
        'pointing_alt_column',
        'pointing_alt_unit',
        'focal_length_column',
        'focal_length_unit',
        'cog_x_column',
        'cog_y_column',
        'delta_column',
        'delta_unit',
        'log_target',
        'project_disp',
        'data_format',
        'output_name',
    ]

    def __init__(self, config):
        model_config = config['disp']
        self.data_format = config.get('data_format', 'simple')
        self.disp_regressor = load_regressor(model_config['disp_regressor'])
        self.sign_classifier = load_classifier(model_config['sign_classifier'])
        self.project_disp = model_config.get('project_disp', False)
        self.log_target = model_config.get('log_target', False)

        self.n_signal = model_config.get('n_signal', None)
        k = 'n_cross_validations'
        setattr(self, k, model_config.get(k, config.get(k, 5)))

        self.features = model_config['features'].copy()

        gen_config = model_config.get('feature_generation')
        source_features = find_used_source_features(self.features, gen_config)
        if len(source_features):
            raise ValueError('Source dependent features used: {}'.format(source_features))

        if gen_config:
            self.feature_generation = FeatureGenerationConfig(**gen_config)
            self.features.extend(self.feature_generation.features.keys())
        else:
            self.feature_generation = None
        self.features.sort()

        # Only used as group name for CTA Data
        # Column names are still source_x_prediction, source_y_prediction, ...
        self.output_name = model_config.get('output_name', 'disp_predictions')
        # ToDo: Throw Exceptions for wrong specifications!
        if self.data_format == 'CTA':
            self.coordinate_transformation = model_config.get('coordinate_transformation', 'CTA')
            self.source_az_column = model_config.get('source_az_column', 'true_az')
            self.source_alt_column = model_config.get('source_alt_column', 'true_alt')
            self.source_zd_column = None
            self.pointing_az_column = model_config.get('pointing_az_column', 'azimuth')
            self.pointing_alt_column = model_config.get('pointing_alt_column', 'altitude')
            self.pointing_zd_column = None
            self.focal_length_column = model_config.get(
                'focal_length_column',
                'equivalent_focal_length'
            )
            self.focal_length_unit = u.Unit(model_config.get('focal_length', 'm'))
            self.cog_x_column = model_config.get('cog_x_column', 'hillas_x')
            self.cog_y_column = model_config.get('cog_y_column', 'hillas_y')
            self.delta_column = model_config.get('delta_column', 'hillas_psi')
            self.delta_unit = u.Unit(model_config.get('delta_unit', 'rad'))
            for coord in ('alt', 'az', 'zd'):
                col = f'source_{coord}_unit'
                setattr(self, col, u.Unit(model_config.get(col, 'deg')))
            for coord in ('alt', 'az', 'zd'):
                col = f'pointing_{coord}_unit'
                setattr(self, col, u.Unit(model_config.get(col, 'rad')))
        elif self.data_format == 'simple':
            self.coordinate_transformation = model_config.get('coordinate_transformation', 'FACT')
            self.source_az_column = model_config.get(
                'source_az_column',
                'source_position_az'
            )
            self.source_zd_column = model_config.get('source_zd_column')
            self.source_alt_column = model_config.get('source_alt_column')
            if (self.source_zd_column is None) is (self.source_alt_column is None):
                raise ValueError(
                        'Need to specify exactly one of'
                        'source_zd_column or source_alt_column.'
                        'source_zd_column: {}, source_alt_column: {}'.format(
                            self.source_zd_column, self.source_alt_column)
                )
            self.pointing_az_column = model_config.get(
                'pointing_az_column',
                'pointing_position_az'
            )
            self.pointing_zd_column = model_config.get('pointing_zd_column')
            self.pointing_alt_column = model_config.get('pointing_alt_column')
            if (self.pointing_zd_column is None) is (self.pointing_alt_column is None):
                raise ValueError(
                        'Need to specify exactly one of'
                        'pointing_zd_column or pointing_alt_column.'
                        'pointing_zd_column: {}, pointing_alt_column: {}'.format(
                            self.pointing_zd_column, self.pointing_alt_column)
                        )
            self.focal_length_column = model_config.get(
                'focal_length_column',
                'focal_length'
            )
            self.focal_length_unit = u.Unit(model_config.get('focal_length', 'm'))
            self.cog_x_column = model_config.get('cog_x_column', 'cog_x')
            self.cog_y_column = model_config.get('cog_y_column', 'cog_y')
            self.delta_column = model_config.get('delta_column', 'delta')
            self.delta_unit = u.Unit(model_config.get('delta_unit', 'rad'))
            for name in ('source', 'pointing'):
                for coord in ('alt', 'az', 'zd'):
                    col = f'{name}_{coord}_unit'
                    setattr(self, col, u.Unit(model_config.get(col, 'deg')))

        cols = {
            self.cog_x_column,
            self.cog_y_column,
            self.delta_column,
        }

        cols.update(model_config['features'])
        if self.feature_generation:
            cols.update(self.feature_generation.needed_columns)
        # Add id's because we generate new tables instead of adding columns
        # and want these to be included
        if self.data_format == 'CTA':
            cols.update(['tel_id', 'event_id', 'obs_id'])
        self.columns_to_read_apply = list(cols)
        cols.update({
            self.pointing_az_column,
            self.pointing_zd_column,
            self.pointing_alt_column,
            self.source_az_column,
            self.source_zd_column,
            self.source_alt_column,
        })
        cols.discard(None)
        # Add the focal length to make sure the coordinate transformations work
        if self.data_format == 'CTA':
            cols.add(self.focal_length_column)

        for col in ('true_energy_column', 'size_column'):
            if col in config:
                cols.add(config[col])

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
        'output_name',
        'log_target',
        'data_format',
    ]

    def __init__(self, config):
        model_config = config['energy']
        self.data_format = config.get('data_format', 'simple')
        self.model = load_regressor(model_config['regressor'])
        self.features = model_config['features'].copy()

        self.n_signal = model_config.get('n_signal', None)
        k = 'n_cross_validations'
        setattr(self, k, model_config.get(k, config.get(k, 5)))

        if self.data_format == 'CTA':
            self.target_column = model_config.get(
                'target_column', 'true_energy'
            )
        elif self.data_format == 'simple':
            self.target_column = model_config.get(
                'target_column', 'corsika_event_header_total_energy'
            )
        self.output_name = model_config.get('output_name', 'gamma_energy_prediction')
        self.log_target = model_config.get('log_target', False)

        gen_config = model_config.get('feature_generation')
        source_features = find_used_source_features(self.features, gen_config)
        if len(source_features):
            raise ValueError('Source dependent features used: {}'.format(source_features))
        if gen_config:
            self.feature_generation = FeatureGenerationConfig(**gen_config)
            self.features.extend(self.feature_generation.features.keys())
        else:
            self.feature_generation = None
        self.features.sort()

        cols = set(model_config['features'])
        if self.feature_generation:
            cols.update(self.feature_generation.needed_columns)

        # Add id's because we generate new tables instead of adding columns
        # and want these to be included
        if self.data_format == 'CTA':
            cols.update(['tel_id', 'event_id', 'obs_id'])
        self.columns_to_read_apply = list(cols)

        for col in ('true_energy_column', 'size_column'):
            if col in config:
                cols.add(config[col])
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
        'output_name',
        'data_format',
    ]

    def __init__(self, config):
        model_config = config['separator']
        self.data_format = config.get('data_format', 'simple')
        self.model = load_classifier(model_config['classifier'])
        self.features = model_config['features'].copy()

        self.n_signal = model_config.get('n_signal', None)
        self.n_background = model_config.get('n_background', None)
        k = 'n_cross_validations'
        setattr(self, k, model_config.get(k, config.get(k, 5)))
        self.calibrate_classifier = model_config.get('calibrate_classifier', False)
        self.output_name = model_config.get('output_name', 'gamma_prediction')

        gen_config = model_config.get('feature_generation')
        source_features = find_used_source_features(self.features, gen_config)
        if len(source_features):
            raise ValueError('Source dependent features used: {}'.format(source_features))
        if gen_config:
            self.feature_generation = FeatureGenerationConfig(**gen_config)
            self.features.extend(self.feature_generation.features.keys())
        else:
            self.feature_generation = None
        self.features.sort()

        cols = set(model_config['features'])
        if self.feature_generation:
            cols.update(self.feature_generation.needed_columns)

        # Add id's because we generate new tables instead of adding columns
        # and want these to be included
        if self.data_format == 'CTA':
            cols.update(['tel_id', 'event_id', 'obs_id'])

        self.columns_to_read_apply = list(cols)
        for col in ('true_energy_column', 'size_column'):
            if col in config:
                cols.add(config[col])
        self.columns_to_read_train = list(cols)
