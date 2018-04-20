from collections import namedtuple
import yaml
from sklearn import ensemble

TrainingConfig = namedtuple('training_config', [
    'n_background',
    'n_signal',
    'n_cross_validations',
    'model',
    'seed',
    'training_variables'
])

FeatureConfig = namedtuple('feature_generation', ['needed_keys', 'features'])


class KlaasConfig():
    feature_generation_config = None
    training_config = None

    telescope_events_key = 'events'
    array_events_key = 'array_events'
    runs_key = 'runs'

    has_multiple_telescopes = False
    class_name = 'gamma'
    seed = 0

    def __init__(self, configuration_path):
        with open(configuration_path) as f:
            config = yaml.load(f)

        self.seed = config.get('seed', 0)
        self.class_name = config.get('class_name', 'gamma')

        training_variables = config['training_variables']
        columns_to_read = [] + config['training_variables']

        generation_config = config.get('feature_generation')
        if generation_config:
            self.feature_generation_config = FeatureConfig(
                generation_config.get('needed_keys'),
                generation_config.get('features'))
            training_variables += generation_config.get('features')

        self.has_multiple_telescopes = config.get('multiple_telescopes', False)

        if self.has_multiple_telescopes:
            self.runs_key = config.get('runs_key', 'runs')
            self.telescope_events_key = config.get('telescope_events_key', 'events')
            self.events_key = config.get('array_events_key', 'array_events')

            self.run_id_key = config.get('run_id_key', 'run_id')
            columns_to_read.append(self.run_id_key)
            self.array_event_id_key = config.get('array_event_id_key', 'array_event_id')
            columns_to_read.append(self.array_event_id_key)

        else:
            self.telescope_events_key = config.get('telescope_events_key', 'events')

        self.log_target = config.get('log_target', False)
        if config.get('target_name', None):
            self.target_name = config['target_name']
            columns_to_read += [config['target_name']]

        self.columns_to_read = columns_to_read

        # trainign configuration
        n_background = config.get('n_background', None)
        n_signal = config.get('n_signal', None)
        n_cross_validations = config.get('n_cross_validations', 10)

        if config.get('classifier'):
            model = eval(config['classifier'])
        else:
            model = eval(config['regressor'])
        model.random_state = self.seed

        self.training_config = TrainingConfig(
            n_background,
            n_signal,
            n_cross_validations,
            model,
            self.seed,
            training_variables,
        )
