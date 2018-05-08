source_dependent_features = {
    'alpha',
    'cos_delta_alpha',
    'distance',
    'theta',
    'theta_deg',
}


def has_source_dependent_features(used_features, generation_config=None):

    used_source_features = find_used_source_features(used_features, generation_config)

    return len(used_source_features) > 0


def find_used_source_features(used_features, generation_config=None):
    used_source_features = set(filter(
        lambda v: v in source_dependent_features, used_features
    ))

    if generation_config:
        used_source_features = used_source_features.union(set(filter(
            lambda v: v in source_dependent_features,
            generation_config['needed_columns']
        )))

    return used_source_features
