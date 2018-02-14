source_dependent_features = {
    'alpha',
    'cos_delta_alpha',
    'distance',
    'theta',
    'theta_deg',
}


def find_used_source_features(used_features, generation_config=None):
    used_source_feautures = set(filter(
        lambda v: v in source_dependent_features,
        used_features
    ))

    if generation_config:
        used_source_feautures = used_source_feautures.union(set(filter(
            lambda v: v in source_dependent_features,
            generation_config['needed_keys']
        )))

    return used_source_feautures
