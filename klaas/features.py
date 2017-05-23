source_dependent_features = {
    'alpha',
    'cos_delta_alpha',
    'distance',
    'theta',
}


def find_used_source_features(used_features):
    used_source_feautures = set(filter(
        lambda v: v in source_dependent_features,
        used_features
    ))

    return used_source_feautures
