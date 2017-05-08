# imported so expressions can use @pi
from numpy import pi


def feature_generation(df, config, inplace=False):
    if not inplace:
        df = df.copy()

    for feature_name, expression in config['features'].items():
        df[feature_name] = df.loc[:, config['needed_keys']].eval(expression)

    if not inplace:
        return df
