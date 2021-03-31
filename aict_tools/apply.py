from astropy.table import Table, join
import numpy as np
import logging
from operator import le, lt, eq, ne, ge, gt
import h5py
import tables
from tables import NaturalNameWarning
from tqdm import tqdm
import warnings

from .preprocessing import convert_to_float32, check_valid_rows
from .io import get_number_of_rows_in_table

log = logging.getLogger(__name__)


OPERATORS = {
    "<": lt,
    "lt": lt,
    "<=": le,
    "le": le,
    "==": eq,
    "eq": eq,
    "=": eq,
    "!=": ne,
    "ne": ne,
    ">": gt,
    "gt": gt,
    ">=": ge,
    "ge": ge,
}

text2symbol = {
    "lt": "<",
    "le": "<=",
    "eq": "==",
    "ne": "!=",
    "gt": ">",
    "ge": ">=",
}


def build_query(selection_config):
    queries = []
    for k, (o, v) in selection_config.items():
        o = text2symbol.get(o, o)

        queries.append(
            "{} {} {}".format(k, o, '"' + v + '"' if isinstance(v, str) else v)
        )

    query = "(" + ") & (".join(queries) + ")"
    return query


def predict_energy(df, model, log_target=False):
    df_features = convert_to_float32(df)
    valid = check_valid_rows(df_features)

    energy_prediction = np.full(len(df_features), np.nan)
    energy_prediction[valid] = model.predict(df_features.loc[valid].values)

    if log_target:
        energy_prediction[valid] = np.exp(energy_prediction[valid])

    return energy_prediction


def predict_disp(df, abs_model, sign_model, log_target=False):
    df_features = convert_to_float32(df)
    valid = check_valid_rows(df_features)

    disp_abs = abs_model.predict(df_features.loc[valid].values)
    disp_sign = sign_model.predict(df_features.loc[valid].values)

    if log_target:
        disp_abs = np.exp(disp_abs)

    disp_prediction = np.full(len(df_features), np.nan)
    disp_prediction[valid] = disp_abs * disp_sign

    return disp_prediction


def predict_dxdy(df, dxdy_model, log_target=False):
    df_features = convert_to_float32(df)
    valid = check_valid_rows(df_features)

    # 2, because prediction will return two values: dx and dy
    dxdy_prediction = np.full((len(df_features), 2), np.nan)
    dxdy_prediction[valid] = dxdy_model.predict(df_features.loc[valid].values)

    if log_target:
        dxdy_prediction[valid] = np.exp(dxdy_prediction[valid])

    return dxdy_prediction


def predict_separator(df, model):
    df_features = convert_to_float32(df)
    valid = check_valid_rows(df_features)

    score = np.full(len(df_features), np.nan)
    score[valid] = model.predict_proba(df_features.loc[valid].values)[:, 1]

    return score


def create_mask_h5py(
    infile,
    selection_config,
    n_events,
    key="events",
    start=None,
    end=None,
):
    """
    Creates a boolean mask for a dataframe in a h5 file based on a
    selection config.

    Parameters:
    -----------
    infile: str, Path
    selection_config: dict
        Dictionary with column names as keys and (operator, value) tuples as value
    n_events: int
        Number of events to select.
    key: str
        Path to the dataframe in the file
    start: int
        If None, select the first row
    end: int
        If None, select the last row

    Returns:
    --------
    Boolean mask with len=n_events or len(df)
    """
    start = start or 0
    end = min(n_events, end) if end else n_events

    n_selected = end - start
    mask = np.ones(n_selected, dtype=bool)

    if isinstance(selection_config, dict):
        raise ValueError('Dictionaries are not supported for the cuts anymore, use a list')

    for c in selection_config:
        if len(c) > 1:
            raise ValueError(
                "Expected dict with single entry column: [operator, value]."
            )
        name, (operator, value) = list(c.items())[0]

        before = np.count_nonzero(mask)
        selection = OPERATORS[operator](infile[key][name][start:end], value)
        mask = np.logical_and(mask, selection)
        after = np.count_nonzero(mask)
        log.debug(
            'Cut "{} {} {}" removed {} events'.format(
                name, operator, value, before - after
            )
        )

    return mask


def create_mask_table(
    table,
    selection_config,
    n_events,
    start=None,
    end=None,
):
    """
    Creates a boolean mask for a pytables.Table object

    Parameters:
    -----------
    table: pytables.Table
        Table to perform selection on
    selection_config: dict
        Dictionary with column names as keys and (operator, value) tuples as value
    n_events: int
        Number of events to select.
    start: int
        If None, select the first row
    end: int
        If None, select the last row

    Returns:
    --------
    Boolean mask with len n_events or len(table) if unspecified
    """
    start = start or 0
    end = min(n_events, end) if end else n_events

    n_selected = end - start
    mask = np.ones(n_selected, dtype=bool)

    for c in selection_config:
        if len(c) > 1:
            raise ValueError(
                "Expected dict with single entry column: [operator, value]."
            )
        name, (operator, value) = list(c.items())[0]

        before = np.count_nonzero(mask)
        if name not in table.colnames:
            raise KeyError(
                f"Cant perform selection based on {name} "
                "Column is missing from parameters table"
            )
        selection = OPERATORS[operator](table.col(name)[start:end], value)
        mask = np.logical_and(mask, selection)
        after = np.count_nonzero(mask)
        log.debug(
            'Cut "{} {} {}" removed {} events'.format(
                name, operator, value, before - after
            )
        )

    return mask


def apply_cuts_h5py_chunked(
    input_path,
    output_path,
    selection_config,
    key="events",
    chunksize=100000,
    progress=True,
):
    """
    Apply cuts defined in selection config to input_path and write result to
    outputpath. Apply cuts to chunksize events at a time.
    """

    n_events = get_number_of_rows_in_table(
        input_path,
        key=key,
    )
    n_chunks = int(np.ceil(n_events / chunksize))
    log.debug("Using {} chunks of size {}".format(n_chunks, chunksize))

    with h5py.File(input_path, "r") as infile, h5py.File(output_path, "w") as outfile:
        group = outfile.create_group(key)

        for chunk in tqdm(range(n_chunks), disable=not progress, total=n_chunks):
            start = chunk * chunksize
            end = min(n_events, (chunk + 1) * chunksize)

            mask = create_mask_h5py(
                infile, selection_config, n_events, key=key, start=start, end=end
            )
            for name, dataset in infile[key].items():
                if chunk == 0:
                    if dataset.ndim == 1:
                        group.create_dataset(
                            name, data=dataset[start:end][mask], maxshape=(None,)
                        )
                    elif dataset.ndim == 2:
                        group.create_dataset(
                            name,
                            data=dataset[start:end, :][mask, :],
                            maxshape=(None, 2),
                        )
                    else:
                        log.warning("Skipping not 1d or 2d column {}".format(name))

                else:

                    n_old = group[name].shape[0]
                    n_new = np.count_nonzero(mask)
                    group[name].resize(n_old + n_new, axis=0)

                    if dataset.ndim == 1:
                        group[name][n_old : n_old + n_new] = dataset[start:end][mask]
                    elif dataset.ndim == 2:
                        group[name][n_old : n_old + n_new, :] = dataset[start:end][
                            mask, :
                        ]
                    else:
                        log.warning("Skipping not 1d or 2d column {}".format(name))


def apply_cuts_cta_dl1(
    input_path,
    output_path,
    selection_config,
    keep_images=True,
):
    """
    Apply cuts from a selection config to a cta dl1 file and write results
    to output_path.
    """
    filters = tables.Filters(
        complevel=5,  # compression medium, tradeoff between speed and compression
        complib="blosc:zstd",  # use modern zstd algorithm
        fletcher32=True,  # add checksums to data chunks
    )
    n_rows_before = 0
    n_rows_after = 0
    with tables.open_file(input_path) as in_, tables.open_file(
        output_path, "w", filters=filters
    ) as out_:
        # perform cuts on the measured parameters
        remaining_showers = set()
        for table in in_.root.dl1.event.telescope.parameters:
            key = "/dl1/event/telescope/parameters"
            mask = create_mask_table(
                table,
                selection_config,
                n_events=len(table),
            )
            new_table = out_.create_table(
                key,
                table.name,
                table.description,
                createparents=True,
                expectedrows=np.count_nonzero(mask),
            )
            # set user attributes
            for name in table.attrs._f_list():
                new_table.attrs[name] = table.attrs[name]
            n_rows_before += len(table)
            data = table.read()
            new_table.append(data[mask])
            remaining_showers.update(data[mask][["obs_id", "event_id"]].tolist())

            n_rows_after += np.count_nonzero(mask)
        selection_table = Table(data=np.array(list(remaining_showers)), names=['obs_id', 'event_id'])
        # copy the other tables disregarding events with no more observations
        for table in in_.walk_nodes():
            # skip groups, we create the parents anyway
            if isinstance(table, tables.Group):
                continue

            if not keep_images:
                if table._v_parent._v_pathname == "/dl1/event/telescope/images":
                    continue
                elif (
                    table._v_parent._v_pathname == "/simulation/event/telescope/images"
                ):
                    continue
            # parameter tables were already processed
            if table._v_parent._v_pathname == "/dl1/event/telescope/parameters":
                continue

            new_table = out_.create_table(
                table._v_parent._v_pathname,
                table.name,
                table.description,
                createparents=True,
            )
            # set user attributes
            for name in table.attrs._f_list():
                new_table.attrs[name] = table.attrs[name]
            mask = np.ones(len(table), dtype=bool)
            # they dont appear individually
            if "event_id" in table.colnames:
                selected = join(selection_table, table.read(), keys=['obs_id', 'event_id'], join_type='left')
                new_table.append(selected.as_array().astype(table.dtype))
            else:
                new_table.append(table[:])


        with warnings.catch_warnings():
            warnings.simplefilter("ignore", NaturalNameWarning)
            for name in in_.root._v_attrs._f_list():
                out_.root._v_attrs[name] = in_.root._v_attrs[name]

    return n_rows_before, n_rows_after
