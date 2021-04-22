import os
import joblib
import logging
import numpy as np
import pandas as pd
import h5py
import click
from sklearn.base import is_classifier
import tables

from fact.io import read_h5py, write_data

from .feature_generation import feature_generation
from .preprocessing import convert_units
from . import __version__

__all__ = [
    "drop_prediction_column",
    "read_telescope_data",
    "read_telescope_data_chunked",
    "save_model",
    "load_model",
]


log = logging.getLogger(__name__)


def write_hdf(data, path, table_name, mode="w", **kwargs):
    write_data(data, path, key=table_name, use_h5py=True, mode=mode, **kwargs)


def get_number_of_rows_in_table(path, key):
    with h5py.File(path, "r") as f:
        element = f.get(key)
        if element is None:
            raise ValueError(f"File {path} does not contain {key}")

        if isinstance(element, h5py.Group):
            return element[next(iter(element.keys()))].shape[0]
        elif isinstance(element, h5py.Dataset):
            return element.shape[0]

        raise ValueError(f'Unsupported object found: {element}')


def read_data(file_path, key=None, columns=None, first=None, last=None, **kwargs):
    """
    This is similar to the read_data function in fact.io
    pandas hdf5:   pd.HDFStore
    h5py hdf5:     fact.io.read_h5py
    """
    _, extension = os.path.splitext(file_path)

    if extension in [".hdf", ".hdf5", ".h5"]:
        try:
            df = pd.read_hdf(
                file_path, key=key, columns=columns, start=first, stop=last, **kwargs
            )
        except (TypeError, ValueError):
            df = read_h5py(
                file_path, key=key, columns=columns, first=first, last=last, **kwargs
            )
        return df
    else:
        raise NotImplementedError(
            f"AICT tools cannot handle data with extension {extension} yet."
        )


def drop_prediction_column(data_path, group_name, column_name, yes=True):
    """
    Deletes prediction columns in a h5py file if the columns exist.
    Including 'mean' and 'std' columns.
    """
    n_del = 0
    with h5py.File(data_path, "r+") as f:

        if group_name not in f.keys():
            return n_del

        columns = f[group_name].keys()
        if column_name in columns:
            if not yes:
                click.confirm(
                    f'Column "{column_name}" exists in file, overwrite?',
                    abort=True,
                )

            del f[group_name][column_name]
            n_del += 1
            log.warn("Deleted {} from the feature set.".format(column_name))

        if column_name + "_std" in columns:
            del f[group_name][column_name + "_std"]
            n_del += 1
            log.warn("Deleted {} from the feature set.".format(column_name + "_std"))
        if column_name + "_mean" in columns:
            del f[group_name][column_name + "_mean"]
            n_del += 1
            log.warn("Deleted {} from the feature set.".format(column_name) + "_mean")
    return n_del


def drop_prediction_groups(data_path, group_name, yes=True):
    """
    Deletes prediction groups in a h5py file if the group exists.
    Including 'mean' and 'std' groups.
    This is pretty hardcoded for the moment as its only used for CTA.
    ToDo: Generalize this.
    """
    n_del = 0
    with h5py.File(data_path, "r+") as f:

        if "dl2" not in f.keys():
            return n_del
        tel_group = f["dl2"]["event"]["telescope"]
        for tel in tel_group:
            if group_name in tel_group[tel]:
                if not yes:
                    click.confirm(
                        f'Group "{group_name}" exists in group /dl2/event/telescope/{tel} overwrite?',
                        abort=True,
                    )
                del tel_group[tel][group_name]
                log.warn(
                    f'Group "{group_name}" deleted from group /dl2/event/telescope/{tel}'
                )
                n_del += 1

        if "subarray" not in f["dl2"]["event"]:
            return n_del
        array_group = f["dl2"]["event"]["subarray"]
        if group_name in array_group:
            if not yes:
                click.confirm(
                    f'Group "{group_name}" exists in group /dl2/event/subarray, overwrite?',
                    abort=True,
                )
            del array_group[group_name]
            log.warn(f'Group "{group_name}" deleted from group /dl2/event/subarray')
            n_del += 1
    return n_del


def read_telescope_data_chunked(
    path, aict_config, chunksize, columns=None, feature_generation_config=None
):
    """
    Reads data from hdf5 file given as PATH and yields merged
    dataframes with feature generation applied for each chunk
    """
    return TelescopeDataIterator(
        path,
        aict_config,
        chunksize,
        columns,
        feature_generation_config=feature_generation_config,
    )


def read_data_chunked(path, table_name, chunksize, columns=None):
    """
    Reads data from hdf5 file given as PATH and yields dataframes for each chunk
    """
    return HDFDataIterator(
        path,
        table_name,
        chunksize,
        columns,
    )


class HDFDataIterator:
    def __init__(
        self,
        path,
        table_name,
        chunksize,
        columns,
    ):
        self.path = path
        self.table_name = table_name
        self.n_rows = get_number_of_rows_in_table(path, table_name)
        self.columns = columns
        if chunksize:
            self.chunksize = chunksize
            self.n_chunks = int(np.ceil(self.n_rows / chunksize))
        else:
            self.n_chunks = 1
            self.chunksize = self.n_rows
        log.info("Splitting data into {} chunks".format(self.n_chunks))

        self._current_chunk = 0

    def __len__(self):
        return self.n_chunks

    def __iter__(self):
        return self

    def __next__(self):
        if self._current_chunk == self.n_chunks:
            raise StopIteration

        chunk = self._current_chunk
        start = chunk * self.chunksize
        end = min(self.n_rows, (chunk + 1) * self.chunksize)
        self._current_chunk += 1
        df = read_data(
            self.path,
            key=self.table_name,
            columns=self.columns,
            first=start,
            last=end,
        )

        return df, start, end


class TelescopeDataIterator:
    def __init__(
        self,
        path,
        aict_config,
        chunksize,
        columns,
        feature_generation_config=None,
    ):
        self.aict_config = aict_config
        self.columns = columns
        self.feature_generation_config = feature_generation_config
        if aict_config.data_format == "CTA":
            with tables.open_file(path) as t:
                if aict_config.telescopes:
                    tables_ = []
                    for key in aict_config.telescopes:
                        if key in t.root.dl1.event.telescope.parameters:
                            tables_.append(
                                (
                                    f"/dl1/event/telescope/parameters/{key}",
                                    len(t.root.dl1.event.telescope.parameters[key]),
                                )
                            )
                        else:
                            log.warning(f"Didnt find telescope: {key}")
                else:
                    tables_ = [
                        (f"/dl1/event/telescope/parameters/{tel.name}", len(tel))
                        for tel in t.root.dl1.event.telescope.parameters
                    ]
            self.n_rows = sum([count for key, count in tables_])
            self.tables = iter(tables_)
        else:
            self.n_rows = get_number_of_rows_in_table(path, aict_config.events_key)
            self.tables = iter([(aict_config.events_key, self.n_rows)])
        self.path = path
        if chunksize:
            self.chunksize = chunksize
            self.n_chunks = int(np.ceil(self.n_rows / chunksize))
        else:
            self.n_chunks = 1
            self.chunksize = self.n_rows
        log.info("Splitting data into {} chunks".format(self.n_chunks))

        self.exhausted_table_lengths = 0
        self.table = next(self.tables)
        self._current_chunk = 0
        self._index_start = 0

    def __len__(self):
        return self.n_chunks

    def __iter__(self):
        return self

    def __next__(self):
        if self._current_chunk == self.n_chunks:
            raise StopIteration

        chunk = self._current_chunk
        start = chunk * self.chunksize
        end = min(self.n_rows, (chunk + 1) * self.chunksize)
        self._current_chunk += 1
        if self.exhausted_table_lengths > 0:
            start -= self.exhausted_table_lengths
            end -= self.exhausted_table_lengths

        df = read_telescope_data(
            self.path,
            aict_config=self.aict_config,
            columns=self.columns,
            first=start,
            last=end,
            key=self.table[0],
        )

        # In case the table is exhausted, continue with the next until
        # the desired chunksize is reached
        while len(df) < self.chunksize:
            start = 0
            end -= self.table[1]
            exhausted_table_length = self.table[1]
            try:
                self.table = next(self.tables)
            except StopIteration:
                break

            next_df = read_telescope_data(
                self.path,
                aict_config=self.aict_config,
                columns=self.columns,
                first=start,
                last=end,
                key=self.table[0],
            )

            self.exhausted_table_lengths += exhausted_table_length
            df = pd.concat([df, next_df])

        df.index = np.arange(self._index_start, self._index_start + len(df))

        index_start = self._index_start
        self._index_start += len(df)
        index_stop = self._index_start

        if self.feature_generation_config:
            feature_generation(df, self.feature_generation_config, inplace=True)

        return df, index_start, index_stop


def get_column_names_in_file(path, table_name):
    """Returns the list of column names in the given group

    Parameters
    ----------
    path : str
        path to hdf5 file
    table_name : str
        name of group/table in file

    Returns
    -------
    list
        list of column names
    """
    with h5py.File(path, "r") as f:
        return list(f[table_name].keys())


def remove_column_from_file(path, table_name, column_to_remove):
    """
    Removes a column from a hdf5 file.

    Note: this is one of the reasons why we decided to not support pytables.
    In case of 'tables' format this needs to copy the entire table
    into memory and then some.


    Parameters
    ----------
    path : str
        path to hdf5 file
    table_name : str
        name of the group/table from which the column should be removed
    column_to_remove : str
        name of column to remove
    """
    with h5py.File(path, "r+") as f:
        del f[table_name][column_to_remove]


def is_sorted(values, stable=False):
    i = 1 if stable else 0
    return (np.diff(values) >= i).all()


def has_holes(values):
    return (np.diff(values) > 1).any()


def read_telescope_data(
    path,
    aict_config,
    columns=None,
    feature_generation_config=None,
    n_sample=None,
    first=None,
    last=None,
    key=None,
):
    """Read columns from data in file given under PATH.
    Returns a single pandas data frame containing all the requested data.

    Parameters
    ----------
    path : str
        path to the hdf5 file to read
    aict_config : AICTConfig
        The configuration object.
        This is needed for gathering the primary keys to merge merge on.
    columns : list, optional
        column names to read, by default None
    feature_generation_config : FeatureGenerationConfig, optional
        The configuration object containing the information
        for feature generation, by default None
    n_sample : int, optional
        number of rows to randomly sample from the file, by default None
    first : int, optional
        first row to read from file, by default None
    last : int, optional
        last row to read form file, by default None
    key: str, optional
        Specify the telescope table to load in case of cta files
        This is used for chunkwise reading
    Returns
    -------
    pd.DataFrame
        Dataframe containing the requested data.

    """
    # In case of a simple DataFrame just call read_data accordingly
    if aict_config.data_format == "simple":
        df = read_data(
            file_path=path,
            key=aict_config.events_key,
            columns=columns,
            first=first,
            last=last,
        )
        df = convert_units(df, aict_config)
    # For cta files multiple tables need to be read and appended/merged
    elif aict_config.data_format == "CTA":
        df = read_cta_dl1(
            path,
            key=key,
            columns=columns,
            aict_config=aict_config,
            first=first,
            last=last,
        )
    if n_sample is not None:
        if n_sample > len(df):
            raise ValueError(
                "number of sampled events"
                " {} must be smaller than number events in file {} ({})".format(
                    n_sample, path, len(df)
                )
            )
        log.info("Randomly sample {} events".format(n_sample))
        state = np.random.RandomState()
        state.set_state(np.random.get_state())
        df = df.sample(n_sample, random_state=state)

    # generate features if given in config
    if feature_generation_config:
        feature_generation(df, feature_generation_config, inplace=True)

    return df


def read_cta_dl1(path, aict_config, key=None, columns=None, first=None, last=None):
    from ctapipe.io import read_table
    from astropy.table import Table, join, vstack
    import astropy.units as u

    # choose the telescope tables to load
    with tables.open_file(path) as file_table:
        if key:
            tels_to_load = (key,)
            print(path, key)
        else:
            if aict_config.telescopes:
                tels_to_load = [
                    f"/dl1/event/telescope/parameters/{tel.name}"
                    for tel in file_table.root.dl1.event.telescope.parameters
                    if tel.name in aict_config.telescopes
                ]
            else:
                tels_to_load = [
                    f"/dl1/event/telescope/parameters/{tel.name}"
                    for tel in file_table.root.dl1.event.telescope.parameters
                ]

        if "equivalent_focal_length" in columns:
            layout = Table.read(path, "/configuration/instrument/subarray/layout")
            optics = Table.read(path, "/configuration/instrument/telescope/optics")
            optics["tel_description"] = optics["description"]
            optics.meta.clear()
            layout = join(
                layout,
                optics[["tel_description", "equivalent_focal_length"]],
                join_type="left",
                keys="tel_description",
            )

        # load the telescope parameter table(s)
        tel_tables = []

        for tel in tels_to_load:
            log.info(f'Loading data for telescope {tel}')
            # as not all columns are located here, we cant just use
            # columns=columns
            tel_table = read_table(path, tel)[first:last]

            # astropy table joins do not preserve input order
            # but sort by the join keys, we add this simple index
            # so we can restore the input order after joining
            tel_table["__index"] = np.arange(len(tel_table))

            # Pointing information has to be loaded from the monitoring tables and interpolated
            # We also need the trigger tables as monitoring is based on time not events
            if columns:
                if "azimuth" in columns or "altitude" in columns:
                    tel_key = tel.split("/")[-1]
                    tel_triggers = read_table(path, "/dl1/event/telescope/trigger")
                    tel_table = join(
                        tel_table,
                        tel_triggers,
                        join_type="inner",
                        keys=["obs_id", "event_id", "tel_id"],
                    )
                    tel_pointings = read_table(
                        path, f"/dl1/monitoring/telescope/pointing/{tel_key}"
                    )
                    if aict_config.datamodel_version > "1.0.0":
                        time_key = "time"
                    else:
                        time_key = "telescopetrigger_time"
                    tel_table["azimuth"] = np.interp(
                        tel_table[time_key].mjd,
                        tel_pointings[time_key].mjd,
                        tel_pointings["azimuth"].quantity.to_value(u.deg),
                    ) * u.deg
                    tel_table["altitude"] = np.interp(
                        tel_table[time_key].mjd,
                        tel_pointings[time_key].mjd,
                        tel_pointings["altitude"].quantity.to_value(u.deg),
                    ) * u.deg
                if "equivalent_focal_length" in columns:
                    tel_table = join(
                        tel_table,
                        layout[["tel_id", "equivalent_focal_length"]],
                        join_type="left",
                        keys="tel_id",
                    )
                if columns:
                    # True / Simulation columns are still missing, so only use the columns already present
                    tel_table = tel_table[
                        list(set(columns).intersection(tel_table.columns)) + ['__index']
                    ].copy()

            # restore the input order that was changed by astropy.table.join
            # and remove the additional column
            tel_table.sort('__index')
            tel_table.remove_column('__index')
            tel_tables.append(tel_table)

        # Monte carlo information is located in the simulation group
        # and we are interested in the array wise true information only
        event_table = vstack(tel_tables)
        event_table = convert_units(event_table, aict_config)
        df = pd.DataFrame(event_table.as_array()) # workaround for #11286 in astropy 4.2
        if columns:
            true_columns = [x for x in columns if x.startswith("true")]
            if true_columns:
                true_information = read_table(
                    path, "/simulation/event/subarray/shower"
                )[true_columns + ["obs_id", "event_id"]]
                true_information = true_information.to_pandas()
                df = df.merge(true_information, on=["obs_id", "event_id"], how="left")
    return df


def save_model(model, feature_names, model_path, label_text="label"):
    p, extension = os.path.splitext(model_path)
    model.feature_names = feature_names
    pickle_path = p + ".pkl"

    if extension == ".pmml":
        try:
            from sklearn2pmml import sklearn2pmml, PMMLPipeline
        except ImportError:
            raise ImportError(
                "You need to install `sklearn2pmml` to store models in pmml format"
            )

        pipeline = PMMLPipeline([("model", model)])
        pipeline.target_field = label_text
        pipeline.active_fields = np.array(feature_names)
        sklearn2pmml(pipeline, model_path)

    elif extension == ".onnx":

        try:
            from skl2onnx import convert_sklearn
            from skl2onnx.common.data_types import FloatTensorType
            from skl2onnx.helpers.onnx_helper import select_model_inputs_outputs
            from onnx.onnx_pb import StringStringEntryProto
        except ImportError:
            raise ImportError(
                "You need to install `skl2onnx` to store models in onnx format"
            )

        onnx = convert_sklearn(
            model,
            name=label_text,
            initial_types=[("input", FloatTensorType((None, len(feature_names))))],
            doc_string="Model created by aict-tools to estimate {}".format(label_text),
        )

        # this makes sure we only get the scores and that they are numpy arrays and not
        # a list of dicts.
        # must come before setting metadata as it clears the metadata_props
        if hasattr(model, "predict_proba"):
            onnx = select_model_inputs_outputs(onnx, ["probabilities"])

        metadata = dict(
            model_author="aict-tools",
            aict_tools_version=__version__,
            feature_names=",".join(feature_names),
            model_type="classifier" if is_classifier(model) else "regressor",
        )
        for key, value in metadata.items():
            onnx.metadata_props.append(StringStringEntryProto(key=key, value=value))

        with open(model_path, "wb") as f:
            f.write(onnx.SerializeToString())
    else:
        pickle_path = model_path

    # Always store the pickle dump,just in case
    joblib.dump(model, pickle_path, compress=4)


def load_model(model_path):
    name, ext = os.path.splitext(model_path)
    if ext == ".onnx":
        from .onnx import ONNXModel

        return ONNXModel(model_path)

    if ext == ".pmml":
        from .pmml import PMMLModel

        return PMMLModel(model_path)

    return joblib.load(model_path)


def append_column_to_hdf5(path, array, table_name, new_column_name):
    """
    Add array of values as a new column to the given file.

    Parameters
    ----------
    path : str
        path to file
    array : array-like
        values to append to the file
    table_name : str
        name of the group to append to
    new_column_name : str
        name of the new column
    """
    with h5py.File(path, "r+") as f:
        group = f.require_group(table_name)  # create if not exists

        max_shape = list(array.shape)
        max_shape[0] = None
        if new_column_name not in group.keys():
            group.create_dataset(
                new_column_name,
                data=array,
                maxshape=tuple(max_shape),
            )
        else:
            n_existing = group[new_column_name].shape[0]
            n_new = array.shape[0]

            group[new_column_name].resize(n_existing + n_new, axis=0)
            group[new_column_name][n_existing : n_existing + n_new] = array


def set_sample_fraction(path, fraction):
    with h5py.File(path, mode="r+") as f:
        before = f.attrs.get("sample_fraction", 1.0)
        f.attrs["sample_fraction"] = before * fraction


def copy_group(inpath, outpath, group):
    with h5py.File(inpath, mode="r") as infile, h5py.File(outpath, "a") as outfile:
        if group in infile:
            group_path = infile[group].parent.name
            out_group = outfile.require_group(group_path)
            log.info('Copying group "{}"'.format(group))
            infile.copy(group, out_group)


def append_predictions_cta(file_path, df, table_path):
    with tables.open_file(file_path, mode="a") as f:
        if table_path not in f:
            group, table_name = os.path.split(table_path)
            f.create_table(
                group,
                table_name,
                df.to_records(),
                createparents=True,
            )
        else:
            f.get_node(table_path).append(df.to_records())
