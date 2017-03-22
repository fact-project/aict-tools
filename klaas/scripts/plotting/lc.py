import logging
import pandas as pd
import matplotlib.pyplot as plt
import h5py
# from matplotlib.ticker import FuncFormatter
# from matplotlib.ticker import FixedLocator
import click
import numpy as np
import sys
import astropy.units as u

native_byteorder = native_byteorder = {'little': '<', 'big': '>'}[sys.byteorder]


def to_native_byteorder(array):
    ''' Convert numpy array to native byteorder '''
    if array.dtype.byteorder not in ('=', native_byteorder):
        return array.byteswap().newbyteorder()
    else:
        return array


def to_native(df):
    ''' Convert data frame to native byteorder '''
    for col in df.columns:
        df[col] = to_native_byteorder(df[col].values)
    return df


def li_ma_significance(N_on, N_off, alpha=0.2):
    if N_on + N_off == 0:
        return 0

    p_on = N_on / (N_on + N_off)
    p_off = N_off / (N_on + N_off)

    if p_on == 0:
        return 0

    t1 = N_on * np.log(((1 + alpha) / alpha) * p_on)
    t2 = N_off * np.log((1 + alpha) * p_off)

    ts = (t1 + t2)

    significance = np.sqrt(ts * 2)

    return significance


def unixtimes_to_pandas_datetime(h5file):
    '''
    unixTimeUTC is the 2D representation of the
    eventtimestamps as produced by the FACT DAQ.
    Returns a pandas datatime thing.
    '''
    seconds = h5file['events/UnixTimeUTC'][:, 0]
    micros = h5file['events/UnixTimeUTC'][:, 1]

    return pd.to_datetime(seconds + micros/1000000, unit='s')

#
# def remove_colons(df):
#     rename_dict = {k:k.replace(":", "_") for k in df.columns}
#     return df.rename_axis(rename_dict, axis='columns')
#

#
# def axis_formatter(seconds, tick_position):
#     dt = datetime.datetime.utcfromtimestamp(seconds)
#     return dt.strftime("%d %b %Y")

# def get_tick_locations(df):
#     min_date = df.index.min()
#     max_date = df.index.max()
#     date_range = pd.date_range(min_date, max_date,  freq='MS', tz='UTC')
#     s = [min_date] + date_range.tolist() + [max_date]
#     return [t.timestamp() for t in s]

    # return [left_tick.timestamp()] + s


def theta_degrees_to_theta_squared_mm(theta):
    '''
    Convert theta from fact-tols output (in mm) to theta^2 in mm.
    This formula contains at two approximations.
    1. The mirror is a perfect parabola
    2. The area around the point source in the camera, aka the 'phase space',  grows
        with r**2. I think its missing a cosine somewhere. but thats okay.
    '''
    pixelsize = 9.5  # mm
    fov_per_pixel = 0.11  # degree
    return (theta * (fov_per_pixel / pixelsize))**2


def signal_thetas(df, threshold):
    return len(df[df.theta_squared < threshold])


def background_thetas(df, threshold):
    l = [df['theta_off_{}_squared'.format(region)] for region in [1, 2, 3, 4, 5]]
    background_thetas = pd.concat(l).values
    return len(background_thetas[background_thetas < threshold])


@click.command()
@click.argument('infile', type=click.Path(file_okay=True, dir_okay=False, exists=True))
@click.argument('out', type=click.Path(file_okay=True, dir_okay=False))
@click.option('--date_range', nargs=2, help='date range to plot')
@click.option(
    '--binning',
    type=click.INT,
    help='Binning in minutes. As in 5, or 30',
    default='20'
    )
@click.option(
    '-c',
    '--prediction_threshold',
    default=0.94,
    help='prediction threshold'
    )
@click.option(
    '-t',
    '--theta_square_cut',
    default=0.025,
    help='size of selected signal region'
    )
def lc(infile, out, date_range, binning, prediction_threshold, theta_square_cut):
    logging.basicConfig(level=logging.INFO)

    threshold = prediction_threshold
    theta_cut = theta_square_cut

    num_off_regions = 5
    alpha = 1.0/num_off_regions

    # f = h5py.File(infile)
    #
    # d = {
    #     'signal_prediction': f['events/signal_prediction'],
    #     'theta': f['events/Theta'],
    #     'theta_off_1': f['events/Theta_Off_1'],
    #     'theta_off_2': f['events/Theta_Off_2'],
    #     'theta_off_3': f['events/Theta_Off_3'],
    #     'theta_off_4': f['events/Theta_Off_4'],
    #     'theta_off_5': f['events/Theta_Off_5'],
    #     'theta_recpos_x': f['events/Theta_recPos'][:, 0],
    #     'theta_recpos_y': f['events/Theta_recPos'][:, 1],
    #     'run_id': f['events/RUNID'],
    #     'night': f['events/NIGHT'],
    #     'timestamp': unixtimes_to_pandas_datetime(f)
    #     }
    #
    # events = pd.DataFrame(d)
    # events = to_native(events)
    # events['theta_squared'] = theta_degrees_to_theta_squared_mm(events.theta)
    # events['theta_off_1_squared'] = theta_degrees_to_theta_squared_mm(events.theta_off_1)
    # events['theta_off_2_squared'] = theta_degrees_to_theta_squared_mm(events.theta_off_2)
    # events['theta_off_3_squared'] = theta_degrees_to_theta_squared_mm(events.theta_off_3)
    # events['theta_off_4_squared'] = theta_degrees_to_theta_squared_mm(events.theta_off_4)
    # events['theta_off_5_squared'] = theta_degrees_to_theta_squared_mm(events.theta_off_5)
    #
    # runs = pd.DataFrame(dict(f['runs']))
    # # runs = to_native(runs)
    # # merge the whole thing to get ontime
    # data = pd.merge(runs, events, on=['night', 'run_id'])
    #

    data = pd.read_hdf(infile, key='table')
    # TODO select dateimt ranges
    # df = df.sort_index()
    # if len(date_range) == 2:
    #     df = df[date_range[0]:date_range[1]]
    # start = df.index.searchsorted(datetime.datetime(2015, 12, 15))
    # end = df.index.searchsorted(datetime.datetime(2015, 12, 18))
    # df = df.ix[start:end]

    # start_event = data.timestamp.min()

    lc = []
    for name, group in data.groupby(['run_id', 'night']):
        if group.empty:
            continue

        first_event = group.timestamp.min()
        latest_event = group.timestamp.max()

        duration = latest_event - first_event

        group = group.query('(signal_prediction > 1 - {})'.format(threshold))

        n_on = signal_thetas(group, theta_cut)
        n_off = background_thetas(group, theta_cut)
        d = {
            'min': first_event,
            'center': first_event + duration * 0.5,
            'max': latest_event,
            'n_on': n_on,
            'n_off': n_off,
            'excess': (n_on - alpha * n_off),
            'yerror': np.sqrt(n_on + alpha**2 * n_off),
            'xerror': duration * 0.5,
            'duration': duration,
            'significance': li_ma_significance(n_on, n_off)
        }
        lc.append(d)

    df_lc = pd.DataFrame(lc)

    from IPython import embed
    embed()


    # print(df_lc.query('excess > 20'))
    # df_lc = df_lc.set_index(df_lc['min'])
    # fig, ax = plt.subplots()
    # ax.errorbar(df_lc['center'], df_lc['excess'], yerr=df_lc['yerror'],
    #             xerr=df_lc['xerror'], fmt='h', color='#71d0ff', markeredgecolor='#4b4b4b')
    # ax.set_ylabel('Excess Events per Hour')
    # ax.set_axis_bgcolor('#585858')
    # ax.grid(color='#7c7c7c')
    # fig.autofmt_xdate()
    # # import matplotlib.dates as mdates
    # # ax.fmt_xdata = mdates.DateFormatter('%b %d %Y %H:%M')
    #
    # plt.savefig(out)


if __name__ == '__main__':
    lc()
