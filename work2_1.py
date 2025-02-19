import os

# from numpy.array_api import float64
from scipy.ndimage import generic_filter
import pandas as pd
import numpy as np
import datetime as dt
from scipy.signal import savgol_filter
from typing import Dict
from typing import List
import matplotlib.pyplot as plt
import matplotlib.axes as axs
import matplotlib.figure as fig
import matplotlib.colors as mplcolors
import matplotlib.cm as mplcm
import reader_cmn as rcmn
import math
from scipy.linalg import lstsq
import h5py
import read_los_file as rlos
import re
import work_site_file as wsite
import work_los_file as wlos
import multiprocessing.pool as multipool
import multiprocessing as mp
import copy
import statistics
from scipy.fft import fft, fftfreq
import learning_f5py as lf5py
import itertools
import work_file_system as wfs

from draw_plot import save_path2

SOURCE_FILE = r"/home/vadymskipa/Documents/PhD_student/temp/SAVE2/267-2020-09-23/G04.txt"
SOURCE_FILE2 = r"/home/vadymskipa/Documents/PhD_student/temp/SAVE2/267-2020-09-23/G02.txt"
SOURCE_FILE2_1 = r"/home/vadymskipa/Documents/PhD_student/temp/SAVE2/268-2020-09-24/G02.txt"
SOURCE_FILE3 = r"/home/vadymskipa/Documents/PhD_student/temp/SAVE2/267-2020-09-23/G08.txt"
SOURCE_FILE16 = r"/home/vadymskipa/Documents/PhD_student/temp/SAVE2/267-2020-09-23/G16.txt"

SOURCE_CMN_FILE1 = r"/home/vadymskipa/Documents/GPS_2023_outcome/BASE057-2023-02-26.Cmn"
SOURCE_CMN_FILE2 = r"/home/vadymskipa/Documents/GPS_2023_outcome/BASE058-2023-02-27.Cmn"
SOURCE_CMN_FILE3 = r"/home/vadymskipa/Documents/GPS_2023_outcome/BASE059-2023-02-28.Cmn"
SOURCE_CMN_FILE4 = r"/home/vadymskipa/Documents/Dyplom/GPS and TEC data/267/BASE267-2020-09-23.Cmn"
SOURCE_CMN_FILE5 = r"/home/vadymskipa/Documents/Dyplom/GPS and TEC data/268/BASE268-2020-09-24.Cmn"


SAVE_PATH = r"/home/vadymskipa/Documents/PhD_student/temp/SAVE3/"
SAVE_PATH2 = r"/home/vadymskipa/Documents/PhD_student/temp/SAVE4/"
SAVE_PATH5 = r"/home/vadymskipa/Documents/PhD_student/temp/SAVE5/"
SAVE_PATH6 = r"/home/vadymskipa/Documents/PhD_student/temp/SAVE6/"


SOURCE_DIRECTORY1 = r"/home/vadymskipa/Documents/PhD_student/temp/SAVE2/267-2020-09-23/"
SOURCE_DIRECTORY2 = r"/home/vadymskipa/Documents/PhD_student/temp/SAVE2/268-2020-09-24/"
SOURCE_DIRECTORY_HDF_EUROPE = r"/home/vadymskipa/Documents/PhD_student/data/Europe/"

PERIOD_CONST = 30


# hour	min 	sec   	los_tec	azm   	elm   	gdlat 	gdlon
def read_sat_file(file=SOURCE_FILE):
    arr_hour = []
    arr_min = []
    arr_sec = []
    arr_los_tec = []
    arr_azm = []
    arr_elm = []
    arr_gdlat = []
    arr_gdlon = []
    arr_of_arr = [arr_hour, arr_min, arr_sec, arr_los_tec, arr_azm, arr_elm, arr_gdlat, arr_gdlon]
    with open(file, "r") as reader:
        reader.readline()
        for line in reader:
            text_tuple = line.split()
            arr_of_arr[0].append(int(text_tuple[0]))
            arr_of_arr[1].append(int(text_tuple[1]))
            arr_of_arr[2].append(int(text_tuple[2]))
            arr_of_arr[3].append(float(text_tuple[3]))
            arr_of_arr[4].append(float(text_tuple[4]))
            arr_of_arr[5].append(float(text_tuple[5]))
            arr_of_arr[6].append(float(text_tuple[6]))
            arr_of_arr[7].append(float(text_tuple[7]))
    outcome_dataframe = pd.DataFrame({"hour": arr_hour, "min": arr_min, "sec": arr_sec, "los_tec": arr_los_tec,
                                      "azm": arr_azm, "elm": arr_elm, "gdlat": arr_gdlat, "gdlon": arr_gdlon, })
    return outcome_dataframe


def read_dtec_file(file):
    arr_hour = []
    arr_min = []
    arr_sec = []
    arr_dtec = []
    arr_azm = []
    arr_elm = []
    arr_gdlat = []
    arr_gdlon = []
    arr_of_arr = [arr_hour, arr_min, arr_sec, arr_dtec, arr_azm, arr_elm, arr_gdlat, arr_gdlon]

    with open(file, "r") as reader:
        reader.readline()
        for line in reader:
            text_tuple = line.split()
            arr_of_arr[0].append(int(text_tuple[0]))
            arr_of_arr[1].append(int(text_tuple[1]))
            arr_of_arr[2].append(int(text_tuple[2]))
            arr_of_arr[3].append(float(text_tuple[3]))
            arr_of_arr[4].append(float(text_tuple[4]))
            arr_of_arr[5].append(float(text_tuple[5]))
            arr_of_arr[6].append(float(text_tuple[6]))
            arr_of_arr[7].append(float(text_tuple[7]))
    outcome_dataframe = pd.DataFrame({"hour": arr_hour, "min": arr_min, "sec": arr_sec, "dtec": arr_dtec,
                                      "azm": arr_azm, "elm": arr_elm, "gdlat": arr_gdlat, "gdlon": arr_gdlon, })
    return outcome_dataframe


# Додає до pandas dataframe, в якому є стовпчики "hour", "min", "sec", стовчик з даними "timestamp"
def add_timestamp_column_to_df(dataframe: pd.DataFrame, date: dt.datetime):
    date_timestamp = date.timestamp()
    new_dataframe = dataframe.assign(timestamp=dataframe["hour"] * 3600 + dataframe["min"] * 60 + dataframe["sec"] +
                                               date_timestamp)
    return new_dataframe


# Додає до pandas dataframe, в якому є стовпчики "timestamp", стовчик з даними "datetime"
def add_datetime_column_to_df(dataframe: pd.DataFrame):
    dataframe.loc[:, "datetime"] = pd.to_datetime(dataframe.loc[:, "timestamp"], unit="s", utc=True)
    return dataframe


# Повертає List з tuples що містять стартовий та кінцевий індекс часових періодів, де:
# їхня тривалість не менша від - min_period,
# максимальна відстань між двома точками всередині періоду - max_diff_between_points
def get_time_periods(time_series: pd.Series, min_period, max_diff_between_points=30):
    diff_time_series = time_series.diff()
    breaking_index_array = diff_time_series.loc[diff_time_series > max_diff_between_points].index
    time_period_list = []
    if not len(breaking_index_array):
        time_period_list.append((time_series.index[0], time_series.index[-1]))
    else:
        start_point = time_series.index[0]
        for index in breaking_index_array:
            time_period_list.append((start_point, index - 1))
            start_point = index
        time_period_list.append((start_point, time_series.index[-1]))
    del_list = []
    for period in time_period_list:
        try:
            if (time_series.loc[period[1]] - time_series.loc[period[0]]) < min_period:
                del_list.append(period)
        except Exception as er:
            pass
    if del_list:
        for period in del_list:
            time_period_list.remove(period)
    return time_period_list


def add_savgol_data_simple(dataframe: pd.DataFrame, window_length=3600, polyorder=2):
    savgol_data = savgol_filter(dataframe.loc[:, "los_tec"], (window_length // PERIOD_CONST + 1), polyorder)
    diff_data = dataframe.loc[:, "los_tec"] - savgol_data
    temp_window_length = window_length // PERIOD_CONST // 2
    new_dataframe = dataframe.assign(savgol=savgol_data,
                                     diff=diff_data.iloc[temp_window_length:-temp_window_length])
    return new_dataframe


#
# def calculate_savgol_diff_data(dataframe: pd.DataFrame, params: Dict):
#     savgov_data = savgol_filter(dataframe.loc[:, "los_tec"], **params)
#     diff_data = dataframe.loc[:, "los_tec"] - savgov_data
#     temp_window_length = (params["window_length"] - 1) // 2
#     new_dataframe = dataframe.assign(savgol=savgol_data,
#                                      diff=diff_data.iloc[temp_window_length:-temp_window_length])
#     return new_dataframe


def fill_small_gaps(dataframe: pd.DataFrame):
    time_series = dataframe.loc[:, "timestamp"]
    diff_time_series = time_series.diff()
    breaking_index_array = diff_time_series.loc[diff_time_series.loc[:] == 2 * PERIOD_CONST].index
    for index in breaking_index_array:
        new_line: pd.Series = dataframe.loc[index - 1:index].mean(0)
        date = dt.datetime.fromtimestamp(new_line.loc["timestamp"], tz=dt.timezone.utc)
        hour = date.hour
        minute = date.minute
        second = date.second
        new_line.loc["hour"] = hour
        new_line.loc["min"] = minute
        new_line.loc["sec"] = second
        dataframe = pd.concat([dataframe, new_line.to_frame().T], ignore_index=True)
    dataframe.sort_values(by="timestamp", inplace=True)
    dataframe.reset_index(inplace=True, drop=True)
    return dataframe


def add_savgol_data_complicate(dataframe: pd.DataFrame, params: Dict, filling=False):
    if filling:
        dataframe = fill_small_gaps(dataframe)
    time_periods = get_time_periods(dataframe.loc[:, "timestamp"], (params["window_length"]))
    new_dataframe = pd.DataFrame()
    for period in time_periods:
        temp_dataframe = dataframe.loc[period[0]:period[1]]
        new_temp_dataframe = add_savgol_data(temp_dataframe, **params)
        new_dataframe = pd.concat([new_dataframe, new_temp_dataframe])
    return new_dataframe


def get_ready_dataframe(file, params: Dict, date=dt.datetime.min, filling=False):
    first_dataframe = read_sat_file(file)
    second_dataframe = add_timestamp_column_to_df(first_dataframe, date)
    third_dataframe = add_savgol_data_complicate(second_dataframe, params, filling)
    return third_dataframe


def get_dataframe_with_timestamp(file, date=dt.datetime.min):
    first_dataframe = read_sat_file(file)
    second_dataframe = add_timestamp_column_to_df(first_dataframe, date)
    return second_dataframe


def plot(save_path, name, dataframe, title=None):
    # title = name
    figure: fig.Figure = plt.figure(layout="tight", figsize=[16.0 * 160 / 300, 9.0 * 160 / 300])
    axes1: axs.Axes = figure.add_subplot(2, 1, 1)
    ytext1 = axes1.set_ylabel("STEC, TEC units")
    xtext1 = axes1.set_xlabel("Time, hrs")
    # time_array = dataframe.loc[:, "timestamp"] / 3600
    time_array = dataframe.loc[:, "datetime"]
    line1, = axes1.plot(time_array, dataframe["los_tec"], label="Data from GPS-TEC", linestyle=" ",
                        marker=".", color="blue", markeredgewidth=1, markersize=1.1)

    line2, = axes1.plot(time_array, dataframe["savgol"], label="Data passed through Savitzki-Golay filter",
                        linestyle=" ", marker=".", color="red", markeredgewidth=1, markersize=1.1)
    axes1.set_xlim(time_array.iloc[0], time_array.iloc[-1])
    axes1.legend()
    axes2: axs.Axes = figure.add_subplot(2, 1, 2)
    line4, = axes2.plot(time_array, dataframe["diff"], linestyle=" ", marker=".",
                        markeredgewidth=1,
                        markersize=1,
                        label="Difference between Madrigal data and GPS-TEC")
    axes2.legend()
    axes2.set_ylim(-1.2, 1.2)
    axes2.set_xlim(*axes1.get_xlim())
    axes2.grid(True)
    if title:
        figure.suptitle(title)
    plt.savefig(os.path.join(save_path, name + ".png"), dpi=300)
    plt.close(figure)


#
# UNREADY
# UNREADY
#
def plot_difference_graph(save_path, name, dataframe, title=None):
    if not title:
        title = name
    figure: fig.Figure = plt.figure(layout="tight", figsize=[16.0 * 120 / 300, 9.0 * 120 / 300])
    axes1: axs.Axes = figure.add_subplot(1, 1, 1)
    # line4, = axes1.plot(dataframe.loc[:, "timestamp"] / 3600, dataframe["diff"], linestyle="-", marker=" ",
    #                     markeredgewidth=0.5, markersize=1.5, linewidth=0.6)
    line4, = axes1.plot(dataframe.loc[:, "datetime"], dataframe["dtec"], linestyle="-", marker=" ",
                        markeredgewidth=0.5, markersize=1.5, linewidth=0.6)
    axes1.set_ylim(-1.1, 1.1)
    axes1.grid(True)
    if title:
        figure.suptitle(title)
    plt.savefig(os.path.join(save_path, name + ".png"), dpi=300)
    plt.close(figure)


def main_data_processing(dataframe, data_save_directory_path, window_length=3600, polyorder=2):
    list_of_params_for_processing = [{"window_length": 3600}, {"window_length": 7200}]


# Шукає файли формату Cmn в папці, але не у внутрішніх папках
def find_all_cmn_files(directory):
    list_of_files = os.listdir(directory)
    list_of_cmn_files = []
    for file in list_of_files:
        if os.path.isfile(os.path.join(directory, file)):
            if os.path.splitext(file)[1] == ".Cmn":
                list_of_cmn_files.append(os.path.join(directory, file))
    return list_of_cmn_files


# Створює pandas dataframe на основі даних Cmn файлу
# Назви стовпчики: "los_tec", "azm", "elm", "gdlat", "gdlon", "time", "sat_id", "hour", "min", "sec"
def create_pd_dataframe_from_cmn_file(file):
    dataframe_from_file = rcmn.read_cmn_file_pd(file)
    new_dataframe = pd.DataFrame({"los_tec": dataframe_from_file.loc[:, "stec"],
                                  "azm": dataframe_from_file.loc[:, "azimuth"],
                                  "elm": dataframe_from_file.loc[:, "elevation"],
                                  "gdlat": dataframe_from_file.loc[:, "latitude"],
                                  "gdlon": dataframe_from_file.loc[:, "longitude"],
                                  "time": dataframe_from_file.loc[:, "time"],
                                  "sat_id": dataframe_from_file.loc[:, "PRN"]})
    new_dataframe: pd.DataFrame = new_dataframe.loc[new_dataframe.loc[:, "time"] >= 0]
    hour_list = []
    min_list = []
    sec_list = []
    time_series = new_dataframe.loc[:, "time"]
    for index in time_series.index:
        hour = int(time_series.loc[index])
        hour_list.append(hour)
        min = int((time_series.loc[index] - hour) * 60)
        min_list.append(min)
        sec = int(((time_series.loc[index] - hour) * 60 - min) * 60 + 0.5)
        sec_list.append(sec)
    new_dataframe.insert(0, "hour", hour_list)
    new_dataframe.insert(0, "min", min_list)
    new_dataframe.insert(0, "sec", sec_list)
    return new_dataframe


# Перетворює dataframe на інший з чітко визначеним періодом між точками та стартом відліку
# НОВІ ЧАСОВИХ КООРДИНАТ ПРИ ПЕРІОДІ 30 СЕКУНД: 00:00:00, 00:00:30, 00:01:00, 00:01:30, 00:02:00, 00:02:30, 00:03:00
# Вхідний Dataframe має стовчики "los_tec", "azm", "elm", "gdlat", "gdlon", "timestamp"
# Вхідний Dataframe є лише даними для одного супутника
def convert_dataframe_to_hard_period(dataframe: pd.DataFrame, max_nonbreakable_period_between_points=PERIOD_CONST):
    series_timestamp = dataframe.loc[:, "timestamp"]
    list_time_periods = get_time_periods(series_timestamp, PERIOD_CONST, max_nonbreakable_period_between_points)
    new_dataframe = pd.DataFrame()
    for start, finish in list_time_periods:
        temp_dataframe = dataframe.loc[start:finish]
        date = dt.datetime.fromtimestamp(temp_dataframe.iloc[0].loc["timestamp"], tz=dt.timezone.utc)
        date = dt.datetime(year=date.year, month=date.month, day=date.day, hour=0, minute=0, second=0, tzinfo=dt.timezone.utc)
        new_start = int((temp_dataframe.iloc[0].loc["timestamp"] - date.timestamp()) // PERIOD_CONST * PERIOD_CONST + \
                         date.timestamp())
        list_new_timestamp = list(range(new_start, int(temp_dataframe.iloc[-1].loc["timestamp"]) + 1, PERIOD_CONST))
        list_new_los_tec = np.interp(list_new_timestamp, temp_dataframe.loc[:, "timestamp"],
                                     temp_dataframe.loc[:, "los_tec"])
        list_new_azm = np.interp(list_new_timestamp, temp_dataframe.loc[:, "timestamp"], temp_dataframe.loc[:, "azm"])
        list_new_elm = np.interp(list_new_timestamp, temp_dataframe.loc[:, "timestamp"], temp_dataframe.loc[:, "elm"])
        list_new_gdlat = np.interp(list_new_timestamp, temp_dataframe.loc[:, "timestamp"],
                                   temp_dataframe.loc[:, "gdlat"])
        list_new_gdlon = np.interp(list_new_timestamp, temp_dataframe.loc[:, "timestamp"],
                                   temp_dataframe.loc[:, "gdlon"])
        new_temp_dataframe = pd.DataFrame(data={"timestamp": list_new_timestamp, "los_tec": list_new_los_tec,
                                                "azm": list_new_azm, "elm": list_new_elm, "gdlat": list_new_gdlat,
                                                "gdlon": list_new_gdlon})
        new_dataframe = pd.concat([new_dataframe, new_temp_dataframe], ignore_index=True)
    return new_dataframe


def test_convert_dataframe_to_hard_period():
    test_file = SOURCE_CMN_FILE1
    dataframe_cmn_file = create_pd_dataframe_from_cmn_file(test_file)
    cmn_file = os.path.basename(test_file)
    temp_date = dt.datetime(year=int(cmn_file[8:12]), month=int(cmn_file[13:15]),
                            day=int(cmn_file[16:18]), tzinfo=dt.timezone.utc)
    dataframe_cmn_file = add_timestamp_column_to_df(dataframe_cmn_file, temp_date)
    dataframe_cmn_file = add_datetime_column_to_df(dataframe_cmn_file)
    list_sad_id = np.unique(dataframe_cmn_file.loc[:, "sat_id"])
    for sat_id in list_sad_id:
    # sat_id = 11
        print(sat_id, dt.datetime.now())
        dataframe_cmn_sat_id = dataframe_cmn_file.loc[dataframe_cmn_file.loc[:, "sat_id"] == sat_id]
        new_dataframe_sat_id = convert_dataframe_to_hard_period(dataframe_cmn_sat_id,
                                                                max_nonbreakable_period_between_points=3 * PERIOD_CONST)
        new_dataframe_sat_id = add_datetime_column_to_df(new_dataframe_sat_id)

        for column in ["los_tec", "azm", "elm", "gdlat", "gdlon"]:
            # PLOTTING
            name = os.path.basename(test_file)
            title = column + "_" + name
            figure: fig.Figure = plt.figure(layout="tight", figsize=[16.0 * 320 / 300, 9.0 * 320 / 300])
            axes1: axs.Axes = figure.add_subplot(1, 1, 1)
            ytext1 = axes1.set_ylabel("column")
            xtext1 = axes1.set_xlabel("time")
            # time_array = dataframe.loc[:, "timestamp"] / 3600
            time_array = new_dataframe_sat_id.loc[:, "datetime"]
            time_array2 = dataframe_cmn_sat_id.loc[:, "datetime"]
            line1, = axes1.plot(time_array, new_dataframe_sat_id[column], label="NEW(interpolated)", color="red",
                                linestyle="-", marker=" ", markeredgewidth=0.05, markersize=0.1, linewidth=0.2)

            line2, = axes1.plot(time_array2, dataframe_cmn_sat_id[column], label="OLD(form Cmn)", linestyle=" ",
                                marker=".", color="blue", markeredgewidth=0.4, markersize=0.5)
            axes1.set_xlim(time_array.iloc[0], time_array.iloc[-1])
            # axes1.set_xlim(dt.datetime(year=temp_date.year, month=temp_date.month, day=temp_date.day, hour=10, minute=3, second=0),
            #                dt.datetime(year=temp_date.year, month=temp_date.month, day=temp_date.day, hour=10, minute=5, second=0))
            # axes1.set_ylim(158, 160)
            axes1.legend()
            axes1.grid(True)
            save_path = r"/home/vadymskipa/Documents/PhD_student/temp/SAVE/test_convert_dataframe_to_hard_period/"
            plt.savefig(os.path.join(save_path, os.path.splitext(name)[0] + "_" + column + f"G{sat_id:0=2}" + ".svg"), dpi=300)
            plt.close(figure)
            # PLOTTING


def add_dataframe_with_leveling(base_dataframe: pd.DataFrame, adding_dataframe: pd.DataFrame, max_difference,
                                number_of_points=2):
    if len(base_dataframe.index) < number_of_points:
        return adding_dataframe
    elif len(adding_dataframe.index) < number_of_points:
        return base_dataframe
    if adding_dataframe.iloc[0].loc["timestamp"] - base_dataframe.iloc[-1].loc["timestamp"] < max_difference:
        flag = True
        if base_dataframe.iloc[-1].loc["timestamp"] - base_dataframe.iloc[-number_of_points].loc["timestamp"] >= (max_difference * (number_of_points - 1)):
            indexes = base_dataframe.iloc[-number_of_points:].index
            base_dataframe.drop(indexes)
            flag = False
        if adding_dataframe.iloc[number_of_points-1].loc["timestamp"] - adding_dataframe.iloc[0].loc["timestamp"] >= (max_difference * (number_of_points - 1)):
            indexes = adding_dataframe.iloc[:number_of_points].index
            base_dataframe.drop(indexes)
            flag = False
        if flag:
            x_temp_base = np.array(base_dataframe.iloc[-number_of_points:].loc[:, "timestamp"])
            x_start_point = x_temp_base[0]
            x_temp_base = x_temp_base - x_start_point
            y_base = np.array(base_dataframe.iloc[-number_of_points:].loc[:, "los_tec"])
            x_base = x_temp_base[:, np.newaxis]**[0, 1]
            solve_base = lstsq(x_base, y_base)
            x_temp_adding = np.array(adding_dataframe.iloc[0: number_of_points].loc[:, "timestamp"])
            x_temp_adding = x_temp_adding - x_start_point
            y_adding = np.array(adding_dataframe.iloc[0: number_of_points].loc[:, "los_tec"])
            x_adding = x_temp_adding[:, np.newaxis] ** [0, 1]
            solve_adding = lstsq(x_adding, y_adding)
            temp_midpoint = ((adding_dataframe.iloc[0].loc["timestamp"] + base_dataframe.iloc[-1].loc["timestamp"]) / 2
                             - x_start_point)
            y_base_midpoint = solve_base[0][0] + solve_base[0][1] * temp_midpoint
            y_adding_midpoint = solve_adding[0][0] + solve_adding[0][1] * temp_midpoint
            level_difference = y_base_midpoint - y_adding_midpoint
            adding_dataframe.loc[:, "los_tec"] = adding_dataframe.loc[:, "los_tec"] + level_difference
            list_new_timestamp = list(range(int(base_dataframe.iloc[-1].loc["timestamp"]) + PERIOD_CONST,
                                            int(adding_dataframe.iloc[0].loc["timestamp"]), PERIOD_CONST))
            if list_new_timestamp:
                list_new_los_tec = np.interp(list_new_timestamp, [base_dataframe.iloc[-1].loc["timestamp"],
                                                                  adding_dataframe.iloc[0].loc["timestamp"]],
                                             [base_dataframe.iloc[-1].loc["los_tec"],
                                              adding_dataframe.iloc[0].loc["los_tec"]])
                list_new_azm = np.interp(list_new_timestamp, [base_dataframe.iloc[-1].loc["timestamp"],
                                                                  adding_dataframe.iloc[0].loc["timestamp"]],
                                             [base_dataframe.iloc[-1].loc["azm"],
                                              adding_dataframe.iloc[0].loc["azm"]])
                list_new_elm = np.interp(list_new_timestamp, [base_dataframe.iloc[-1].loc["timestamp"],
                                                                  adding_dataframe.iloc[0].loc["timestamp"]],
                                             [base_dataframe.iloc[-1].loc["elm"],
                                              adding_dataframe.iloc[0].loc["elm"]])
                list_new_gdlat = np.interp(list_new_timestamp, [base_dataframe.iloc[-1].loc["timestamp"],
                                                                  adding_dataframe.iloc[0].loc["timestamp"]],
                                             [base_dataframe.iloc[-1].loc["gdlat"],
                                              adding_dataframe.iloc[0].loc["gdlat"]])
                list_new_gdlon = np.interp(list_new_timestamp, [base_dataframe.iloc[-1].loc["timestamp"],
                                                                  adding_dataframe.iloc[0].loc["timestamp"]],
                                             [base_dataframe.iloc[-1].loc["gdlon"],
                                              adding_dataframe.iloc[0].loc["gdlon"]])
                list_new_datetime = [dt.datetime.fromtimestamp(timestamp, tz=dt.timezone.utc) for timestamp in list_new_timestamp]
                new_small_dataframe = pd.DataFrame(data={"timestamp": list_new_timestamp, "los_tec": list_new_los_tec,
                                                        "azm": list_new_azm, "elm": list_new_elm, "gdlat": list_new_gdlat,
                                                        "gdlon": list_new_gdlon, "datetime": list_new_datetime})
                base_dataframe = pd.concat([base_dataframe, new_small_dataframe], ignore_index=True)
    new_dataframe = pd.concat([base_dataframe, adding_dataframe], ignore_index=True)
    return new_dataframe


def test_add_dataframe_with_leveling(source_directory_path):
    # list_cmn_files = find_all_cmn_files(directory=source_directory_path)
    # list_cmn_files.sort()
    # list_params = [{"window_length": 3600}, {"window_length": 7200}]
    list_cmn_files = [SOURCE_CMN_FILE1, SOURCE_CMN_FILE2, SOURCE_CMN_FILE3]
    list_params = [{"window_length": 3600}]
    max_nonbreakable_period_between_points = 5 * PERIOD_CONST
    for params in list_params:
        dict_dataframe_sat_id = {}
        for cmn_file in list_cmn_files:
            temp_dataframe_cmn_file = create_pd_dataframe_from_cmn_file(cmn_file)
            file_name = os.path.basename(cmn_file)
            temp_date = dt.datetime(year=int(file_name[8:12]), month=int(file_name[13:15]),
                                    day=int(file_name[16:18]), tzinfo=dt.timezone.utc)
            temp_dataframe_cmn_file = add_timestamp_column_to_df(temp_dataframe_cmn_file, temp_date)
            list_sad_id = np.unique(temp_dataframe_cmn_file.loc[:, "sat_id"])
            for sat_id in list_sad_id:
                temp_dataframe_sat_id = temp_dataframe_cmn_file.loc[temp_dataframe_cmn_file.loc[:, "sat_id"] == sat_id]
                new_dataframe_sat_id = convert_dataframe_to_hard_period(temp_dataframe_sat_id,
                                                                        max_nonbreakable_period_between_points)
                new_dataframe_sat_id = add_datetime_column_to_df(new_dataframe_sat_id)
                if sat_id in dict_dataframe_sat_id:
                    dict_dataframe_sat_id[sat_id] = add_dataframe_with_leveling(dict_dataframe_sat_id[sat_id],
                                                                                new_dataframe_sat_id, 5 * PERIOD_CONST)
                else:
                    dict_dataframe_sat_id[sat_id] = new_dataframe_sat_id

        for sat_id, dataframe_sat_id in dict_dataframe_sat_id.items():

            # PLOTTING
            name = f"G{sat_id:0=2}"
            title = f"los_tec for {name}"
            figure: fig.Figure = plt.figure(layout="tight", figsize=[16.0 * 320 / 300, 9.0 * 320 / 300])
            axes1: axs.Axes = figure.add_subplot(1, 1, 1)
            ytext1 = axes1.set_ylabel("column")
            xtext1 = axes1.set_xlabel("time")
            # time_array = dataframe.loc[:, "timestamp"] / 3600
            time_array = dataframe_sat_id.loc[:, "datetime"]
            line1, = axes1.plot(time_array, dataframe_sat_id["los_tec"], label="los_tec",
                                color="blue",
                                linestyle="-", marker=".", markeredgewidth=0.5, markersize=0.6, linewidth=0.2)
            ytext1 = axes1.set_ylabel("STEC, TEC units")
            axes1.set_xlim(time_array.iloc[0], time_array.iloc[-1])
            # axes1.set_xlim(dt.datetime(year=temp_date.year, month=temp_date.month, day=temp_date.day, hour=10, minute=3, second=0),
            #                dt.datetime(year=temp_date.year, month=temp_date.month, day=temp_date.day, hour=10, minute=5, second=0))
            # axes1.set_ylim(158, 160)
            axes1.legend()
            axes1.grid(True)
            save_path = r"/home/vadymskipa/Documents/PhD_student/temp/SAVE/test_add_dataframe_with_leveling/"
            plt.savefig(
                os.path.join(save_path, f"G{sat_id:0=2}" + ".svg"),
                dpi=300)
            plt.close(figure)
            # PLOTTING


def get_directory_path(existing_directory_path, new_folder):
    new_path = os.path.join(existing_directory_path, new_folder)
    if not os.path.exists(new_path):
        os.mkdir(new_path)
    return new_path


def get_date_str(datetime: dt.datetime):
    return f"{datetime.timetuple().tm_yday:0=3}-{datetime.year}-{datetime.month:0=2}-{datetime.day:0=2}"


def add_savgol_data(dataframe: pd.DataFrame, params: Dict):
    time_periods = get_time_periods(dataframe.loc[:, "timestamp"], (params["window_length"]))
    new_dataframe = pd.DataFrame()
    for period in time_periods:
        temp_dataframe = dataframe.loc[period[0]:period[1]]
        new_temp_dataframe = add_savgol_data_simple(temp_dataframe, **params)
        new_dataframe = pd.concat([new_dataframe, new_temp_dataframe])
    return new_dataframe


def save_diff_dataframe_txt(save_file_path, dataframe: pd.DataFrame):
    dataframe_without_nan = dataframe.loc[dataframe.loc[:, "diff"].notna()]
    with open(save_file_path, "w") as write_file:
        write_file.write(f"{'hour':<4}\t{'min':<4}\t{'sec':<6}\t{'dTEC':<6}\t{'azm':<6}\t{'elm':<6}"
                         f"\t{'gdlat':<6}\t{'gdlon':<6}\n")
        for index in dataframe_without_nan.index:
            hour = dataframe_without_nan.loc[index, "datetime"].hour
            minute = dataframe_without_nan.loc[index, "datetime"].minute
            sec = dataframe_without_nan.loc[index, "datetime"].second
            dtec = dataframe_without_nan.loc[index, "diff"]
            azm = dataframe_without_nan.loc[index, "azm"]
            elm = dataframe_without_nan.loc[index, "elm"]
            gdlat = dataframe_without_nan.loc[index, "gdlat"]
            gdlon = dataframe_without_nan.loc[index, "gdlon"]
            write_file.write(f"{hour:<4}\t{minute:<4}\t{sec:<4.0f}\t{dtec:<6.3f}\t{azm:<6.2f}\t{elm:<6.2f}"
                             f"\t{gdlat:<6.2f}\t{gdlon:<6.2f}\n")


SAVE_DIRECTORY1 = r"/home/vadymskipa/Documents/PhD_student/temp/SAVE/main1/"
SOURCE_CMN_DIRECTORY1 = r"/home/vadymskipa/Documents/GPS_2023_outcome/"
def main1(source_directory_path, save_directory_path):
    list_cmn_files = find_all_cmn_files(directory=source_directory_path)
    list_cmn_files.sort()
    # list_cmn_files = [SOURCE_CMN_FILE4, SOURCE_CMN_FILE5]
    list_params = [{"window_length": 3600}, {"window_length": 7200}]
    max_nonbreakable_period_between_points = 5 * PERIOD_CONST
    save_path_txt = get_directory_path(save_directory_path, "TXT")
    save_path_csv = get_directory_path(save_directory_path, "CSV")
    for params in list_params:
        dict_dataframe_sat_id = {}
        for cmn_file in list_cmn_files:
            temp_dataframe_cmn_file = create_pd_dataframe_from_cmn_file(cmn_file)
            file_name = os.path.basename(cmn_file)
            temp_date = dt.datetime(year=int(file_name[8:12]), month=int(file_name[13:15]),
                                    day=int(file_name[16:18]), tzinfo=dt.timezone.utc)
            temp_dataframe_cmn_file = add_timestamp_column_to_df(temp_dataframe_cmn_file, temp_date)
            list_sad_id = np.unique(temp_dataframe_cmn_file.loc[:, "sat_id"])
            for sat_id in list_sad_id:
                temp_dataframe_sat_id = temp_dataframe_cmn_file.loc[temp_dataframe_cmn_file.loc[:, "sat_id"] == sat_id]
                new_dataframe_sat_id = convert_dataframe_to_hard_period(temp_dataframe_sat_id,
                                                                        max_nonbreakable_period_between_points)
                new_dataframe_sat_id = add_datetime_column_to_df(new_dataframe_sat_id)
                if sat_id in dict_dataframe_sat_id:
                    dict_dataframe_sat_id[sat_id] = add_dataframe_with_leveling(dict_dataframe_sat_id[sat_id],
                                                                                new_dataframe_sat_id,
                                                                                max_nonbreakable_period_between_points,
                                                                                number_of_points=4)
                else:
                    dict_dataframe_sat_id[sat_id] = new_dataframe_sat_id
        for sat_id, dataframe_sat_id in dict_dataframe_sat_id.items():
            dataframe_sat_id = add_savgol_data(dataframe_sat_id, params)
            first_timestamp = dataframe_sat_id.iloc[0].loc["timestamp"]
            first_datetime = dt.datetime.fromtimestamp(first_timestamp, tz=dt.timezone.utc)
            first_day_timestamp = dt.datetime(year=first_datetime.year, month=first_datetime.month,
                                              day=first_datetime.day, hour=0, minute=0, second=0,
                                              tzinfo=dt.timezone.utc).timestamp()
            series_timestamp = (dataframe_sat_id.loc[:, "timestamp"] - first_day_timestamp) // (3600 * 24) * \
                               (3600 * 24) + first_day_timestamp
            array_timestamp = np.unique(series_timestamp)
            for timestamp in array_timestamp:
                date = dt.datetime.fromtimestamp(timestamp)
                save_path_txt_date = get_directory_path(save_path_txt, get_date_str(date))
                save_path_csv_date = get_directory_path(save_path_csv, get_date_str(date))
                save_path_txt_params = get_directory_path(save_path_txt_date,
                                                          f"Window_{params['window_length']}_Seconds")
                save_path_csv_params = get_directory_path(save_path_csv_date,
                                                          f"Window_{params['window_length']}_Seconds")
                save_path_txt_sat_id = os.path.join(save_path_txt_params, f"G{sat_id:0=2}.txt")
                save_path_csv_sat_id = os.path.join(save_path_csv_params, f"G{sat_id:0=2}.csv")
                mask = dataframe_sat_id.loc[:, "timestamp"] >= timestamp
                mask = np.logical_and(mask, dataframe_sat_id.loc[:, "timestamp"] < (timestamp + 3600 * 24))
                temp_dataframe: pd.DataFrame = dataframe_sat_id.loc[mask]
                temp_dataframe.to_csv(save_path_csv_sat_id)
                save_diff_dataframe_txt(save_path_txt_sat_id, temp_dataframe)


SOURCE_DIRECTORY_NEW1 = r"/home/vadymskipa/Documents/PhD_student/temp/SAVE/main1/TXT/2020-09-23/"
SOURCE_DIRECTORY_OLD1 = r"/home/vadymskipa/Documents/PhD_student/temp/SAVE6/267-2020-09-23/"
SOURCE_DIRECTORY_NEW2 = r"/home/vadymskipa/Documents/PhD_student/temp/SAVE/main1/TXT/2020-09-24/"
SOURCE_DIRECTORY_OLD2 = r"/home/vadymskipa/Documents/PhD_student/temp/SAVE6/268-2020-09-24/"
SAVE_DIRECTORY_TEST1 = r"/home/vadymskipa/Documents/PhD_student/temp/SAVE/test_plot_new_savgov_vs_old_savgov/"
def test_plot_new_savgov_vs_old_savgov():
    list_directory = [(SOURCE_DIRECTORY_OLD1, SOURCE_DIRECTORY_NEW1),
                      (SOURCE_DIRECTORY_OLD2, SOURCE_DIRECTORY_NEW2)]
    for old_directory_path, new_directory_path in list_directory:
        list_inner_directory = os.listdir(old_directory_path)
        year = int(os.path.basename(os.path.dirname(old_directory_path))[4:8])
        month = int(os.path.basename(os.path.dirname(old_directory_path))[9:11])
        day = int(os.path.basename(os.path.dirname(old_directory_path))[12:14])
        date = dt.datetime(year=year, month=month, day=day, tzinfo=dt.timezone.utc)
        for inner_directory in list_inner_directory:
            list_dtec_file = os.listdir(os.path.join(old_directory_path, inner_directory))
            save_directory_path = get_directory_path(SAVE_DIRECTORY_TEST1,
                                                     os.path.basename(os.path.dirname(old_directory_path)))
            save_directory_path = get_directory_path(save_directory_path, inner_directory)
            for dtec_file in list_dtec_file:
                str_sat_id = os.path.splitext(dtec_file)[0]
                dtec_file_old_path = os.path.join(old_directory_path, inner_directory, dtec_file)
                dtec_file_new_path = os.path.join(new_directory_path, inner_directory, dtec_file)
                dataframe_old = read_dtec_file(dtec_file_old_path)
                dataframe_old = add_timestamp_column_to_df(dataframe_old, date)
                dataframe_old = add_datetime_column_to_df(dataframe_old)
                dataframe_new = read_dtec_file(dtec_file_new_path)
                dataframe_new = add_timestamp_column_to_df(dataframe_new, date)
                dataframe_new = add_datetime_column_to_df(dataframe_new)
                title = f"Comparison of two savgol approaches (without midnight and with) for {str_sat_id}" \
                        f"\n{inner_directory}, {dt.date(year=year, month=month, day=day)}"
                figure: fig.Figure = plt.figure(layout="tight", figsize=[16.0 * 160 / 300, 9.0 * 160 / 300])
                axes1: axs.Axes = figure.add_subplot(1, 1, 1)
                line1, = axes1.plot(dataframe_new.loc[:, "datetime"], dataframe_new["dtec"], linestyle="-", marker=" ",
                                    linewidth=0.3, color="red", label="with midnight")
                line2, = axes1.plot(dataframe_old.loc[:, "datetime"], dataframe_old["dtec"], linestyle=" ", marker=".",
                                    markersize=1.0, color="blue", label="without midnight")
                axes1.set_ylim(-1.1, 1.1)
                axes1.set_xlim(dataframe_new.iloc[0].loc["datetime"], dataframe_new.iloc[-1].loc["datetime"])
                axes1.grid(True)
                axes1.set_title(title)
                axes1.legend()
                plt.savefig(os.path.join(save_directory_path, f"{str_sat_id}.png"), dpi=300)
                plt.close(figure)
                print(str_sat_id)










def test_lstsq():
    x1 = np.array([3, 4, 5])
    y1 = np.array([5, 7.1, 5])
    x = x1[:, np.newaxis]**[0, 1]
    a = lstsq(x, y1)
    print(a)




def save_los_tec_txt_from_dataframe(save_path, dataframe: pd.DataFrame):
    with open(save_path, "w") as write_file:
        write_file.write(f"{'hour':<4}\t{'min':<4}\t{'sec':<6}\t{'los_tec':<6}\t{'azm':<6}\t{'elm':<6}"
                         f"\t{'gdlat':<6}\t{'gdlon':<6}\n")
        for index in dataframe.index:
            hour = dataframe.loc[index, "hour"]
            min = dataframe.loc[index, "min"]
            sec = dataframe.loc[index, "sec"]
            los_tec = dataframe.loc[index, "los_tec"]
            azm = dataframe.loc[index, "azm"]
            elm = dataframe.loc[index, "elm"]
            gdlat = dataframe.loc[index, "gdlat"]
            gdlon = dataframe.loc[index, "glon"]
            write_file.write(f"{hour:<4}\t{min:<4}\t{sec:<4.0f}\t{los_tec:<6.2f}\t{azm:<6.2f}\t{elm:<6.2f}"
                             f"\t{gdlat:<6.2f}\t{gdlon:<6.2f}\n")


SITES1 = ['bute', 'bor1', 'fra2', 'plnd', 'polv', 'krrs', 'mikl', 'pryl', 'cfrm']
LOS_FILE_PATH1 = r"/home/vadymskipa/Documents/PhD_student/data/data1/los_20230226.001.h5.hdf5"
SAVE_PATH7 = r"/home/vadymskipa/Documents/PhD_student/temp/SAVE/DTEC_FROM_MADRIGAL_FILES/"
def save_los_tec_txt_file_for_some_sites_from_los_file(los_file_path, list_gps_sites, save_path):
    new_save_path = get_directory_path(save_path, "los_tec_txt")
    start = dt.datetime.now()
    print(f"Reading \"gps_site\" column from los_fie ------------ {start}")
    with h5py.File(los_file_path, "r") as file:
        site_array = file["Data"]["Table Layout"]["gps_site"]
    print(f"End of reading \"gps_site\" column from los_file ----- {dt.datetime.now()} -- {dt.datetime.now()-start}")

    mask = np.full(len(site_array), False)
    for site in list_gps_sites:
        temp_mask = site_array == site.encode("ascii")
        mask = np.logical_or(mask, temp_mask)

    print(f"Reading data for sites ------------------------------- {dt.datetime.now()}")
    los_dataframe = rlos.get_data_by_indeces_GPS_pd(los_file_path, mask)
    print(f"End of reading data for sites ------------------------ {dt.datetime.now()} -- {dt.datetime.now()-start}")

    max_nonbreakable_period_between_points = 5 * PERIOD_CONST
    name_los_file = os.path.basename(los_file_path)
    date = dt.datetime(int(name_los_file[4:8]), int(name_los_file[8:10]), int(name_los_file[10:12]))
    los_dataframe = add_timestamp_column_to_df(los_dataframe, date)
    for site in list_gps_sites:
        print(f"Saving los_tec_txt for site {site} ----------------- {dt.datetime.now()} -- {dt.datetime.now()-start}")
        save_path_site_1 = get_directory_path(new_save_path, site)
        site_dataframe = los_dataframe.loc[los_dataframe.loc[:, "gps_site"] == site.encode("ascii")]
        save_path_date_2 = get_directory_path(save_path_site_1, get_date_str(date))
        sat_id_array = np.unique(site_dataframe.loc[:, "sat_id"])
        for sat_id in sat_id_array:
            save_path_sat_id_3 = os.path.join(save_path_date_2, f"G{sat_id:0=2}.txt")
            sat_id_dataframe = site_dataframe.loc[site_dataframe.loc[:, "sat_id"] == sat_id]
            save_los_tec_txt_from_dataframe(save_path_sat_id_3, sat_id_dataframe)


def get_los_file_paths_from_directory(directory_path):
    list_of_objects_in_directory = os.listdir(directory_path)
    result_list = [os.path.join(directory_path, el) for el in list_of_objects_in_directory
                   if (os.path.isfile(os.path.join(directory_path, el)) and
                       re.search("^los_(\d{8})\S*(\.h5|\.hdf5)$", el))]
    return result_list


def get_date_directory_paths_from_directory(directory_path):
    list_of_objects_in_directory = os.listdir(directory_path)
    result_list = [os.path.join(directory_path, el) for el in list_of_objects_in_directory
                   if (os.path.isdir(os.path.join(directory_path, el)) and
                       re.search("^(\d{3})-(\d{4})-(\d{2})-(\d{2})$", el))]
    return result_list


def get_sat_id_file_paths_from_directory(directory_path):
    list_of_objects_in_directory = os.listdir(directory_path)
    result_list = [os.path.join(directory_path, el) for el in list_of_objects_in_directory
                   if (os.path.isfile(os.path.join(directory_path, el)) and
                       re.search("^G(\d{2})\.txt$", el))]
    return result_list



DIRECTORY_PATH1 = r"/home/vadymskipa/Documents/PhD_student/data/data1/"
def save_los_tec_txt_file_for_some_sites_from_directory_with_los_files(directory_path, list_gps_sites, save_path):
    list_los_file_paths = get_los_file_paths_from_directory(directory_path)
    for los_file_path in list_los_file_paths:
        save_los_tec_txt_file_for_some_sites_from_los_file(los_file_path, list_gps_sites, save_path)


def get_site_directory_paths_from_directory(directory_path):
    list_of_objects_in_directory = os.listdir(directory_path)
    result_list = [os.path.join(directory_path, el) for el in list_of_objects_in_directory
                   if (os.path.isdir(os.path.join(directory_path, el)) and
                       re.search("^(\S{4,4})$", el))]
    return result_list


def get_date_from_date_directory_name(date_directory_name):
    date = dt.datetime(int(date_directory_name[4:8]), int(date_directory_name[9:11]), int(date_directory_name[12:14]),
                       tzinfo=dt.timezone.utc)
    return date



SAVE_PATH8 = r"/home/vadymskipa/Documents/PhD_student/temp/SAVE/DTEC_FROM_MADRIGAL_FILES/dtec_txt/bor1/"
SOURCE_DIRECTORY_PATH8 = r"/home/vadymskipa/Documents/PhD_student/temp/SAVE/DTEC_FROM_MADRIGAL_FILES/los_tec_txt/bor1/"
LIST_PARAMS1 = ({"window_length": 3600}, {"window_length": 7200})
SOURCE_DIRECTORY_PATH8_1 = r"/home/vadymskipa/Documents/PhD_student/temp/SAVE/DTEC_FROM_MADRIGAL_FILES/los_tec_txt/krrs/"

def get_dtec_from_los_tec_txt_1(source_directory_path, save_directory_path, list_params=LIST_PARAMS1):
    max_nonbreakable_period_between_points = 8 * PERIOD_CONST
    list_of_date_directory_paths = get_date_directory_paths_from_directory(source_directory_path)
    list_of_date_directory_paths.sort()
    dict_dataframe_sat_id = {}
    for date_directory_path in list_of_date_directory_paths:
        name_date_directory = os.path.basename(date_directory_path)
        date = get_date_from_date_directory_name(name_date_directory)
        list_of_sat_id_los_tec_file_paths = get_sat_id_file_paths_from_directory(date_directory_path)
        for sat_id_los_tec_file_path in list_of_sat_id_los_tec_file_paths:
            name_sat_id_file = os.path.basename(sat_id_los_tec_file_path)
            los_tec_dataframe = read_sat_file(sat_id_los_tec_file_path)
            los_tec_dataframe = add_timestamp_column_to_df(los_tec_dataframe, date)
            los_tec_dataframe = convert_dataframe_to_hard_period(los_tec_dataframe,
                                                                 max_nonbreakable_period_between_points)
            if name_sat_id_file in dict_dataframe_sat_id:
                dict_dataframe_sat_id[name_sat_id_file] = \
                    add_dataframe_with_leveling(dict_dataframe_sat_id[name_sat_id_file],
                                                los_tec_dataframe, max_nonbreakable_period_between_points,
                                                number_of_points=4)
            else:
                dict_dataframe_sat_id[name_sat_id_file] = los_tec_dataframe
    for name_sat_id_file, los_tec_dataframe in dict_dataframe_sat_id.items():
        for params in list_params:
            dataframe_sat_id = add_savgol_data(los_tec_dataframe, params)
            dataframe_sat_id = add_datetime_column_to_df(dataframe_sat_id)
            first_timestamp = dataframe_sat_id.iloc[0].loc["timestamp"]
            first_datetime = dt.datetime.fromtimestamp(first_timestamp, tz=dt.timezone.utc)
            first_day_timestamp = dt.datetime(year=first_datetime.year, month=first_datetime.month,
                                              day=first_datetime.day, hour=0, minute=0, second=0,
                                              tzinfo=dt.timezone.utc).timestamp()
            series_timestamp = (dataframe_sat_id.loc[:, "timestamp"] - first_day_timestamp) // (3600 * 24) * \
                               (3600 * 24) + first_day_timestamp
            array_timestamp = np.unique(series_timestamp)
            for timestamp in array_timestamp:
                date = dt.datetime.fromtimestamp(timestamp)
                save_path_date_1 = get_directory_path(save_directory_path, get_date_str(date))
                save_path_params_2 = get_directory_path(save_path_date_1,
                                                        f"Window_{params['window_length']}_Seconds")
                save_path_sat_id = os.path.join(save_path_params_2, name_sat_id_file)
                mask = dataframe_sat_id.loc[:, "timestamp"] >= timestamp
                mask = np.logical_and(mask, dataframe_sat_id.loc[:, "timestamp"] < (timestamp + 3600 * 24))
                temp_dataframe: pd.DataFrame = dataframe_sat_id.loc[mask]
                save_diff_dataframe_txt(save_path_sat_id, temp_dataframe)





SAVE_PATH10 = r"/home/vadymskipa/Documents/PhD_student/temp/SAVE/DTEC_FROM_MADRIGAL_FILES/dtec_txt/"
SOURCE_DIRECTORY_PATH10 = r"/home/vadymskipa/Documents/PhD_student/temp/SAVE/DTEC_FROM_MADRIGAL_FILES/los_tec_txt/"
def get_dtec_from_los_tec_txt_many_sites(source_path, save_path, params):
    list_site_directory_path = get_site_directory_paths_from_directory(source_path)
    for site_directory_path in list_site_directory_path:
        name_site = os.path.basename(site_directory_path)
        save_path_site_1 = get_directory_path(save_path, name_site)
        get_dtec_from_los_tec_txt_1(site_directory_path, save_path_site_1, params)


SAVE_PATH9 = r"/home/vadymskipa/Documents/PhD_student/temp/SAVE/DTEC_FROM_MADRIGAL_FILES/dtec_plots/bor1"
SAVE_PATH11 = r"/home/vadymskipa/Documents/PhD_student/temp/SAVE/DTEC_FROM_MADRIGAL_FILES/dtec_plots/krrs"
SOURCE_DIRECTORY_PATH11 = r"/home/vadymskipa/Documents/PhD_student/temp/SAVE/DTEC_FROM_MADRIGAL_FILES/dtec_txt/krrs"
def plot_graphs_for_site_directory(source_path, save_path):
    list_of_date_directory_paths = get_date_directory_paths_from_directory(source_path)
    for path_date_directory in list_of_date_directory_paths:
        name_date_directory = os.path.basename(path_date_directory)
        date = get_date_from_date_directory_name(name_date_directory)
        save_path_date_1 = get_directory_path(save_path, name_date_directory)
        list_od_window_directory_paths = [os.path.join(path_date_directory, directory_name) for directory_name in
                                          os.listdir(path_date_directory)]
        for path_window_directory in list_od_window_directory_paths:
            name_window_directory = os.path.basename(path_window_directory)
            save_path_window_2 = get_directory_path(save_path_date_1, name_window_directory)
            list_od_sat_id_file_paths = get_sat_id_file_paths_from_directory(path_window_directory)
            for sat_id_file_path in list_od_sat_id_file_paths:
                name_sat_id = os.path.basename(sat_id_file_path)[0:3]
                temp_dataframe = read_dtec_file(sat_id_file_path)
                temp_dataframe = add_timestamp_column_to_df(temp_dataframe, date)
                temp_dataframe = add_datetime_column_to_df(temp_dataframe)
                plot_difference_graph(save_path_window_2, name_sat_id, temp_dataframe)
                print(f"ploted {name_date_directory} - {name_window_directory} - {name_sat_id}")







def get_autocorr_dataframe_for_dtec_dataframe(dataframe_dtec, autocorr_limits):
    list_time = list(range(int(dataframe_dtec.iloc[0].loc["timestamp"]), int(dataframe_dtec.iloc[-1].loc["timestamp"]) + 1,
                            PERIOD_CONST))
    dataframe_dtec = dataframe_dtec.set_index("timestamp")
    dupl = dataframe_dtec.index.duplicated()
    if dupl[-1]:
        dataframe_dtec.drop([dataframe_dtec.index[-1]], inplace=True)
    dataframe_dtec = dataframe_dtec.reindex(list_time)
    series_dtec: pd.Series = dataframe_dtec.loc[:, "dtec"]
    bottom_limit = autocorr_limits[0] // PERIOD_CONST * PERIOD_CONST
    list_shift = list(range(bottom_limit, autocorr_limits[1], PERIOD_CONST))
    list_autocorr = []
    for shift in list_shift:
        series_shifted = series_dtec.shift(shift // PERIOD_CONST)
        koef_autocorr = series_dtec.corr(series_shifted)
        list_autocorr.append(koef_autocorr)
    data = {"shift": list_shift, "autocorr": list_autocorr}
    dataframe_autocorr = pd.DataFrame(data)
    dataframe_autocorr = dataframe_autocorr.set_index("shift")
    return dataframe_autocorr


SAVE_PATH12 = r"/home/vadymskipa/Documents/PhD_student/temp/SAVE/DTEC_FROM_MADRIGAL_FILES/autocorr/krrs/"
def get_autocorr_for_site_directory(source_path, save_path, autocorr_limits):
    list_of_date_directory_paths = get_date_directory_paths_from_directory(source_path)
    for path_date_directory in list_of_date_directory_paths:
        name_date_directory = os.path.basename(path_date_directory)
        date = get_date_from_date_directory_name(name_date_directory)
        save_path_date_1 = get_directory_path(save_path, name_date_directory)
        list_od_window_directory_paths = [os.path.join(path_date_directory, directory_name) for directory_name in
                                          os.listdir(path_date_directory)]
        for path_window_directory in list_od_window_directory_paths:
            name_window_directory = os.path.basename(path_window_directory)
            save_path_window_2 = get_directory_path(save_path_date_1, name_window_directory)
            list_od_sat_id_file_paths = get_sat_id_file_paths_from_directory(path_window_directory)
            dataframe_autocorr = pd.DataFrame()
            for sat_id_file_path in list_od_sat_id_file_paths:
                name_sat_id = os.path.basename(sat_id_file_path)[0:3]
                temp_dataframe = read_dtec_file(sat_id_file_path)
                temp_dataframe = add_timestamp_column_to_df(temp_dataframe, date)
                temp_dataframe = temp_dataframe.drop_duplicates()
                dataframe_sat_autocorr = get_autocorr_dataframe_for_dtec_dataframe(temp_dataframe, autocorr_limits)
                dataframe_sat_autocorr = dataframe_sat_autocorr.rename(columns={"shift": "shift",
                                                                                "autocorr": name_sat_id})
                dataframe_autocorr = pd.concat([dataframe_autocorr, dataframe_sat_autocorr], axis=1)
            save_path_autocorr_file = os.path.join(save_path_window_2, "autocorr.txt")
            with open(save_path_autocorr_file, "w") as file_autocorr:
                list_column_names = list(dataframe_autocorr.columns)
                first_line = ""
                for name_column in list_column_names:
                    first_line += f"{name_column:6}\t"
                file_autocorr.write(first_line + "\n")
                for index in dataframe_autocorr.index:
                    list_values = list(dataframe_autocorr.loc[index].values)
                    line = f"{index:6}\t"
                    for value in list_values:
                        line += f"{value:<6.3f}\t"
                    file_autocorr.write(line + "\n")


def temp_main1():
    list_los_file_paths = [r"/home/vadymskipa/Documents/PhD_student/data/data1/los_20230224.001.h5.hdf5",
                  r"/home/vadymskipa/Documents/PhD_student/data/data1/los_20230225.001.h5.hdf5",
                  r"/home/vadymskipa/Documents/PhD_student/data/data1/los_20230301.001.h5_0.hdf5",
                  r"/home/vadymskipa/Documents/PhD_student/data/data1/los_20230302.001.h5.hdf5"]
    save_path = SAVE_PATH7
    list_gps_sites = SITES1
    for los_file_path in list_los_file_paths:
        save_los_tec_txt_file_for_some_sites_from_los_file(los_file_path, list_gps_sites, save_path)


def save_los_tec_txt_file_for_region_from_los_files_directory(source_los_hdf_directory_path,
                                                              source_site_hdf_directory_path, save_path, lats, lons):
    list_of_objects_in_los_directory = os.listdir(source_los_hdf_directory_path)
    list_los_hdf_file_names = [el for el in list_of_objects_in_los_directory
                           if (os.path.isfile(os.path.join(source_los_hdf_directory_path, el)) and
                               wlos.check_los_file_name(el))]
    list_of_objects_in_site_directory = os.listdir(source_site_hdf_directory_path)
    list_site_hdf_file_names = [el for el in list_of_objects_in_site_directory
                            if (os.path.isfile(os.path.join(source_site_hdf_directory_path, el)) and
                                wsite.check_site_file_name(el))]
    for los_hdf_file_name in list_los_hdf_file_names:
        date = wlos.get_date_by_los_file_name(los_hdf_file_name)
        site_file_hdf_name = None
        for site_file_name_check in list_site_hdf_file_names:
            if wsite.check_site_file_name_by_date(site_file_name_check, date):
                site_file_hdf_name = site_file_name_check
        if not site_file_hdf_name:
            continue
        list_of_sites = wsite.get_sites_dataframe_by_coordinates(site_file_hdf_name, lons=lons, lats=lats)
        save_los_tec_txt_file_for_some_sites_from_los_file(get_directory_path(source_los_hdf_directory_path,
                                                                              los_hdf_file_name), list_of_sites,
                                                           get_directory_path(save_path, get_date_str(date)))


SAVE_PATH13 = r"/home/vadymskipa/Documents/PhD_student/data/los_txt/ukraine_and_around/"


def get_dtec_from_los_tec_txt_2(source_directory_path, save_directory_path, site_name, list_params=LIST_PARAMS1):
    max_nonbreakable_period_between_points = 8 * PERIOD_CONST
    list_of_date_directory_paths = get_date_directory_paths_from_directory(source_directory_path)
    list_of_date_directory_paths.sort()
    dict_dataframe_sat_id = {}
    for date_directory_path in list_of_date_directory_paths:
        name_date_directory = os.path.basename(date_directory_path)
        date = get_date_from_date_directory_name(name_date_directory)
        list_of_sat_id_los_tec_file_paths = get_sat_id_file_paths_from_directory(date_directory_path)
        for sat_id_los_tec_file_path in list_of_sat_id_los_tec_file_paths:
            name_sat_id_file = os.path.basename(sat_id_los_tec_file_path)
            los_tec_dataframe = read_sat_file(sat_id_los_tec_file_path)
            los_tec_dataframe = add_timestamp_column_to_df(los_tec_dataframe, date)
            los_tec_dataframe = convert_dataframe_to_hard_period(los_tec_dataframe,
                                                                 max_nonbreakable_period_between_points)
            if name_sat_id_file in dict_dataframe_sat_id:
                dict_dataframe_sat_id[name_sat_id_file] = \
                    add_dataframe_with_leveling(dict_dataframe_sat_id[name_sat_id_file],
                                                los_tec_dataframe, max_nonbreakable_period_between_points,
                                                number_of_points=8)
            else:
                dict_dataframe_sat_id[name_sat_id_file] = los_tec_dataframe
    for name_sat_id_file, los_tec_dataframe in dict_dataframe_sat_id.items():
        for params in list_params:
            dataframe_sat_id = add_savgol_data(los_tec_dataframe, params)
            dataframe_sat_id = add_datetime_column_to_df(dataframe_sat_id)
            first_timestamp = dataframe_sat_id.iloc[0].loc["timestamp"]
            first_datetime = dt.datetime.fromtimestamp(first_timestamp, tz=dt.timezone.utc)
            first_day_timestamp = dt.datetime(year=first_datetime.year, month=first_datetime.month,
                                              day=first_datetime.day, hour=0, minute=0, second=0,
                                              tzinfo=dt.timezone.utc).timestamp()
            series_timestamp = (dataframe_sat_id.loc[:, "timestamp"] - first_day_timestamp) // (3600 * 24) * \
                               (3600 * 24) + first_day_timestamp
            array_timestamp = np.unique(series_timestamp)
            for timestamp in array_timestamp:
                date = dt.datetime.fromtimestamp(timestamp)
                save_path_date_1 = get_directory_path(save_directory_path, get_date_str(date))
                save_path_site_2 = get_directory_path(save_path_date_1, site_name)
                save_path_params_3 = get_directory_path(save_path_site_2,
                                                        f"Window_{params['window_length']}_Seconds")
                save_path_sat_id = os.path.join(save_path_params_3, name_sat_id_file)
                mask = dataframe_sat_id.loc[:, "timestamp"] >= timestamp
                mask = np.logical_and(mask, dataframe_sat_id.loc[:, "timestamp"] < (timestamp + 3600 * 24))
                temp_dataframe: pd.DataFrame = dataframe_sat_id.loc[mask]
                save_diff_dataframe_txt(save_path_sat_id, temp_dataframe)

SOURCE_PATH14 = r"/home/vadymskipa/Documents/PhD_student/data/los_txt/ukraine_and_around/los_tec_txt/"
SAVE_PATH14 = r"/home/vadymskipa/Documents/PhD_student/data/los_txt/ukraine_and_around/dtec_txt/"
def get_dtec_from_los_tec_txt_many_sites2(source_path, save_path, params):
    list_site_directory_path = get_site_directory_paths_from_directory(source_path)
    for site_directory_path in list_site_directory_path:
        name_site = os.path.basename(site_directory_path)
        get_dtec_from_los_tec_txt_2(site_directory_path, save_path, name_site, params)


def main2():
    list1 = [r"/home/vadymskipa/Documents/PhD_student/data/Europe_los_hdf/los_20221114.europe.001.h5.hdf5",
             r"/home/vadymskipa/Documents/PhD_student/data/Europe_los_hdf/los_20221115.europe.001.h5.hdf5",
             r"/home/vadymskipa/Documents/PhD_student/data/Europe_los_hdf/los_20221116.europe.001.h5.hdf5",
             r"/home/vadymskipa/Documents/PhD_student/data/Europe_los_hdf/los_20221117.europe.001.h5.hdf5"]
    lats = (44.38, 52.38)
    lons = (22.14, 40.23)
    site_hdf_directory_path = r"/home/vadymskipa/Documents/PhD_student/data/site_hdf/"
    save_path = r"/home/vadymskipa/Documents/PhD_student/data/ukraine_and_around/los_tec_txt/13-17nov2022/"
    los_txt_source = (r"/home/vadymskipa/Documents/PhD_student/data/ukraine_and_around/los_tec_txt/13-17nov2022/"
                      r"los_tec_txt/")
    save_dtec_path = r"/home/vadymskipa/Documents/PhD_student/data/ukraine_and_around/dtec_txt/"

    # save_los_tec_txt_file_for_region_from_los_files(list1, site_hdf_directory_path, save_path, lons=lons,
    #                                                 lats=lats)
    get_dtec_from_los_tec_txt_many_sites2(los_txt_source, save_dtec_path, LIST_PARAMS1)


def main3():
    lats = (44.38, 52.38)
    lons = (22.14, 40.23)
    list1 = [r"/home/vadymskipa/Documents/PhD_student/data/site_hdf/site_20221114.001.h5.hdf5",
             r"/home/vadymskipa/Documents/PhD_student/data/site_hdf/site_20221115.001.h5.hdf5",
             r"/home/vadymskipa/Documents/PhD_student/data/site_hdf/site_20221116.001.h5.hdf5",
             r"/home/vadymskipa/Documents/PhD_student/data/site_hdf/site_20221117.001.h5.hdf5"]
    list2 = [r"/home/vadymskipa/Documents/PhD_student/data/site_hdf/site_20230224.001.h5",
             r"/home/vadymskipa/Documents/PhD_student/data/site_hdf/site_20230323.001.h5.hdf5",
             r"/home/vadymskipa/Documents/PhD_student/data/site_hdf/site_20230423.001.h5.hdf5",
             r"/home/vadymskipa/Documents/PhD_student/data/site_hdf/site_20230605.001.h5.hdf5"]
    for path in list1:
        print(path)
        temp_list = wsite.get_sites_dataframe_by_coordinates(path, lons=lons, lats=lats)
        print(temp_list)
        print(len(temp_list))
    for path in list2:
        print(path)
        temp_list = wsite.get_sites_dataframe_by_coordinates(path, lons=lons, lats=lats)
        print(temp_list)
        print(len(temp_list))


def get_str_for_los_tec_txt(dataframe: pd.DataFrame):
    result_str = (f"{'hour':<4}\t{'min':<4}\t{'sec':<6}\t{'los_tec':<6}\t{'azm':<6}\t{'elm':<6}"
                  f"\t{'gdlat':<6}\t{'gdlon':<6}\n")
    for index in dataframe.index:
        hour = dataframe.loc[index, "hour"]
        min = dataframe.loc[index, "min"]
        sec = dataframe.loc[index, "sec"]
        los_tec = dataframe.loc[index, "los_tec"]
        azm = dataframe.loc[index, "azm"]
        elm = dataframe.loc[index, "elm"]
        gdlat = dataframe.loc[index, "gdlat"]
        gdlon = dataframe.loc[index, "glon"]
        result_str += (f"{hour:<4}\t{min:<4}\t{sec:<4.0f}\t{los_tec:<6.2f}\t{azm:<6.2f}\t{elm:<6.2f}"
                       f"\t{gdlat:<6.2f}\t{gdlon:<6.2f}\n")
    return result_str


def get_str_for_los_tec_txt_process(input_tuple):
    result_str = get_str_for_los_tec_txt(input_tuple[1])
    return input_tuple[0], result_str



def save_los_tec_txt_file_for_lots_of_sites_from_los_file(los_file_path, list_gps_sites, save_path, step_of_sites=100,
                                                          multiprocessing=False, gps_only=True):
    length_of_text = 120
    new_save_path = get_directory_path(save_path, "los_tec_txt")
    start = dt.datetime.now()
    print_text = "Reading \"gps_site\" column from los_fie"
    print(f"{print_text:-<{length_of_text}s}{start}")
    with h5py.File(los_file_path, "r") as file:
        site_array = file["Data"]["Table Layout"]["gps_site"]
    print_text = "End of reading \"gps_site\" column from los_file"
    print(f"{print_text:-<{length_of_text}s}{dt.datetime.now()} -- {dt.datetime.now()-start}")

    for start_index in range(0, len(list_gps_sites), step_of_sites):
        end_index = start_index + step_of_sites
        if end_index > len(list_gps_sites):
            end_index = len(list_gps_sites)
        sites = list_gps_sites[start_index:end_index]
        mask = wlos.get_indices_for_sites_from_site_array_multiprocessing(site_array, sites)

        print_text = f"Reading data for sites {start_index + 1}-{end_index} of {len(list_gps_sites)}"
        print(f"{print_text:-<{length_of_text}s}{dt.datetime.now()}")
        if gps_only:
            los_dataframe = wlos.get_data_by_indices_GPS_pd(los_file_path, mask)
        else:
            los_dataframe = wlos.get_data_by_indices_pd(los_file_path, mask)
        # los_dataframe = rlos.get_data_by_indeces_GPS_pd(los_file_path, mask)
        print_text = f"End of reading data for sites {start_index + 1}-{end_index} of {len(list_gps_sites)}"
        print(f"{print_text:-<{length_of_text}s}{dt.datetime.now()} -- {dt.datetime.now()-start}")

        max_nonbreakable_period_between_points = 5 * PERIOD_CONST
        name_los_file = os.path.basename(los_file_path)
        date = wlos.get_date_by_los_file_name(name_los_file)
        los_dataframe = add_timestamp_column_to_df(los_dataframe, date)
        for site in sites:
            print_text = f"    Saving los_tec_txt for site {site}"
            print(f"{print_text:-<{length_of_text}s}{dt.datetime.now()} -- {dt.datetime.now()-start}")
            save_path_site_1 = get_directory_path(new_save_path, site)
            site_dataframe = los_dataframe.loc[los_dataframe.loc[:, "gps_site"] == site.encode("ascii")]
            save_path_date_2 = get_directory_path(save_path_site_1, get_date_str(date))
            sat_id_array = np.unique(site_dataframe.loc[:, "sat_id"])
            if multiprocessing:
                list_sat_id_tuple = [(sat_id, site_dataframe.loc[site_dataframe.loc[:, "sat_id"] == sat_id])
                                     for sat_id in sat_id_array]
                pool = multipool.Pool(mp.cpu_count())
                for sat_id, save_str in pool.imap(get_str_for_los_tec_txt_process, list_sat_id_tuple):
                    save_path_sat_id_3 = os.path.join(save_path_date_2, f"G{sat_id:0=2}.txt")
                    with open(save_path_sat_id_3, "w") as file:
                        file.write(save_str)
                pool.close()
            else:
                for sat_id in sat_id_array:
                    save_path_sat_id_3 = os.path.join(save_path_date_2, f"G{sat_id:0=2}.txt")
                    sat_id_dataframe = site_dataframe.loc[site_dataframe.loc[:, "sat_id"] == sat_id]
                    save_los_tec_txt_from_dataframe(save_path_sat_id_3, sat_id_dataframe)


def save_los_tec_txt_file_for_region_from_los_files(list_los_hdf_file_paths, source_site_hdf_directory_path,
                                                    save_path, lats, lons, multiprocessing=False, gps_only=True):
    list_of_objects_in_site_directory = os.listdir(source_site_hdf_directory_path)
    list_site_hdf_file_names = [el for el in list_of_objects_in_site_directory
                                if (os.path.isfile(os.path.join(source_site_hdf_directory_path, el)) and
                                    wsite.check_site_file_name(el))]
    for los_hdf_file_path in list_los_hdf_file_paths:
        los_hdf_file_name = os.path.basename(los_hdf_file_path)
        date = wlos.get_date_by_los_file_name(los_hdf_file_name)
        site_file_hdf_name = None
        for site_file_name_check in list_site_hdf_file_names:
            if wsite.check_site_file_name_by_date(site_file_name_check, date):
                site_file_hdf_name = site_file_name_check
        if not site_file_hdf_name:
            continue
        list_of_sites_bytes = list(wsite.get_sites_dataframe_by_coordinates(os.path.join(source_site_hdf_directory_path,
                                                                                   site_file_hdf_name),
                                                                            lons=lons, lats=lats).loc[:, "gps_site"])
        list_of_sites = [site_bytes.decode("ascii") for site_bytes in list_of_sites_bytes]
        save_los_tec_txt_file_for_lots_of_sites_from_los_file(los_hdf_file_path, list_of_sites, save_path,
                                                              multiprocessing=multiprocessing, gps_only=gps_only)


def main2023jan1():
    lats = (36, 70)
    lons = (-10, 20)
    source_site_hdf_directory_path = r"/home/vadymskipa/PhD_student/data/site_hdf/"
    list_los_hdf = [r"/home/vadymskipa/PhD_student/data/Europe_los_hdf/los_20200923.europe.001.h5"]
    save_path = r"/home/vadymskipa/PhD_student/data/Europe_36-70__-10-20"
    save_los_tec_txt_file_for_region_from_los_files(list_los_hdf, source_site_hdf_directory_path, save_path,
                                                    lats=lats, lons=lons)


def get_str_for_dtec_txt(dataframe: pd.DataFrame):
    dataframe_without_nan = dataframe.loc[dataframe.loc[:, "diff"].notna()]
    result_str = (f"{'hour':<4}\t{'min':<4}\t{'sec':<6}\t{'dTEC':<6}\t{'azm':<6}\t{'elm':<6}\t{'gdlat':<6}\t"
                  f"{'gdlon':<6}\n")
    for index in dataframe_without_nan.index:
        hour = dataframe_without_nan.loc[index, "datetime"].hour
        minute = dataframe_without_nan.loc[index, "datetime"].minute
        sec = dataframe_without_nan.loc[index, "datetime"].second
        dtec = dataframe_without_nan.loc[index, "diff"]
        azm = dataframe_without_nan.loc[index, "azm"]
        elm = dataframe_without_nan.loc[index, "elm"]
        gdlat = dataframe_without_nan.loc[index, "gdlat"]
        gdlon = dataframe_without_nan.loc[index, "gdlon"]
        result_str += (f"{hour:<4}\t{minute:<4}\t{sec:<4.0f}\t{dtec:<6.3f}\t{azm:<6.2f}\t{elm:<6.2f}\t{gdlat:<6.2f}\t"
                       f"{gdlon:<6.2f}\n")
    return result_str


def get_dtec_txt_from_los_dataframe_process(data_tuple):
    name_sat_id_file: str = data_tuple[0]
    params = data_tuple[1]
    input_dataframe = data_tuple[2]
    temp_dataframe = add_savgol_data(input_dataframe, params)
    if temp_dataframe.empty:
        return None
    result_dataframe = add_datetime_column_to_df(temp_dataframe)
    # return sat_id, params, output_dataframe

    first_timestamp = result_dataframe.iloc[0].loc["timestamp"]
    first_datetime = dt.datetime.fromtimestamp(first_timestamp, tz=dt.timezone.utc)
    first_day_timestamp = dt.datetime(year=first_datetime.year, month=first_datetime.month,
                                      day=first_datetime.day, hour=0, minute=0, second=0,
                                      tzinfo=dt.timezone.utc).timestamp()
    series_timestamp = (result_dataframe.loc[:, "timestamp"] - first_day_timestamp) // (3600 * 24) * \
                       (3600 * 24) + first_day_timestamp
    array_timestamp = np.unique(series_timestamp)
    result_list = []
    for timestamp in array_timestamp:
        date = dt.datetime.fromtimestamp(timestamp)
        # save_path_date_1 = get_directory_path(save_directory_path, get_date_str(date))
        # save_path_site_2 = get_directory_path(save_path_date_1, site_name)
        # save_path_params_3 = get_directory_path(save_path_site_2,
        #                                         f"Window_{params['window_length']}_Seconds")
        # save_path_sat_id = os.path.join(save_path_params_3, name_sat_id_file)
        mask = result_dataframe.loc[:, "timestamp"] >= timestamp
        mask = np.logical_and(mask, result_dataframe.loc[:, "timestamp"] < (timestamp + 3600 * 24))
        temp_dataframe: pd.DataFrame = result_dataframe.loc[mask]
        result_str = get_str_for_dtec_txt(temp_dataframe)
        result_list.append((name_sat_id_file, params, date, result_str))
    return result_list



def get_dtec_from_los_tec_txt_for_site_directory(source_directory_path, save_directory_path, site_name,
                                                 list_params=LIST_PARAMS1, min_date=None, max_date=None):
    max_nonbreakable_period_between_points = 8 * PERIOD_CONST
    list_of_date_directory_paths = get_date_directory_paths_from_directory(source_directory_path)
    list_of_date_directory_paths.sort()
    dict_dataframe_sat_id = {}
    for date_directory_path in list_of_date_directory_paths:
        name_date_directory = os.path.basename(date_directory_path)
        date = get_date_from_date_directory_name(name_date_directory)
        if min_date and min_date > date:
            continue
        if max_date and max_date < date:
            continue
        list_of_sat_id_los_tec_file_paths = get_sat_id_file_paths_from_directory(date_directory_path)
        for sat_id_los_tec_file_path in list_of_sat_id_los_tec_file_paths:
            name_sat_id_file = os.path.basename(sat_id_los_tec_file_path)
            los_tec_dataframe = read_sat_file(sat_id_los_tec_file_path)
            los_tec_dataframe = add_timestamp_column_to_df(los_tec_dataframe, date)
            los_tec_dataframe = convert_dataframe_to_hard_period(los_tec_dataframe,
                                                                 max_nonbreakable_period_between_points)
            if name_sat_id_file in dict_dataframe_sat_id:
                dict_dataframe_sat_id[name_sat_id_file] = \
                    add_dataframe_with_leveling(dict_dataframe_sat_id[name_sat_id_file],
                                                los_tec_dataframe, max_nonbreakable_period_between_points,
                                                number_of_points=8)
            else:
                dict_dataframe_sat_id[name_sat_id_file] = los_tec_dataframe
    list_of_process_tuples = [(name_sat_id_file, params, los_tec_dataframe)
                              for name_sat_id_file, los_tec_dataframe in dict_dataframe_sat_id.items()
                              for params in list_params]
    pool = multipool.Pool(mp.cpu_count())
    for temp_list in pool.imap(get_dtec_txt_from_los_dataframe_process, list_of_process_tuples):
        if not temp_list:
            continue
        for name_sat_id_file, params, date, result_str in temp_list:
            save_path_date_1 = get_directory_path(save_directory_path, get_date_str(date))
            save_path_site_2 = get_directory_path(save_path_date_1, site_name)
            save_path_params_3 = get_directory_path(save_path_site_2,
                                                    f"Window_{params['window_length']}_Seconds")
            save_path_sat_id = os.path.join(save_path_params_3, name_sat_id_file)
            with open(save_path_sat_id, "w") as file:
                file.write(result_str)
    pool.close()

def get_dtec_from_los_tec_txt_for_lots_of_sites(source_path, save_path, params, min_date=None, max_date=None):
    list_site_directory_path = get_site_directory_paths_from_directory(source_path)
    for site_directory_path in list_site_directory_path:
        name_site = os.path.basename(site_directory_path)
        print_text = f"Work with site {name_site}"
        print(f"{print_text:-<{120}s}{dt.datetime.now()}")
        get_dtec_from_los_tec_txt_for_site_directory(site_directory_path, save_path, name_site, params,
                                                     min_date=min_date, max_date=max_date)


def main2023jan2():
    source_path = r"/home/vadymskipa/Documents/PhD_student/data/Europe_36-70__-10-20/los_tec_txt"
    save_path = r"/home/vadymskipa/Documents/PhD_student/data/Europe_36-70__-10-20/dtec_txt"
    # save_path = r"/home/vadymskipa/Documents/PhD_student/data/Europe_36-70__-10-20/dtec_txt_monoprocess"
    params = LIST_PARAMS1
    get_dtec_from_los_tec_txt_for_lots_of_sites(source_path, save_path, params)


def get_str_for_sites_txt(dataframe: pd.DataFrame):
    result_str = f"{'site':<4}\t{'lat':<6}\t{'lon':<6}\n"
    for index in dataframe.index:
        site = dataframe.loc[index, "gps_site"].decode("ascii")
        lat = dataframe.loc[index, "gdlatr"]
        lon = dataframe.loc[index, "gdlonr"]
        result_str += f"{site:<4s}\t{lat:<6.2f}\t{lon:<6.2f}\n"
    return result_str


def read_sites_txt(path):
    with open(path, "r") as file:
        file.readline()
        list_site = []
        list_lat = []
        list_lon = []
        for line in file:
            text_tuple = line.split()
            site = text_tuple[0]
            lat = float(text_tuple[1])
            lon = float(text_tuple[2])
            list_site.append(site)
            list_lat.append(lat)
            list_lon.append(lon)
        dataframe = pd.DataFrame.from_dict({"gps_site": list_site, "gdlatr": list_lat, "gdlonr": list_lon})
    return dataframe




def save_sites_txt_for_site_directories(source_directory_path, site_hdf_path, save_path):
    list_site_directory_path = get_site_directory_paths_from_directory(source_directory_path)
    list_site = [os.path.basename(site_directory_path).encode("ascii") for site_directory_path
                 in list_site_directory_path]
    list_site.sort()
    site_dataframe = wsite.get_sites_dataframe(site_hdf_path)
    result_dataframe = pd.DataFrame()
    for site in list_site:
        result_dataframe = pd.concat([result_dataframe, site_dataframe.loc[site_dataframe.loc[:, "gps_site"] == site]],
                                     axis=0)
    result_str = get_str_for_sites_txt(result_dataframe)
    save_file_path = os.path.join(save_path, f"Sites.txt")
    with open(save_file_path, "w") as file:
        file.write(result_str)


def main2023jan3():
    source_directory_patrh1 = (r"/home/vadymskipa/Documents/PhD_student/data/Europe_36-70__-10-20/dtec_txt/"
                               r"267-2020-09-23/")
    source_directory_patrh2 = (r"/home/vadymskipa/Documents/PhD_student/data/Europe_36-70__-10-20/dtec_txt/"
                               r"268-2020-09-24/")
    site_hdf_path1 = r"/home/vadymskipa/Documents/PhD_student/data/site_hdf/site_20200923.001.h5"
    site_hdf_path2 = r"/home/vadymskipa/Documents/PhD_student/data/site_hdf/site_20200924.001.h5"
    save_sites_txt_for_site_directories(source_directory_patrh1, site_hdf_path1, source_directory_patrh1)
    save_sites_txt_for_site_directories(source_directory_patrh2, site_hdf_path2, source_directory_patrh2)



def main2023jan4():
    lats = (36, 70)
    lons = (-10, 20)
    source_site_hdf_directory_path = r"/home/vadymskipa/Documents/PhD_student/data/site_hdf/"
    list_los_hdf = [r"/home/vadymskipa/Documents/PhD_student/data/Europe_los_hdf/los_20230225.europe.001.h5.hdf5",
                    r"/home/vadymskipa/Documents/PhD_student/data/Europe_los_hdf/los_20230226.europe.001.h5.hdf5",
                    r"/home/vadymskipa/Documents/PhD_student/data/Europe_los_hdf/los_20230227.europe.001.h5.hdf5",
                    r"/home/vadymskipa/Documents/PhD_student/data/Europe_los_hdf/los_20230228.europe.001.h5.hdf5",
                    r"/home/vadymskipa/Documents/PhD_student/data/Europe_los_hdf/los_20230301.europe.001.h5.hdf5",
                    r"/home/vadymskipa/Documents/PhD_student/data/Europe_los_hdf/los_20230302.europe.001.h5.hdf5"]
    save_path = r"/home/vadymskipa/Documents/PhD_student/data/Europe_36-70__-10-20"
    save_los_tec_txt_file_for_region_from_los_files(list_los_hdf, source_site_hdf_directory_path, save_path,
                                                    lats=lats, lons=lons, multiprocessing=True)


def main2023jan5():
    source_path = r"/home/vadymskipa/Documents/PhD_student/data/Europe_36-70__-10-20/los_tec_txt"
    save_path = r"/home/vadymskipa/Documents/PhD_student/data/Europe_36-70__-10-20/dtec_txt"
    params = LIST_PARAMS1
    min_date = dt.datetime(year=2023, month=2, day=24, tzinfo=dt.timezone.utc)
    max_date = dt.datetime(year=2023, month=3, day=2, tzinfo=dt.timezone.utc)
    get_dtec_from_los_tec_txt_for_lots_of_sites(source_path, save_path, params, min_date=min_date, max_date=max_date)


def main2023jan6():
    source_directories = [r"/home/vadymskipa/Documents/PhD_student/data/Europe_36-70__-10-20/dtec_txt/055-2023-02-24",
                          r"/home/vadymskipa/Documents/PhD_student/data/Europe_36-70__-10-20/dtec_txt/056-2023-02-25",
                          r"/home/vadymskipa/Documents/PhD_student/data/Europe_36-70__-10-20/dtec_txt/057-2023-02-26",
                          r"/home/vadymskipa/Documents/PhD_student/data/Europe_36-70__-10-20/dtec_txt/058-2023-02-27",
                          r"/home/vadymskipa/Documents/PhD_student/data/Europe_36-70__-10-20/dtec_txt/059-2023-02-28",
                          r"/home/vadymskipa/Documents/PhD_student/data/Europe_36-70__-10-20/dtec_txt/060-2023-03-01",
                          r"/home/vadymskipa/Documents/PhD_student/data/Europe_36-70__-10-20/dtec_txt/061-2023-03-02",]
    site_directory = r"/home/vadymskipa/Documents/PhD_student/data/site_hdf/"
    site_paths = wsite.get_site_file_paths_from_directory(site_directory)
    for source in source_directories:
        date = get_date_from_date_directory_name(os.path.basename(source))
        site_file = None
        for site_path in site_paths:
            if wsite.check_site_file_name_by_date(os.path.basename(site_path), date):
                site_file = site_path
                break
        save_sites_txt_for_site_directories(source, site_file, source)


def main2023jan7():
    path = r"/home/vadymskipa/Documents/PhD_student/data/site_hdf/site_20230226.001.h5.hdf5"
    lats = (36, 70)
    lons = (-10, -5)
    sites = wsite.get_sites_dataframe_by_coordinates(path, lats=lats, lons=lons)
    print(sites)
    path = r"/home/vadymskipa/Documents/PhD_student/data/site_hdf/site_20230224.001.h5"
    sites = wsite.get_sites_dataframe_by_coordinates(path, lats=lats, lons=lons)
    print(sites)


def main2023jan8():
    list_los_hdf = [r"/home/vadymskipa/Documents/PhD_student/data/Europe_los_hdf/los_20230225.europe.001.h5.hdf5",
                    r"/home/vadymskipa/Documents/PhD_student/data/Europe_los_hdf/los_20230226.europe.001.h5.hdf5",
                    r"/home/vadymskipa/Documents/PhD_student/data/Europe_los_hdf/los_20230227.europe.001.h5.hdf5",
                    r"/home/vadymskipa/Documents/PhD_student/data/Europe_los_hdf/los_20230228.europe.001.h5.hdf5",
                    r"/home/vadymskipa/Documents/PhD_student/data/Europe_los_hdf/los_20230301.europe.001.h5.hdf5",
                    r"/home/vadymskipa/Documents/PhD_student/data/Europe_los_hdf/los_20230302.europe.001.h5.hdf5"]
    for los_hdf in list_los_hdf:
        with h5py.File(los_hdf, "r") as file:
            lons = file["Data"]["Table Layout"]["gdlonr"]
        print(lons.min(), lons.max())


def main2023jan9():
    lats = (36, 70)
    lons = (-10, 20)

    source_site_hdf_directory_path = r"/home/vadymskipa/Documents/PhD_student/data/site_hdf/"
    list_los_hdf = [r"/home/vadymskipa/Documents/PhD_student/data/Europe_los_hdf/los_20230226.europe.001.h5"]
    save_path = r"/home/vadymskipa/Documents/PhD_student/data/Europe_36-70__-10-20"
    save_los_tec_txt_file_for_region_from_los_files(list_los_hdf, source_site_hdf_directory_path, save_path,
                                                    lats=lats, lons=lons, multiprocessing=True)


def main2023jan10():
    lats = (36, 70)
    lons = (-10, 20)
    min_date = dt.datetime(year=2023, month=3, day=21, tzinfo=dt.timezone.utc)
    max_date = dt.datetime(year=2023, month=3, day=28, tzinfo=dt.timezone.utc)
    source_site_hdf_directory_path = r"/home/vadymskipa/Documents/PhD_student/data/site_hdf/"
    source_los_hdf_directory = r"/home/vadymskipa/Documents/PhD_student/data/Europe_los_hdf/"
    los_hdfs_in_directory = os.listdir(source_los_hdf_directory)
    list_los_hdf_paths = []
    for los_hdf in los_hdfs_in_directory:
        temp_date = wlos.get_date_by_los_file_name(los_hdf)
        if temp_date < min_date:
            continue
        if temp_date > max_date:
            continue
        list_los_hdf_paths.append(os.path.join(source_los_hdf_directory, los_hdf))
    save_path = r"/home/vadymskipa/Documents/PhD_student/data/Europe_36-70__-10-20"
    save_los_tec_txt_file_for_region_from_los_files(list_los_hdf_paths, source_site_hdf_directory_path, save_path,
                                                    lats=lats, lons=lons, multiprocessing=True)


def main2023jan11():
    source_los_hdf_directory = r"/home/vadymskipa/Documents/PhD_student/data/Europe_los_hdf/"
    min_date = dt.datetime(year=2023, month=3, day=21, tzinfo=dt.timezone.utc)
    max_date = dt.datetime(year=2023, month=3, day=28, tzinfo=dt.timezone.utc)
    los_hdfs_in_directory = os.listdir(source_los_hdf_directory)
    list_los_hdf_paths = []
    for los_hdf in los_hdfs_in_directory:
        temp_date = wlos.get_date_by_los_file_name(los_hdf)
        if temp_date < min_date:
            continue
        if temp_date > max_date:
            continue
        list_los_hdf_paths.append(os.path.join(source_los_hdf_directory, los_hdf))
    for los_hdf in list_los_hdf_paths:
        with h5py.File(los_hdf, "r") as file:
            lons = file["Data"]["Table Layout"]["gdlonr"]
        print(lons.min(), lons.max(), los_hdf)
        with h5py.File(los_hdf, "r") as file:
            lats = file["Data"]["Table Layout"]["gdlatr"]
        print(lats.min(), lats.max(), los_hdf)


def main2023jan12():
    source_path = r"/home/vadymskipa/Documents/PhD_student/data/Europe_36-70__-10-20/los_tec_txt"
    save_path = r"/home/vadymskipa/Documents/PhD_student/data/Europe_36-70__-10-20/dtec_txt"
    params = LIST_PARAMS1
    min_date = dt.datetime(year=2023, month=3, day=21, tzinfo=dt.timezone.utc)
    max_date = dt.datetime(year=2023, month=3, day=28, tzinfo=dt.timezone.utc)
    get_dtec_from_los_tec_txt_for_lots_of_sites(source_path, save_path, params, min_date=min_date, max_date=max_date)


def main2023jan13():
    source_directories = [r"/home/vadymskipa/Documents/PhD_student/data/Europe_36-70__-10-20/dtec_txt/080-2023-03-21",
                          r"/home/vadymskipa/Documents/PhD_student/data/Europe_36-70__-10-20/dtec_txt/081-2023-03-22",
                          r"/home/vadymskipa/Documents/PhD_student/data/Europe_36-70__-10-20/dtec_txt/082-2023-03-23",
                          r"/home/vadymskipa/Documents/PhD_student/data/Europe_36-70__-10-20/dtec_txt/083-2023-03-24",
                          r"/home/vadymskipa/Documents/PhD_student/data/Europe_36-70__-10-20/dtec_txt/084-2023-03-25",
                          r"/home/vadymskipa/Documents/PhD_student/data/Europe_36-70__-10-20/dtec_txt/085-2023-03-26",
                          r"/home/vadymskipa/Documents/PhD_student/data/Europe_36-70__-10-20/dtec_txt/086-2023-03-27",
                          r"/home/vadymskipa/Documents/PhD_student/data/Europe_36-70__-10-20/dtec_txt/087-2023-03-28"]
    site_directory = r"/home/vadymskipa/Documents/PhD_student/data/site_hdf/"
    site_paths = wsite.get_site_file_paths_from_directory(site_directory)
    for source in source_directories:
        date = get_date_from_date_directory_name(os.path.basename(source))
        site_file = None
        for site_path in site_paths:
            if wsite.check_site_file_name_by_date(os.path.basename(site_path), date):
                site_file = site_path
                break
        save_sites_txt_for_site_directories(source, site_file, source)


def main2023jan14():
    source_los_hdf_directory = r"/home/vadymskipa/Documents/PhD_student/data/Europe_los_hdf/"
    min_date = dt.datetime(year=2023, month=4, day=21, tzinfo=dt.timezone.utc)
    max_date = dt.datetime(year=2023, month=4, day=28, tzinfo=dt.timezone.utc)
    los_hdfs_in_directory = os.listdir(source_los_hdf_directory)
    list_los_hdf_paths = []
    for los_hdf in los_hdfs_in_directory:
        temp_date = wlos.get_date_by_los_file_name(los_hdf)
        if temp_date < min_date:
            continue
        if temp_date > max_date:
            continue
        list_los_hdf_paths.append(os.path.join(source_los_hdf_directory, los_hdf))
    for los_hdf in list_los_hdf_paths:
        with h5py.File(los_hdf, "r") as file:
            lons = file["Data"]["Table Layout"]["gdlonr"]
        print(lons.min(), lons.max(), los_hdf)
        with h5py.File(los_hdf, "r") as file:
            lats = file["Data"]["Table Layout"]["gdlatr"]
        print(lats.min(), lats.max(), los_hdf)


def main2023jan15():
    lats = (36, 70)
    lons = (-10, 20)
    min_date = dt.datetime(year=2023, month=4, day=22, tzinfo=dt.timezone.utc)
    max_date = dt.datetime(year=2023, month=4, day=26, tzinfo=dt.timezone.utc)
    source_site_hdf_directory_path = r"/home/vadymskipa/Documents/PhD_student/data/site_hdf/"
    source_los_hdf_directory = r"/home/vadymskipa/Documents/PhD_student/data/Europe_los_hdf/"
    los_hdfs_in_directory = os.listdir(source_los_hdf_directory)
    list_los_hdf_paths = []
    for los_hdf in los_hdfs_in_directory:
        temp_date = wlos.get_date_by_los_file_name(los_hdf)
        if temp_date < min_date:
            continue
        if temp_date > max_date:
            continue
        list_los_hdf_paths.append(os.path.join(source_los_hdf_directory, los_hdf))
    save_path = r"/home/vadymskipa/Documents/PhD_student/data/Europe_36-70__-10-20"
    save_los_tec_txt_file_for_region_from_los_files(list_los_hdf_paths, source_site_hdf_directory_path, save_path,
                                                    lats=lats, lons=lons, multiprocessing=True)


def main2023jan16():
    source_path = r"/home/vadymskipa/Documents/PhD_student/data/Europe_36-70__-10-20/los_tec_txt"
    save_path = r"/home/vadymskipa/Documents/PhD_student/data/Europe_36-70__-10-20/dtec_txt"
    params = LIST_PARAMS1
    min_date = dt.datetime(year=2023, month=4, day=22, tzinfo=dt.timezone.utc)
    max_date = dt.datetime(year=2023, month=4, day=26, tzinfo=dt.timezone.utc)
    get_dtec_from_los_tec_txt_for_lots_of_sites(source_path, save_path, params, min_date=min_date, max_date=max_date)


def main2023jan17():
    source_directories = [r"/home/vadymskipa/Documents/PhD_student/data/Europe_36-70__-10-20/dtec_txt/112-2023-04-22",
                          r"/home/vadymskipa/Documents/PhD_student/data/Europe_36-70__-10-20/dtec_txt/113-2023-04-23",
                          r"/home/vadymskipa/Documents/PhD_student/data/Europe_36-70__-10-20/dtec_txt/114-2023-04-24",
                          r"/home/vadymskipa/Documents/PhD_student/data/Europe_36-70__-10-20/dtec_txt/115-2023-04-25",
                          r"/home/vadymskipa/Documents/PhD_student/data/Europe_36-70__-10-20/dtec_txt/116-2023-04-26"]
    site_directory = r"/home/vadymskipa/Documents/PhD_student/data/site_hdf/"
    site_paths = wsite.get_site_file_paths_from_directory(site_directory)
    for source in source_directories:
        date = get_date_from_date_directory_name(os.path.basename(source))
        site_file = None
        for site_path in site_paths:
            if wsite.check_site_file_name_by_date(os.path.basename(site_path), date):
                site_file = site_path
                break
        save_sites_txt_for_site_directories(source, site_file, source)


def main2024jan18():
    los_txt1 = r"/home/vadymskipa/Documents/PhD_student/data/Europe_36-70__-10-20/los_tec_txt/cral/055-2023-02-24/G01.txt"
    los_txt2 = r"/home/vadymskipa/Documents/PhD_student/data/Europe_36-70__-10-20/los_tec_txt/cral/056-2023-02-25/G01.txt"
    los_dataframe1 = read_sat_file(los_txt1)
    los_dataframe2 = read_sat_file(los_txt2)
    max_nonbreakable_period_between_points = 8 * PERIOD_CONST
    date1 = dt.datetime(year=2023, month=2, day=24, tzinfo=dt.timezone.utc)
    date2 = dt.datetime(year=2023, month=2, day=25, tzinfo=dt.timezone.utc)
    los_dataframe1 = add_timestamp_column_to_df(los_dataframe1, date1)
    los_dataframe2 = add_timestamp_column_to_df(los_dataframe2, date2)
    los_dataframe1 = add_datetime_column_to_df(los_dataframe1)
    los_dataframe2 = add_datetime_column_to_df(los_dataframe2)
    los_dataframe = pd.concat([los_dataframe1, los_dataframe2], axis=0)
    new_los_dataframe1 = convert_dataframe_to_hard_period(los_dataframe1, max_nonbreakable_period_between_points)
    new_los_dataframe2 = convert_dataframe_to_hard_period(los_dataframe2, max_nonbreakable_period_between_points)
    new_los_dataframe = add_dataframe_with_leveling(new_los_dataframe1, new_los_dataframe2,
                                                    max_nonbreakable_period_between_points, number_of_points=8)
    new_dataframe = add_savgol_data(new_los_dataframe, LIST_PARAMS1[0])
    new_dataframe = add_datetime_column_to_df(new_dataframe)
    save_path = r"/home/vadymskipa/Documents/PhD_student/temp/"
    start = dt.datetime(year=2023, month=2, day=24, hour=23, tzinfo=dt.timezone.utc)
    end = dt.datetime(year=2023, month=2, day=25, hour=1, tzinfo=dt.timezone.utc)
    x_tics = [dt.datetime(year=2023, month=2, day=24, hour=23, minute=30, tzinfo=dt.timezone.utc),
              dt.datetime(year=2023, month=2, day=25, tzinfo=dt.timezone.utc),
              dt.datetime(year=2023, month=2, day=25, minute=30, tzinfo=dt.timezone.utc)]
    x_ticks_name = ["23:30", "00:00", "00:30"]
    figure: fig.Figure = plt.figure(layout="tight", figsize=[9.0 * 120 / 300, 9.0 * 120 / 300])
    axes1: axs.Axes = figure.add_subplot(1, 1, 1)
    line1, = axes1.plot(los_dataframe.loc[:, "datetime"], los_dataframe["los_tec"], linestyle=" ", marker=".",
                        markeredgewidth=0.5, markersize=1.5, linewidth=0.6)
    axes1.set_xlim(start, end)
    axes1.set_ylim(7, 18)
    axes1.set_xticks(x_tics, x_ticks_name)
    axes1.grid(True)
    plt.savefig(os.path.join(save_path, "picture1.png"), dpi=300)
    plt.close(figure)
    figure: fig.Figure = plt.figure(layout="tight", figsize=[9.0 * 120 / 300, 9.0 * 120 / 300])
    axes1: axs.Axes = figure.add_subplot(1, 1, 1)
    line1, = axes1.plot(new_dataframe.loc[:, "datetime"], new_dataframe["los_tec"], linestyle=" ", marker=".",
                        markeredgewidth=0.5, markersize=1.5, linewidth=0.6)
    axes1.set_xlim(start, end)
    axes1.set_ylim(7, 18)
    axes1.set_xticks(x_tics, x_ticks_name)
    axes1.grid(True)
    plt.savefig(os.path.join(save_path, "picture2.png"), dpi=300)
    plt.close(figure)
    figure: fig.Figure = plt.figure(layout="tight", figsize=[9.0 * 120 / 300, 9.0 * 120 / 300])
    axes1: axs.Axes = figure.add_subplot(1, 1, 1)
    line1, = axes1.plot(new_dataframe.loc[:, "datetime"], new_dataframe["savgol"], linestyle=" ", marker=".",
                        markeredgewidth=0.5, markersize=1.5, linewidth=0.6)
    axes1.set_xlim(start, end)
    axes1.set_ylim(7, 18)
    axes1.set_xticks(x_tics, x_ticks_name)
    axes1.grid(True)
    plt.savefig(os.path.join(save_path, "picture3.png"), dpi=300)
    plt.close(figure)
    figure: fig.Figure = plt.figure(layout="tight", figsize=[9.0 * 120 / 300, 9.0 * 120 / 300])
    axes1: axs.Axes = figure.add_subplot(1, 1, 1)
    line1, = axes1.plot(new_dataframe.loc[:, "datetime"], new_dataframe["diff"], linestyle="-", marker=".",
                        markeredgewidth=0.5, markersize=1.5, linewidth=0.6)
    axes1.set_xlim(start, end)
    axes1.set_ylim(-0.2, 0.2)
    axes1.set_xticks(x_tics, x_ticks_name)
    axes1.grid(True)
    plt.savefig(os.path.join(save_path, "picture4.png"), dpi=300)
    plt.close(figure)


def main2024feb1():
    lats = (25, 70)
    lons = (-125, -65)
    min_date = dt.datetime(year=2023, month=2, day=24, tzinfo=dt.timezone.utc)
    max_date = dt.datetime(year=2023, month=3, day=3, tzinfo=dt.timezone.utc)
    source_site_hdf_directory_path = r"/home/vadymskipa/PhD_student/data/site_hdf/"
    source_los_hdf_directory = r"/home/vadymskipa/PhD_student/data/America_25-70__-125--65/los_hdf/"
    los_hdfs_in_directory = os.listdir(source_los_hdf_directory)
    list_los_hdf_paths = []
    for los_hdf in los_hdfs_in_directory:
        temp_date = wlos.get_date_by_los_file_name(los_hdf)
        if temp_date < min_date:
            continue
        if temp_date > max_date:
            continue
        list_los_hdf_paths.append(os.path.join(source_los_hdf_directory, los_hdf))
    save_path = r"/home/vadymskipa/PhD_student/data/America_25-70__-125--65/"
    save_los_tec_txt_file_for_region_from_los_files(list_los_hdf_paths, source_site_hdf_directory_path, save_path,
                                                    lats=lats, lons=lons, multiprocessing=True)


def main2024feb2():
    source_path = r"/home/vadymskipa/PhD_student/data/America_25-70__-125--65/los_tec_txt"
    save_path = r"/home/vadymskipa/PhD_student/data/America_25-70__-125--65/dtec_txt"
    params = LIST_PARAMS1
    min_date = dt.datetime(year=2023, month=2, day=24, tzinfo=dt.timezone.utc)
    max_date = dt.datetime(year=2023, month=3, day=3, tzinfo=dt.timezone.utc)
    get_dtec_from_los_tec_txt_for_lots_of_sites(source_path, save_path, params, min_date=min_date, max_date=max_date)


def main2024feb3():
    source_directories = [r"/home/vadymskipa/PhD_student/data/America_25-70__-125--65/dtec_txt/055-2023-02-24",
                          r"/home/vadymskipa/PhD_student/data/America_25-70__-125--65/dtec_txt/056-2023-02-25",
                          r"/home/vadymskipa/PhD_student/data/America_25-70__-125--65/dtec_txt/057-2023-02-26",
                          r"/home/vadymskipa/PhD_student/data/America_25-70__-125--65/dtec_txt/058-2023-02-27",
                          r"/home/vadymskipa/PhD_student/data/America_25-70__-125--65/dtec_txt/059-2023-02-28",
                          r"/home/vadymskipa/PhD_student/data/America_25-70__-125--65/dtec_txt/060-2023-03-01",
                          r"/home/vadymskipa/PhD_student/data/America_25-70__-125--65/dtec_txt/061-2023-03-02",
                          r"/home/vadymskipa/PhD_student/data/America_25-70__-125--65/dtec_txt/062-2023-03-03"]
    site_directory = r"/home/vadymskipa/PhD_student/data/site_hdf/"
    site_paths = wsite.get_site_file_paths_from_directory(site_directory)
    for source in source_directories:
        date = get_date_from_date_directory_name(os.path.basename(source))
        site_file = None
        for site_path in site_paths:
            if wsite.check_site_file_name_by_date(os.path.basename(site_path), date):
                site_file = site_path
                break
        save_sites_txt_for_site_directories(source, site_file, source)


def main2024feb4():
    lats = (-35, -20)
    lons = (15, 35)
    min_date = dt.datetime(year=2023, month=2, day=24, tzinfo=dt.timezone.utc)
    max_date = dt.datetime(year=2023, month=3, day=3, tzinfo=dt.timezone.utc)
    source_site_hdf_directory_path = r"/home/vadymskipa/PhD_student/data/site_hdf/"
    source_los_hdf_directory = r"/home/vadymskipa/PhD_student/data/South_Africa_-20--35__15-35/los_hdf/"
    los_hdfs_in_directory = os.listdir(source_los_hdf_directory)
    list_los_hdf_paths = []
    for los_hdf in los_hdfs_in_directory:
        temp_date = wlos.get_date_by_los_file_name(los_hdf)
        if temp_date < min_date:
            continue
        if temp_date > max_date:
            continue
        list_los_hdf_paths.append(os.path.join(source_los_hdf_directory, los_hdf))
    save_path = r"/home/vadymskipa/PhD_student/data/South_Africa_-20--35__15-35/"
    save_los_tec_txt_file_for_region_from_los_files(list_los_hdf_paths, source_site_hdf_directory_path, save_path,
                                                    lats=lats, lons=lons, multiprocessing=True)


def main2024feb5():
    source_path = r"/home/vadymskipa/PhD_student/data/South_Africa_-20--35__15-35/los_tec_txt"
    save_path = r"/home/vadymskipa/PhD_student/data/South_Africa_-20--35__15-35/dtec_txt"
    params = LIST_PARAMS1
    min_date = dt.datetime(year=2023, month=2, day=24, tzinfo=dt.timezone.utc)
    max_date = dt.datetime(year=2023, month=3, day=3, tzinfo=dt.timezone.utc)
    get_dtec_from_los_tec_txt_for_lots_of_sites(source_path, save_path, params, min_date=min_date, max_date=max_date)


def main2024feb6():
    source_directories = [r"/home/vadymskipa/PhD_student/data/South_Africa_-20--35__15-35/dtec_txt/055-2023-02-24",
                          r"/home/vadymskipa/PhD_student/data/South_Africa_-20--35__15-35/dtec_txt/056-2023-02-25",
                          r"/home/vadymskipa/PhD_student/data/South_Africa_-20--35__15-35/dtec_txt/057-2023-02-26",
                          r"/home/vadymskipa/PhD_student/data/South_Africa_-20--35__15-35/dtec_txt/058-2023-02-27",
                          r"/home/vadymskipa/PhD_student/data/South_Africa_-20--35__15-35/dtec_txt/059-2023-02-28",
                          r"/home/vadymskipa/PhD_student/data/South_Africa_-20--35__15-35/dtec_txt/060-2023-03-01",
                          r"/home/vadymskipa/PhD_student/data/South_Africa_-20--35__15-35/dtec_txt/061-2023-03-02",
                          r"/home/vadymskipa/PhD_student/data/South_Africa_-20--35__15-35/dtec_txt/062-2023-03-03"]
    site_directory = r"/home/vadymskipa/PhD_student/data/site_hdf/"
    site_paths = wsite.get_site_file_paths_from_directory(site_directory)
    for source in source_directories:
        date = get_date_from_date_directory_name(os.path.basename(source))
        site_file = None
        for site_path in site_paths:
            if wsite.check_site_file_name_by_date(os.path.basename(site_path), date):
                site_file = site_path
                break
        save_sites_txt_for_site_directories(source, site_file, source)


def main2024apr1():
    # source_directories = [r"/home/vadymskipa/PhD_student/data/Europe_los_hdf/los_20200923.europe.001.h5",
    #                       r"/home/vadymskipa/PhD_student/data/Europe_los_hdf/los_20200924.europe.001.h5"]
    # site_directory = r"/home/vadymskipa/PhD_student/data/site_hdf/"
    lats = (30, 80)
    lons = (-10, 50)
    min_date = dt.datetime(year=2020, month=9, day=24, tzinfo=dt.timezone.utc)
    max_date = dt.datetime(year=2020, month=9, day=24, tzinfo=dt.timezone.utc)
    source_site_hdf_directory_path = r"/home/vadymskipa/PhD_student/data/site_hdf/"
    source_los_hdf_directory = r"/home/vadymskipa/PhD_student/data/Europe_los_hdf/"
    los_hdfs_in_directory = os.listdir(source_los_hdf_directory)
    list_los_hdf_paths = []
    for los_hdf in los_hdfs_in_directory:
        temp_date = wlos.get_date_by_los_file_name(los_hdf)
        if temp_date < min_date:
            continue
        if temp_date > max_date:
            continue
        list_los_hdf_paths.append(os.path.join(source_los_hdf_directory, los_hdf))
    save_path = r"/home/vadymskipa/PhD_student/data/Europe_30-80__-10-50/"
    save_los_tec_txt_file_for_region_from_los_files(list_los_hdf_paths, source_site_hdf_directory_path, save_path,
                                                    lats=lats, lons=lons, multiprocessing=True)


def main2024apr2():
    source_path = r"/home/vadymskipa/PhD_student/data/Europe_30-80__-10-50/los_tec_txt"
    save_path = r"/home/vadymskipa/PhD_student/data/Europe_30-80__-10-50/dtec_txt"
    params = LIST_PARAMS1
    min_date = dt.datetime(year=2020, month=9, day=24, tzinfo=dt.timezone.utc)
    max_date = dt.datetime(year=2020, month=9, day=24, tzinfo=dt.timezone.utc)
    get_dtec_from_los_tec_txt_for_lots_of_sites(source_path, save_path, params, min_date=min_date, max_date=max_date)


def main2024apr3():
    source_directories = [r"/home/vadymskipa/PhD_student/data/Europe_30-80__-10-50/dtec_txt/268-2020-09-24"]
    site_directory = r"/home/vadymskipa/PhD_student/data/site_hdf/"
    site_paths = wsite.get_site_file_paths_from_directory(site_directory)
    for source in source_directories:
        date = get_date_from_date_directory_name(os.path.basename(source))
        site_file = None
        for site_path in site_paths:
            if wsite.check_site_file_name_by_date(os.path.basename(site_path), date):
                site_file = site_path
                break
        save_sites_txt_for_site_directories(source, site_file, source)


def main2024apr4():
    source_directory = r"/home/vadymskipa/PhD_student/data/Europe_30-80__-10-50/dtec_txt"
    save_directory = r"/home/vadymskipa/PhD_student/data/Europe_30-80__-10-50/dtec_smooth_txt"
    inner_directories = os.listdir(source_directory)
    min_date = dt.datetime(year=2020, month=9, day=24, hour=10, tzinfo=dt.timezone.utc)
    max_date = dt.datetime(year=2020, month=9, day=24, hour=11, minute=30, tzinfo=dt.timezone.utc)
    min_date_ts = min_date.timestamp()
    max_date_ts = max_date.timestamp()
    for date_directory in inner_directories:
        date = get_date_from_date_directory_name(date_directory)
        date_directory_path = os.path.join(source_directory, date_directory)
        save_path_1 = get_directory_path(save_directory, date_directory)
        inner_directories_2 = os.listdir(date_directory_path)
        for gps_site in inner_directories_2:
            gps_site_path = os.path.join(date_directory_path, gps_site)
            save_path_2 = get_directory_path(save_path_1, gps_site)
            inner_directories_3 = os.listdir(gps_site_path)
            for window in inner_directories_3:
                window_path = os.path.join(gps_site_path, window)
                save_path_3 = get_directory_path(save_path_2, window)
                inner_files_4 = os.listdir(window_path)
                for sat_id_file in inner_files_4:
                    sat_id_path = os.path.join(window_path, sat_id_file)
                    dtec_dataframe = read_dtec_file(sat_id_path)
                    dtec_dataframe = add_timestamp_column_to_df(dtec_dataframe, date)
                    dtec_dataframe = add_datetime_column_to_df(dtec_dataframe)
                    dtec_dataframe = dtec_dataframe.loc[dtec_dataframe.loc[:, "timestamp"] >= min_date_ts]
                    dtec_dataframe = dtec_dataframe.loc[dtec_dataframe.loc[:, "timestamp"] <= max_date_ts]
                    dtec_dataframe = dtec_dataframe.loc[dtec_dataframe.loc[:, "elm"] > 35.0]
                    dtec_dataframe.rename(columns={"dtec": "los_tec"}, inplace=True)
                    try:
                        new_dataframe = add_savgol_data(dtec_dataframe, {"window_length": 600})
                        new_dataframe.drop(["diff"], axis=1, inplace=True)
                        new_dataframe.rename(columns={"savgol": "diff"}, inplace=True)
                        save_path_4 = os.path.join(save_path_3, sat_id_file)
                        save_diff_dataframe_txt(save_path_4, new_dataframe)
                        print(f"save {gps_site} {window} {sat_id_file}")
                    except Exception as er:
                        print("\t", er)
                        print("\t", gps_site, window, sat_id_file, "\n")


def get_angle(elevation_angle, altitude):
    rad_a = math.radians(elevation_angle)
    cos_a = math.cos(rad_a)
    sin_a = math.sin(rad_a)
    earth = 6371
    # full version
    sin_b = (-earth * sin_a * cos_a + cos_a *
             math.sqrt((earth ** 2) * (sin_a ** 2) + 2 * earth * altitude + altitude ** 2)) / (earth + altitude)
    # b = (-earth * sin_a * cos_a + cos_a *
    #          math.sqrt((earth ** 2) * (sin_a ** 2) + 2 * earth * altitude)) / (earth + altitude)
    b = math.asin(sin_b)
    return b

def get_endpoint(lat1, lon1, bearing, distance):
    # Earth's radius in kilometers
    R = 6371

    # Convert bearing from degrees to radians
    brng = math.radians(bearing)

    # Convert lat1 and lon1 to radians
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)

    # Calculate new latitude and longitude
    lat2 = math.asin(math.sin(lat1_rad) * math.cos(distance / R) + math.cos(lat1_rad) * math.sin(distance / R) * math.cos(brng))
    lon2 = lon1_rad + math.atan2(math.sin(brng) * math.sin(distance / R) * math.cos(lat1_rad), math.cos(distance / R) - math.sin(lat1_rad) * math.sin(lat2))

    # Convert lat2 and lon2 back to degrees
    lat2_deg = math.degrees(lat2)
    lon2_deg = math.degrees(lon2)

    return lat2_deg, lon2_deg


def calc_vtec(dtec, elm, rad_b):
    rad_elm = math.radians(elm)
    vtec = dtec * math.sin(rad_b + rad_elm)
    return vtec


def main2024apr5():
    source_directory = r"/home/vadymskipa/PhD_student/data/Europe_30-80__-10-50/dtec_smooth_txt"
    save_directory = r"/home/vadymskipa/PhD_student/data/Europe_30-80__-10-50/temp"
    list_of_altitudes = [200, 250, 300, 350, 400, 450, 500, 600]
    lats = [46.5, 49.5]
    lons = [12.7, 17.3]
    min_date = dt.datetime(year=2020, month=9, day=24, hour=10, tzinfo=dt.timezone.utc)
    max_date = dt.datetime(year=2020, month=9, day=24, hour=11, minute=30, tzinfo=dt.timezone.utc)
    min_date_ts = min_date.timestamp()
    max_date_ts = max_date.timestamp()
    inner_directories = os.listdir(source_directory)
    for altitude in list_of_altitudes:
        altitude_name = f"dvtec_txt_alt_{altitude}"
        save_path_0 = get_directory_path(save_directory, altitude_name)
        for date_directory in inner_directories:
            date = get_date_from_date_directory_name(date_directory)
            date_directory_path = os.path.join(source_directory, date_directory)
            save_path_1 = get_directory_path(save_path_0, date_directory)
            sites_path = os.path.join(date_directory_path, "Sites.txt")
            sites_dataframe = read_sites_txt(sites_path)
            inner_directories_2 = os.listdir(date_directory_path)
            for gps_site in inner_directories_2:
                gps_site_path = os.path.join(date_directory_path, gps_site)
                if not os.path.isdir(gps_site_path):
                    continue
                inner_directories_3 = os.listdir(gps_site_path)
                lat_site = float(sites_dataframe.loc[sites_dataframe.loc[:, "gps_site"] == gps_site, "gdlatr"].iloc[0])
                lon_site = float(sites_dataframe.loc[sites_dataframe.loc[:, "gps_site"] == gps_site, "gdlonr"].iloc[0])
                for window in inner_directories_3:
                    window_path = os.path.join(gps_site_path, window)
                    inner_files_4 = os.listdir(window_path)
                    for sat_id_file in inner_files_4:
                        sat_id_path = os.path.join(window_path, sat_id_file)
                        dtec_dataframe = read_dtec_file(sat_id_path)
                        dtec_dataframe = add_timestamp_column_to_df(dtec_dataframe, date)
                        flag = False
                        for i in range(0, len(dtec_dataframe.index), 60):
                            elm = dtec_dataframe.loc[i, "elm"]
                            distance = get_angle(elm, altitude) * 6371
                            # lat_site = sites_dataframe.loc[sites_dataframe.loc[:, "gps_site"] == gps_site].loc["gdlatr"]
                            # lon_site = sites_dataframe.loc[sites_dataframe.loc[:, "gps_site"] == gps_site].loc["gdlonr"]
                            azm = dtec_dataframe.loc[i, "azm"]
                            end_point = get_endpoint(lat_site, lon_site, azm, distance)
                            if (lats[0] <= end_point[0] <= lats[1]) and (lons[0] <= end_point[1] <= lons[1]):
                                flag = True
                                break
                        temp_dt1: pd.DataFrame = dtec_dataframe.loc[dtec_dataframe.loc[:, "timestamp"] == (min_date_ts + 1800)]
                        temp_dt2: pd.DataFrame = dtec_dataframe.loc[dtec_dataframe.loc[:, "timestamp"] == (min_date_ts + 3600)]
                        if temp_dt1.empty or temp_dt2.empty:
                            continue
                        if flag:
                            save_path_2 = get_directory_path(save_path_1, gps_site)
                            save_path_3 = get_directory_path(save_path_2, window)
                            result_str = (
                                f"{'hour':<4}\t{'min':<4}\t{'sec':<6}\t{'dvTEC':<6}\t{'azm':<6}\t{'elm':<6}\t{'gdlat':<6}\t"
                                f"{'gdlon':<6}\n")
                            for timestamp in range(int(min_date_ts), int(max_date_ts), 300):
                                try:
                                    temp_dt: pd.DataFrame = dtec_dataframe.loc[dtec_dataframe.loc[:, "timestamp"] >= timestamp]
                                    temp_dt = temp_dt.loc[temp_dt.loc[:, "timestamp"] < (timestamp + 300)]
                                    elm = temp_dt.loc[:, "elm"].mean()
                                    rad_b = get_angle(elm, altitude)
                                    distance = rad_b * 6371
                                    azm = temp_dt.loc[:, "azm"].mean()
                                    dtec = temp_dt.loc[:, "dtec"].mean()
                                    gdlat, gdlon = get_endpoint(lat_site, lon_site, azm, distance)
                                    hour = temp_dt.iloc[0].loc["hour"]
                                    minute = temp_dt.iloc[0].loc["min"]
                                    sec = 0
                                    vtec = calc_vtec(dtec, elm, rad_b)
                                    if (lats[0] <= gdlat <= lats[1]) and (lons[0] <= gdlon <= lons[1]):
                                        result_str += (f"{int(hour):<4}\t{int(minute):<4}\t{sec:<4.0f}\t{vtec:<6.3f}\t{azm:<6.2f}\t"
                                                       f"{elm:<6.2f}\t{gdlat:<6.2f}\t{gdlon:<6.2f}\n")
                                except Exception as er:
                                    print("\t", er)
                                    print("\t", gps_site, window, sat_id_file, "\n")
                            save_path_4 = os.path.join(save_path_3, sat_id_file)
                            with open(save_path_4, "w") as file:
                                file.write(result_str)
                            print(f"save {gps_site} {window} {sat_id_file}")


def haversine(lat1, lon1, lat2, lon2, alt):
    R = 6371 + alt  # Earth's radius in kilometers

    # Convert latitude and longitude from degrees to radians
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    # Calculate differences in latitudes and longitudes
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    # Haversine formula
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c

    return distance


def main2024apr6():
    source_directory = r"/home/vadymskipa/PhD_student/data/Europe_30-80__-10-50/temp"
    save_directory = r"/home/vadymskipa/PhD_student/data/Europe_30-80__-10-50/temp"
    list_of_altitudes = [200, 250, 300, 350, 400, 450, 500, 600]
    lats = [43.5, 48.5]
    lons = [11.5, 18.5]
    min_date = dt.datetime(year=2020, month=9, day=24, hour=10, tzinfo=dt.timezone.utc)
    max_date = dt.datetime(year=2020, month=9, day=24, hour=11, minute=30, tzinfo=dt.timezone.utc)
    min_date_ts = min_date.timestamp()
    max_date_ts = max_date.timestamp()
    for altitude in list_of_altitudes:
        data_dict = {}
        for timestamp in range(int(min_date_ts), int(max_date_ts), 300):
            data_dict[timestamp] = []
        altitude_name = f"dvtec_txt_alt_{altitude}"
        altitude_path = get_directory_path(source_directory, altitude_name)
        inner_directories = os.listdir(altitude_path)
        for date_directory in inner_directories:
            date = get_date_from_date_directory_name(date_directory)
            date_directory_path = os.path.join(altitude_path, date_directory)
            inner_directories_2 = os.listdir(date_directory_path)
            for gps_site in inner_directories_2:
                gps_site_path = os.path.join(date_directory_path, gps_site)
                window_path = os.path.join(gps_site_path, "Window_7200_Seconds")
                if not os.path.exists(window_path):
                    continue
                inner_files_4 = os.listdir(window_path)
                for sat_id_file in inner_files_4:
                    sat_id_path = os.path.join(window_path, sat_id_file)
                    dtec_dataframe = read_dtec_file(sat_id_path)
                    dtec_dataframe = add_timestamp_column_to_df(dtec_dataframe, date)
                    for timestamp in range(int(min_date_ts), int(max_date_ts), 300):
                        try:
                            mask = dtec_dataframe.loc[:, "timestamp"] == timestamp
                            temp_dt = dtec_dataframe.loc[mask]
                            elm = temp_dt.iloc[0].loc["elm"]
                            azm = temp_dt.iloc[0].loc["azm"]
                            dtec = temp_dt.iloc[0].loc["dtec"]
                            gdlat = temp_dt.iloc[0].loc["gdlat"]
                            gdlon = temp_dt.iloc[0].loc["gdlon"]
                            hour = temp_dt.iloc[0].loc["hour"]
                            minute = temp_dt.iloc[0].loc["min"]
                            sec = temp_dt.iloc[0].loc["sec"]
                            temp_dict = {"elm": elm, "azm": azm, "dtec": dtec, "gdlat": gdlat, "gdlon": gdlon,
                                         "hour": hour, "min": minute, "sec": sec}
                            data_dict[timestamp].append(temp_dict)
                        except Exception as er:
                            print("\t", er)
                            print("\t", gps_site, sat_id_file, "\n")
        filename = f"Parameter_alt_{altitude}.txt"
        temp_save_path = os.path.join(save_directory, filename)
        text = f"{'hour':<4}\t{'min':<4}\t{'sec':<6}\t{'param':<9}\t{'nod':<9}\n"
        for timestamp, datalist in data_dict.items():
            hour = datalist[0]["hour"]
            minute = datalist[0]["min"]
            sec = datalist[0]["sec"]
            diff_list = []
            while len(datalist) > 1:
                point_dict = datalist.pop(0)
                lat1 = point_dict["gdlat"]
                lon1 = point_dict["gdlon"]
                dtec1 = point_dict["dtec"]
                for second_point_dict in datalist:
                    lat2 = second_point_dict["gdlat"]
                    lon2 = second_point_dict["gdlon"]
                    distance = haversine(lat1, lon1, lat2, lon2, altitude)
                    if distance < 150:
                        dtec2 = second_point_dict["dtec"]
                        diff = (dtec2 - dtec1) ** 2
                        diff_list.append(diff)
            number_od_diff = len(diff_list)
            parameter = sum(diff_list) / number_od_diff
            text += f"{hour:<4}\t{minute:<4}\t{sec:<4.0f}\t{parameter:<9.5f}\t{number_od_diff:<9}\n"
        with open(temp_save_path, "w") as file:
            file.write(text)


def main2024apr7():
    source_directory = r"/home/vadymskipa/PhD_student/data/Europe_30-80__-10-50/temp"
    save_directory = r"/home/vadymskipa/PhD_student/data/Europe_30-80__-10-50"
    list_of_altitudes = [200, 250, 300, 350, 400, 450, 500, 600]
    lats = [43.5, 48.5]
    lons = [11.5, 18.5]
    min_date = dt.datetime(year=2020, month=9, day=24, hour=10, tzinfo=dt.timezone.utc)
    max_date = dt.datetime(year=2020, month=9, day=24, hour=11, minute=30, tzinfo=dt.timezone.utc)
    min_date_ts = min_date.timestamp()
    max_date_ts = max_date.timestamp()
    for altitude in list_of_altitudes:
        data_dict = {}
        for timestamp in range(int(min_date_ts), int(max_date_ts), 300):
            data_dict[timestamp] = []
        altitude_name = f"dvtec_txt_alt_{altitude}"
        altitude_path = get_directory_path(source_directory, altitude_name)
        save_path_1 = get_directory_path(source_directory, altitude_name + "_norm")
        inner_directories = os.listdir(altitude_path)
        for date_directory in inner_directories:
            date = get_date_from_date_directory_name(date_directory)
            date_directory_path = os.path.join(altitude_path, date_directory)
            save_path_2 = get_directory_path(save_path_1, date_directory)
            inner_directories_2 = os.listdir(date_directory_path)
            for gps_site in inner_directories_2:
                gps_site_path = os.path.join(date_directory_path, gps_site)
                save_path_3 = get_directory_path(save_path_2, gps_site)
                inner_directories_3 = os.listdir(gps_site_path)
                for window in inner_directories_3:
                    window_path = os.path.join(gps_site_path, window)
                    save_path_4 = get_directory_path(save_path_3, window)
                    inner_files_4 = os.listdir(window_path)
                    for sat_id_file in inner_files_4:
                        sat_id_path = os.path.join(window_path, sat_id_file)
                        dtec_dataframe = read_dtec_file(sat_id_path)
                        dtec_dataframe = add_timestamp_column_to_df(dtec_dataframe, date)
                        dtec_dataframe = add_datetime_column_to_df(dtec_dataframe)
                        loc_dataframe = dtec_dataframe.loc[dtec_dataframe.loc[:, "timestamp"] >= min_date_ts + 1800]
                        loc_dataframe = loc_dataframe.loc[loc_dataframe.loc[:, "timestamp"] < min_date_ts + 3600]
                        dtec = dtec_dataframe.loc[:, "dtec"]
                        loc_dtec = loc_dataframe.loc[:, "dtec"]
                        dtec_max = loc_dtec.max()
                        new_dtec = dtec / dtec_max
                        # dtec_dataframe.drop(["dtec"], axis=1, inplace=True)
                        dtec_dataframe.insert(0, "diff", new_dtec)
                        text = get_str_for_dtec_txt(dtec_dataframe)
                        save_path_5 = os.path.join(save_path_4, sat_id_file)
                        with open(save_path_5, "w") as file:
                            file.write(text)


def main2024apr8():
    source_directory = r"/home/vadymskipa/PhD_student/data/Europe_30-80__-10-50/temp"
    save_directory = r"/home/vadymskipa/PhD_student/data/Europe_30-80__-10-50/temp"
    list_of_altitudes = [200, 250, 300, 350, 400, 450, 500, 600]
    lats = [43.5, 48.5]
    lons = [11.5, 18.5]
    min_date = dt.datetime(year=2020, month=9, day=24, hour=10, tzinfo=dt.timezone.utc)
    max_date = dt.datetime(year=2020, month=9, day=24, hour=11, minute=30, tzinfo=dt.timezone.utc)
    min_date_ts = min_date.timestamp()
    max_date_ts = max_date.timestamp()
    for altitude in list_of_altitudes:
        data_dict = {}
        for timestamp in range(int(min_date_ts), int(max_date_ts), 300):
            data_dict[timestamp] = []
        altitude_name = f"dvtec_txt_alt_{altitude}_norm"
        altitude_path = get_directory_path(source_directory, altitude_name)
        inner_directories = os.listdir(altitude_path)
        for date_directory in inner_directories:
            date = get_date_from_date_directory_name(date_directory)
            date_directory_path = os.path.join(altitude_path, date_directory)
            inner_directories_2 = os.listdir(date_directory_path)
            for gps_site in inner_directories_2:
                gps_site_path = os.path.join(date_directory_path, gps_site)
                window_path = os.path.join(gps_site_path, "Window_7200_Seconds")
                if not os.path.exists(window_path):
                    continue
                inner_files_4 = os.listdir(window_path)
                for sat_id_file in inner_files_4:
                    sat_id_path = os.path.join(window_path, sat_id_file)
                    dtec_dataframe = read_dtec_file(sat_id_path)
                    dtec_dataframe = add_timestamp_column_to_df(dtec_dataframe, date)
                    for timestamp in range(int(min_date_ts), int(max_date_ts), 300):
                        try:
                            mask = dtec_dataframe.loc[:, "timestamp"] == timestamp
                            temp_dt = dtec_dataframe.loc[mask]
                            elm = temp_dt.iloc[0].loc["elm"]
                            azm = temp_dt.iloc[0].loc["azm"]
                            dtec = temp_dt.iloc[0].loc["dtec"]
                            gdlat = temp_dt.iloc[0].loc["gdlat"]
                            gdlon = temp_dt.iloc[0].loc["gdlon"]
                            hour = temp_dt.iloc[0].loc["hour"]
                            minute = temp_dt.iloc[0].loc["min"]
                            sec = temp_dt.iloc[0].loc["sec"]
                            temp_dict = {"elm": elm, "azm": azm, "dtec": dtec, "gdlat": gdlat, "gdlon": gdlon,
                                         "hour": hour, "min": minute, "sec": sec}
                            data_dict[timestamp].append(temp_dict)
                        except Exception as er:
                            print("\t", er)
                            print("\t", gps_site, sat_id_file, "\n")
        filename = f"Parameter_norm_alt_{altitude}.txt"
        temp_save_path = os.path.join(save_directory, filename)
        text = f"{'hour':<4}\t{'min':<4}\t{'sec':<6}\t{'param':<9}\t{'nod':<9}\n"
        for timestamp, datalist in data_dict.items():
            hour = int(datalist[0]["hour"])
            minute = int(datalist[0]["min"])
            sec = datalist[0]["sec"]
            diff_list = []
            while len(datalist) > 1:
                point_dict = datalist.pop(0)
                lat1 = point_dict["gdlat"]
                lon1 = point_dict["gdlon"]
                dtec1 = point_dict["dtec"]
                for second_point_dict in datalist:
                    lat2 = second_point_dict["gdlat"]
                    lon2 = second_point_dict["gdlon"]
                    distance = haversine(lat1, lon1, lat2, lon2, altitude)
                    if distance <= 70:
                        dtec2 = second_point_dict["dtec"]
                        diff = (dtec2 - dtec1) ** 2
                        diff_list.append(diff)
            number_od_diff = len(diff_list)
            parameter = sum(diff_list) / number_od_diff
            text += f"{hour:<4}\t{minute:<4}\t{sec:<4.0f}\t{parameter:<9.5f}\t{number_od_diff:<9}\n"
        with open(temp_save_path, "w") as file:
            file.write(text)


def main2024aprread(file):
    arr_hour = []
    arr_min = []
    arr_sec = []
    arr_par = []
    arr_nod = []
    arr_of_arr = [arr_hour, arr_min, arr_sec, arr_par, arr_nod]
    with open(file, "r") as reader:
        reader.readline()
        for line in reader:
            text_tuple = line.split()
            arr_of_arr[0].append(int(text_tuple[0]))
            arr_of_arr[1].append(int(text_tuple[1]))
            arr_of_arr[2].append(int(text_tuple[2]))
            arr_of_arr[3].append(float(text_tuple[3]))
            arr_of_arr[4].append(float(text_tuple[4]))
    outcome_dataframe = pd.DataFrame({"hour": arr_hour, "min": arr_min, "sec": arr_sec, "par": arr_par, "nod": arr_nod})
    return outcome_dataframe


def main2024apr9():
    source_directory = r"/home/vadymskipa/PhD_student/data/Europe_30-80__-10-50/temp"
    list_of_altitudes = [200, 250, 300, 350, 400, 450, 500, 600]
    date = dt.datetime(year=2020, month=9, day=24, tzinfo=dt.timezone.utc)
    min_date = dt.datetime(year=2020, month=9, day=24, hour=10, tzinfo=dt.timezone.utc)
    max_date = dt.datetime(year=2020, month=9, day=24, hour=11, minute=30, tzinfo=dt.timezone.utc)
    min_date_ts = min_date.timestamp()
    max_date_ts = max_date.timestamp()
    main_dict = {}
    for timestamp in range(int(min_date_ts), int(max_date_ts), 300):
        main_dict[timestamp] = {}
    for altitude in list_of_altitudes:
        filename = f"Parameter_norm_alt_{altitude}.txt"
        filepath = os.path.join(source_directory, filename)
        dataframe = main2024aprread(filepath)
        dataframe = add_timestamp_column_to_df(dataframe, date)
        for timestamp in main_dict.keys():
            temp_dt = dataframe.loc[dataframe.loc[:, "timestamp"] == timestamp]
            par = temp_dt.iloc[0].loc["par"]
            main_dict[timestamp][altitude] = par
    text = f"{'hour':<4}\t{'min':<4}\t{'sec':<6}\t{'200':<9}\t{'250':<9}\t{'300':<9}\t{'350':<9}\t{'400':<9}\t{'450':<9}\t{'500':<9}\t{'600':<9}\n"
    for timestamp, datadict in main_dict.items():
        temp_date = dt.datetime.fromtimestamp(timestamp)
        hour = temp_date.hour
        minute = temp_date.minute
        sec = temp_date.second
        p200 = datadict[200]
        p250 = datadict[250]
        p300 = datadict[300]
        p350 = datadict[350]
        p400 = datadict[400]
        p450 = datadict[450]
        p500 = datadict[500]
        p600 = datadict[600]
        text += f"{hour:<4}\t{minute:<4}\t{sec:<4.0f}\t{p200:<9.5f}\t{p250:<9.5f}\t{p300:<9.5f}\t{p350:<9.5f}\t{p400:<9.5f}\t{p450:<9.5f}\t{p500:<9.5f}\t{p600:<9.5f}\n"
    svpath = os.path.join(source_directory, "table_norm_par.txt")
    with open(svpath, "w") as file:
        file.write(text)


def main2024apr10():
    # create dtec_txt directory in particular time period
    source_directory = r"/home/vadymskipa/PhD_student/data/Europe_30-80__-10-50/dtec_txt"
    save_directory = r"/home/vadymskipa/PhD_student/data/Europe_30-80__-10-50/dtec_txt_period"
    inner_directories = os.listdir(source_directory)
    min_date = dt.datetime(year=2020, month=9, day=24, hour=9, minute=30, tzinfo=dt.timezone.utc)
    max_date = dt.datetime(year=2020, month=9, day=24, hour=12, minute=0, tzinfo=dt.timezone.utc)
    min_date_ts = min_date.timestamp()
    max_date_ts = max_date.timestamp()
    for date_directory in inner_directories:
        date = get_date_from_date_directory_name(date_directory)
        date_directory_path = os.path.join(source_directory, date_directory)
        save_path_1 = get_directory_path(save_directory, date_directory)
        inner_directories_2 = os.listdir(date_directory_path)
        for gps_site in inner_directories_2:
            gps_site_path = os.path.join(date_directory_path, gps_site)
            save_path_2 = get_directory_path(save_path_1, gps_site)
            inner_directories_3 = os.listdir(gps_site_path)
            for window in inner_directories_3:
                window_path = os.path.join(gps_site_path, window)
                save_path_3 = get_directory_path(save_path_2, window)
                inner_files_4 = os.listdir(window_path)
                for sat_id_file in inner_files_4:
                    sat_id_path = os.path.join(window_path, sat_id_file)
                    dtec_dataframe: pd.DataFrame = read_dtec_file(sat_id_path)
                    dtec_dataframe = add_timestamp_column_to_df(dtec_dataframe, date)
                    dtec_dataframe = add_datetime_column_to_df(dtec_dataframe)
                    dtec_dataframe = dtec_dataframe.loc[dtec_dataframe.loc[:, "timestamp"] >= min_date_ts]
                    dtec_dataframe = dtec_dataframe.loc[dtec_dataframe.loc[:, "timestamp"] <= max_date_ts]
                    dtec_dataframe = dtec_dataframe.loc[dtec_dataframe.loc[:, "elm"] > 30.0]
                    if dtec_dataframe.empty:
                        continue
                    try:
                        dtec_dataframe.rename(columns={"dtec": "diff"}, inplace=True)
                        save_path_4 = os.path.join(save_path_3, sat_id_file)
                        save_diff_dataframe_txt(save_path_4, dtec_dataframe)
                        print(f"save {gps_site} {window} {sat_id_file}")
                    except Exception as er:
                        print("\t", er)
                        print("\t", gps_site, window, sat_id_file, "\n")


def main2024apr11():
    start = dt.datetime.now()
    # crate dvtec_txt for different altitudes with data every 5 minutes
    source_directory = r"/home/vadymskipa/PhD_student/data/Europe_30-80__-10-50/dtec_txt_period"
    save_directory = r"/home/vadymskipa/PhD_student/data/Europe_30-80__-10-50/dvtec_altitude1"
    list_of_altitudes = [200, 250, 300, 350, 400, 450]
    # list_of_altitudes = [150, 250]
    # lats = [46.5, 49.5]
    # lons = [12.7, 17.3]
    min_date = dt.datetime(year=2020, month=9, day=24, hour=9, minute=30, tzinfo=dt.timezone.utc)
    max_date = dt.datetime(year=2020, month=9, day=24, hour=12, minute=0, tzinfo=dt.timezone.utc)
    min_date_ts = min_date.timestamp()
    max_date_ts = max_date.timestamp()
    period_of_averaging = 300
    inner_directories = os.listdir(source_directory)
    for altitude in list_of_altitudes:
        altitude_name = f"dvtec_txt_alt_{altitude}"
        save_path_0 = get_directory_path(save_directory, altitude_name)
        for date_directory in inner_directories:
            date = get_date_from_date_directory_name(date_directory)
            date_directory_path = os.path.join(source_directory, date_directory)
            save_path_1 = get_directory_path(save_path_0, date_directory)
            sites_path = os.path.join(date_directory_path, "Sites.txt")
            sites_dataframe = read_sites_txt(sites_path)
            inner_directories_2 = os.listdir(date_directory_path)
            for gps_site in inner_directories_2:
                gps_site_path = os.path.join(date_directory_path, gps_site)
                if not os.path.isdir(gps_site_path):
                    continue
                inner_directories_3 = os.listdir(gps_site_path)
                lat_site = float(sites_dataframe.loc[sites_dataframe.loc[:, "gps_site"] == gps_site, "gdlatr"].iloc[0])
                lon_site = float(sites_dataframe.loc[sites_dataframe.loc[:, "gps_site"] == gps_site, "gdlonr"].iloc[0])
                for window in inner_directories_3:
                    window_path = os.path.join(gps_site_path, window)
                    inner_files_4 = os.listdir(window_path)
                    for sat_id_file in inner_files_4:
                        sat_id_path = os.path.join(window_path, sat_id_file)
                        dtec_dataframe = read_dtec_file(sat_id_path)
                        dtec_dataframe = add_timestamp_column_to_df(dtec_dataframe, date)
                        if dtec_dataframe.empty:
                            continue
                        save_path_2 = get_directory_path(save_path_1, gps_site)
                        save_path_3 = get_directory_path(save_path_2, window)
                        result_str = (
                            f"{'hour':<4}\t{'min':<4}\t{'sec':<6}\t{'dvTEC':<6}\t{'azm':<6}\t{'elm':<6}\t{'gdlat':<6}\t"
                            f"{'gdlon':<6}\n")
                        for timestamp in range(int(min_date_ts), int(max_date_ts), period_of_averaging):
                            try:
                                temp_dt: pd.DataFrame = dtec_dataframe.loc[dtec_dataframe.loc[:, "timestamp"] >= timestamp]
                                temp_dt = temp_dt.loc[temp_dt.loc[:, "timestamp"] < (timestamp + period_of_averaging)]
                                elm = temp_dt.loc[:, "elm"].mean()
                                rad_b = get_angle(elm, altitude)
                                distance = rad_b * 6371
                                if temp_dt.empty:
                                    continue
                                azm = temp_dt.loc[:, "azm"].mean()
                                dtec = temp_dt.loc[:, "dtec"].mean()
                                gdlat, gdlon = get_endpoint(lat_site, lon_site, azm, distance)
                                hour = temp_dt.iloc[0].loc["hour"]
                                minute = temp_dt.iloc[0].loc["min"]
                                sec = 0
                                vtec = calc_vtec(dtec, elm, rad_b)
                                result_str += (f"{int(hour):<4}\t{int(minute):<4}\t{sec:<4.0f}\t{vtec:<6.3f}\t{azm:<6.2f}\t"
                                               f"{elm:<6.2f}\t{gdlat:<6.2f}\t{gdlon:<6.2f}\n")
                            except Exception as er:
                                print("\t", er)
                                print("\t", gps_site, window, sat_id_file, "\n")
                        save_path_4 = os.path.join(save_path_3, sat_id_file)
                        with open(save_path_4, "w") as file:
                            file.write(result_str)
                print(f"save {altitude} {gps_site} {dt.datetime.now() - start}")


def main2024apr12():
    anglekm = math.degrees(1 / 20000 * math.pi)
    source_directory = r"/home/vadymskipa/PhD_student/data/Europe_30-80__-10-50/dvtec_altitude"
    save_directory = r"/home/vadymskipa/PhD_student/data/Europe_30-80__-10-50/dvtec_altitude/parameter_lists"
    # list_of_altitudes = [150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700]
    list_of_altitudes = [200, 250, 300, 350, 400, 450, 550, 600, 650, 700]
    # list_of_altitudes = [150]
    lats = [43.5, 48.5]
    lons = [11.5, 18.5]
    min_date = dt.datetime(year=2020, month=9, day=24, hour=9, minute=30, tzinfo=dt.timezone.utc)
    max_date = dt.datetime(year=2020, month=9, day=24, hour=12, minute=0, tzinfo=dt.timezone.utc)
    # min_date = dt.datetime(year=2020, month=9, day=24, hour=11, minute=38, tzinfo=dt.timezone.utc)
    # max_date = dt.datetime(year=2020, month=9, day=24, hour=12, minute=0, tzinfo=dt.timezone.utc)
    min_date_ts = min_date.timestamp()
    max_date_ts = max_date.timestamp()
    period_of_averaging = 120
    for altitude in list_of_altitudes:
        data_dict = {}
        for timestamp in range(int(min_date_ts), int(max_date_ts), period_of_averaging):
            data_dict[timestamp] = []
        altitude_name = f"dvtec_txt_alt_{altitude}"
        altitude_path = get_directory_path(source_directory, altitude_name)
        save_path_1 = get_directory_path(save_directory, f"alt_{altitude}")
        inner_directories = os.listdir(altitude_path)
        for date_directory in inner_directories:
            date = get_date_from_date_directory_name(date_directory)
            date_directory_path = os.path.join(altitude_path, date_directory)
            inner_directories_2 = os.listdir(date_directory_path)
            for gps_site in inner_directories_2:
                gps_site_path = os.path.join(date_directory_path, gps_site)
                window_path = os.path.join(gps_site_path, "Window_7200_Seconds")
                if not os.path.exists(window_path):
                    continue
                inner_files_4 = os.listdir(window_path)
                # data_dict[gps_site] = {}
                for sat_id_file in inner_files_4:
                    sat_id_path = os.path.join(window_path, sat_id_file)
                    sat_id = sat_id_file[0:3]
                    # data_dict[gps_site][sat_id] = {}
                    dtec_dataframe = read_dtec_file(sat_id_path)
                    dtec_dataframe = add_timestamp_column_to_df(dtec_dataframe, date)
                    # for index in dtec_dataframe.index:
                    #     dtec = dtec_dataframe.loc[index].loc["dtec"]
                    #     gdlat = dtec_dataframe.loc[index].loc["gdlat"]
                    #     gdlon = dtec_dataframe.loc[index].loc["gdlon"]
                    #     timestamp = dtec_dataframe.loc[index].loc["timestamp"]
                    #     temp_dict = {"dtec": dtec, "gdlat": gdlat, "gdlon": gdlon, "sat_id": sat_id,
                    #                  "gps_site": gps_site}
                    #     data_dict[timestamp].append(temp_dict)
                    for timestamp in range(int(min_date_ts), int(max_date_ts), period_of_averaging):
                        try:
                            mask = dtec_dataframe.loc[:, "timestamp"] == timestamp
                            temp_dt = dtec_dataframe.loc[mask]
                            if temp_dt.empty:
                                continue
                            dtec = temp_dt.iloc[0].loc["dtec"]
                            gdlat = temp_dt.iloc[0].loc["gdlat"]
                            gdlon = temp_dt.iloc[0].loc["gdlon"]
                            temp_dict = {"dtec": dtec, "gdlat": gdlat, "gdlon": gdlon, "sat_id": sat_id, "gps_site": gps_site}
                            data_dict[timestamp].append(temp_dict)
                        except Exception as er:
                            print("\t", er)
                            print("\t", gps_site, sat_id_file, "\n")
        dist_dict = {}
        dist_dict2 = {}
        # distances = [50, 70, 100, 150]
        distances = [150]
        print("read")
        for distance in distances:
            dlat = (distance + 20) * anglekm
            copy_data_dict = copy.deepcopy(data_dict)
            dist_dict[distance] = copy_data_dict
            copy_data_dict2 = copy.deepcopy(data_dict)
            dist_dict2[distance] = copy_data_dict2
            for timestamp, datalist in dist_dict2[distance].items():
                for elem in datalist:
                    elem["diff_list"] = []

            for timestamp, datalist in data_dict.items():
                date = dt.datetime.fromtimestamp(timestamp, tz=dt.timezone.utc)
                for index in range(len(datalist)):
                    diff_list = []
                    lat1 = datalist[index]["gdlat"]
                    lon1 = datalist[index]["gdlon"]
                    dtec1 = datalist[index]["dtec"]
                    if (index % 1000) == 0:
                        print(index, len(datalist))
                    for second_pd in datalist:
                        if datalist[index] == second_pd:
                            continue
                        lat2 = second_pd["gdlat"]
                        if not ((lat1-dlat) <= lat2 <= (lat1 + dlat)):
                            continue
                        lon2 = second_pd["gdlon"]
                        if lat1 < 60:
                            if not ((lon1 - 2 * dlat) <= lon2 <= (lon1 + 2 * dlat)):
                                continue
                        d = haversine(lat1, lon1, lat2, lon2, altitude)
                        if d <= distance:
                            dtec2 = second_pd["dtec"]
                            diff = (dtec2 - dtec1) ** 2
                            diff_list.append(diff)
                            # dist_dict2[distance][timestamp][index]["diff_list"].append({"diff": diff,
                            #                                                             "sat_id": second_pd["sat_id"],
                            #                                                             "gps_site": second_pd["gps_site"]})
                            dist_dict2[distance][timestamp][index]["diff_list"].append((diff, second_pd["sat_id"],
                                                                                        second_pd["gps_site"]))
                    dist_dict[distance][timestamp][index]["ncp"] = len(diff_list)
                    if len(diff_list) == 0:
                        dist_dict[distance][timestamp][index]["param"] = 9999
                    else:
                        dist_dict[distance][timestamp][index]["param"] = sum(diff_list)/len(diff_list)
        print("start writting")
        for distance, datadict in dist_dict.items():
            directory_name = f"distance_{distance}"
            save_path_2 = get_directory_path(save_path_1, directory_name)
            for timestamp, datalist in datadict.items():
                time = dt.datetime.fromtimestamp(timestamp, tz=dt.timezone.utc)
                time_file = f"{time.hour:0=2}{time.minute:0=2}.txt"
                text = f"{'gps_site':<8}\t{'sat_id':<8}\t{'param':<9}\t{'ncp':<6}\t{'gdlat':<6}\t{'gdlon':<6}\n"
                for item in datalist:
                    try:
                        text += (f"{item['gps_site']:<8}\t{item['sat_id']:<8}\t{item['param']:<9.5f}\t{item['ncp']:<6}\t"
                                 f"{item['gdlat']:<6.2f}\t{item['gdlon']:<6.2f}\n")
                    except Exception as er:
                        print(item)
                        raise er
                save_path_3 = os.path.join(save_path_2, time_file)
                with open(save_path_3, "w") as file:
                    file.write(text)
            print("save", altitude, distance)
        for distance, datadict in dist_dict2.items():
            directory_name = f"distance_{distance}"
            save_path_2 = get_directory_path(save_path_1, directory_name)
            for timestamp, datalist in datadict.items():
                time = dt.datetime.fromtimestamp(timestamp, tz=dt.timezone.utc)
                time_file = f"large_{time.hour:0=2}{time.minute:0=2}.txt"
                text = f"{'gps_site':<8}\t{'sat_id':<8}\t{'diff^2':<9}\t{'gdlat':<6}\t{'gdlon':<6}\t{'gps_sit2':<8}\t{'sat_id2':<8}\n"
                for item in datalist:
                    for el in item["diff_list"]:
                        # text += (f"{item['gps_site']:<8}\t{item['sat_id']:<8}\t{el['diff']:<9.5f}\t"
                        #          f"{item['gdlat']:<6.2f}\t{item['gdlon']:<6.2f}\t{el['gps_site']:<8}\t{el['sat_id']:<8}\n")
                        text += (f"{item['gps_site']:<8}\t{item['sat_id']:<8}\t{el[0]:<9.5f}\t"
                                 f"{item['gdlat']:<6.2f}\t{item['gdlon']:<6.2f}\t{el[2]:<8}\t{el[1]:<8}\n")
                save_path_3 = os.path.join(save_path_2, time_file)
                with open(save_path_3, "w") as file:
                    file.write(text)


def main2024apr12_mp():
    anglekm = math.degrees(1 / 20000 * math.pi)
    source_directory = r"/home/vadymskipa/PhD_student/data/Europe_30-80__-10-50/dvtec_altitude"
    save_directory = r"/home/vadymskipa/PhD_student/data/Europe_30-80__-10-50/dvtec_altitude/parameter_lists"
    # list_of_altitudes = [150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700]
    list_of_altitudes = [150, 200, 250, 300, 350, 400, 450, 550, 600, 650, 700]
    # list_of_altitudes = [150]
    lats = [43.5, 48.5]
    lons = [11.5, 18.5]
    min_date = dt.datetime(year=2020, month=9, day=24, hour=9, minute=30, tzinfo=dt.timezone.utc)
    max_date = dt.datetime(year=2020, month=9, day=24, hour=12, minute=0, tzinfo=dt.timezone.utc)
    # min_date = dt.datetime(year=2020, month=9, day=24, hour=11, minute=38, tzinfo=dt.timezone.utc)
    # max_date = dt.datetime(year=2020, month=9, day=24, hour=12, minute=0, tzinfo=dt.timezone.utc)
    min_date_ts = min_date.timestamp()
    max_date_ts = max_date.timestamp()
    period_of_averaging = 120
    for altitude in list_of_altitudes:
        data_dict = {}
        for timestamp in range(int(min_date_ts), int(max_date_ts), period_of_averaging):
            data_dict[timestamp] = []
        altitude_name = f"dvtec_txt_alt_{altitude}"
        altitude_path = get_directory_path(source_directory, altitude_name)
        save_path_1 = get_directory_path(save_directory, f"alt_{altitude}")
        inner_directories = os.listdir(altitude_path)
        for date_directory in inner_directories:
            date = get_date_from_date_directory_name(date_directory)
            date_directory_path = os.path.join(altitude_path, date_directory)
            inner_directories_2 = os.listdir(date_directory_path)
            for gps_site in inner_directories_2:
                gps_site_path = os.path.join(date_directory_path, gps_site)
                window_path = os.path.join(gps_site_path, "Window_7200_Seconds")
                if not os.path.exists(window_path):
                    continue
                inner_files_4 = os.listdir(window_path)
                # data_dict[gps_site] = {}
                for sat_id_file in inner_files_4:
                    sat_id_path = os.path.join(window_path, sat_id_file)
                    sat_id = sat_id_file[0:3]
                    # data_dict[gps_site][sat_id] = {}
                    dtec_dataframe = read_dtec_file(sat_id_path)
                    dtec_dataframe = add_timestamp_column_to_df(dtec_dataframe, date)
                    # for index in dtec_dataframe.index:
                    #     dtec = dtec_dataframe.loc[index].loc["dtec"]
                    #     gdlat = dtec_dataframe.loc[index].loc["gdlat"]
                    #     gdlon = dtec_dataframe.loc[index].loc["gdlon"]
                    #     timestamp = dtec_dataframe.loc[index].loc["timestamp"]
                    #     temp_dict = {"dtec": dtec, "gdlat": gdlat, "gdlon": gdlon, "sat_id": sat_id,
                    #                  "gps_site": gps_site}
                    #     data_dict[timestamp].append(temp_dict)
                    for timestamp in range(int(min_date_ts), int(max_date_ts), period_of_averaging):
                        try:
                            mask = dtec_dataframe.loc[:, "timestamp"] == timestamp
                            temp_dt = dtec_dataframe.loc[mask]
                            if temp_dt.empty:
                                continue
                            dtec = temp_dt.iloc[0].loc["dtec"]
                            gdlat = temp_dt.iloc[0].loc["gdlat"]
                            gdlon = temp_dt.iloc[0].loc["gdlon"]
                            temp_dict = {"dtec": dtec, "gdlat": gdlat, "gdlon": gdlon, "sat_id": sat_id, "gps_site": gps_site}
                            data_dict[timestamp].append(temp_dict)
                        except Exception as er:
                            print("\t", er)
                            print("\t", gps_site, sat_id_file, "\n")
        dist_dict = {}
        dist_dict2 = {}
        # distances = [50, 70, 100, 150]
        distances = [150]
        print("read")

        inputlist = [(distance, anglekm, altitude, data_dict) for distance in distances]
        pool = multipool.Pool(1)
        for distance, dist_dict_in, dist_dict2_in in pool.imap(main2024apr12_p, inputlist):
            dist_dict[distance] = dist_dict_in
            dist_dict2[distance] = dist_dict2_in
        pool.close()

        print("start writting")
        for distance, datadict in dist_dict.items():
            directory_name = f"distance_{distance}"
            save_path_2 = get_directory_path(save_path_1, directory_name)
            for timestamp, datalist in datadict.items():
                time = dt.datetime.fromtimestamp(timestamp, tz=dt.timezone.utc)
                time_file = f"{time.hour:0=2}{time.minute:0=2}.txt"
                text = f"{'gps_site':<8}\t{'sat_id':<8}\t{'param':<9}\t{'ncp':<6}\t{'gdlat':<6}\t{'gdlon':<6}\n"
                for item in datalist:
                    try:
                        text += (f"{item['gps_site']:<8}\t{item['sat_id']:<8}\t{item['param']:<9.5f}\t{item['ncp']:<6}\t"
                                 f"{item['gdlat']:<6.2f}\t{item['gdlon']:<6.2f}\n")
                    except Exception as er:
                        print(item)
                        raise er
                save_path_3 = os.path.join(save_path_2, time_file)
                with open(save_path_3, "w") as file:
                    file.write(text)
            print("save", altitude, distance)
        for distance, datadict in dist_dict2.items():
            directory_name = f"distance_{distance}"
            save_path_2 = get_directory_path(save_path_1, directory_name)
            for timestamp, datalist in datadict.items():
                time = dt.datetime.fromtimestamp(timestamp, tz=dt.timezone.utc)
                time_file = f"large_{time.hour:0=2}{time.minute:0=2}.txt"
                text = f"{'gps_site':<8}\t{'sat_id':<8}\t{'diff^2':<9}\t{'gdlat':<6}\t{'gdlon':<6}\t{'gps_sit2':<8}\t{'sat_id2':<8}\n"
                for item in datalist:
                    for el in item["diff_list"]:
                        # text += (f"{item['gps_site']:<8}\t{item['sat_id']:<8}\t{el['diff']:<9.5f}\t"
                        #          f"{item['gdlat']:<6.2f}\t{item['gdlon']:<6.2f}\t{el['gps_site']:<8}\t{el['sat_id']:<8}\n")
                        text += (f"{item['gps_site']:<8}\t{item['sat_id']:<8}\t{el[0]:<9.5f}\t"
                                 f"{item['gdlat']:<6.2f}\t{item['gdlon']:<6.2f}\t{el[2]:<8}\t{el[1]:<8}\n")
                save_path_3 = os.path.join(save_path_2, time_file)
                with open(save_path_3, "w") as file:
                    file.write(text)


def main2024apr12_p(datatuple):
    distance, anglekm, altitude, data_dict = datatuple
    dlat = (distance + 20) * anglekm
    copy_data_dict = copy.deepcopy(data_dict)
    dist_dict = copy_data_dict
    copy_data_dict2 = copy.deepcopy(data_dict)
    dist_dict2 = copy_data_dict2
    i = 0
    for timestamp, datalist in dist_dict2.items():
        for elem in datalist:
            elem["diff_list"] = []

    for timestamp, datalist in data_dict.items():
        print(distance, i)
        i += 1
        date = dt.datetime.fromtimestamp(timestamp, tz=dt.timezone.utc)
        for index in range(len(datalist)):
            diff_list = []
            lat1 = datalist[index]["gdlat"]
            lon1 = datalist[index]["gdlon"]
            dtec1 = datalist[index]["dtec"]
            for second_pd in datalist:
                if datalist[index] == second_pd:
                    continue
                lat2 = second_pd["gdlat"]
                if not ((lat1 - dlat) <= lat2 <= (lat1 + dlat)):
                    continue
                lon2 = second_pd["gdlon"]
                if lat1 < 60:
                    if not ((lon1 - 2 * dlat) <= lon2 <= (lon1 + 2 * dlat)):
                        continue
                d = haversine(lat1, lon1, lat2, lon2, altitude)
                if d <= distance:
                    dtec2 = second_pd["dtec"]
                    diff = (dtec2 - dtec1) ** 2
                    diff_list.append(diff)
                    # dist_dict2[distance][timestamp][index]["diff_list"].append({"diff": diff,
                    #                                                             "sat_id": second_pd["sat_id"],
                    #                                                             "gps_site": second_pd["gps_site"]})
                    dist_dict2[timestamp][index]["diff_list"].append((diff, second_pd["sat_id"],
                                                                                second_pd["gps_site"]))
            dist_dict[timestamp][index]["ncp"] = len(diff_list)
            if len(diff_list) == 0:
                dist_dict[timestamp][index]["param"] = 9999
            else:
                dist_dict[timestamp][index]["param"] = sum(diff_list) / len(diff_list)
    return (distance, dist_dict, dist_dict2)



def read_paramfile(file):
    arr_site = []
    arr_sat_id = []
    arr_param = []
    arr_ncp = []
    arr_gdlat = []
    arr_gdlon = []
    arr_of_arr = [arr_site, arr_sat_id, arr_param, arr_ncp, arr_gdlat, arr_gdlon]
    with open(file, "r") as reader:
        reader.readline()
        for line in reader:
            text_tuple = line.split()
            if int(text_tuple[3]) == 0:
                continue
            arr_of_arr[0].append((text_tuple[0]))
            arr_of_arr[1].append((text_tuple[1]))
            arr_of_arr[2].append(float(text_tuple[2]))
            arr_of_arr[3].append(int(text_tuple[3]))
            arr_of_arr[4].append(float(text_tuple[4]))
            arr_of_arr[5].append(float(text_tuple[5]))
    outcome_dataframe = pd.DataFrame({"gps_site": arr_of_arr[0], "sat_id": arr_of_arr[1], "param": arr_of_arr[2],
                                      "ncp": arr_of_arr[3], "gdlat": arr_of_arr[4], "gdlon": arr_of_arr[5]})
    return outcome_dataframe


def read_paramfile_coord(file, lats, lons):
    arr_site = []
    arr_sat_id = []
    arr_param = []
    arr_ncp = []
    arr_gdlat = []
    arr_gdlon = []
    arr_of_arr = [arr_site, arr_sat_id, arr_param, arr_ncp, arr_gdlat, arr_gdlon]
    with open(file, "r") as reader:
        reader.readline()
        for line in reader:
            text_tuple = line.split()
            if not (lats[0] <= float(text_tuple[4]) <= lats[1]):
                continue
            if not (lons[0] <= float(text_tuple[5]) <= lons[1]):
                continue
            if int(text_tuple[3]) == 0:
                continue
            arr_of_arr[0].append((text_tuple[0]))
            arr_of_arr[1].append((text_tuple[1]))
            arr_of_arr[2].append(float(text_tuple[2]))
            arr_of_arr[3].append(int(text_tuple[3]))
            arr_of_arr[4].append(float(text_tuple[4]))
            arr_of_arr[5].append(float(text_tuple[5]))
    outcome_dataframe = pd.DataFrame({"gps_site": arr_of_arr[0], "sat_id": arr_of_arr[1], "param": arr_of_arr[2],
                                      "ncp": arr_of_arr[3], "gdlat": arr_of_arr[4], "gdlon": arr_of_arr[5]})
    return outcome_dataframe


def read_large_paramfilecoord(file, lats, lons):
    arr_site = []
    arr_sat_id = []
    arr_param = []
    arr_gdlat = []
    arr_gdlon = []
    arr_site2 = []
    arr_sat_id2 = []
    arr_of_arr = [arr_site, arr_sat_id, arr_param, arr_gdlat, arr_gdlon, arr_site2, arr_sat_id2]
    with open(file, "r") as reader:
        reader.readline()
        for line in reader:
            text_tuple = line.split()
            if not (lats[0] <= float(text_tuple[3]) <= lats[1]):
                continue
            if not (lons[0] <= float(text_tuple[4]) <= lons[1]):
                continue
            arr_of_arr[0].append((text_tuple[0]))
            arr_of_arr[1].append((text_tuple[1]))
            arr_of_arr[2].append(float(text_tuple[2]))
            arr_of_arr[3].append(float(text_tuple[3]))
            arr_of_arr[4].append(float(text_tuple[4]))
            arr_of_arr[5].append((text_tuple[5]))
            arr_of_arr[6].append((text_tuple[6]))
    outcome_dataframe = pd.DataFrame({"gps_site": arr_of_arr[0], "sat_id": arr_of_arr[1], "param": arr_of_arr[2],
                                      "gdlat": arr_of_arr[3], "gdlon": arr_of_arr[4], "gps_site2": arr_of_arr[5],
                                      "sat_id2": arr_of_arr[6]})
    return outcome_dataframe


def main2024apr13(lats, lons):
    sd_name = f"lats{lats[0]}{lats[1]}_lons{lons[0]}{lons[1]}"
    source_directory = r"/home/vadymskipa/PhD_student/data/Europe_30-80__-10-50/dvtec_altitude/parameter_lists"
    save_directory = r"/home/vadymskipa/PhD_student/data/Europe_30-80__-10-50/dvtec_altitude/parameter_tables"
    save_directory = get_directory_path(save_directory, sd_name)
    list_of_altitudes = [150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700]
    distances = [150]
    min_date = dt.datetime(year=2020, month=9, day=24, hour=9, minute=30, tzinfo=dt.timezone.utc)
    max_date = dt.datetime(year=2020, month=9, day=24, hour=12, minute=0, tzinfo=dt.timezone.utc)
    min_date_ts = min_date.timestamp()
    max_date_ts = max_date.timestamp()
    # average_dtec_list = main2024apr15(lats=lats, lons=lons)

    period_of_averaging = 120
    list_date_txt = []
    for timestamp in range(int(min_date_ts), int(max_date_ts), period_of_averaging):
        temp_date = dt.datetime.fromtimestamp(timestamp, tz=dt.timezone.utc)
        list_date_txt.append(f"{temp_date.hour:0=2}{temp_date.minute:0=2}_ds.txt")
    for distance in distances:
        distance_name = f"distance_{distance}"
        save_path = get_directory_path(save_directory, distance_name)

        param_o1_dict = {}
        std_o1_dict = {}
        param_o2_dict = {}
        for date_txt in list_date_txt:
            param_o1_dict[date_txt[:4]] = {}
            std_o1_dict[date_txt[:4]] = {}
            param_o2_dict[date_txt[:4]] = {}
        for altitude in list_of_altitudes:
            altitude_name = f"alt_{altitude}"
            print(distance, altitude)
            altitude_path = get_directory_path(source_directory, altitude_name)
            distnace_path = get_directory_path(altitude_path, distance_name)
            # save_path_1 = get_directory_path(save_directory, f"alt_{altitude}")
            for date_txt in list_date_txt:
                file_path = os.path.join(distnace_path, date_txt)
                dataframe = read_paramfile_coord(file_path, lats=lats, lons=lons)
                param_o1_series = dataframe.loc[:, "param"].apply(math.sqrt)
                param_o1 = param_o1_series.mean() * 10
                std_o1 = param_o1_series.std(ddof=0) * 10
                series_o2 = dataframe.loc[:, "param"] * dataframe.loc[:, "ncp"]
                param_o2 = math.sqrt(series_o2.sum() / dataframe.loc[:, "ncp"].sum()) * 10

                param_o1_dict[date_txt[:4]][altitude] = param_o1
                std_o1_dict[date_txt[:4]][altitude] = std_o1
                param_o2_dict[date_txt[:4]][altitude] = param_o2
        text = f"{'time':<8}\t"
        for altitude in list_of_altitudes:
            text += f"{altitude:<9}\t"
        text += "\n"
        text_param_o1 = text
        for time, datadict in param_o1_dict.items():
            text_param_o1 += f"{time:<8}\t"
            for altitude in list_of_altitudes:
                text_param_o1 += f"{datadict[altitude]:<9.5f}\t"
            text_param_o1 += "\n"
        save_param_o1_path = os.path.join(save_path, "param_o1_ds.txt")
        with open(save_param_o1_path, "w") as file:
            file.write(text_param_o1)
        text_param_o2 = text
        for time, datadict in param_o2_dict.items():
            text_param_o2 += f"{time:<8}\t"
            for altitude in list_of_altitudes:
                text_param_o2 += f"{datadict[altitude]:<9.5f}\t"
            text_param_o2 += "\n"
        save_param_o2_path = os.path.join(save_path, "param_o2_ds.txt")
        with open(save_param_o2_path, "w") as file:
            file.write(text_param_o2)
        text_std_o1 = text
        for time, datadict in std_o1_dict.items():
            text_std_o1 += f"{time:<8}\t"
            for altitude in list_of_altitudes:
                text_std_o1 += f"{datadict[altitude]:<9.5f}\t"
            text_std_o1 += "\n"
        save_std_o1_path = os.path.join(save_path, "std_o1_ds.txt")
        text_param_o1_norm = text
        with open(save_std_o1_path, "w") as file:
            file.write(text_std_o1)

        # for time, datadict in param_o1_dict.items():
        #     text_param_o1_norm += f"{time:<8}\t"
        #     for altitude in list_of_altitudes:
        #         norm = datadict[altitude] / average_dtec_list[time][altitude]
        #         text_param_o1_norm += f"{norm:<9.5f}\t"
        #     text_param_o1_norm += "\n"
        # save_param_o1_norm_path = os.path.join(save_path, "param_o1_norm.txt")
        # with open(save_param_o1_norm_path, "w") as file:
        #     file.write(text_param_o1_norm)
        # text_param_o2_norm = text
        # for time, datadict in param_o2_dict.items():
        #     text_param_o2_norm += f"{time:<8}\t"
        #     for altitude in list_of_altitudes:
        #         norm = datadict[altitude] / average_dtec_list[time][altitude]
        #         text_param_o2_norm += f"{norm:<9.5f}\t"
        #     text_param_o2_norm += "\n"
        # save_param_o2_norm_path = os.path.join(save_path, "param_o2_norm.txt")
        # with open(save_param_o2_norm_path, "w") as file:
        #     file.write(text_param_o2_norm)
        # text_std_norm_o1 = text
        # for time, datadict in std_o1_dict.items():
        #     text_std_norm_o1 += f"{time:<8}\t"
        #     for altitude in list_of_altitudes:
        #         norm = datadict[altitude] / average_dtec_list[time][altitude]
        #         text_std_norm_o1 += f"{norm:<9.5f}\t"
        #     text_std_norm_o1 += "\n"
        # save_std_o1_norm_path = os.path.join(save_path, "std_o1_norm.txt")
        # with open(save_std_o1_norm_path, "w") as file:
        #     file.write(text_std_norm_o1)


def main2024apr15(lats, lons):
    # get mean dvtec for every time-altitude couple
    source_directory = r"/home/vadymskipa/PhD_student/data/Europe_30-80__-10-50/dvtec_altitude"
    list_of_altitudes = [150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700]
    min_date = dt.datetime(year=2020, month=9, day=24, hour=9, minute=30, tzinfo=dt.timezone.utc)
    max_date = dt.datetime(year=2020, month=9, day=24, hour=12, minute=0, tzinfo=dt.timezone.utc)
    min_date_ts = min_date.timestamp()
    max_date_ts = max_date.timestamp()
    period_of_averaging = 120
    data_dict = {}
    for altitude in list_of_altitudes:
        print(altitude)
        temp_data_dict = {}
        data_dict[altitude] = temp_data_dict
        for timestamp in range(int(min_date_ts), int(max_date_ts), period_of_averaging):
            temp_data_dict[timestamp] = []
        altitude_name = f"dvtec_txt_alt_{altitude}"
        altitude_path = get_directory_path(source_directory, altitude_name)
        inner_directories = os.listdir(altitude_path)
        for date_directory in inner_directories:
            date = get_date_from_date_directory_name(date_directory)
            date_directory_path = os.path.join(altitude_path, date_directory)
            inner_directories_2 = os.listdir(date_directory_path)
            i = 0
            for gps_site in inner_directories_2:
                i += 1
                if (i % 200) == 0:
                    print("\t", i)
                gps_site_path = os.path.join(date_directory_path, gps_site)
                window_path = os.path.join(gps_site_path, "Window_7200_Seconds")
                if not os.path.exists(window_path):
                    continue
                inner_files_4 = os.listdir(window_path)
                for sat_id_file in inner_files_4:
                    sat_id_path = os.path.join(window_path, sat_id_file)
                    dtec_dataframe = read_dtec_file(sat_id_path)
                    dtec_dataframe = add_timestamp_column_to_df(dtec_dataframe, date)
                    for timestamp in range(int(min_date_ts), int(max_date_ts), period_of_averaging):
                        try:
                            mask = dtec_dataframe.loc[:, "timestamp"] == timestamp
                            temp_dt = dtec_dataframe.loc[mask]
                            if temp_dt.empty:
                                continue
                            dtec = temp_dt.iloc[0].loc["dtec"]
                            gdlat = temp_dt.iloc[0].loc["gdlat"]
                            if not (lats[0] <= gdlat <= lats[1]):
                                continue
                            gdlon = temp_dt.iloc[0].loc["gdlon"]
                            if not (lons[0] <= gdlon <= lons[1]):
                                continue
                            temp_data_dict[timestamp].append(dtec)
                        except Exception as er:
                            print("\t", er)
                            print("\t", gps_site, sat_id_file, "\n")
    result_dict = {}
    for timestamp in range(int(min_date_ts), int(max_date_ts), period_of_averaging):
        time = dt.datetime.fromtimestamp(timestamp, tz=dt.timezone.utc)
        time_file = f"{time.hour:0=2}{time.minute:0=2}"
        result_dict[time_file] = {}
        for altitude in list_of_altitudes:
            try:
                average_dtec = sum(data_dict[altitude][timestamp]) / len(data_dict[altitude][timestamp])
            except Exception as er:
                raise er
            result_dict[time_file][altitude] = average_dtec
    return result_dict


def main2024apr16(lats, lons):
    data_dict = main2024apr15(lats, lons)
    save_directory = r"/home/vadymskipa/PhD_student/data/Europe_30-80__-10-50/dvtec_altitude/average_dvtec_1"
    sd_name = f"lats{lats[0]}{lats[1]}_lons{lons[0]}{lons[1]}.txt"
    save_path_1 = os.path.join(save_directory, sd_name)
    list_of_altitudes = [150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700]
    text = f"{'time':<8}\t"
    for altitude in list_of_altitudes:
        text += f"{altitude:<9}\t"
    text += "\n"
    for time, time_datadict in data_dict.items():
        text += f"{time:<8}\t"
        for altitude in list_of_altitudes:
            text += f"{time_datadict[altitude]:<9.5f}\t"
        text += "\n"
    with open(save_path_1, "w") as file:
        file.write(text)





def main2024apr14():
    # lats = [49.5, 52.5]
    # lons = [12.5, 17.5]
    # lats = [53, 57]
    lons = [13.5, 16.5]
    # for i in [39, 43, 47, 51, 55, 59]:
    #     lats = [i - 2, i + 2]
    #     main2024apr13(lats=lats, lons=lons)
    # for coords in [(50, 14.6), (54.6, 13.4)]:
    #     lats = [coords[0] - 2, coords[0] + 2]
    #     lons = [coords[1] - 3, coords[1] + 3]
    #     main2024apr13(lats=lats, lons=lons)

    for i in range(42, 60, 2):
        lats = [i - 1, i + 1]
        main2024apr13(lats=lats, lons=lons)
        # main2024apr16(lats=lats, lons=lons)
        main2024apr18(lats=lats, lons=lons)
    # lats = [44, 46]
    # lons = [7, 10]
    # lats = [46, 48]
    # lons1 = [8, 11]
    # # lons2 = [15, 18]
    # main2024apr13(lats=lats, lons=lons1)
    # main2024apr18(lats=lats, lons=lons1)
    # main2024apr13(lats=lats, lons=lons2)
    # main2024apr18(lats=lats, lons=lons2)
    # main2024apr16(lats=lats, lons=lons)



def main2024apr17():
    # for coords in [(50, 14.6), (54.6, 13.4)]:
    #     lats = [coords[0] - 2, coords[0] + 2]
    #     lons = [coords[1] - 3, coords[1] + 3]
    #     main2024apr16(lats=lats, lons=lons)
    lons = [12, 18]
    for i in [39, 43, 47, 51, 55, 59]:
        lats = [i - 2, i + 2]
        # main2024apr16(lats=lats, lons=lons)
        # main2024apr18(lats=lats, lons=lons)
        # main2024apr19(lats=lats, lons=lons)
        main2024apr20(lats=lats, lons=lons)



def main2024apr18(lats, lons):
    sd_name = f"lats{lats[0]}{lats[1]}_lons{lons[0]}{lons[1]}"
    source_directory = r"/home/vadymskipa/PhD_student/data/Europe_30-80__-10-50/dvtec_altitude/parameter_lists/"
    save_directory = r"/home/vadymskipa/PhD_student/data/Europe_30-80__-10-50/dvtec_altitude/parameter_tables/"
    save_directory = get_directory_path(save_directory, sd_name)
    list_of_altitudes = [150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700]
    min_date = dt.datetime(year=2020, month=9, day=24, hour=9, minute=30, tzinfo=dt.timezone.utc)
    max_date = dt.datetime(year=2020, month=9, day=24, hour=12, minute=0, tzinfo=dt.timezone.utc)
    min_date_ts = min_date.timestamp()
    max_date_ts = max_date.timestamp()
    period_of_averaging = 120
    list_date_txt = []
    distances = [150]
    for timestamp in range(int(min_date_ts), int(max_date_ts), period_of_averaging):
        temp_date = dt.datetime.fromtimestamp(timestamp, tz=dt.timezone.utc)
        list_date_txt.append(f"{temp_date.hour:0=2}{temp_date.minute:0=2}_ds.txt")
    for distance in distances:
        distance_name = f"distance_{distance}"
        save_path = get_directory_path(save_directory, distance_name)

        ncp_dict = {}
        mean_ncp_dict = {}
        for date_txt in list_date_txt:
            ncp_dict[date_txt[:4]] = {}
            mean_ncp_dict[date_txt[:4]] = {}
        for altitude in list_of_altitudes:
            altitude_name = f"alt_{altitude}"
            altitude_path = get_directory_path(source_directory, altitude_name)
            distnace_path = get_directory_path(altitude_path, distance_name)
            # save_path_1 = get_directory_path(save_directory, f"alt_{altitude}")
            for date_txt in list_date_txt:
                file_path = os.path.join(distnace_path, date_txt)
                dataframe = read_paramfile_coord(file_path, lats=lats, lons=lons)
                series_ncp = dataframe.loc[:, "ncp"]
                ncp_dict[date_txt[:4]][altitude] = series_ncp.sum()
                mean_ncp_dict[date_txt[:4]][altitude] = series_ncp.mean()


        text = f"{'time':<8}\t"
        for altitude in list_of_altitudes:
            text += f"{altitude:<9}\t"
        text += "\n"
        text_ncp = text
        for time, datadict in ncp_dict.items():
            text_ncp += f"{time:<8}\t"
            for altitude in list_of_altitudes:
                text_ncp += f"{datadict[altitude]:<9}\t"
            text_ncp += "\n"
        save_ncp_path = os.path.join(save_path, "ncp_ds.txt")
        with open(save_ncp_path, "w") as file:
            file.write(text_ncp)
        text_mean_ncp = text
        for time, datadict in mean_ncp_dict.items():
            text_mean_ncp += f"{time:<8}\t"
            for altitude in list_of_altitudes:
                text_mean_ncp += f"{datadict[altitude]:<9.5f}\t"
            text_mean_ncp += "\n"
        save_mean_ncp_path = os.path.join(save_path, "mean_ncp_ds.txt")
        with open(save_mean_ncp_path, "w") as file:
            file.write(text_mean_ncp)


def main2024apr19(lats, lons):
    sd_name = f"lats{lats[0]}{lats[1]}_lons{lons[0]}{lons[1]}"
    source_directory = r"/home/vadymskipa/PhD_student/data/Europe_30-80__-10-50/dvtec_altitude/parameters"
    save_directory = r"/home/vadymskipa/PhD_student/data/Europe_30-80__-10-50/dvtec_altitude/parameters/param_tables"
    save_directory = get_directory_path(save_directory, sd_name)
    list_of_altitudes = [150, 200, 250, 300, 350, 400, 450, 500, 550, 600]
    min_date = dt.datetime(year=2020, month=9, day=24, hour=10, tzinfo=dt.timezone.utc)
    max_date = dt.datetime(year=2020, month=9, day=24, hour=11, minute=30, tzinfo=dt.timezone.utc)
    min_date_ts = min_date.timestamp()
    max_date_ts = max_date.timestamp()
    list_date_txt = []
    distances = [70, 100, 150]
    for timestamp in range(int(min_date_ts), int(max_date_ts), 300):
        temp_date = dt.datetime.fromtimestamp(timestamp)
        list_date_txt.append(f"{temp_date.hour:0=2}{temp_date.minute:0=2}.txt")
    for distance in distances:
        distance_name = f"distance_{distance}"
        save_path = get_directory_path(save_directory, distance_name)

        param_std_o1_dict = {}
        param_std_o2_dict = {}
        std_std_dict = {}
        ncp_dict = {}
        mean_ncp_dict = {}
        for date_txt in list_date_txt:
            param_std_o1_dict[date_txt[:4]] = {}
            param_std_o2_dict[date_txt[:4]] = {}
            std_std_dict[date_txt[:4]] = {}
            ncp_dict[date_txt[:4]] = {}
            mean_ncp_dict[date_txt[:4]] = {}
        for altitude in list_of_altitudes:
            print(altitude)
            altitude_name = f"alt_{altitude}"
            altitude_path = get_directory_path(source_directory, altitude_name)
            distnace_path = get_directory_path(altitude_path, distance_name)
            # save_path_1 = get_directory_path(save_directory, f"alt_{altitude}")
            for date_txt in list_date_txt:
                print("\t", date_txt)
                paramlist = []
                ncplist = []
                text = f"{'gps_site':<8}\t{'sat_id':<8}\t{'param':<9}\t{'ncp':<6}\t{'gdlat':<6}\t{'gdlon':<6}\n"
                file_path = os.path.join(distnace_path, date_txt)
                large_file_path = os.path.join(distnace_path, "large_" + date_txt)
                dataframe = read_paramfile_coord(file_path, lats=lats, lons=lons)
                large_dataframe = read_large_paramfilecoord(large_file_path, lats=lats, lons=lons)
                param_o1 = dataframe.loc[:, "param"].mean()
                std_o1 = dataframe.loc[:, "param"].std(ddof=0)
                series_o2 = dataframe.loc[:, "param"] * dataframe.loc[:, "ncp"]
                # param_o2 = series_o2.sum() / dataframe.loc[:, "ncp"].sum() * 100

                mask_o1 = dataframe.loc[:, "param"] >= param_o1 - std_o1
                mask_o1 = np.logical_and(mask_o1, dataframe.loc[:, "param"] <= param_o1 + std_o1)
                new_dataframe = dataframe.loc[mask_o1]
                good_pairs = {}
                good_gps_sites_list = np.unique(new_dataframe.loc[:, 'gps_site'])
                for gps_site in good_gps_sites_list:
                    one_site_dataframe = new_dataframe.loc[new_dataframe.loc[:, "gps_site"] == gps_site]
                    sat_id_list = np.unique(one_site_dataframe.loc[:, "sat_id"])
                    good_pairs[gps_site] = sat_id_list
                for gps_site, sat_id_list in good_pairs.items():
                    gps_site_large_dataframe = large_dataframe.loc[large_dataframe.loc[:, "gps_site"] == gps_site]
                    for sat_id in sat_id_list:
                        pair_large_dataframe = gps_site_large_dataframe.loc[large_dataframe.loc[:, "sat_id"] == sat_id]
                        param = 0
                        ncp = 0
                        for index in pair_large_dataframe.index:
                            temp_series = pair_large_dataframe.loc[index]
                            if temp_series.loc["gps_site2"] in good_pairs.keys():
                                if temp_series.loc["sat_id2"] in good_pairs[temp_series.loc["gps_site2"]]:
                                    param += temp_series.loc["param"]
                                    ncp += 1
                        if ncp != 0:
                            paramlist.append(param / ncp)
                            ncplist.append(ncp)
                param_std_o1 = sum(paramlist) / len(paramlist)
                param_std_o2 = 0
                for i in range(len(paramlist)):
                    param_std_o2 += paramlist[i] * ncplist[i]
                param_std_o2 = param_std_o2 / sum(ncplist)
                std_std = statistics.pstdev(paramlist)
                ncp = sum(ncplist)
                mean_ncp = ncp / len(ncplist)
                param_std_o1_dict[date_txt[:4]][altitude] = param_std_o1 * 100
                param_std_o2_dict[date_txt[:4]][altitude] = param_std_o2 * 100
                std_std_dict[date_txt[:4]][altitude] = std_std * 100
                ncp_dict[date_txt[:4]][altitude] = ncp
                mean_ncp_dict[date_txt[:4]][altitude] = mean_ncp
        text = f"{'time':<8}\t"
        for altitude in list_of_altitudes:
            text += f"{altitude:<9}\t"
        text += "\n"

        text_param_std_o1 = text
        for time, datadict in param_std_o1_dict.items():
            text_param_std_o1 += f"{time:<8}\t"
            for altitude in list_of_altitudes:
                text_param_std_o1 += f"{datadict[altitude]:<9.5f}\t"
            text_param_std_o1 += "\n"
        save_param_std_o1_path = os.path.join(save_path, "param-std_o1.txt")
        with open(save_param_std_o1_path, "w") as file:
            file.write(text_param_std_o1)
        text_param_std_o2 = text
        for time, datadict in param_std_o2_dict.items():
            text_param_std_o2 += f"{time:<8}\t"
            for altitude in list_of_altitudes:
                text_param_std_o2 += f"{datadict[altitude]:<9.5f}\t"
            text_param_std_o2 += "\n"
        save_param_std_o2_path = os.path.join(save_path, "param-std_o2.txt")
        with open(save_param_std_o2_path, "w") as file:
            file.write(text_param_std_o2)
        text_std_std = text
        for time, datadict in std_std_dict.items():
            text_std_std += f"{time:<8}\t"
            for altitude in list_of_altitudes:
                text_std_std += f"{datadict[altitude]:<9.5f}\t"
            text_std_std += "\n"
        save_std_std_path = os.path.join(save_path, "std-std.txt")
        with open(save_std_std_path, "w") as file:
            file.write(text_std_std)
        text_ncp = text
        for time, datadict in ncp_dict.items():
            text_ncp += f"{time:<8}\t"
            for altitude in list_of_altitudes:
                text_ncp += f"{datadict[altitude]:<9}\t"
            text_ncp += "\n"
        save_ncp_path = os.path.join(save_path, "ncp-std.txt")
        with open(save_ncp_path, "w") as file:
            file.write(text_ncp)
        text_mean_ncp = text
        for time, datadict in mean_ncp_dict.items():
            text_mean_ncp += f"{time:<8}\t"
            for altitude in list_of_altitudes:
                text_mean_ncp += f"{datadict[altitude]:<9.5f}\t"
            text_mean_ncp += "\n"
        save_mean_ncp_path = os.path.join(save_path, "mean_ncp-std.txt")
        with open(save_mean_ncp_path, "w") as file:
            file.write(text_mean_ncp)


def read_param_file(file, lats, lons):
    arr_alt = []
    outcome_dict = {}
    with open(file, "r") as reader:
        text_tuple = reader.readline().split()
        for str_alt in text_tuple[1:]:
            arr_alt.append(int(str_alt))
        for line in reader:
            local_dict = {}
            text_tuple = line.split()
            time = text_tuple[0]
            for index in range(len(text_tuple[1:])):
                local_dict[arr_alt[index]] = float(text_tuple[index+1])
            outcome_dict[time] = local_dict

    return outcome_dict


def main2024apr20(lats, lons):
    # norm of ***-std.txt
    sd_name = f"lats{lats[0]}{lats[1]}_lons{lons[0]}{lons[1]}"
    source_param_directory = r"/home/vadymskipa/PhD_student/data/Europe_30-80__-10-50/dvtec_altitude/parameters/param_tables"
    source_average_dvtec_directory = r"/home/vadymskipa/PhD_student/data/Europe_30-80__-10-50/dvtec_altitude/average_dvetc"
    save_directory = r"/home/vadymskipa/PhD_student/data/Europe_30-80__-10-50/dvtec_altitude/parameters/param_tables"
    save_directory = get_directory_path(save_directory, sd_name)
    distances = [70, 100, 150]
    file_name_list = ["param-std_o1.txt", "param-std_o2.txt", "std-std.txt"]
    source_param_path_1 = get_directory_path(source_param_directory, sd_name)
    source_average_dvtec_file = os.path.join(source_average_dvtec_directory, sd_name + ".txt")
    average_dvtec_dict = read_param_file(source_average_dvtec_file, lats=lats, lons=lons)
    for distance in distances:
        distance_name = f"distance_{distance}"
        source_param_path_2 = get_directory_path(source_param_path_1, distance_name)
        for file_name in file_name_list:
            file_path = os.path.join(source_param_path_2, file_name)
            param_dict = read_param_file(file_path, lats=lats, lons=lons)
            text = f"{'time':<8}\t"
            for time, time_dict in param_dict.items():
                for altitude in time_dict.keys():
                    text += f"{altitude:<9}\t"
                break

            text += "\n"
            for time, datadict in param_dict.items():
                text += f"{time:<8}\t"
                for altitude, param in datadict.items():
                    p = param / average_dvtec_dict[time][altitude]
                    text += f"{p:<9.5f}\t"
                text += "\n"
            save_path = os.path.join(source_param_path_2, file_name[:-4] + "_norm.txt")
            with open(save_path, "w") as file:
                file.write(text)


def main2024apr21():
    source_directory = r"/home/vadymskipa/PhD_student/data/Europe_30-80__-10-50/dvtec_altitude/parameter_lists/"
    list_of_altitudes = [150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700]
    min_date = dt.datetime(year=2020, month=9, day=24, hour=9, minute=30, tzinfo=dt.timezone.utc)
    max_date = dt.datetime(year=2020, month=9, day=24, hour=12, minute=0, tzinfo=dt.timezone.utc)
    min_date_ts = min_date.timestamp()
    max_date_ts = max_date.timestamp()
    period_of_averaging = 120
    list_time_txt = []
    distances = [150]
    for timestamp in range(int(min_date_ts), int(max_date_ts), period_of_averaging):
        temp_time = dt.datetime.fromtimestamp(timestamp, tz=dt.timezone.utc)
        list_time_txt.append(f"large_{temp_time.hour:0=2}{temp_time.minute:0=2}.txt")
    for distance in distances:
        distance_name = f"distance_{distance}"
        for altitude in list_of_altitudes:
            print(altitude)
            altitude_name = f"alt_{altitude}"
            altitude_path = get_directory_path(source_directory, altitude_name)
            distnace_path = get_directory_path(altitude_path, distance_name)
            for time_txt in list_time_txt:
                print("\t", time_txt)
                file_path = os.path.join(distnace_path, time_txt)
                save_file_path = os.path.join(distnace_path, time_txt[:10] + "_ds.txt")
                with open(file_path, "r") as file_read:
                    with open(save_file_path, "w") as save_file:
                        line = file_read.readline()
                        save_file.write(line)
                        for line in file_read:
                            text_tuple = line.split()
                            if text_tuple[1] == text_tuple[6]:
                                continue
                            save_file.write(line)


def main2024apr22():
    source_directory = r"/home/vadymskipa/PhD_student/data/Europe_30-80__-10-50/dvtec_altitude/parameter_lists/"
    list_of_altitudes = [150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700]
    min_date = dt.datetime(year=2020, month=9, day=24, hour=9, minute=30, tzinfo=dt.timezone.utc)
    max_date = dt.datetime(year=2020, month=9, day=24, hour=12, minute=0, tzinfo=dt.timezone.utc)
    min_date_ts = min_date.timestamp()
    max_date_ts = max_date.timestamp()
    period_of_averaging = 120
    list_time_txt = []
    distances = [150]
    for timestamp in range(int(min_date_ts), int(max_date_ts), period_of_averaging):
        temp_time = dt.datetime.fromtimestamp(timestamp, tz=dt.timezone.utc)
        list_time_txt.append(f"large_{temp_time.hour:0=2}{temp_time.minute:0=2}_ds.txt")
    for distance in distances:
        distance_name = f"distance_{distance}"
        for altitude in list_of_altitudes:
            print(altitude)
            altitude_name = f"alt_{altitude}"
            altitude_path = get_directory_path(source_directory, altitude_name)
            distnace_path = get_directory_path(altitude_path, distance_name)
            for time_txt in list_time_txt:
                print("\t", time_txt)
                text = f"{'gps_site':<8}\t{'sat_id':<8}\t{'param':<9}\t{'ncp':<6}\t{'gdlat':<6}\t{'gdlon':<6}\n"
                file_path = os.path.join(distnace_path, time_txt)
                save_path = os.path.join(distnace_path, time_txt[6:])
                large_dataframe = read_large_paramfile(file_path)
                site_list = np.unique(large_dataframe.loc[:, "gps_site"])
                for gps_site in site_list:
                    site_dataframe = large_dataframe.loc[large_dataframe.loc[:, "gps_site"] == gps_site]
                    sat_list = np.unique(site_dataframe.loc[:, "sat_id"])
                    for sat_id in sat_list:
                        sat_dataframe = site_dataframe.loc[site_dataframe.loc[:, "sat_id"] == sat_id]
                        param = sat_dataframe.loc[:, "param"].mean()
                        ncp = len(sat_dataframe.loc[:, "param"])
                        gdlat = sat_dataframe.iloc[0].loc["gdlat"]
                        gdlon = sat_dataframe.iloc[0].loc["gdlon"]
                        text += (
                            f"{gps_site:<8}\t{sat_id:<8}\t{param:<9.5f}\t{ncp:<6}\t"
                            f"{gdlat:<6.2f}\t{gdlon:<6.2f}\n")
                with open(save_path, "w") as file:
                    file.write(text)


def read_large_paramfile(file):
    arr_site = []
    arr_sat_id = []
    arr_param = []
    arr_gdlat = []
    arr_gdlon = []
    arr_site2 = []
    arr_sat_id2 = []
    arr_of_arr = [arr_site, arr_sat_id, arr_param, arr_gdlat, arr_gdlon, arr_site2, arr_sat_id2]
    with open(file, "r") as reader:
        reader.readline()
        for line in reader:
            text_tuple = line.split()
            arr_of_arr[0].append((text_tuple[0]))
            arr_of_arr[1].append((text_tuple[1]))
            arr_of_arr[2].append(float(text_tuple[2]))
            arr_of_arr[3].append(float(text_tuple[3]))
            arr_of_arr[4].append(float(text_tuple[4]))
            arr_of_arr[5].append((text_tuple[5]))
            arr_of_arr[6].append((text_tuple[6]))
    outcome_dataframe = pd.DataFrame({"gps_site": arr_of_arr[0], "sat_id": arr_of_arr[1], "param": arr_of_arr[2],
                                      "gdlat": arr_of_arr[3], "gdlon": arr_of_arr[4], "gps_site2": arr_of_arr[5],
                                      "sat_id2": arr_of_arr[6]})
    return outcome_dataframe







def main2024apr():
    # main2024apr10()
    # main2024apr11()
    # main2024apr12()
    main2024apr14()

    # main2024apr17()
    # main2024apr10()
    # main2024apr11()
    # main2024apr12_mp()
    # main2024apr13()
    # main2024apr21()
    # main2024apr22()


def main2024jun():
    # main2024jun1()
    # main2024jun2()
    # main2024jun3()
    # main2024jun4()
    # main2024jun5()
    # main2024jun6()
    # main2024jun7()
    main2024jun8()


def main2024jun1():
    lats = (36, 70)
    lons = (-10, 20)
    min_date = dt.datetime(year=2016, month=3, day=4, tzinfo=dt.timezone.utc)
    max_date = dt.datetime(year=2016, month=3, day=11, tzinfo=dt.timezone.utc)
    source_site_hdf_directory_path = r"/home/vadymskipa/PhD_student/data/site_hdf/"
    source_los_hdf_directory = r"/home/vadymskipa/PhD_student/data/Europe_los_hdf/"
    los_hdfs_in_directory = os.listdir(source_los_hdf_directory)
    list_los_hdf_paths = []
    for los_hdf in los_hdfs_in_directory:
        temp_date = wlos.get_date_by_los_file_name(los_hdf)
        if temp_date < min_date:
            continue
        if temp_date > max_date:
            continue
        list_los_hdf_paths.append(os.path.join(source_los_hdf_directory, los_hdf))
    save_path = r"/home/vadymskipa/PhD_student/data/Europe_36-70__-10-20/"
    save_los_tec_txt_file_for_region_from_los_files(list_los_hdf_paths, source_site_hdf_directory_path, save_path,
                                                    lats=lats, lons=lons, multiprocessing=True, gps_only=False)


def main2024jun2():
    los_list = [r"/home/vadymskipa/HDD1/big_los_hdf/los_20160305.001.h5.hdf5",
                r"/home/vadymskipa/HDD1/big_los_hdf/los_20160306.001.h5.hdf5",
                r"/home/vadymskipa/HDD1/big_los_hdf/los_20160307.001.h5.hdf5",
                r"/home/vadymskipa/HDD1/big_los_hdf/los_20160308.001.h5.hdf5",
                r"/home/vadymskipa/HDD1/big_los_hdf/los_20160309.001.h5.hdf5",
                r"/home/vadymskipa/HDD1/big_los_hdf/los_20160310.001.h5.hdf5",
                r"/home/vadymskipa/HDD1/big_los_hdf/los_20160311.001.h5.hdf5"]
    save = r"/home/vadymskipa/PhD_student/data/site_hdf/"
    for los in los_list:
        wlos.create_site_file_from_los(los, save)


def main2024jun3():
    source_path = r"/home/vadymskipa/PhD_student/data/Europe_36-70__-10-20/los_tec_txt/"
    save_path = r"/home/vadymskipa/PhD_student/data/Europe_36-70__-10-20/dtec_txt/"
    params = LIST_PARAMS1
    min_date = dt.datetime(year=2016, month=3, day=4, tzinfo=dt.timezone.utc)
    max_date = dt.datetime(year=2016, month=3, day=11, tzinfo=dt.timezone.utc)
    get_dtec_from_los_tec_txt_for_lots_of_sites(source_path, save_path, params, min_date=min_date, max_date=max_date)


def main2024jun4():
    source_directories = [r"/home/vadymskipa/PhD_student/data/Europe_36-70__-10-20/dtec_txt/064-2016-03-04",
                          r"/home/vadymskipa/PhD_student/data/Europe_36-70__-10-20/dtec_txt/065-2016-03-05",
                          r"/home/vadymskipa/PhD_student/data/Europe_36-70__-10-20/dtec_txt/066-2016-03-06",
                          r"/home/vadymskipa/PhD_student/data/Europe_36-70__-10-20/dtec_txt/067-2016-03-07",
                          r"/home/vadymskipa/PhD_student/data/Europe_36-70__-10-20/dtec_txt/068-2016-03-08",
                          r"/home/vadymskipa/PhD_student/data/Europe_36-70__-10-20/dtec_txt/069-2016-03-09",
                          r"/home/vadymskipa/PhD_student/data/Europe_36-70__-10-20/dtec_txt/070-2016-03-10",
                          r"/home/vadymskipa/PhD_student/data/Europe_36-70__-10-20/dtec_txt/071-2016-03-11"]
    site_directory = r"/home/vadymskipa/PhD_student/data/site_hdf/"
    site_paths = wsite.get_site_file_paths_from_directory(site_directory)
    for source in source_directories:
        date = get_date_from_date_directory_name(os.path.basename(source))
        site_file = None
        for site_path in site_paths:
            if wsite.check_site_file_name_by_date(os.path.basename(site_path), date):
                site_file = site_path
                break
        save_sites_txt_for_site_directories(source, site_file, source)


def main2024jun5():
    los_list = [r"/home/vadymskipa/HDD2/big_los_hdf/los_20160830.001.h5.hdf5",
                r"/home/vadymskipa/HDD2/big_los_hdf/los_20160831.001.h5.hdf5",
                r"/home/vadymskipa/HDD2/big_los_hdf/los_20160901.001.h5.hdf5",
                r"/home/vadymskipa/HDD2/big_los_hdf/los_20160902.001.h5.hdf5",
                r"/home/vadymskipa/HDD2/big_los_hdf/los_20160903.001.h5.hdf5",
                r"/home/vadymskipa/HDD2/big_los_hdf/los_20160904.001.h5.hdf5"]
    save = r"/home/vadymskipa/PhD_student/data/site_hdf/"
    for los in los_list:
        wlos.create_site_file_from_los(los, save)


def main2024jun6():
    lats = (36, 70)
    lons = (-10, 20)
    min_date = dt.datetime(year=2016, month=8, day=30, tzinfo=dt.timezone.utc)
    max_date = dt.datetime(year=2016, month=9, day=4, tzinfo=dt.timezone.utc)
    source_site_hdf_directory_path = r"/home/vadymskipa/PhD_student/data/site_hdf/"
    source_los_hdf_directory = r"/home/vadymskipa/PhD_student/data/Europe_los_hdf/"
    los_hdfs_in_directory = os.listdir(source_los_hdf_directory)
    list_los_hdf_paths = []
    for los_hdf in los_hdfs_in_directory:
        temp_date = wlos.get_date_by_los_file_name(los_hdf)
        if temp_date < min_date:
            continue
        if temp_date > max_date:
            continue
        list_los_hdf_paths.append(os.path.join(source_los_hdf_directory, los_hdf))
    save_path = r"/home/vadymskipa/PhD_student/data/Europe_36-70__-10-20/"
    save_los_tec_txt_file_for_region_from_los_files(list_los_hdf_paths, source_site_hdf_directory_path, save_path,
                                                    lats=lats, lons=lons, multiprocessing=True, gps_only=False)


def main2024jun7():
    source_path = r"/home/vadymskipa/PhD_student/data/Europe_36-70__-10-20/los_tec_txt/"
    save_path = r"/home/vadymskipa/PhD_student/data/Europe_36-70__-10-20/dtec_txt/"
    params = LIST_PARAMS1
    min_date = dt.datetime(year=2016, month=8, day=30, tzinfo=dt.timezone.utc)
    max_date = dt.datetime(year=2016, month=9, day=4, tzinfo=dt.timezone.utc)
    get_dtec_from_los_tec_txt_for_lots_of_sites(source_path, save_path, params, min_date=min_date, max_date=max_date)


def main2024jun8():
    source_directories = [r"/home/vadymskipa/PhD_student/data/Europe_36-70__-10-20/dtec_txt/243-2016-08-30",
                          r"/home/vadymskipa/PhD_student/data/Europe_36-70__-10-20/dtec_txt/244-2016-08-31",
                          r"/home/vadymskipa/PhD_student/data/Europe_36-70__-10-20/dtec_txt/245-2016-09-01",
                          r"/home/vadymskipa/PhD_student/data/Europe_36-70__-10-20/dtec_txt/246-2016-09-02",
                          r"/home/vadymskipa/PhD_student/data/Europe_36-70__-10-20/dtec_txt/247-2016-09-03",
                          r"/home/vadymskipa/PhD_student/data/Europe_36-70__-10-20/dtec_txt/248-2016-09-04"]
    site_directory = r"/home/vadymskipa/PhD_student/data/site_hdf/"
    site_paths = wsite.get_site_file_paths_from_directory(site_directory)
    for source in source_directories:
        date = get_date_from_date_directory_name(os.path.basename(source))
        site_file = None
        for site_path in site_paths:
            if wsite.check_site_file_name_by_date(os.path.basename(site_path), date):
                site_file = site_path
                break
        save_sites_txt_for_site_directories(source, site_file, source)


def main2024jul1():
    source_directories = [r"/home/vadymskipa/PhD_student/data/Europe_30-80__-10-50/dvtec_altitude/dvtec_txt_alt_400/268-2020-09-24"]
    site_directory = r"/home/vadymskipa/PhD_student/data/site_hdf/"
    site_paths = wsite.get_site_file_paths_from_directory(site_directory)
    for source in source_directories:
        date = get_date_from_date_directory_name(os.path.basename(source))
        site_file = None
        for site_path in site_paths:
            if wsite.check_site_file_name_by_date(os.path.basename(site_path), date):
                site_file = site_path
                break
        save_sites_txt_for_site_directories(source, site_file, source)


def get_gps_directions_for_some_sites(sites, source_directory):
    sites_file_path = os.path.join(source_directory, "Sites.txt")
    date = get_date_from_date_directory_name(os.path.basename(source_directory))
    sites_dataframe = read_sites_txt(sites_file_path)
    dataframe = pd.DataFrame()
    for site in sites:
        site_directory = os.path.join(source_directory, site, "Window_3600_Seconds")
        site_dataframe = pd.DataFrame()
        gps_id_list = os.listdir(site_directory)
        for sat_id in gps_id_list:
            gps_id_file = os.path.join(site_directory, sat_id)
            gps_id_dataframe = read_dtec_file(gps_id_file)
            gps_id_dataframe = add_timestamp_column_to_df(gps_id_dataframe, date)
            gps_id_dataframe = add_datetime_column_to_df(gps_id_dataframe)
            gps_id_dataframe.insert(0, "sat_id", sat_id[:3])
            site_dataframe = pd.concat([site_dataframe, gps_id_dataframe], ignore_index=True)
        gdlonr = sites_dataframe.loc[sites_dataframe.loc[:, "gps_site"] == site].iloc[0].loc["gdlonr"]
        gdlatr = sites_dataframe.loc[sites_dataframe.loc[:, "gps_site"] == site].iloc[0].loc["gdlatr"]
        site_dataframe.insert(0, "gps_site", site)
        site_dataframe.insert(0, "gdlonr", gdlonr)
        site_dataframe.insert(0, "gdlatr", gdlatr)
        dataframe = pd.concat([dataframe, site_dataframe], ignore_index=True)
    return dataframe


def main2024jul2():
    main2024apr11()


def main2024jul3():
    deviation = (3, 2)
    places = [(15, 60), (15, 48), (-5, 40), (35, 40)]
    source_directory = r"/home/vadymskipa/PhD_student/data/Europe_30-80__-10-50/dvtec_altitude1/dvtec_txt_alt_400/268-2020-09-24"
    sites_file_path = os.path.join(source_directory, "Sites.txt")
    sites_dataframe = read_sites_txt(sites_file_path)
    correct_sites_list = []
    for lon, lat in places:
        temp_dataframe = sites_dataframe.loc[sites_dataframe.loc[:, "gdlonr"] < lon + deviation[0]]
        temp_dataframe = temp_dataframe.loc[sites_dataframe.loc[:, "gdlonr"] > lon - deviation[0]]
        temp_dataframe = temp_dataframe.loc[sites_dataframe.loc[:, "gdlatr"] < lat + deviation[1]]
        temp_dataframe = temp_dataframe.loc[sites_dataframe.loc[:, "gdlatr"] > lat - deviation[1]]
        correct_sites_list.append(temp_dataframe.iloc[0].loc["gps_site"])
    dataframe = get_gps_directions_for_some_sites(correct_sites_list, source_directory)
    return dataframe


def main2024jul4():
    source_directories = [r"/home/vadymskipa/PhD_student/data/Europe_30-80__-10-50/dvtec_altitude1/dvtec_txt_alt_400/268-2020-09-24"]
    site_directory = r"/home/vadymskipa/PhD_student/data/site_hdf/"
    site_paths = wsite.get_site_file_paths_from_directory(site_directory)
    for source in source_directories:
        date = get_date_from_date_directory_name(os.path.basename(source))
        site_file = None
        for site_path in site_paths:
            if wsite.check_site_file_name_by_date(os.path.basename(site_path), date):
                site_file = site_path
                break
        save_sites_txt_for_site_directories(source, site_file, source)


def main2024aug1():
    source_directory = r"/home/vadymskipa/PhD_student/data/Europe_30-80__-10-50/dvtec_altitude1"
    save_directory = r"/home/vadymskipa/PhD_student/data/Europe_30-80__-10-50/small_region_data1"

    date = dt.datetime(year=2020, month=9, day=24, tzinfo=dt.timezone.utc)
    # region_list = [((13, 16), (58, 60))]
    # region_list = [((12, 15), (42, 44))]
    # region_list = [((7, 12), (44, 47))]
    region_list = [((7, 13), (46, 50))]


    list_of_altitudes = [200, 250, 300, 350, 400, 450]
    for altitude in list_of_altitudes:
        altitude_name = f"dvtec_txt_alt_{altitude}"
        source_altitude_directory_1 = get_directory_path(source_directory, altitude_name)
        source_date_dierectory_1 = os.path.join(source_altitude_directory_1, "268-2020-09-24")
        for lons, lats in region_list:
            # save_region_name = f"lats{lats[0]}{lats[1]}_lons{lons[0]}{lons[1]}"
            save_region_name = f"lats{lats[0]}{lats[1]}_lons{lons[0]}{lons[1]}!"

            save_region_directory_1 = get_directory_path(save_directory, save_region_name)
            for i in range(32):
                sat_id = f"G{i+1}"
                print(f"start_{sat_id}", dt.datetime.now())
                sat_id_file = f"{sat_id}.txt"
                save_sat_id_directory_2 = get_directory_path(save_region_directory_1, sat_id)
                sat_id_dataframe = pd.DataFrame()
                for site in os.listdir(source_date_dierectory_1):
                    if os.path.isdir(os.path.join(source_date_dierectory_1, site)):
                        source_site_directory_2 = os.path.join(source_date_dierectory_1, site)
                        # source_window_directory_3 = os.path.join(source_site_directory_2, "Window_3600_Seconds")
                        source_window_directory_3 = os.path.join(source_site_directory_2, "Window_7200_Seconds")

                        existing_sat_id_files = os.listdir(source_window_directory_3)
                        if sat_id_file in existing_sat_id_files:
                            temp_dataframe = read_dtec_file(os.path.join(source_window_directory_3, sat_id_file))
                            temp_dataframe = add_timestamp_column_to_df(temp_dataframe, date)
                            temp_dataframe = add_datetime_column_to_df(temp_dataframe)
                            temp_dataframe.insert(0, "gps_site", site)
                            sat_id_dataframe = pd.concat([sat_id_dataframe, temp_dataframe], ignore_index=True)
                    else:
                        continue
                if len(sat_id_dataframe) > 0:
                    save_sat_id_dataframe_3 = os.path.join(save_sat_id_directory_2, f"{sat_id}_dataframe_alt_{altitude}.hdf5")
                    region_sat_id_dataframe: pd.DataFrame = sat_id_dataframe.loc[sat_id_dataframe.loc[:, "gdlon"] >= lons[0]]
                    region_sat_id_dataframe = region_sat_id_dataframe.loc[region_sat_id_dataframe.loc[:, "gdlon"] <= lons[1]]
                    region_sat_id_dataframe = region_sat_id_dataframe.loc[region_sat_id_dataframe.loc[:, "gdlat"] >= lats[0]]
                    region_sat_id_dataframe = region_sat_id_dataframe.loc[region_sat_id_dataframe.loc[:, "gdlat"] <= lats[1]]
                    if len(region_sat_id_dataframe) > 0:
                        region_sat_id_dataframe.to_hdf(save_sat_id_dataframe_3, key="df")
                print(f"end_{sat_id}", dt.datetime.now())


def main2024aug2():
    source_directory = r"/home/vadymskipa/PhD_student/data/Europe_30-80__-10-50/dvtec_altitude1"
    save_directory = r"/home/vadymskipa/PhD_student/data/Europe_30-80__-10-50/small_region_data1"

    date = dt.datetime(year=2020, month=9, day=24, tzinfo=dt.timezone.utc)
    # region_list = [((13, 16), (58, 60))]
    # region_list = [((12, 15), (42, 44))]
    # region_list = [((7, 12), (44, 47))]
    region_list = [((7, 13), (46, 50))]


    list_of_altitudes = [200, 250, 300, 350, 400, 450]
    for altitude in list_of_altitudes:
        altitude_name = f"dvtec_txt_alt_{altitude}"
        source_altitude_directory_1 = get_directory_path(source_directory, altitude_name)
        source_date_directory_1 = os.path.join(source_altitude_directory_1, "268-2020-09-24")
        for lons, lats in region_list:
            # save_region_name = f"lats{lats[0]}{lats[1]}_lons{lons[0]}{lons[1]}"
            save_region_name = f"lats{lats[0]}{lats[1]}_lons{lons[0]}{lons[1]}!"
            save_region_directory_1 = get_directory_path(save_directory, save_region_name)
            for i in range(32):
                sat_id = f"G{i+1}"
                print(f"start_{sat_id}", dt.datetime.now())
                sat_id_file = f"{sat_id}.txt"
                save_sat_id_directory_2 = get_directory_path(save_region_directory_1, sat_id)
                sat_id_dataframe = pd.DataFrame()
                for site in os.listdir(source_date_directory_1):
                    if os.path.isdir(os.path.join(source_date_directory_1, site)):
                        # source_window_directory_3 = os.path.join(source_site_directory_2, "Window_3600_Seconds")
                        source_window_directory_3 = os.path.join(source_site_directory_2, "Window_7200_Seconds")
                        existing_sat_id_files = os.listdir(source_window_directory_3)
                        if sat_id_file in existing_sat_id_files:
                            temp_dataframe = read_dtec_file(os.path.join(source_window_directory_3, sat_id_file))
                            temp_dataframe = add_timestamp_column_to_df(temp_dataframe, date)
                            temp_dataframe = add_datetime_column_to_df(temp_dataframe)
                            temp_dataframe.insert(0, "gps_site", site)
                            sat_id_dataframe = pd.concat([sat_id_dataframe, temp_dataframe], ignore_index=True)
                    else:
                        continue
                if len(sat_id_dataframe) > 0:
                    save_sat_id_dataframe_3 = os.path.join(save_sat_id_directory_2, f"{sat_id}_dataframe_alt_{altitude}.csv")
                    region_sat_id_dataframe: pd.DataFrame = sat_id_dataframe.loc[sat_id_dataframe.loc[:, "gdlon"] >= lons[0]]
                    region_sat_id_dataframe = region_sat_id_dataframe.loc[region_sat_id_dataframe.loc[:, "gdlon"] <= lons[1]]
                    region_sat_id_dataframe = region_sat_id_dataframe.loc[region_sat_id_dataframe.loc[:, "gdlat"] >= lats[0]]
                    region_sat_id_dataframe = region_sat_id_dataframe.loc[region_sat_id_dataframe.loc[:, "gdlat"] <= lats[1]]
                    if len(region_sat_id_dataframe) > 0:
                        region_sat_id_dataframe.to_csv(save_sat_id_dataframe_3)
                print(f"end_{sat_id}", dt.datetime.now())



def main2024jul():
    main2024jul3()


def main2024aug3():
    # source_directory = r"/home/vadymskipa/PhD_student/data/Europe_30-80__-10-50/small_region_data1/lats5860_lons1316"
    # source_directory = r"/home/vadymskipa/PhD_student/data/Europe_30-80__-10-50/small_region_data1/lats4244_lons1215"
    # source_directory = r"/home/vadymskipa/PhD_student/data/Europe_30-80__-10-50/small_region_data1/lats4447_lons712"
    # source_directory = r"/home/vadymskipa/PhD_student/data/Europe_30-80__-10-50/small_region_data1/lats4650_lons713"
    source_directory = r"/home/vadymskipa/PhD_student/data/Europe_30-80__-10-50/small_region_data1/lats4650_lons713!"


    date = dt.datetime(year=2020, month=9, day=24, tzinfo=dt.timezone.utc)
    # region_list = [((13, 16), (58, 60))]
    list_of_altitudes = [200, 250, 300, 350, 400, 450]
    list_attr = ["timestamp"]
    for alt in list_of_altitudes:
        list_attr.append(f"alt_{alt}")
    # for altitude in list_of_altitudes:
    #     altitude_name = f"dvtec_txt_alt_{altitude}"
    #     source_altitude_directory_1 = get_directory_path(source_directory, altitude_name)
    min_date = dt.datetime(year=2020, month=9, day=24, hour=9, minute=30, tzinfo=dt.timezone.utc)
    max_date = dt.datetime(year=2020, month=9, day=24, hour=12, minute=0, tzinfo=dt.timezone.utc)
    min_date_ts = min_date.timestamp()
    max_date_ts = max_date.timestamp()
    period_of_averaging = 300
    timestamp_list = []
    for timestamp in range(int(min_date_ts), int(max_date_ts), period_of_averaging):
        timestamp_list.append(timestamp)
    for i in range(32):
        sat_id = f"G{i + 1}"
        sat_dataframe: pd.DataFrame = pd.DataFrame()
        source_sat_directory = get_directory_path(source_directory, sat_id)
        if len(os.listdir(source_sat_directory)) == 0:
            continue
        save_dataframe = pd.DataFrame(data={"timestamp": timestamp_list}, columns=list_attr)
        for altitude in list_of_altitudes:
            save_sat_id_file = os.path.join(source_sat_directory, f"{sat_id}_dataframe_alt_{altitude}.hdf5")
            try:
                temp_dataframe = pd.read_hdf(save_sat_id_file, "df")
            except Exception as err:
                print(err)
                continue
            sat_dataframe = pd.concat([sat_dataframe, temp_dataframe], ignore_index=True)
            for timestamp in timestamp_list:
                dvtec = sat_dataframe.loc[sat_dataframe.loc[:, "timestamp"] == timestamp].loc[:, "dtec"].mean()
                index = save_dataframe.loc[save_dataframe.loc[:, "timestamp"] == timestamp].index[0]
                save_dataframe.loc[index, f"alt_{altitude}"] = dvtec
        save_file = os.path.join(source_sat_directory, f"{sat_id}_mean.csv")
        save_dataframe = add_datetime_column_to_df(save_dataframe)
        save_dataframe.to_csv(save_file)



def main2024aug4():
    lats = (36, 70)
    lons = (-10, 20)
    min_date = dt.datetime(year=2024, month=5, day=9, tzinfo=dt.timezone.utc)
    max_date = dt.datetime(year=2024, month=5, day=13, tzinfo=dt.timezone.utc)
    source_site_hdf_directory_path = r"/home/vadymskipa/PhD_student/data/site_hdf/"
    source_los_hdf_directory = r"/home/vadymskipa/PhD_student/data/Europe_los_hdf/"
    los_hdfs_in_directory = os.listdir(source_los_hdf_directory)
    list_los_hdf_paths = []
    for los_hdf in los_hdfs_in_directory:
        temp_date = wlos.get_date_by_los_file_name(los_hdf)
        if temp_date < min_date:
            continue
        if temp_date > max_date:
            continue
        list_los_hdf_paths.append(os.path.join(source_los_hdf_directory, los_hdf))
    save_path = r"/home/vadymskipa/PhD_student/data/Europe_36-70__-10-20/"
    save_los_tec_txt_file_for_region_from_los_files(list_los_hdf_paths, source_site_hdf_directory_path, save_path,
                                                    lats=lats, lons=lons, multiprocessing=True, gps_only=True)


def main2024aug5():
    source_path = r"/home/vadymskipa/PhD_student/data/Europe_36-70__-10-20/los_tec_txt/"
    save_path = r"/home/vadymskipa/PhD_student/data/Europe_36-70__-10-20/dtec_txt/"
    params = LIST_PARAMS1
    min_date = dt.datetime(year=2024, month=5, day=9, tzinfo=dt.timezone.utc)
    max_date = dt.datetime(year=2024, month=5, day=13, tzinfo=dt.timezone.utc)
    get_dtec_from_los_tec_txt_for_lots_of_sites(source_path, save_path, params, min_date=min_date, max_date=max_date)




def main2024aug():
    # main2024aug1()
    # main2024aug2()
    # main2024aug3()
    main2024aug4()
    main2024aug5()


def main2024sep1():
    save_dir = r"/home/vadymskipa/PhD_student/data/Europe_36-70__-10-20/dvtec_csv"
    source_dirs = [r"/home/vadymskipa/PhD_student/data/Europe_36-70__-10-20/dtec_txt/130-2024-05-09",
                   r"/home/vadymskipa/PhD_student/data/Europe_36-70__-10-20/dtec_txt/131-2024-05-10",
                   r"/home/vadymskipa/PhD_student/data/Europe_36-70__-10-20/dtec_txt/132-2024-05-11",
                   r"/home/vadymskipa/PhD_student/data/Europe_36-70__-10-20/dtec_txt/133-2024-05-12",
                   r"/home/vadymskipa/PhD_student/data/Europe_36-70__-10-20/dtec_txt/134-2024-05-13"]
    window = "Window_3600_Seconds"
    for source_dir in source_dirs:
        date = get_date_from_date_directory_name(os.path.basename(source_dir))
        save_dir_1 = get_directory_path(save_dir, get_date_str(date))
        entries = os.listdir(source_dir)
        sites = []
        for entry in entries:
            if os.path.isdir(os.path.join(source_dir, entry)):
                sites.append(entry)
        for site in sites:
            print(site, dt.datetime.now())
            source_dir_1 = os.path.join(source_dir, site, window)
            sats = [text[:3] for text in os.listdir(source_dir_1)]
            sat_data_list = []
            for sat in sats:
                read_path = os.path.join(source_dir_1, sat + ".txt")
                sat_dataframe = read_dtec_file(read_path)
                sat_data_list.append((sat, sat_dataframe))
            pool = multipool.Pool(7)
            outcome_list = []
            for tup in pool.imap(main2024sep1_p, sat_data_list):
                outcome_list.append(tup)
            save_path_2 = get_directory_path(save_dir_1, site)
            save_path_3 = get_directory_path(save_path_2, window)
            for sat, dataframe in outcome_list:
                save_path_4 = os.path.join(save_path_3, sat + ".csv")
                dataframe.to_csv(save_path_4)


def main2024sep1_p(tup):
    sat, dataframe = tup
    dvtec_list = []
    for i in dataframe.index:
        dvtec = calc_vtec_2(dataframe.loc[i, "dtec"], dataframe.loc[i, "elm"])
        dvtec_list.append(dvtec)
    dataframe.insert(4, "dvtec", dvtec_list)
    return (sat, dataframe)


def calc_vtec_2(dtec, elm):
    b = get_angle(elm, 350)
    dtec = calc_vtec(dtec, elm, b)
    return dtec


def main2024sep2():
    save_dir = r"/home/vadymskipa/PhD_student/data/Europe_36-70__-10-20/dvtec_csv_5min"
    source_dirs = [r"/home/vadymskipa/PhD_student/data/Europe_36-70__-10-20/dvtec_csv/130-2024-05-09",
                   r"/home/vadymskipa/PhD_student/data/Europe_36-70__-10-20/dvtec_csv/131-2024-05-10",
                   r"/home/vadymskipa/PhD_student/data/Europe_36-70__-10-20/dvtec_csv/132-2024-05-11",
                   r"/home/vadymskipa/PhD_student/data/Europe_36-70__-10-20/dvtec_csv/133-2024-05-12",
                   r"/home/vadymskipa/PhD_student/data/Europe_36-70__-10-20/dvtec_csv/134-2024-05-13"]
    window = "Window_3600_Seconds"

    for source_dir in source_dirs:
        date = get_date_from_date_directory_name(os.path.basename(source_dir))
        save_dir_1 = get_directory_path(save_dir, get_date_str(date))
        entries = os.listdir(source_dir)
        sites = []
        for entry in entries:
            if os.path.isdir(os.path.join(source_dir, entry)):
                sites.append(entry)
        for site in sites:
            print(site, dt.datetime.now())
            source_dir_1 = os.path.join(source_dir, site, window)
            sats = [text[:3] for text in os.listdir(source_dir_1)]
            sat_data_list = []
            for sat in sats:
                read_path = os.path.join(source_dir_1, sat + ".csv")
                sat_dataframe = pd.read_csv(read_path)
                sat_data_list.append((date, sat, sat_dataframe))
            pool = multipool.Pool(3)
            outcome_list = []
            for tup in pool.imap(main2024sep2_p, sat_data_list):
                outcome_list.append(tup)
            save_path_2 = get_directory_path(save_dir_1, site)
            save_path_3 = get_directory_path(save_path_2, window)
            for sat, dataframe in outcome_list:
                save_path_4 = os.path.join(save_path_3, sat + ".csv")
                dataframe.to_csv(save_path_4)


def main2024sep2_p(tup):
    date, sat, dataframe = tup
    min_date_ts = int(date.timestamp())
    max_date_ts = min_date_ts + 24 * 3600
    period_of_averaging = 300
    timestamp_list = []
    for timestamp in range(int(min_date_ts), int(max_date_ts), period_of_averaging):
        timestamp_list.append(timestamp)
    dataframe = add_timestamp_column_to_df(dataframe, date)
    outcome_dataframe = pd.DataFrame()
    for timestamp in timestamp_list:
        temp_dataframe = dataframe.loc[dataframe.loc[:, "timestamp"] >= timestamp]
        temp_dataframe = temp_dataframe.loc[temp_dataframe.loc[:, "timestamp"] < timestamp + period_of_averaging]
        if temp_dataframe.empty:
            continue
        azm = temp_dataframe.iloc[0].loc["azm"]
        dvtec = temp_dataframe.loc[:, "dvtec"].mean()
        elm = temp_dataframe.iloc[0].loc["elm"]
        gdlat = temp_dataframe.loc[:, "gdlat"].mean()
        gdlon = temp_dataframe.loc[:, "gdlon"].mean()
        datetime = dt.datetime.fromtimestamp(timestamp, tz=dt.timezone.utc)
        hour = datetime.hour
        minute = datetime.minute
        sec= datetime.second
        temp = pd.DataFrame({"hour": hour, "min": minute, "sec": sec, "dvtec": dvtec, "elm": elm, "azm": azm,
                                 "gdlat": gdlat, "gdlon": gdlon}, index=[1])
        outcome_dataframe = pd.concat((outcome_dataframe, temp), ignore_index=True)
    return (sat, outcome_dataframe)


def main2024sep3():
    source_directories = [r"/home/vadymskipa/PhD_student/data/Europe_36-70__-10-20/dvtec_csv_5min/130-2024-05-09",
                   r"/home/vadymskipa/PhD_student/data/Europe_36-70__-10-20/dvtec_csv_5min/131-2024-05-10",
                   r"/home/vadymskipa/PhD_student/data/Europe_36-70__-10-20/dvtec_csv_5min/132-2024-05-11",
                   r"/home/vadymskipa/PhD_student/data/Europe_36-70__-10-20/dvtec_csv_5min/133-2024-05-12",
                   r"/home/vadymskipa/PhD_student/data/Europe_36-70__-10-20/dvtec_csv_5min/134-2024-05-13"]
    site_directory = r"/home/vadymskipa/PhD_student/data/site_hdf/"
    site_paths = wsite.get_site_file_paths_from_directory(site_directory)
    for source in source_directories:
        date = get_date_from_date_directory_name(os.path.basename(source))
        site_file = None
        for site_path in site_paths:
            if wsite.check_site_file_name_by_date(os.path.basename(site_path), date):
                site_file = site_path
                break
        save_sites_txt_for_site_directories(source, site_file, source)


def main2024sep4():
    source_path = r"/home/vadymskipa/PhD_student/data/123/los_txt"
    save_path = r"/home/vadymskipa/PhD_student/data/123/dtec_txt"
    params = LIST_PARAMS1
    min_date = dt.datetime(year=2024, month=5, day=9, tzinfo=dt.timezone.utc)
    max_date = dt.datetime(year=2024, month=5, day=13, tzinfo=dt.timezone.utc)
    get_dtec_from_los_tec_txt_for_lots_of_sites(source_path, save_path, params, min_date=min_date, max_date=max_date)


def main2024sep5():
    save_dir = r"/home/vadymskipa/PhD_student/data/123/dvtec_csv"
    source_dirs = [r"/home/vadymskipa/PhD_student/data/123/dtec_txt/130-2024-05-09",
                   r"/home/vadymskipa/PhD_student/data/123/dtec_txt/131-2024-05-10",
                   r"/home/vadymskipa/PhD_student/data/123/dtec_txt/132-2024-05-11",
                   r"/home/vadymskipa/PhD_student/data/123/dtec_txt/133-2024-05-12",
                   r"/home/vadymskipa/PhD_student/data/123/dtec_txt/134-2024-05-13"]
    window = "Window_7200_Seconds"
    for source_dir in source_dirs:
        date = get_date_from_date_directory_name(os.path.basename(source_dir))
        save_dir_1 = get_directory_path(save_dir, get_date_str(date))
        entries = os.listdir(source_dir)
        sites = []
        for entry in entries:
            if os.path.isdir(os.path.join(source_dir, entry)):
                sites.append(entry)
        for site in sites:
            print(site, dt.datetime.now())
            source_dir_1 = os.path.join(source_dir, site, window)
            sats = [text[:3] for text in os.listdir(source_dir_1)]
            sat_data_list = []
            for sat in sats:
                read_path = os.path.join(source_dir_1, sat + ".txt")
                sat_dataframe = read_dtec_file(read_path)
                sat_data_list.append((sat, sat_dataframe))
            pool = multipool.Pool(7)
            outcome_list = []
            for tup in pool.imap(main2024sep1_p, sat_data_list):
                outcome_list.append(tup)
            save_path_2 = get_directory_path(save_dir_1, site)
            save_path_3 = get_directory_path(save_path_2, window)
            for sat, dataframe in outcome_list:
                save_path_4 = os.path.join(save_path_3, sat + ".csv")
                dataframe.to_csv(save_path_4)


def main2024sep6():
    save_dir = r"/home/vadymskipa/PhD_student/data/123/dvtec_csv_5min"
    source_dirs = [r"/home/vadymskipa/PhD_student/data/123/dvtec_csv/130-2024-05-09",
                   r"/home/vadymskipa/PhD_student/data/123/dvtec_csv/131-2024-05-10",
                   r"/home/vadymskipa/PhD_student/data/123/dvtec_csv/132-2024-05-11",
                   r"/home/vadymskipa/PhD_student/data/123/dvtec_csv/133-2024-05-12",
                   r"/home/vadymskipa/PhD_student/data/123/dvtec_csv/134-2024-05-13"]
    window = "Window_3600_Seconds"

    for source_dir in source_dirs:
        date = get_date_from_date_directory_name(os.path.basename(source_dir))
        save_dir_1 = get_directory_path(save_dir, get_date_str(date))
        entries = os.listdir(source_dir)
        sites = []
        for entry in entries:
            if os.path.isdir(os.path.join(source_dir, entry)):
                sites.append(entry)
        for site in sites:
            print(site, dt.datetime.now())
            source_dir_1 = os.path.join(source_dir, site, window)
            sats = [text[:3] for text in os.listdir(source_dir_1)]
            sat_data_list = []
            for sat in sats:
                read_path = os.path.join(source_dir_1, sat + ".csv")
                sat_dataframe = pd.read_csv(read_path)
                sat_data_list.append((date, sat, sat_dataframe))
            pool = multipool.Pool(3)
            outcome_list = []
            for tup in pool.imap(main2024sep2_p, sat_data_list):
                outcome_list.append(tup)
            save_path_2 = get_directory_path(save_dir_1, site)
            save_path_3 = get_directory_path(save_path_2, window)
            for sat, dataframe in outcome_list:
                save_path_4 = os.path.join(save_path_3, sat + ".csv")
                dataframe.to_csv(save_path_4)


def main2024sep7():
    save_dir = r"/home/vadymskipa/PhD_student/data/123/dvtec_site_sat_observation"
    source_dirs = [r"/home/vadymskipa/PhD_student/data/123/dvtec_csv/130-2024-05-09",
                   r"/home/vadymskipa/PhD_student/data/123/dvtec_csv/131-2024-05-10",
                   r"/home/vadymskipa/PhD_student/data/123/dvtec_csv/132-2024-05-11",
                   r"/home/vadymskipa/PhD_student/data/123/dvtec_csv/133-2024-05-12",
                   r"/home/vadymskipa/PhD_student/data/123/dvtec_csv/134-2024-05-13"]
    window = "Window_3600_Seconds"
    sites = ["BASE", "Bas3"]
    sats = [f"G{i:0=2}" for i in range(1, 33)]
    sat_data_list = []
    for site in sites:
        for sat in sats:
            sat_dataframe = pd.DataFrame()
            for source_dir in source_dirs:
                date = get_date_from_date_directory_name(os.path.basename(source_dir))
                read_path = os.path.join(source_dir, site, window, sat + ".csv")
                try:
                    temp_sat_dataframe = pd.read_csv(read_path)
                    temp_sat_dataframe = add_timestamp_column_to_df(temp_sat_dataframe, date)
                    sat_dataframe = pd.concat([sat_dataframe, temp_sat_dataframe], ignore_index=True)
                except Exception as er:
                    print(er)
            if sat_dataframe.empty:
                continue
            timestamp_series: pd.Series = sat_dataframe.loc[:, "timestamp"]
            timestamp_series_shifted = timestamp_series.shift()
            timestamp_diff = timestamp_series - timestamp_series_shifted
            breaking_index = timestamp_diff.loc[timestamp_diff >= 60].index
            list_obs_dataframe = []
            start_index = 0
            if len(breaking_index) == 0:
                list_obs_dataframe.append(sat_dataframe)
            else:
                for index in breaking_index:
                    list_obs_dataframe.append(sat_dataframe.loc[start_index:index-1])
                    start_index = index
                list_obs_dataframe.append(sat_dataframe.loc[start_index:])
            for dataframe in list_obs_dataframe:
                save_dir_1 = get_directory_path(save_dir, window)
                save_dir_2 = get_directory_path(save_dir_1, sat)
                save_dir_3 = get_directory_path(save_dir_2, site)
                start_date = dt.datetime.fromtimestamp(dataframe.iloc[0].loc["timestamp"], tz=dt.timezone.utc)
                end_date = dt.datetime.fromtimestamp(dataframe.iloc[-1].loc["timestamp"], tz=dt.timezone.utc)
                filename = (f"{sat}_{site}_{start_date.year:0=4}{start_date.month:0=2}{start_date.day:0=2}_"
                            f"{start_date.hour:0=2}{start_date.minute:0=2}{end_date.hour:0=2}{end_date.minute:0=2}.csv")
                dataframe.to_csv(os.path.join(save_dir_3, filename))
                print(filename)


def create_blocks_table(lats, lons, lat_tick, lon_tick):
    list_blocks = []
    list_lats = [round(lats[0] + lat_tick * i, 2) for i in range(math.ceil((lats[1] - lats[0]) / lat_tick))]
    list_lons = [round(lons[0] + lon_tick * i, 2) for i in range(math.ceil((lons[1] - lons[0]) / lon_tick))]
    for lat in list_lats:
        list_blocks.append([(lat, lon) for lon in list_lons])
    return list_blocks




def main2024sep8():
    save_dir = r"/home/vadymskipa/PhD_student/data/Europe_36-70__-10-20/dvtec_csv_5min/blocks"
    source_dir = r"/home/vadymskipa/PhD_student/data/Europe_36-70__-10-20/dvtec_csv_5min"
    window = "Window_3600_Seconds"
    sats = [f"G{i:0=2}" for i in range(1, 33)]
    lats = (36, 70)
    lons = (-10, 20)
    lat_tick = 0.15
    lon_tick = 0.15
    blocks_table= create_blocks_table(lats=lats, lons=lons, lat_tick=lat_tick, lon_tick=lon_tick)
    min_elm = 30



    entries = os.listdir(source_dir)
    list_date_dir = [dirname for dirname in entries if
                       (len(dirname) == 14 and os.path.isdir(os.path.join(source_dir, dirname)))]
    for date_dir in list_date_dir:
        date = get_date_from_date_directory_name(date_dir)
        source_path_1 = os.path.join(source_dir, date_dir)
        save_path_1 = get_directory_path(save_dir, date_dir)
        save_path_2 = get_directory_path(save_path_1, window)
        for sat in sats:
            save_path_3 = get_directory_path(save_path_2, sat)
            dataframe_1 = pd.DataFrame()
            entries_1 = os.listdir(source_path_1)
            list_site_dir = [dirname for dirname in entries_1 if os.path.isdir(os.path.join(source_path_1, dirname))]
            print(f"--- start reading {sat}", dt.datetime.now())
            for site in list_site_dir:
                try:
                    source_path_2 = os.path.join(source_path_1, site, window, f"{sat}.csv")
                    sat_dataframe: pd.Dataframe = pd.read_csv(source_path_2)
                    sat_dataframe.insert(1, "gps_site", site)
                    dataframe_1 = pd.concat([dataframe_1, sat_dataframe], ignore_index=True)
                except Exception as er:
                    print(er, dt.datetime.now())
            print(f"--- end reading {sat}  ", dt.datetime.now())
            dataframe_1 = add_timestamp_column_to_df(dataframe_1, date)
            dataframe_1 = dataframe_1.loc[dataframe_1.loc[:, "elm"] >= min_elm]
            list_timestamp = np.unique(dataframe_1.loc[:, "timestamp"])
            for timestamp in list_timestamp:
                d = dt.datetime.fromtimestamp(timestamp, tz=dt.timezone.utc)
                hour = d.hour
                minute = d.minute
                # dataframe_2 = pd.DataFrame()
                timestamp_dataframe_1 = dataframe_1.loc[dataframe_1.loc[:, "timestamp"] == timestamp]
                if timestamp_dataframe_1.empty:
                    continue
                timestamp_dataframe_1 = timestamp_dataframe_1.loc[timestamp_dataframe_1.loc[:, "gdlat"] >= lats[0]]
                timestamp_dataframe_1 = timestamp_dataframe_1.loc[timestamp_dataframe_1.loc[:, "gdlat"] < lats[1]]
                timestamp_dataframe_1 = timestamp_dataframe_1.loc[timestamp_dataframe_1.loc[:, "gdlon"] >= lons[0]]
                timestamp_dataframe_1 = timestamp_dataframe_1.loc[timestamp_dataframe_1.loc[:, "gdlon"] < lons[1]]
                if timestamp_dataframe_1.empty:
                    continue
                list_sites = []
                list_nos = []
                list_dvtec = []
                list_lats = []
                list_lons = []
                list_skip_lon_i = []
                for lon_i in range(len(blocks_table[0]) - 1):
                    lon_min = blocks_table[0][lon_i][1]
                    lon_max = blocks_table[0][lon_i + 1][1]
                    temp2_dataframe_1 = timestamp_dataframe_1.loc[timestamp_dataframe_1.loc[:, "gdlon"] >= lon_min]
                    temp2_dataframe_1 = temp2_dataframe_1.loc[temp2_dataframe_1.loc[:, "gdlon"] < lon_max]
                    if temp2_dataframe_1.empty:
                        list_skip_lon_i.append(lon_i)
                for lat_i in range(len(blocks_table) - 1):
                    lat_min = blocks_table[lat_i][0][0]
                    lat_max = blocks_table[lat_i + 1][0][0]
                    temp_dataframe_1 = timestamp_dataframe_1.loc[timestamp_dataframe_1.loc[:, "gdlat"] >= lat_min]
                    temp_dataframe_1 = temp_dataframe_1.loc[temp_dataframe_1.loc[:, "gdlat"] < lat_max]
                    if temp_dataframe_1.empty:
                        continue
                    for lon_i in range(len(blocks_table[lat_i]) - 1):
                        if lon_i in list_skip_lon_i:
                            continue
                        lon_min = blocks_table[lat_i][lon_i][1]
                        lon_max = blocks_table[lat_i][lon_i + 1][1]
                        temp2_dataframe_1 = temp_dataframe_1.loc[temp_dataframe_1.loc[:, "gdlon"] >= lon_min]
                        temp2_dataframe_1 = temp2_dataframe_1.loc[temp2_dataframe_1.loc[:, "gdlon"] < lon_max]
                        if temp2_dataframe_1.empty:
                            continue
                        sites = tuple(temp2_dataframe_1.loc[:, "gps_site"])
                        nos = len(sites)
                        dvtec = temp2_dataframe_1.loc[:, "dvtec"].mean()
                        # temp_dataframe_2 = pd.DataFrame({"gdlat": lat_min, "gdlon": lon_min, "dvtec": dvtec, "nos": nos,
                        #                                  "sites": sites})
                        list_sites.append(sites)
                        list_nos.append(nos)
                        list_dvtec.append(dvtec)
                        list_lats.append(lat_min)
                        list_lons.append(lon_min)
                        # dataframe_2 = pd.concat([dataframe_2, temp_dataframe_2], ignore_index=True)
                dataframe_2 = pd.DataFrame({"gdlat": list_lats, "gdlon": list_lons, "dvtec": list_dvtec,
                                            "nos": list_nos, "sites": list_sites})
                save_path_4 = os.path.join(save_path_3, f"{hour:0=2}{minute:0=2}.csv")
                dataframe_2.to_csv(save_path_4)
                print(date_dir, window, sat, f"{hour:0=2}{minute:0=2}", dt.datetime.now())



def main2024sep8_mp():
    save_dir = r"/home/vadymskipa/PhD_student/data/Europe_36-70__-10-20/dvtec_csv_5min/blocks"
    source_dir = r"/home/vadymskipa/PhD_student/data/Europe_36-70__-10-20/dvtec_csv_5min"
    window = "Window_7200_Seconds"
    sats = [f"G{i:0=2}" for i in range(1, 33)]
    lats = (36, 70)
    lons = (-10, 20)
    lat_tick = 0.15
    lon_tick = 0.15
    blocks_table= create_blocks_table(lats=lats, lons=lons, lat_tick=lat_tick, lon_tick=lon_tick)
    min_elm = 30



    entries = os.listdir(source_dir)
    list_date_dir = [dirname for dirname in entries if
                       (len(dirname) == 14 and os.path.isdir(os.path.join(source_dir, dirname)))]
    for date_dir in list_date_dir:
        date = get_date_from_date_directory_name(date_dir)
        # if date <= dt.datetime(year=2024, month=5, day=9, tzinfo=dt.timezone.utc):
        #     continue
        source_path_1 = os.path.join(source_dir, date_dir)
        save_path_1 = get_directory_path(save_dir, date_dir)
        save_path_2 = get_directory_path(save_path_1, window)
        for sat in sats:
            save_path_3 = get_directory_path(save_path_2, sat)
            dataframe_1 = pd.DataFrame()
            entries_1 = os.listdir(source_path_1)
            list_site_dir = [dirname for dirname in entries_1 if os.path.isdir(os.path.join(source_path_1, dirname))]
            print(f"--- start reading {sat}", dt.datetime.now())
            for site in list_site_dir:
                try:
                    source_path_2 = os.path.join(source_path_1, site, window, f"{sat}.csv")
                    sat_dataframe: pd.Dataframe = pd.read_csv(source_path_2)
                    sat_dataframe.insert(1, "gps_site", site)
                    dataframe_1 = pd.concat([dataframe_1, sat_dataframe], ignore_index=True)
                except Exception as er:
                    print(er, dt.datetime.now())
            print(f"--- end reading {sat}  ", dt.datetime.now())
            dataframe_1 = add_timestamp_column_to_df(dataframe_1, date)
            dataframe_1 = dataframe_1.loc[dataframe_1.loc[:, "elm"] >= min_elm]
            list_timestamp = np.unique(dataframe_1.loc[:, "timestamp"])
            input_mp_list = [(timestamp, dataframe_1.loc[dataframe_1.loc[:, "timestamp"] == timestamp]
                              , lats, lons, blocks_table) for timestamp in list_timestamp]

            pool = multipool.Pool(6)
            for dataframe_2, hour, minute in pool.imap(main2024sep8_proccess, input_mp_list):
                if type(dataframe_2) == pd.DataFrame:
                    save_path_4 = os.path.join(save_path_3, f"{hour:0=2}{minute:0=2}.csv")
                    dataframe_2.to_csv(save_path_4)
                print(date_dir, window, sat, f"{hour:0=2}{minute:0=2}", dt.datetime.now())



def main2024sep8_proccess(tup):
    timestamp, dataframe_1, lats, lons, blocks_table = tup
    d = dt.datetime.fromtimestamp(timestamp, tz=dt.timezone.utc)
    hour = d.hour
    minute = d.minute
    # dataframe_2 = pd.DataFrame()
    timestamp_dataframe_1 = dataframe_1.loc[dataframe_1.loc[:, "timestamp"] == timestamp]
    if timestamp_dataframe_1.empty:
        return None, hour, minute
    timestamp_dataframe_1 = timestamp_dataframe_1.loc[timestamp_dataframe_1.loc[:, "gdlat"] >= lats[0]]
    timestamp_dataframe_1 = timestamp_dataframe_1.loc[timestamp_dataframe_1.loc[:, "gdlat"] < lats[1]]
    timestamp_dataframe_1 = timestamp_dataframe_1.loc[timestamp_dataframe_1.loc[:, "gdlon"] >= lons[0]]
    timestamp_dataframe_1 = timestamp_dataframe_1.loc[timestamp_dataframe_1.loc[:, "gdlon"] < lons[1]]
    if timestamp_dataframe_1.empty:
        return None, hour, minute
    list_sites = []
    list_nos = []
    list_dvtec = []
    list_lats = []
    list_lons = []
    list_skip_lon_i = []
    for lon_i in range(len(blocks_table[0]) - 1):
        lon_min = blocks_table[0][lon_i][1]
        lon_max = blocks_table[0][lon_i + 1][1]
        temp2_dataframe_1 = timestamp_dataframe_1.loc[timestamp_dataframe_1.loc[:, "gdlon"] >= lon_min]
        temp2_dataframe_1 = temp2_dataframe_1.loc[temp2_dataframe_1.loc[:, "gdlon"] < lon_max]
        if temp2_dataframe_1.empty:
            list_skip_lon_i.append(lon_i)
    for lat_i in range(len(blocks_table) - 1):
        lat_min = blocks_table[lat_i][0][0]
        lat_max = blocks_table[lat_i + 1][0][0]
        temp_dataframe_1 = timestamp_dataframe_1.loc[timestamp_dataframe_1.loc[:, "gdlat"] >= lat_min]
        temp_dataframe_1 = temp_dataframe_1.loc[temp_dataframe_1.loc[:, "gdlat"] < lat_max]
        if temp_dataframe_1.empty:
            continue
        for lon_i in range(len(blocks_table[lat_i]) - 1):
            if lon_i in list_skip_lon_i:
                continue
            lon_min = blocks_table[lat_i][lon_i][1]
            lon_max = blocks_table[lat_i][lon_i + 1][1]
            temp2_dataframe_1 = temp_dataframe_1.loc[temp_dataframe_1.loc[:, "gdlon"] >= lon_min]
            temp2_dataframe_1 = temp2_dataframe_1.loc[temp2_dataframe_1.loc[:, "gdlon"] < lon_max]
            if temp2_dataframe_1.empty:
                continue
            sites = tuple(temp2_dataframe_1.loc[:, "gps_site"])
            nos = len(sites)
            dvtec = temp2_dataframe_1.loc[:, "dvtec"].mean()
            # temp_dataframe_2 = pd.DataFrame({"gdlat": lat_min, "gdlon": lon_min, "dvtec": dvtec, "nos": nos,
            #                                  "sites": sites})
            list_sites.append(sites)
            list_nos.append(nos)
            list_dvtec.append(dvtec)
            list_lats.append(lat_min)
            list_lons.append(lon_min)
            # dataframe_2 = pd.concat([dataframe_2, temp_dataframe_2], ignore_index=True)
    dataframe_2 = pd.DataFrame({"gdlat": list_lats, "gdlon": list_lons, "dvtec": list_dvtec,
                                "nos": list_nos, "sites": list_sites})
    return dataframe_2, hour, minute









def main2024sep():
    # main2024sep1()
    # main2024sep2()
    # main2024sep3()
    # main2024sep4()
    # main2024sep5()
    # main2024sep6()
    # main2024sep7()
    # main2024sep8()
    main2024sep8_mp()


def main2024oct1():
    source_path = r"/home/vadymskipa/PhD_student/data/Europe_36-70__-10-20/dvtec_csv_5min/blocks/"
    save_path = r"/home/vadymskipa/PhD_student/data/Europe_36-70__-10-20/dvtec_csv_5min/blocks_maps/"
    window = "Window_3600_Seconds"
    lats = (36, 70)
    lons = (-10, 20)
    lat_tick = 0.15
    lon_tick = 0.15
    ampl=1

    entries_1 = os.listdir(source_path)
    list_datedir = [datedir for datedir in entries_1 if len(datedir) == 14]
    for datedir in list_datedir:
        date = get_date_from_date_directory_name(datedir)
        source_path_1 = os.path.join(source_path, datedir, window)
        save_path_1 = get_directory_path(save_path, datedir)
        save_path_2 = get_directory_path(save_path_1, window)
        save_path_3 = get_directory_path(save_path_2, f"Amplitude_{ampl}_tecu.csv")
        date_dataframe = pd.DataFrame()
        list_sat = os.listdir(source_path_1)
        for sat in list_sat:
            print(f"reading {date.year}-{date.month}-{date.day}  {sat}")
            source_path_2 = os.path.join(source_path_1, sat)
            list_time_file = os.listdir(source_path_2)
            sat_dataframe = pd.DataFrame()
            for time_file in list_time_file:
                source_path_3 = os.path.join(source_path_2, time_file)
                temp_datetime = dt.datetime(year=date.year, month=date.month, day=date.day,
                                            hour=int(time_file[:2]), minute=int(time_file[2:4], tzinfo=dt.timezone.utc))
                temp_dataframe: pd.DataFrame = pd.read_csv(source_path_3)
                if temp_dataframe.empty:
                    continue
                temp_dataframe.insert(1, "timestamp", temp_datetime.timestamp())
                sat_dataframe = pd.concat((sat_dataframe, temp_dataframe), ignore_index=True)
            if sat_dataframe.empty:
                continue
            sat_dataframe.insert(2, "sat_id", int(sat[1:3]))
            date_dataframe = pd.concat((date_dataframe, sat_dataframe), ignore_index=True)
        list_timestamp = np.unique(date_dataframe.loc[:, "timestamp"])

        date_dataframe_2 = pd.DataFrame()
        for timestamp in list_timestamp:
            timestamp_dataframe = date_dataframe.loc[date_dataframe.loc[:, "timestamp"] == timestamp]
            temp_dataframe = main2024oct1_proccess(timestamp_dataframe)
            date_dataframe_2 = pd.concat((date_dataframe_2, temp_dataframe), ignore_index=True)
        date_dataframe_2.sort_values(by=["timestamp"], inplace=True)
        date_dataframe_2.to_csv(save_path_3)
        print(save_path_3, dt.datetime.now())



def main2024oct1_proccess(dataframe, nos_min=0):

    outcome_dataframe = pd.DataFrame()
    list_data_lat = np.unique(dataframe.loc[:, "gdlat"])

    for lat in list_data_lat:
        lat_dataframe = dataframe.loc[dataframe.loc[:, "gdlat"] == lat]
        list_lon = np.unique(lat_dataframe.loc[:, "gdlon"])
        for lon in list_lon:
            lon_dataframe = lat_dataframe.loc[lat_dataframe.loc[:, "gdlon"] == lon]
            nos_sum = lon_dataframe.loc[:, "nos"].sum()
            dvtec = lon_dataframe.loc[:, "dvtec"].sum() / nos_sum
            if nos_sum >= nos_min:
                temp_dataframe = pd.DataFrame({"gdlat": lat, "gdlon": lon, "dvtec": dvtec,
                                               "nos": nos_sum, "timestamp": dataframe.iloc[0].loc["timestamp"]},
                                              index = (1,))
                outcome_dataframe = pd.concat((outcome_dataframe, temp_dataframe), ignore_index=True)


    return outcome_dataframe


def main2024oct1_mp():
    source_path = r"/home/vadymskipa/PhD_student/data/Europe_36-70__-10-20/dvtec_csv_5min/blocks/"
    save_path = r"/home/vadymskipa/PhD_student/data/Europe_36-70__-10-20/dvtec_csv_5min/blocks/all_sats"
    window = "Window_7200_Seconds"

    entries_1 = os.listdir(source_path)
    list_datedir = [datedir for datedir in entries_1 if len(datedir) == 14]
    for datedir in list_datedir:
        date = get_date_from_date_directory_name(datedir)
        source_path_1 = os.path.join(source_path, datedir, window)
        save_path_1 = get_directory_path(save_path, datedir)
        save_path_2 = os.path.join(save_path_1, f"{window}.csv")
        date_dataframe = pd.DataFrame()
        list_sat = os.listdir(source_path_1)
        for sat in list_sat:
            print(f"reading {date.year}-{date.month}-{date.day}  {sat}")
            source_path_2 = os.path.join(source_path_1, sat)
            list_time_file = os.listdir(source_path_2)
            sat_dataframe = pd.DataFrame()
            for time_file in list_time_file:
                source_path_3 = os.path.join(source_path_2, time_file)
                temp_datetime = dt.datetime(year=date.year, month=date.month, day=date.day,
                                            hour=int(time_file[:2]), minute=int(time_file[2:4]),
                                            tzinfo=dt.timezone.utc)
                temp_dataframe: pd.DataFrame = pd.read_csv(source_path_3)
                if temp_dataframe.empty:
                    continue
                temp_dataframe.insert(1, "timestamp", temp_datetime.timestamp())
                sat_dataframe = pd.concat((sat_dataframe, temp_dataframe), ignore_index=True)
            if sat_dataframe.empty:
                continue
            sat_dataframe.insert(2, "sat_id", int(sat[1:3]))
            date_dataframe = pd.concat((date_dataframe, sat_dataframe), ignore_index=True)

        date_dataframe_2 = pd.DataFrame()
        pool = multipool.Pool(7)
        list_timestamp = np.unique(date_dataframe.loc[:, "timestamp"])
        input_list = [date_dataframe.loc[date_dataframe.loc[:, "timestamp"] == timestamp] for timestamp
                      in list_timestamp]
        for temp_dataframe in pool.imap(main2024oct1_proccess, input_list):
            date_dataframe_2 = pd.concat((date_dataframe_2, temp_dataframe), ignore_index=True)
            timestamp = temp_dataframe.iloc[0].loc["timestamp"]
            timestamp_datetime = dt.datetime.fromtimestamp(timestamp, tz=dt.timezone.utc)
            print(f"{timestamp_datetime.hour:0=2}{timestamp_datetime.minute:0=2} done! {dt.datetime.now()}")
        date_dataframe_2.sort_values(by=["timestamp"], inplace=True)
        date_dataframe_2.to_csv(save_path_2)
        print(save_path_2, dt.datetime.now())


def main2024oct2():
    source_path = r"/home/vadymskipa/PhD_student/data/Europe_36-70__-10-20/dvtec_csv_5min/blocks/all_sats"
    save_path = r"/home/vadymskipa/PhD_student/data/Europe_36-70__-10-20/dvtec_csv_5min/blocks/all_sats"
    window = "Window_3600_Seconds"
    lats = (36, 70)
    lons = (-10, 20)
    lat_tick = 0.15
    lon_tick = 0.15
    list_lats = [round((lats[0] + lat_tick * i), 2) for i in range(math.ceil((lats[1] - lats[0]) / lat_tick))]
    list_lons = [round((lons[0] + lon_tick * i), 2) for i in range(math.ceil((lons[1] - lons[0]) / lon_tick))]
    stock_blocks_dataframe = pd.DataFrame({"lats": list_lats}, columns=["lats", *list_lons])

    entries = os.listdir(source_path)
    list_datedir = [entry for entry in entries if len(entry) == 14]
    for datedir in list_datedir:
        source_path_1 = os.path.join(source_path, datedir)
        source_path_2 = os.path.join(source_path_1, f"{window}.csv")
        save_path_1 = os.path.join(save_path, datedir)
        save_path_2 = get_directory_path(save_path_1, window)
        source_dataframe: pd.DataFrame = pd.read_csv(source_path_2)

        list_timestamp = np.unique(source_dataframe.loc[:, "timestamp"])
        list_input = [(timestamp, source_dataframe.loc[source_dataframe.loc[:, "timestamp"] == timestamp],
                       stock_blocks_dataframe.copy())
                      for timestamp in list_timestamp]
        pool = multipool.Pool(7)
        for timestamp, timestamp_dataframe in pool.imap(main2024cot2_convert_proccess, list_input):
            timestamp_datetime = dt.datetime.fromtimestamp(timestamp, tz=dt.timezone.utc)
            filename = f"{timestamp_datetime.hour:0=2}{timestamp_datetime.minute:0=2}.csv"
            save_path_3 = os.path.join(save_path_2, filename)
            timestamp_dataframe.to_csv(save_path_3)
            print(f"{datedir} {filename} done! {dt.datetime.now}")



def main2024cot2_convert_proccess(tup):
    lats = (36, 70)
    lons = (-10, 20)
    lat_tick = 0.15
    lon_tick = 0.15

    timestamp, dataframe, new_dataframe = tup
    for index in dataframe.index:
        lat_index = int(round((dataframe.loc[index, "gdlat"] - lats[0]) / lat_tick))
        lon = round(dataframe.loc[index, "gdlon"], 2)
        dvtec = dataframe.loc[index, "dvtec"]
        new_dataframe.loc[lat_index, lon] = dvtec
    return timestamp, new_dataframe


def main2024oct3():
    source_path = r"/home/vadymskipa/PhD_student/data/Europe_36-70__-10-20/dvtec_csv_5min/blocks/all_sats"
    save_path = r"/home/vadymskipa/PhD_student/data/Europe_36-70__-10-20/dvtec_csv_5min/blocks/all_sats_smooth"
    window = "Window_3600_Seconds"
    lats = (36, 70)
    lons = (-10, 20)
    lat_tick = 0.15
    lon_tick = 0.15
    list_lats = [round((lats[0] + lat_tick * i), 2) for i in range(math.ceil((lats[1] - lats[0]) / lat_tick))]
    list_lons = [round((lons[0] + lon_tick * i), 2) for i in range(math.ceil((lons[1] - lons[0]) / lon_tick))]
    stock_blocks_dataframe = pd.DataFrame({"lats": list_lats}, columns=["lats", *list_lons])
    smoothing_range = 5
    smoothing_limits_dataframes = get_smoothing_limits_dataframe(stock_blocks_dataframe, smoothing_range)


    entries = os.listdir(source_path)
    list_datedir = [entry for entry in entries if len(entry) == 14]
    for datedir in list_datedir:
        if get_date_from_date_directory_name(datedir) <= dt.datetime(year=2024, month=5, day=10, tzinfo=dt.timezone.utc):
            continue
        source_path_1 = os.path.join(source_path, datedir)
        source_path_2 = os.path.join(source_path_1, window)
        dict_source_blocks_dataframe = {}
        list_blocks_files = os.listdir(source_path_2)
        for block_file in list_blocks_files:
            source_path_3 = os.path.join(source_path_2, block_file)
            blocks_dataframe = pd.read_csv(source_path_3)
            dict_source_blocks_dataframe[block_file] = blocks_dataframe

        save_path_1 = get_directory_path(save_path, datedir)
        save_path_2 = get_directory_path(save_path_1, window)
        # for filename, blocks_dataframe in dict_source_blocks_dataframe.items():
        #     save_path_3 = os.path.join(save_path_2, filename)
        #     start = dt.datetime.now()
        #     nothing, dataframe = main2024oct3_proccess((filename, blocks_dataframe, stock_blocks_dataframe,
        #                                                 smoothing_limits_dataframes))
        #     dataframe.to_csv(save_path_3)
        #     last = dt.datetime.now() - start
        #     print(filename, last)


        list_input = [(filename, blocks_dataframe, stock_blocks_dataframe, smoothing_limits_dataframes)
                      for filename, blocks_dataframe in dict_source_blocks_dataframe.items()]
        pool = multipool.Pool(4)
        for filename, blocks_dataframe_smooth in pool.imap(main2024oct3_proccess, list_input):
            save_path_3 = os.path.join(save_path_2, filename)
            blocks_dataframe_smooth.to_csv(save_path_3)
            print(filename, dt.datetime.now())
        pool.close()



def main2024oct3_proccess(tup):
    filename, dataframe, new_dataframe, smoothing_limits_dataframes = tup
    dataframe.drop([dataframe.columns[0]], axis=1, inplace=True)
    list_index = new_dataframe.index
    len_columns = len(new_dataframe.columns)
    for index in list_index:
        for i_column in range(1, len_columns, 1):
            new_dataframe.iloc[index, i_column] = dataframe.iloc[
                smoothing_limits_dataframes[0].iloc[index, i_column]:
                smoothing_limits_dataframes[1].iloc[index, i_column],
                                                  smoothing_limits_dataframes[2].iloc[index, i_column]:
                                                  smoothing_limits_dataframes[3].iloc[index, i_column]].stack().dropna().mean()
    return filename, new_dataframe





def bigger_or_x(input_number, x):
    if x >= input_number:
        return x
    else:
        return input_number


def lower_or_x(input_number, x):
    if x <= input_number:
        return x
    else:
        return input_number



def get_smoothing_limits_dataframe(stock_blocks_dataframe, smoothing_range):
    left_dataframe: pd.DataFrame = stock_blocks_dataframe.copy()
    right_dataframe: pd.DataFrame = stock_blocks_dataframe.copy()
    top_dataframe: pd.DataFrame = stock_blocks_dataframe.copy()
    bottom_dataframe: pd.DataFrame = stock_blocks_dataframe.copy()
    list_index = left_dataframe.index
    list_columns = left_dataframe.columns
    len_list_index = len(list_index)
    len_list_column = len(list_columns)
    row_range = (-((smoothing_range - 1) // 2), (smoothing_range - 1) // 2 + 1)
    for i_index in range(len(list_index)):
        lat = left_dataframe.iloc[i_index].loc["lats"]
        columns_range = (-(round(smoothing_range / math.cos(math.radians(lat)) - 1) // 2),
                         round((smoothing_range / math.cos(math.radians(lat))) - 1) // 2 + 1)
        for i_column in range(1, len(list_columns), 1):
            top_dataframe.iloc[i_index, i_column] = bigger_or_x(i_index + row_range[0], 0)
            bottom_dataframe.iloc[i_index, i_column] = lower_or_x(i_index + row_range[1], len_list_index)
            left_dataframe.iloc[i_index, i_column] = bigger_or_x(i_column + columns_range[0], 1)
            right_dataframe.iloc[i_index, i_column] = lower_or_x(i_column + columns_range[1], len_list_column)

    return top_dataframe, bottom_dataframe, left_dataframe, right_dataframe


def main2024oct4():
    source_directory = r"/home/vadymskipa/PhD_student/data/Europe_36-70__-10-20/dvtec_csv_5min/blocks/all_sats_smooth"
    save_directory = r"/home/vadymskipa/PhD_student/data/Europe_36-70__-10-20/temp"
    # point = (longitude, latitude)
    list_points = [(2.0, 42.0), (2.0, 45.0), (2.0, 48.0), (2.0, 51.0), (2.0, 54.0), (2.0, 57.0),
                   (6.5, 42.0), (6.5, 45.0), (6.5, 48.0), (6.5, 51.0), (6.5, 54.0), (6.5, 57.0),
                   (11.0, 42.0), (11.0, 45.0), (11.0, 48.0), (11.0, 51.0), (11.0, 54.0), (11.0, 57.0),
                   (15.5, 42.0), (15.5, 45.0), (15.5, 48.0), (15.5, 51.0), (15.5, 54.0), (15.5, 57.0),]
    window = "Window_3600_Seconds"

    for point in list_points:
        entries = os.listdir(source_directory)
        list_datedir = [entry for entry in entries if len(entry) == 14]
        for datedir in list_datedir:
            date = get_date_from_date_directory_name(datedir)
            if date < dt.datetime(year=2024, month=5, day=10,
                                                                         tzinfo=dt.timezone.utc):
                continue
            if date > dt.datetime(year=2024, month=5, day=11,
                                                                         tzinfo=dt.timezone.utc):
                continue
            point_date_dataframe = pd.DataFrame()
            source_path_1 = os.path.join(source_directory, datedir, window)
            save_path_1 = get_directory_path(save_directory, datedir)
            save_path_2 = get_directory_path(save_path_1, window)
            list_files = os.listdir(source_path_1)
            for file in list_files:
                source_path_2 = os.path.join(source_path_1, file)
                hour = int(file[:2])
                minute = int(file[2:4])
                source_dataframe = pd.read_csv(source_path_2)
                temp_dt = dt.datetime(year=date.year, month=date.month, day=date.day, hour=hour, minute=minute,
                                      tzinfo=dt.timezone.utc)
                # temp = source_dataframe.loc[source_dataframe.loc[:, "lats"] == point[1]]
                dvtec = source_dataframe.loc[source_dataframe.loc[:, "lats"] == point[1]].iloc[0].loc[str(point[0])]
                temp_series = pd.DataFrame({"timestamp": temp_dt.timestamp(), "dvtec": dvtec, "datetime": temp_dt},
                                           index=(0,))
                point_date_dataframe = pd.concat([point_date_dataframe, temp_series], ignore_index=True)
            filename = f"lat_{point[1]:+07.2f}_lon_{point[0]:+06.2f}.csv"
            save_path_3 = os.path.join(save_path_2, filename)
            point_date_dataframe.to_csv(save_path_3)
            print(save_path_3, dt.datetime.now())


def main2024oct5():
    directory = r"/home/vadymskipa/PhD_student/data/Europe_36-70__-10-20/temp"
    window = "Window_3600_Seconds"

    entries = os.listdir(directory)
    for datedir in entries:
        source_path_1 = os.path.join(directory, datedir, window)
        list_files = os.listdir(source_path_1)
        for file in list_files:
            source_path_2 = os.path.join(source_path_1, file)
            dataframe = pd.read_csv(source_path_2)
            dvtec_without_missing = np.interp((dataframe.loc[:, "timestamp"] - dataframe.iloc[0].loc["timestamp"]),
                                              (dataframe.loc[dataframe.loc[:, "dvtec"].notnull()].loc[:, "timestamp"] -
                                              dataframe.iloc[0].loc["timestamp"]),
                                              dataframe.loc[dataframe.loc[:, "dvtec"].notnull()].loc[:, "dvtec"])
            dataframe.loc[:, "dvtec"] = dvtec_without_missing
            dataframe.drop([dataframe.columns[0]], axis=1, inplace=True)
            dataframe.to_csv(source_path_2)


def coord_tuple_to_str(coord_tuple):
    outcome = f"lat_{coord_tuple[0]:+07.2f}_lon_{coord_tuple[1]:+06.2f}"
    return outcome


def str_to_coord_tuple(string: str):
    list_els = string.split("_")
    lat = 0
    lon = 0
    for i in range(len(list_els)):
        if list_els[i] == "lat":
            lat = float(list_els[i + 1])
        if list_els[i] == "lon":
            lon = float(list_els[i + 1])
    return (lat, lon)


def plot_1(dataframe, dataname):
    figure: fig.Figure = plt.figure(layout="tight", figsize=[16.0 * 160 / 300, 9.0 * 160 / 300])
    axes1: axs.Axes = figure.add_subplot(1, 1, 1)
    ytext1 = axes1.set_ylabel(f"{dataname}, TEC units")
    xtext1 = axes1.set_xlabel("Time, hrs")
    time_array = [dt.datetime.strptime(dataframe.iloc[i].loc["datetime"], "%Y-%m-%d %H:%M:%S%z") for i in range(len(dataframe))]
    timeticks = time_array[::36]
    timeticks_name = [i.strftime("%H:%M:%S") for i in timeticks]
    line1, = axes1.plot(time_array, dataframe[dataname], label="Data from GPS-TEC", linestyle="-",
                        marker=" ", color="blue", markeredgewidth=1, markersize=1.1)
    axes1.set_ylim(-5, 5)
    axes1.set_xticks(timeticks, labels=timeticks_name)

    axes1.legend()
    return figure


def main2024oct6():
    directory = r"/home/vadymskipa/PhD_student/data/Europe_36-70__-10-20/temp/1"
    save_directory = r"/home/vadymskipa/PhD_student/data/Europe_36-70__-10-20/temp/2"
    window = "Window_3600_Seconds"
    dict_coord = {}
    start = dt.datetime(year=2024, month=5, day=10, hour=18, tzinfo=dt.timezone.utc)
    end = dt.datetime(year=2024, month=5, day=11, hour=6, tzinfo=dt.timezone.utc)

    entries = os.listdir(directory)
    for datedir in entries:
        source_path_1 = os.path.join(directory, datedir, window)
        list_files = os.listdir(source_path_1)
        for file in list_files:
            source_path_2 = os.path.join(source_path_1, file)
            dataframe = pd.read_csv(source_path_2)
            coord_tuple = str_to_coord_tuple(file[:-4])
            if coord_tuple in dict_coord.keys():
                dict_coord[coord_tuple] = pd.concat([dict_coord[coord_tuple], dataframe], ignore_index=True)
            else:
                dict_coord[coord_tuple] = dataframe
    for key, value in dict_coord.items():
        save_path_1 = get_directory_path(save_directory, window)
        save_path_2 = os.path.join(save_path_1, f"{coord_tuple_to_str(key)}.jpg")
        dataframe = value.loc[value.loc[:, "timestamp"] >= start.timestamp()]
        dataframe = dataframe.loc[dataframe.loc[:, "timestamp"] < end.timestamp()]
        figure = plot_1(dataframe, "dvtec")
        figure.savefig(save_path_2)
        plt.close(figure)
        print(coord_tuple_to_str(key), dt.datetime.now())


def main2024oct7():
    directory = r"/home/vadymskipa/PhD_student/data/Europe_36-70__-10-20/temp/1"
    save_directory = r"/home/vadymskipa/PhD_student/data/Europe_36-70__-10-20/temp/3"
    window = "Window_3600_Seconds"
    dict_coord = {}
    start = dt.datetime(year=2024, month=5, day=10, hour=18, tzinfo=dt.timezone.utc)
    end = dt.datetime(year=2024, month=5, day=11, hour=6, tzinfo=dt.timezone.utc)

    entries = os.listdir(directory)
    for datedir in entries:
        source_path_1 = os.path.join(directory, datedir, window)
        list_files = os.listdir(source_path_1)
        for file in list_files:
            source_path_2 = os.path.join(source_path_1, file)
            dataframe = pd.read_csv(source_path_2)
            coord_tuple = str_to_coord_tuple(file[:-4])
            if coord_tuple in dict_coord.keys():
                dict_coord[coord_tuple] = pd.concat([dict_coord[coord_tuple], dataframe], ignore_index=True)
            else:
                dict_coord[coord_tuple] = dataframe
    for key, value in dict_coord.items():
        save_path_1 = get_directory_path(save_directory, window)
        save_path_2 = os.path.join(save_path_1, f"{coord_tuple_to_str(key)}.csv")
        dataframe = value.loc[value.loc[:, "timestamp"] >= start.timestamp()]
        dataframe = dataframe.loc[dataframe.loc[:, "timestamp"] < end.timestamp()]
        fftdataframe = fft2024oct7(dataframe)
        fftdataframe.to_csv(save_path_2)
        print(coord_tuple_to_str(key), dt.datetime.now())


def fft2024oct7(dataframe: pd.DataFrame):
    data = dataframe.loc[:, "dvtec"].values

    # Apply Fourier Transform
    n = len(data)
    t = (dataframe.iloc[1].loc["timestamp"] - dataframe.iloc[0].loc["timestamp"]) / 3600  # Sample spacing
    yf = fft(data)
    xf = fftfreq(n, t)[:n // 2]
    new_dataframe = pd.DataFrame({"freq": xf, "ampl": 2.0/n * np.abs(yf[0:n//2])})
    return new_dataframe


def main2024oct8():
    directory = r"/home/vadymskipa/PhD_student/data/Europe_36-70__-10-20/temp/3"
    save_directory = r"/home/vadymskipa/PhD_student/data/Europe_36-70__-10-20/temp/4"
    window = "Window_3600_Seconds"

    source_path_1 = os.path.join(directory, window)
    entries = os.listdir(source_path_1)
    save_path_1 = get_directory_path(save_directory, window)
    for file in entries:
        source_path_2 = os.path.join(source_path_1, file)
        save_path_2 = os.path.join(save_path_1, f"{file[:-4]}.jpg")
        dataframe = pd.read_csv(source_path_2)
        figure: fig.Figure = plt.figure(layout="tight")
        axes1: axs.Axes = figure.add_subplot(1, 1, 1)
        ytext1 = axes1.set_ylabel("Amplitude")
        xtext1 = axes1.set_xlabel("Frequency, 1/hour")
        freq_array = [dataframe.iloc[i].loc["freq"] for i in range(len(dataframe))]
        freq_ticks = freq_array[::6]
        line1, = axes1.plot(freq_array, dataframe.loc[:, "ampl"], label="Data from GPS-TEC", linestyle="-",
                            marker=" ", color="blue", markeredgewidth=1, markersize=1.1)
        axes1.set_xticks(freq_ticks)
        axes1.legend()

        figure.savefig(save_path_2, dpi=300)
        plt.close(figure)


def main2024oct9_mp():
    save_dir = r"/home/vadymskipa/PhD_student/data/Europe_36-70__-10-20/dvtec_csv_blocks/per_site"
    source_dir = r"/home/vadymskipa/PhD_student/data/Europe_36-70__-10-20/dvtec_csv"
    window = "Window_3600_Seconds"
    sats = [f"G{i:0=2}" for i in range(1, 33)]
    lats = (36, 70)
    lons = (-10, 20)
    lat_tick = 0.15
    lon_tick = 0.15
    blocks_table= create_blocks_table(lats=lats, lons=lons, lat_tick=lat_tick, lon_tick=lon_tick)
    min_elm = 30



    entries = os.listdir(source_dir)
    list_date_dir = [dirname for dirname in entries if
                       (len(dirname) == 14 and os.path.isdir(os.path.join(source_dir, dirname)))]
    for date_dir in list_date_dir:
        date = get_date_from_date_directory_name(date_dir)
        # if date <= dt.datetime(year=2024, month=5, day=9, tzinfo=dt.timezone.utc):
        #     continue
        source_path_1 = os.path.join(source_dir, date_dir)
        save_path_1 = get_directory_path(save_dir, date_dir)
        save_path_2 = get_directory_path(save_path_1, window)
        for sat in sats:
            save_path_3 = get_directory_path(save_path_2, sat)
            dataframe_1 = pd.DataFrame()
            entries_1 = os.listdir(source_path_1)
            list_site_dir = [dirname for dirname in entries_1 if os.path.isdir(os.path.join(source_path_1, dirname))]
            print(f"--- start reading {sat}", dt.datetime.now())
            for site in list_site_dir:
                try:
                    source_path_2 = os.path.join(source_path_1, site, window, f"{sat}.csv")
                    sat_dataframe: pd.Dataframe = pd.read_csv(source_path_2)
                    sat_dataframe.insert(1, "gps_site", site)
                    dataframe_1 = pd.concat([dataframe_1, sat_dataframe], ignore_index=True)
                except Exception as er:
                    print(er, dt.datetime.now())
            print(f"--- end reading {sat}  ", dt.datetime.now())
            dataframe_1 = add_timestamp_column_to_df(dataframe_1, date)
            dataframe_1 = dataframe_1.loc[dataframe_1.loc[:, "elm"] >= min_elm]
            list_timestamp = np.unique(dataframe_1.loc[:, "timestamp"])
            input_mp_list = [(timestamp, dataframe_1.loc[dataframe_1.loc[:, "timestamp"] == timestamp]
                              , lats, lons, blocks_table) for timestamp in list_timestamp]

            pool = multipool.Pool(6)
            for dataframe_2, hour, minute, second in pool.imap(main2024oct9_proccess, input_mp_list):
                if type(dataframe_2) == pd.DataFrame:
                    save_path_4 = os.path.join(save_path_3, f"{hour:0=2}{minute:0=2}{second:0=2}.csv")
                    dataframe_2.to_csv(save_path_4)
                print(date_dir, window, sat, f"{hour:0=2}{minute:0=2}{second:0=2}", dt.datetime.now())



def main2024oct9_proccess(tup):
    timestamp, dataframe_1, lats, lons, blocks_table = tup
    d = dt.datetime.fromtimestamp(timestamp, tz=dt.timezone.utc)
    hour = d.hour
    minute = d.minute
    second = d.second
    # dataframe_2 = pd.DataFrame()
    timestamp_dataframe_1 = dataframe_1.loc[dataframe_1.loc[:, "timestamp"] == timestamp]
    if timestamp_dataframe_1.empty:
        return None, hour, minute, second
    timestamp_dataframe_1 = timestamp_dataframe_1.loc[timestamp_dataframe_1.loc[:, "gdlat"] >= lats[0]]
    timestamp_dataframe_1 = timestamp_dataframe_1.loc[timestamp_dataframe_1.loc[:, "gdlat"] < lats[1]]
    timestamp_dataframe_1 = timestamp_dataframe_1.loc[timestamp_dataframe_1.loc[:, "gdlon"] >= lons[0]]
    timestamp_dataframe_1 = timestamp_dataframe_1.loc[timestamp_dataframe_1.loc[:, "gdlon"] < lons[1]]
    if timestamp_dataframe_1.empty:
        return None, hour, minute, second
    list_sites = []
    list_nos = []
    list_dvtec = []
    list_lats = []
    list_lons = []
    list_skip_lon_i = []
    for lon_i in range(len(blocks_table[0]) - 1):
        lon_min = blocks_table[0][lon_i][1]
        lon_max = blocks_table[0][lon_i + 1][1]
        temp2_dataframe_1 = timestamp_dataframe_1.loc[timestamp_dataframe_1.loc[:, "gdlon"] >= lon_min]
        temp2_dataframe_1 = temp2_dataframe_1.loc[temp2_dataframe_1.loc[:, "gdlon"] < lon_max]
        if temp2_dataframe_1.empty:
            list_skip_lon_i.append(lon_i)
    for lat_i in range(len(blocks_table) - 1):
        lat_min = blocks_table[lat_i][0][0]
        lat_max = blocks_table[lat_i + 1][0][0]
        temp_dataframe_1 = timestamp_dataframe_1.loc[timestamp_dataframe_1.loc[:, "gdlat"] >= lat_min]
        temp_dataframe_1 = temp_dataframe_1.loc[temp_dataframe_1.loc[:, "gdlat"] < lat_max]
        if temp_dataframe_1.empty:
            continue
        for lon_i in range(len(blocks_table[lat_i]) - 1):
            if lon_i in list_skip_lon_i:
                continue
            lon_min = blocks_table[lat_i][lon_i][1]
            lon_max = blocks_table[lat_i][lon_i + 1][1]
            temp2_dataframe_1 = temp_dataframe_1.loc[temp_dataframe_1.loc[:, "gdlon"] >= lon_min]
            temp2_dataframe_1 = temp2_dataframe_1.loc[temp2_dataframe_1.loc[:, "gdlon"] < lon_max]
            if temp2_dataframe_1.empty:
                continue
            sites = tuple(temp2_dataframe_1.loc[:, "gps_site"])
            nos = len(sites)
            dvtec = temp2_dataframe_1.loc[:, "dvtec"].mean()
            # temp_dataframe_2 = pd.DataFrame({"gdlat": lat_min, "gdlon": lon_min, "dvtec": dvtec, "nos": nos,
            #                                  "sites": sites})
            list_sites.append(sites)
            list_nos.append(nos)
            list_dvtec.append(dvtec)
            list_lats.append(lat_min)
            list_lons.append(lon_min)
            # dataframe_2 = pd.concat([dataframe_2, temp_dataframe_2], ignore_index=True)
    dataframe_2 = pd.DataFrame({"gdlat": list_lats, "gdlon": list_lons, "dvtec": list_dvtec,
                                "nos": list_nos, "sites": list_sites})
    return dataframe_2, hour, minute, second


def main2024oct10():
    save_dir = r"/home/vadymskipa/PhD_student/data/Europe_36-70__-10-20/dvtec_csv_blocks/per_site"
    source_dir = r"/home/vadymskipa/PhD_student/data/Europe_36-70__-10-20/dvtec_csv"
    window = "Window_7200_Seconds"
    sats = [f"G{i:0=2}" for i in range(1, 33)]
    lats = (36, 70)
    lons = (-10, 20)
    lat_tick = 0.15
    lon_tick = 0.15
    min_elm = 30



    entries = os.listdir(source_dir)
    list_date_dir = [dirname for dirname in entries if
                       (len(dirname) == 14 and os.path.isdir(os.path.join(source_dir, dirname)))]
    for date_dir in list_date_dir:
        date = get_date_from_date_directory_name(date_dir)
        # if date <= dt.datetime(year=2024, month=5, day=9, tzinfo=dt.timezone.utc):
        #     continue
        source_path_1 = os.path.join(source_dir, date_dir)
        save_path_1 = get_directory_path(save_dir, date_dir)
        save_path_2 = get_directory_path(save_path_1, window)
        for sat in sats:
            save_path_3 = os.path.join(save_path_2, f"{sat}.csv")
            dataframe_1 = pd.DataFrame()
            entries_1 = os.listdir(source_path_1)
            list_site_dir = [dirname for dirname in entries_1 if os.path.isdir(os.path.join(source_path_1, dirname))]
            print(f"--- start reading {sat}", dt.datetime.now())
            for site in list_site_dir:
                try:
                    source_path_2 = os.path.join(source_path_1, site, window, f"{sat}.csv")
                    sat_dataframe: pd.Dataframe = pd.read_csv(source_path_2)
                    sat_dataframe.insert(1, "gps_site", site)
                    dataframe_1 = pd.concat([dataframe_1, sat_dataframe], ignore_index=True)
                except Exception as er:
                    print(er, dt.datetime.now())
            print(f"--- end reading {sat}  ", dt.datetime.now())
            dataframe_1 = add_timestamp_column_to_df(dataframe_1, date)
            dataframe_1 = dataframe_1.loc[dataframe_1.loc[:, "elm"] >= min_elm]
            dataframe_1 = dataframe_1.loc[dataframe_1.loc[:, "gdlat"] >= lats[0]]
            dataframe_1 = dataframe_1.loc[dataframe_1.loc[:, "gdlat"] < lats[1]]
            dataframe_1 = dataframe_1.loc[dataframe_1.loc[:, "gdlon"] >= lons[0]]
            dataframe_1 = dataframe_1.loc[dataframe_1.loc[:, "gdlon"] < lons[1]]
            dataframe_1.loc[:, "gdlon1"] = dataframe_1.loc[:, "gdlon"].apply(
                lambda gdlon: round(lons[0] + round(((gdlon - lons[0]) // lon_tick) * lon_tick, 2), 2))
            dataframe_1.loc[:, "gdlat1"] = dataframe_1.loc[:, "gdlat"].apply(
                lambda gdlat: round(lats[0] + round(((gdlat - lats[0]) // lat_tick) * lat_tick, 2), 2))
            dataframe_1.drop(columns=["gdlat", "gdlon"], axis=1, inplace=True)
            dataframe_1.rename(columns={"gdlon1": "gdlon", "gdlat1": "gdlat"}, inplace=True)
            new_dataframe = (dataframe_1.loc[:, ("timestamp", "gdlon", "gdlat", "dvtec")]
                         .groupby(["timestamp", "gdlon", "gdlat"]).mean())
            nos_dataframe = (dataframe_1.loc[:, ("timestamp", "gdlon", "gdlat", "gps_site")]
                         .groupby(["timestamp", "gdlon", "gdlat"]).apply(lambda sites: len(sites))).to_frame(name="nos")
            new_dataframe = pd.concat([new_dataframe, nos_dataframe], axis=1).reset_index()
            new_dataframe.to_csv(save_path_3)
            print(save_path_3, dt.datetime.now())


def main2024oct11():
    source_dir = r"/home/vadymskipa/PhD_student/data/Europe_36-70__-10-20/dvtec_csv_blocks/per_site"
    save_dir = r"/home/vadymskipa/PhD_student/data/Europe_36-70__-10-20/dvtec_csv_blocks/all_sats"
    window = "Window_7200_Seconds"

    entries = os.listdir(source_dir)
    list_date_dir = [dirname for dirname in entries if
                     (len(dirname) == 14 and os.path.isdir(os.path.join(source_dir, dirname)))]
    for date_dir in list_date_dir:
        source_path_1 = os.path.join(source_dir, date_dir)
        save_path_1 = get_directory_path(save_dir, date_dir)
        save_path_2 = os.path.join(save_path_1, f"{window}.hdf5")
        source_path_2 = os.path.join(source_path_1, window)
        list_sat_files = os.listdir(source_path_2)
        source_dataframe = pd.DataFrame()
        for sat_file in list_sat_files:
            source_path_3 = os.path.join(source_path_2, sat_file)
            sat_dataframe = pd.read_csv(source_path_3, index_col=0)
            source_dataframe = pd.concat([source_dataframe, sat_dataframe], ignore_index=True)
        source_dataframe.reset_index()
        result_dataframe = (source_dataframe.groupby(["timestamp", "gdlon", "gdlat"]).sum())
        result_dataframe.loc[:, "avg_dvtec"] = result_dataframe.loc[:, "dvtec"] / result_dataframe.loc[:, "nos"]
        result_dataframe.drop(columns=["dvtec", "nos"], axis=1, inplace=True)
        result_dataframe.rename(columns={"avg_dvtec": "dvtec"}, inplace=True)
        result_dataframe.reset_index(inplace=True)
        result_dataframe.to_hdf(save_path_2, key="df")
        print(save_path_2, dt.datetime.now())


def main2024oct12():
    source_dir = r"/home/vadymskipa/PhD_student/data/Europe_36-70__-10-20/dvtec_csv_blocks/all_sats"
    save_dir = r"/home/vadymskipa/PhD_student/data/Europe_36-70__-10-20/dvtec_csv_blocks/block_tables"
    window = "Window_7200_Seconds"

    entries = os.listdir(source_dir)
    list_date_dir = [dirname for dirname in entries if
                     (len(dirname) == 14 and os.path.isdir(os.path.join(source_dir, dirname)))]
    for date_dir in list_date_dir:
        source_path_1 = os.path.join(source_dir, date_dir)
        save_path_1 = get_directory_path(save_dir, date_dir)
        save_path_2 = get_directory_path(save_path_1, window)
        source_path_2 = os.path.join(source_path_1, f"{window}.hdf5")
        source_dataframe = pd.read_hdf(source_path_2, key="df")
        list_timestamp = np.unique(source_dataframe.loc[:, "timestamp"])
        for timestamp in list_timestamp:
            timestamp_dataframe = source_dataframe.loc[source_dataframe.loc[:, "timestamp"] == timestamp]
            date = dt.datetime.fromtimestamp(timestamp, tz=dt.timezone.utc)
            savefile = f"{date.hour:0=2}{date.minute:0=2}{date.second:0=2}.hdf5"
            new_dataframe = timestamp_dataframe.pivot(index="gdlat", columns="gdlon", values="dvtec")
            save_path_3 = os.path.join(save_path_2, savefile)
            new_dataframe.to_hdf(save_path_3, key="df")
            print(savefile, dt.datetime.now())


def main2024oct13():
    source_dir = r"/home/vadymskipa/PhD_student/data/Europe_36-70__-10-20/dvtec_csv_blocks/block_tables"
    save_dir = r"/home/vadymskipa/PhD_student/data/Europe_36-70__-10-20/dvtec_csv_blocks/block_tables_smooth"
    window = "Window_3600_Seconds"
    lats = (36, 70)
    lons = (-10, 20)
    lat_tick = 0.15
    lon_tick = 0.15
    list_lats = [round((lats[0] + lat_tick * i), 2) for i in range(math.ceil((lats[1] - lats[0]) / lat_tick))]
    list_lons = [round((lons[0] + lon_tick * i), 2) for i in range(math.ceil((lons[1] - lons[0]) / lon_tick))]
    stock_blocks_dataframe = pd.DataFrame(columns=list_lons, index=list_lats)
    smoothing_range = 5
    lon_window_dataframe = get_smoothing_lon_window(stock_blocks_dataframe, smoothing_range)
    # smoothing_limits_dataframes = get_smoothing_limits_dataframe2(stock_blocks_dataframe, smoothing_range)

    entries = os.listdir(source_dir)
    list_date_dir = [dirname for dirname in entries if
                     (len(dirname) == 14 and os.path.isdir(os.path.join(source_dir, dirname)))]
    for date_dir in list_date_dir:
        source_path_1 = os.path.join(source_dir, date_dir)
        save_path_1 = get_directory_path(save_dir, date_dir)
        save_path_2 = get_directory_path(save_path_1, window)
        source_path_2 = os.path.join(source_path_1, window)
        list_timefile = os.listdir(source_path_2)
        for timefile in list_timefile:
            source_path_3 = os.path.join(source_path_2, timefile)
            save_path_3 = os.path.join(save_path_2, timefile)
            file_dataframe = pd.read_hdf(source_path_3, key="df")
            new_dataframe = main2024oct13_smoothing_proccess2(file_dataframe, lon_window_dataframe, smoothing_range, stock_blocks_dataframe)
            new_dataframe.to_hdf(save_path_3, key="df")
            print(save_path_3, dt.datetime.now())


def get_smoothing_limits_dataframe2(stock_blocks_dataframe, smoothing_range):
    left_dataframe: pd.DataFrame = stock_blocks_dataframe.copy()
    right_dataframe: pd.DataFrame = stock_blocks_dataframe.copy()
    top_dataframe: pd.DataFrame = stock_blocks_dataframe.copy()
    bottom_dataframe: pd.DataFrame = stock_blocks_dataframe.copy()
    len_list_index = len(left_dataframe.index)
    len_list_column = len(left_dataframe.columns)
    row_range = (-((smoothing_range - 1) // 2), (smoothing_range - 1) // 2)
    for i_index in range(len_list_index):
        lat = left_dataframe.iloc[i_index].name
        if lat > 69.0:
            pass
        columns_range = (-(round((smoothing_range / math.cos(math.radians(lat)) - 1) / 2)),
                         round((smoothing_range / math.cos(math.radians(lat)) - 1) / 2))
        for i_column in range(0, len_list_column, 1):
            top_dataframe.iloc[i_index, i_column] = bigger_or_x(i_index + row_range[0], 0)
            bottom_dataframe.iloc[i_index, i_column] = lower_or_x(i_index + row_range[1] + 1, len_list_index)
            left_dataframe.iloc[i_index, i_column] = bigger_or_x(i_column + columns_range[0], 0)
            right_dataframe.iloc[i_index, i_column] = lower_or_x(i_column + columns_range[1] + 1, len_list_column)

    return top_dataframe, bottom_dataframe, left_dataframe, right_dataframe


def get_smoothing_lon_window(stock_blocks_dataframe, smoothing_range):
    lon_window_dataframe = pd.DataFrame(columns=["window"], index=stock_blocks_dataframe.index)
    for lat in lon_window_dataframe.index:
        lon_window_dataframe.loc[lat, "window"] = round(smoothing_range / math.cos(math.radians(lat)) / 2 - 0.5) * 2 + 1
    return lon_window_dataframe


def main2024oct13_smoothing_proccess(dvtec_dataframe, stock_blocks_dataframe, smoothing_limits_dataframes):
    len_index = len(stock_blocks_dataframe.index)
    len_columns = len(stock_blocks_dataframe.columns)
    new_dataframe = stock_blocks_dataframe
    dvtec_dataframe = dvtec_dataframe.reindex(index=new_dataframe.index, columns=new_dataframe.columns)
    for i_index in range(0, len_index, 1):
        for i_column in range(0, len_columns, 1):
            new_dataframe.iloc[i_index, i_column] = dvtec_dataframe.iloc[
                smoothing_limits_dataframes[0].iloc[i_index, i_column]:
                smoothing_limits_dataframes[1].iloc[i_index, i_column],
                                                  smoothing_limits_dataframes[2].iloc[i_index, i_column]:
                                                  smoothing_limits_dataframes[3].iloc[i_index, i_column]].stack().mean(numeric_only=True)
    return new_dataframe


def main2024oct13_smoothing_proccess2(dvtec_dataframe, lon_window_dataframe, smoothing_range, stock_blocks_dataframe):
    def sliding_avg(values):
        valid_values = values[~np.isnan(values)]
        return np.mean(valid_values) if len(valid_values) > 0 else np.nan
    dvtec_dataframe = dvtec_dataframe.reindex(index=stock_blocks_dataframe.index, columns=stock_blocks_dataframe.columns)

    list_lon_window = np.unique(lon_window_dataframe.loc[:, "window"])
    dict_window_dataframe = {}
    for lon_window in list_lon_window:
        window_shape = (smoothing_range, lon_window)
        temp_result = generic_filter(dvtec_dataframe.values, sliding_avg, size=window_shape)
        temp_dataframe = pd.DataFrame(temp_result, index=dvtec_dataframe.index, columns=dvtec_dataframe.columns)
        dict_window_dataframe[lon_window] = temp_dataframe

    result_dataframe = pd.DataFrame()
    for lon_window, window_dataframe in dict_window_dataframe.items():
        indexes = lon_window_dataframe.loc[lon_window_dataframe.loc[:, "window"] == lon_window].index
        temp_dataframe = window_dataframe.loc[indexes]
        result_dataframe = pd.concat([result_dataframe, temp_dataframe])

    return result_dataframe


def main2024oct14():
    source_dir = r"/home/vadymskipa/PhD_student/data/Europe_36-70__-10-20/dvtec_csv_blocks/block_tables"
    save_dir_sum = r"/home/vadymskipa/PhD_student/data/Europe_36-70__-10-20/dvtec_csv_blocks/block_tables_smooth_sum"
    save_dir_nos = r"/home/vadymskipa/PhD_student/data/Europe_36-70__-10-20/dvtec_csv_blocks/block_tables_smooth_nos"
    window = "Window_3600_Seconds"
    lats = (36, 70)
    lons = (-10, 20)
    lat_tick = 0.15
    lon_tick = 0.15
    list_lats = [round((lats[0] + lat_tick * i), 2) for i in range(math.ceil((lats[1] - lats[0]) / lat_tick))]
    list_lons = [round((lons[0] + lon_tick * i), 2) for i in range(math.ceil((lons[1] - lons[0]) / lon_tick))]
    stock_blocks_dataframe = pd.DataFrame(columns=list_lons, index=list_lats)
    smoothing_range = 5
    lon_window_dataframe = get_smoothing_lon_window(stock_blocks_dataframe, smoothing_range)
    # smoothing_limits_dataframes = get_smoothing_limits_dataframe2(stock_blocks_dataframe, smoothing_range)

    entries = os.listdir(source_dir)
    list_date_dir = [dirname for dirname in entries if
                     (len(dirname) == 14 and os.path.isdir(os.path.join(source_dir, dirname)))]
    for date_dir in list_date_dir:
        source_path_1 = os.path.join(source_dir, date_dir)
        save_path_1 = get_directory_path(save_dir_sum, date_dir)
        save_path_2 = get_directory_path(save_path_1, window)
        save_path_nos_1 = get_directory_path(save_dir_nos, date_dir)
        save_path_nos_2 = get_directory_path(save_path_nos_1, window)
        source_path_2 = os.path.join(source_path_1, window)
        list_timefile = os.listdir(source_path_2)
        for timefile in list_timefile:
            source_path_3 = os.path.join(source_path_2, timefile)
            save_path_3 = os.path.join(save_path_2, timefile)
            save_path_nos_3 = os.path.join(save_path_nos_2, timefile)
            file_dataframe = pd.read_hdf(source_path_3, key="df")
            new_dataframe_sum, new_dataframe_count = main2024oct14_smoothing_proccess(file_dataframe, lon_window_dataframe, smoothing_range, stock_blocks_dataframe)
            new_dataframe_sum.to_hdf(save_path_3, key="df")
            new_dataframe_count.to_hdf(save_path_nos_3, key="df")
            print(date_dir, timefile, dt.datetime.now())


def main2024oct14_smoothing_proccess(dvtec_dataframe, lon_window_dataframe, smoothing_range, stock_blocks_dataframe):
    def sliding_sum(values):
        valid_values = values[~np.isnan(values)]
        return np.sum(valid_values) if len(valid_values) > 0 else np.nan
    def sliding_count(values):
        valid_values = values[~np.isnan(values)]
        return len(valid_values)
    dvtec_dataframe = dvtec_dataframe.reindex(index=stock_blocks_dataframe.index, columns=stock_blocks_dataframe.columns)

    list_lon_window = np.unique(lon_window_dataframe.loc[:, "window"])
    dict_window_dataframe_sum = {}
    dict_window_dataframe_count = {}
    for lon_window in list_lon_window:
        window_shape = (smoothing_range, lon_window)
        temp_result = generic_filter(dvtec_dataframe.values, sliding_sum, size=window_shape, mode="constant", cval=np.nan)
        temp_dataframe = pd.DataFrame(temp_result, index=dvtec_dataframe.index, columns=dvtec_dataframe.columns)
        dict_window_dataframe_sum[lon_window] = temp_dataframe
        temp_result = generic_filter(dvtec_dataframe.values, sliding_count, size=window_shape, mode="constant", cval=np.nan)
        temp_dataframe = pd.DataFrame(temp_result, index=dvtec_dataframe.index, columns=dvtec_dataframe.columns, dtype=np.int16)
        dict_window_dataframe_count[lon_window] = temp_dataframe

    result_dataframe_sum = pd.DataFrame()
    result_dataframe_count = pd.DataFrame()
    for lon_window, window_dataframe in dict_window_dataframe_sum.items():
        indexes = lon_window_dataframe.loc[lon_window_dataframe.loc[:, "window"] == lon_window].index
        temp_dataframe = window_dataframe.loc[indexes]
        result_dataframe_sum = pd.concat([result_dataframe_sum, temp_dataframe])
    for lon_window, window_dataframe in dict_window_dataframe_count.items():
        indexes = lon_window_dataframe.loc[lon_window_dataframe.loc[:, "window"] == lon_window].index
        temp_dataframe = window_dataframe.loc[indexes]
        result_dataframe_count = pd.concat([result_dataframe_count, temp_dataframe])

    return result_dataframe_sum, result_dataframe_count


def main2024oct14_mp():
    source_dir = r"/home/vadymskipa/PhD_student/data/Europe_36-70__-10-20/dvtec_csv_blocks/block_tables"
    save_dir_sum = r"/home/vadymskipa/PhD_student/data/Europe_36-70__-10-20/dvtec_csv_blocks/block_tables_smooth_sum"
    save_dir_nos = r"/home/vadymskipa/PhD_student/data/Europe_36-70__-10-20/dvtec_csv_blocks/block_tables_smooth_nos"
    window = "Window_7200_Seconds"
    lats = (36, 70)
    lons = (-10, 20)
    lat_tick = 0.15
    lon_tick = 0.15
    list_lats = [round((lats[0] + lat_tick * i), 2) for i in range(math.ceil((lats[1] - lats[0]) / lat_tick))]
    list_lons = [round((lons[0] + lon_tick * i), 2) for i in range(math.ceil((lons[1] - lons[0]) / lon_tick))]
    stock_blocks_dataframe = pd.DataFrame(columns=list_lons, index=list_lats)
    smoothing_range = 5
    lon_window_dataframe = get_smoothing_lon_window(stock_blocks_dataframe, smoothing_range)
    # smoothing_limits_dataframes = get_smoothing_limits_dataframe2(stock_blocks_dataframe, smoothing_range)\
    number_of_proccesses = nop = 8

    entries = os.listdir(source_dir)
    list_date_dir = [dirname for dirname in entries if
                     (len(dirname) == 14 and os.path.isdir(os.path.join(source_dir, dirname)))]
    for date_dir in list_date_dir:
        source_path_1 = os.path.join(source_dir, date_dir)
        save_path_1 = get_directory_path(save_dir_sum, date_dir)
        save_path_2 = get_directory_path(save_path_1, window)
        save_path_nos_1 = get_directory_path(save_dir_nos, date_dir)
        save_path_nos_2 = get_directory_path(save_path_nos_1, window)
        source_path_2 = os.path.join(source_path_1, window)
        list_timefile = os.listdir(source_path_2)
        for i in range(0, len(list_timefile), nop):
            list_input = []
            end = i + nop
            if end > len(list_timefile):
                end = len(list_timefile)
            for j in range(i, end, 1):
                timefile = list_timefile[j]
                source_path_3 = os.path.join(source_path_2, timefile)
                save_path_3 = os.path.join(save_path_2, timefile)
                save_path_nos_3 = os.path.join(save_path_nos_2, timefile)
                file_dataframe = pd.read_hdf(source_path_3, key="df")
                list_input.append((file_dataframe, lon_window_dataframe, smoothing_range, stock_blocks_dataframe,
                                   save_path_3, save_path_nos_3))
            pool = multipool.Pool(nop)
            for new_dataframe_sum, new_dataframe_count, save_path_3, save_path_nos_3 in pool.imap(main2024oct14_proccess, list_input):
                new_dataframe_sum.to_hdf(save_path_3, key="df")
                new_dataframe_count.to_hdf(save_path_nos_3, key="df")
                print(save_path_3, dt.datetime.now())


def main2024oct14_proccess(tup):
    file_dataframe, lon_window_dataframe, smoothing_range, stock_blocks_dataframe, save_path_3, save_path_nos_3 = tup
    new_dataframe_sum, new_dataframe_count = main2024oct14_smoothing_proccess(file_dataframe, lon_window_dataframe,
                                                                              smoothing_range, stock_blocks_dataframe)
    return new_dataframe_sum, new_dataframe_count, save_path_3, save_path_nos_3


def main2024oct15():
    lats = (20, 50)
    lons = (130, 160)

    min_date = dt.datetime(year=2023, month=2, day=24, tzinfo=dt.timezone.utc)
    max_date = dt.datetime(year=2023, month=3, day=3, tzinfo=dt.timezone.utc)
    source_site_hdf_directory_path = r"/home/vadymskipa/PhD_student/data/site_hdf/"
    source_los_hdf_directory = r"/home/vadymskipa/PhD_student/data/Japan_los_hdf/"
    los_hdfs_in_directory = os.listdir(source_los_hdf_directory)
    list_los_hdf_paths = []
    for los_hdf in los_hdfs_in_directory:
        temp_date = wlos.get_date_by_los_file_name(los_hdf)
        if temp_date < min_date:
            continue
        if temp_date > max_date:
            continue
        list_los_hdf_paths.append(os.path.join(source_los_hdf_directory, los_hdf))
    save_path = r"/home/vadymskipa/PhD_student/data/Japan_20-50__130-160/"
    save_los_tec_txt_file_for_region_from_los_files(list_los_hdf_paths, source_site_hdf_directory_path, save_path,
                                                    lats=lats, lons=lons, multiprocessing=True)


def main2024oct16():
    source_path = r"/home/vadymskipa/PhD_student/data/Japan_20-50__130-160/los_tec_txt"
    save_path = r"/home/vadymskipa/PhD_student/data/Japan_20-50__130-160/dtec_txt"
    params = LIST_PARAMS1
    min_date = dt.datetime(year=2023, month=2, day=24, tzinfo=dt.timezone.utc)
    max_date = dt.datetime(year=2023, month=3, day=3, tzinfo=dt.timezone.utc)
    get_dtec_from_los_tec_txt_for_lots_of_sites(source_path, save_path, params, min_date=min_date, max_date=max_date)


def main2024oct17():
    source_directories = [r"/home/vadymskipa/PhD_student/data/Japan_20-50__130-160/dtec_txt/055-2023-02-24",
                          r"/home/vadymskipa/PhD_student/data/Japan_20-50__130-160/dtec_txt/056-2023-02-25",
                          r"/home/vadymskipa/PhD_student/data/Japan_20-50__130-160/dtec_txt/057-2023-02-26",
                          r"/home/vadymskipa/PhD_student/data/Japan_20-50__130-160/dtec_txt/058-2023-02-27",
                          r"/home/vadymskipa/PhD_student/data/Japan_20-50__130-160/dtec_txt/059-2023-02-28",
                          r"/home/vadymskipa/PhD_student/data/Japan_20-50__130-160/dtec_txt/060-2023-03-01",
                          r"/home/vadymskipa/PhD_student/data/Japan_20-50__130-160/dtec_txt/061-2023-03-02",
                          r"/home/vadymskipa/PhD_student/data/Japan_20-50__130-160/dtec_txt/062-2023-03-03"]
    site_directory = r"/home/vadymskipa/PhD_student/data/site_hdf/"
    site_paths = wsite.get_site_file_paths_from_directory(site_directory)
    for source in source_directories:
        date = get_date_from_date_directory_name(os.path.basename(source))
        site_file = None
        for site_path in site_paths:
            if wsite.check_site_file_name_by_date(os.path.basename(site_path), date):
                site_file = site_path
                break
        save_sites_txt_for_site_directories(source, site_file, source)


def main2024oct18():
    save_dir = r"/home/vadymskipa/PhD_student/data/Europe_36-70__-10-20/dvtec_csv_blocks/latitude_time_map/hdf"
    source_dir_sum = r"/home/vadymskipa/PhD_student/data/Europe_36-70__-10-20/dvtec_csv_blocks/block_tables_smooth_sum"
    source_dir_nos = r"/home/vadymskipa/PhD_student/data/Europe_36-70__-10-20/dvtec_csv_blocks/block_tables_smooth_nos"
    window = "Window_3600_Seconds"

    entries = os.listdir(source_dir_sum)
    list_date_dir = [dirname for dirname in entries if
                     (len(dirname) == 14 and os.path.isdir(os.path.join(source_dir_sum, dirname)))]
    for date_dir in list_date_dir:
        date = get_date_from_date_directory_name(date_dir)
        source_path_sum_1 = os.path.join(source_dir_sum, date_dir)
        source_path_nos_1 = os.path.join(source_dir_nos, date_dir)
        source_path_sum_2 = os.path.join(source_path_sum_1, window)
        source_path_nos_2 = os.path.join(source_path_nos_1, window)
        save_path_1 = os.path.join(save_dir, window, f"{date_dir}.hdf5")
        list_timefile = os.listdir(source_path_sum_2)
        result_dataframe = pd.DataFrame()
        for timefile in list_timefile:
            datetime = dt.datetime(year=date.year, month=date.month, day=date.day,
                                   hour=int(timefile[:2]), minute=int(timefile[2:4]), second=int(timefile[4:6]),
                                   tzinfo=dt.timezone.utc)
            source_path_sum_3 = os.path.join(source_path_sum_2, timefile)
            source_path_nos_3 = os.path.join(source_path_nos_2, timefile)
            sum_dataframe = pd.read_hdf(source_path_sum_3, key="df")
            nos_dataframe = pd.read_hdf(source_path_nos_3, key="df")
            temp_dataframe = convert_lat_lon_table_to_lat_time_row(sum_dataframe, nos_dataframe, datetime)
            result_dataframe = pd.concat([result_dataframe, temp_dataframe])
        result_dataframe.to_hdf(save_path_1, key="df")



def convert_lat_lon_table_to_lat_time_row(sum_dataframe, nos_dataframe, datetime):
    avg_dataframe = sum_dataframe / nos_dataframe
    result_dataframe = avg_dataframe.mean(axis=1).to_frame(name=datetime).T
    return result_dataframe









def main2024oct():
    # main2024oct1_mp()
    # main2024oct2()
    # main2024oct3()
    # main2024oct4()
    # main2024oct5()
    # main2024oct6()
    # main2024oct7()
    # main2024oct8()
    # main2024oct9_mp()
    # main2024oct10()
    # main2024oct11()
    main2024oct12()
    # main2024oct13()
    # main2024oct14()
    main2024oct14_mp()
    # main2024oct15()
    # main2024oct16()
    # main2024oct17()
    # main2024oct18()


def main2024nov1():
    save_dir = r"/home/vadymskipa/PhD_student/data/Europe_36-70__-10-20/dvtec_csv_blocks/latitude_time_map/hdf"
    source_dir_sum = r"/home/vadymskipa/PhD_student/data/Europe_36-70__-10-20/dvtec_csv_blocks/block_tables_smooth_sum"
    source_dir_nos = r"/home/vadymskipa/PhD_student/data/Europe_36-70__-10-20/dvtec_csv_blocks/block_tables_smooth_nos"
    window = "Window_7200_Seconds"
    time_step = 120
    time_step_str = f"Time_step_{time_step}_seconds"

    entries = os.listdir(source_dir_sum)
    list_date_dir = [dirname for dirname in entries if
                     (len(dirname) == 14 and os.path.isdir(os.path.join(source_dir_sum, dirname)))]
    for date_dir in list_date_dir:
        date = get_date_from_date_directory_name(date_dir)
        source_path_sum_1 = os.path.join(source_dir_sum, date_dir)
        source_path_nos_1 = os.path.join(source_dir_nos, date_dir)
        source_path_sum_2 = os.path.join(source_path_sum_1, window)
        source_path_nos_2 = os.path.join(source_path_nos_1, window)
        save_path_1 = get_directory_path(save_dir, time_step_str)
        save_path_2 = get_directory_path(save_path_1, window)
        save_path_3 = os.path.join(save_path_2, f"{date_dir}.hdf5")
        list_timefile = os.listdir(source_path_sum_2)
        list_time = [dt.datetime(year=date.year, month=date.month, day=date.day,
                                 hour=int(timefile[:2]), minute=int(timefile[2:4]), second=int(timefile[4:6]),
                                 tzinfo=dt.timezone.utc).timestamp() for timefile in list_timefile]
        timefile_dataframe = pd.DataFrame({"timestamp": list_time, "timefile": list_timefile})

        result_dataframe = pd.DataFrame()
        for timestamp in range(int(date.timestamp()), int(date.timestamp()) + 24 * 3600, time_step):
            temp_timefile_dataframe = timefile_dataframe.loc[timefile_dataframe.loc[:, "timestamp"] >= timestamp]
            temp_timefile_dataframe = temp_timefile_dataframe.loc[temp_timefile_dataframe.loc[:, "timestamp"] <
                                                                  timestamp + time_step]
            if temp_timefile_dataframe.empty:
                continue
            sum_dataframe = pd.DataFrame()
            nos_dataframe = pd.DataFrame()
            for index in temp_timefile_dataframe.index:
                temp_timefile = temp_timefile_dataframe.loc[index, "timefile"]
                temp_source_path_sum_3 = os.path.join(source_path_sum_2, temp_timefile)
                temp_source_path_nos_3 = os.path.join(source_path_nos_2, temp_timefile)
                temp_sum_dataframe = pd.read_hdf(temp_source_path_sum_3, key="df")
                temp_nos_dataframe = pd.read_hdf(temp_source_path_nos_3, key="df")
                if sum_dataframe.empty:
                    sum_dataframe = pd.DataFrame(np.nan, index=temp_sum_dataframe.index,
                                                 columns=temp_sum_dataframe.columns)
                    nos_dataframe = pd.DataFrame(np.nan, index=temp_nos_dataframe.index,
                                                 columns=temp_nos_dataframe.columns)
                sum_dataframe = sum_dataframe.add(temp_sum_dataframe, fill_value=0)
                nos_dataframe = nos_dataframe.add(temp_nos_dataframe, fill_value=0)
            datetime = dt.datetime.fromtimestamp(timestamp, tz=dt.timezone.utc)
            temp_dataframe = convert_lat_lon_table_to_lat_time_row(sum_dataframe, nos_dataframe, datetime)
            result_dataframe = pd.concat([result_dataframe, temp_dataframe])
            print(dt.datetime.fromtimestamp(timestamp, tz=dt.timezone.utc), dt.datetime.now())
        result_dataframe.to_hdf(save_path_3, key="df")


def main2024nov2():

    lats = (36, 70)
    lons = (-10, 20)
    min_date = dt.datetime(year=2020, month=9, day=23, tzinfo=dt.timezone.utc)
    max_date = dt.datetime(year=2020, month=9, day=23, tzinfo=dt.timezone.utc)
    source_site_hdf_directory_path = r"/home/vadymskipa/PhD_student/data/site_hdf/"
    source_los_hdf_directory = r"/home/vadymskipa/PhD_student/data/Europe_los_hdf/"
    los_hdfs_in_directory = os.listdir(source_los_hdf_directory)
    list_los_hdf_paths = []
    for los_hdf in los_hdfs_in_directory:
        temp_date = wlos.get_date_by_los_file_name(los_hdf)
        if temp_date < min_date:
            continue
        if temp_date > max_date:
            continue
        list_los_hdf_paths.append(os.path.join(source_los_hdf_directory, los_hdf))
    save_path = r"/home/vadymskipa/PhD_student/data/Europe_36-70__-10-20"
    save_los_tec_txt_file_for_region_from_los_files(list_los_hdf_paths, source_site_hdf_directory_path, save_path,
                                                    lats=lats, lons=lons, multiprocessing=True)


def main2024nov3():
    source_path = r"/home/vadymskipa/PhD_student/data/Europe_36-70__-10-20/los_tec_txt"
    save_path = r"/home/vadymskipa/PhD_student/data/Europe_36-70__-10-20/dtec_txt"
    params = LIST_PARAMS1
    min_date = dt.datetime(year=2020, month=9, day=23, tzinfo=dt.timezone.utc)
    max_date = dt.datetime(year=2020, month=9, day=23, tzinfo=dt.timezone.utc)
    get_dtec_from_los_tec_txt_for_lots_of_sites(source_path, save_path, params, min_date=min_date, max_date=max_date)


def main2024nov4():
    directory = r"/home/vadymskipa/PhD_student/data/Europe_36-70__-10-20/dvtec_csv_blocks/latitude_time_map/"
    window = "Window_3600_Seconds"
    time_step = 120
    time_step_str = f"Time_step_{time_step}_seconds"
    amplitude=0.5
    amplitude_str = f"Amplitude_{amplitude}"

    source_path_1 = os.path.join(directory, "hdf")
    save_path_1 = get_directory_path(directory, "png")
    source_path_2 = os.path.join(source_path_1, time_step_str)
    save_path_2 = get_directory_path(save_path_1, time_step_str)
    source_path_3 = os.path.join(source_path_2, window)
    save_path_3 = get_directory_path(save_path_2, window)
    save_path_4 = get_directory_path(save_path_3, amplitude_str)
    list_files = os.listdir(source_path_3)
    for file in list_files:
        filename_root = os.path.splitext(file)[1]
        source_path_4 = os.path.join(source_path_3, file)
        source_dataframe = pd.read_hdf(source_path_4, key="df")
        result = plot_time_lat_dvtec_variation(source_dataframe, save_path_4, amplitude=amplitude, time_step_sec=time_step,
                                               lat_step=0.15)
        print(result, dt.datetime.now())


def plot_time_lat_dvtec_variation(time_lat_table_dataframe, save_directory, amplitude, time_step_sec, lat_step):
    list_dataframe = time_lat_table_dataframe.stack().reset_index()
    list_dataframe.columns = ["datetime", "gdlat", "dvtec"]
    list_dataframe = list_dataframe.loc[list_dataframe.loc[:, "dvtec"].notna()]
    time_step = dt.timedelta(seconds=time_step_sec)

    min_datetime = list_dataframe.loc[:, "datetime"].min()
    max_datetime = list_dataframe.loc[:, "datetime"].max()
    start_datetime = dt.datetime(year=min_datetime.year, month=min_datetime.month, day=min_datetime.day,
                                 hour=min_datetime.hour, tzinfo=dt.timezone.utc)
    temp_datetime = start_datetime
    time_tick = dt.timedelta(hours=round((max_datetime - min_datetime).seconds / 3600 / 12))
    list_time_ticks = []
    while temp_datetime <= max_datetime:
        list_time_ticks.append(temp_datetime)
        temp_datetime += time_tick
    list_time_tick_name = [datetime.hour for datetime in list_time_ticks]
    if (max_datetime.minute == 0 and max_datetime.second == 0):
        time_limits = (start_datetime, dt.datetime(year=max_datetime.year, month=max_datetime.month,
                                                   day=max_datetime.day, hour=max_datetime.hour, tzinfo=dt.timezone.utc)
    )
    else:
        time_limits = (start_datetime, dt.datetime(year=max_datetime.year, month=max_datetime.month,
                                                   day=max_datetime.day, hour=max_datetime.hour,
                                                   tzinfo=dt.timezone.utc) + dt.timedelta(seconds=3600))

    min_lat = list_dataframe.loc[:, "gdlat"].min()
    max_lat = list_dataframe.loc[:, "gdlat"].max()
    start_lat = math.ceil(min_lat)
    temp_lat = start_lat
    lat_tick = round((max_lat - min_lat) / 6)
    list_lat_ticks = []
    while temp_lat <= max_lat:
        list_lat_ticks.append(temp_lat)
        temp_lat += lat_tick
    lat_limits = (int(min_lat), math.ceil(max_lat))

    fig = plt.figure(layout="tight", figsize = [12, 4.8])
    ax = fig.add_subplot(1, 1, 1)
    color_normalize = mplcolors.Normalize(-amplitude, amplitude)
    colormap = plt.colormaps["viridis"]
    fig.colorbar(mplcm.ScalarMappable(norm=color_normalize, cmap=colormap), ax=ax)
    ax.set(xticks=list_time_ticks, xticklabels=list_time_tick_name, xlabel="time, hours", xlim=time_limits,
           yticks=list_lat_ticks, ylabel="dvtec, TECU", ylim=lat_limits)
    ax.grid(visible=True)

    for index in list_dataframe.index:
        lat = list_dataframe.loc[index, "gdlat"]
        time = list_dataframe.loc[index, "datetime"]
        color = colormap(color_normalize(list_dataframe.loc[index, "dvtec"]))
        list_x = [time, time, time + time_step, time + time_step, time]
        list_y = [lat, lat + lat_step, lat + lat_step, lat, lat]
        ax.fill(list_x, list_y, color=color)
        if index % 10000 == 0:
            print(index, dt.datetime.now())

    save_path = os.path.join(save_directory, f"{get_date_str(min_datetime)}.png")
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    return save_path


def main2024nov5():
    source_los_directory_path = r"/home/vadymskipa/HDD2/big_los_hdf/"
    source_site_directory_path = r"/home/vadymskipa/PhD_student/data/site_hdf/"
    save_region_directory_path = r"/home/vadymskipa/PhD_student/data/Europe_los_hdf/"
    start_date = dt.datetime(year=2023, month=3, day=30, tzinfo=dt.timezone.utc)
    end_date = dt.datetime(year=2023, month=4, day=5, tzinfo=dt.timezone.utc)
    lf5py.save_region_los_files_from_directory(source_los_directory_path, source_site_directory_path,
                                         save_region_directory_path, region_name="europe", lats=(30, 80),
                                         lons=(-10, 50),
                                         repeatition_of_region_files=True,
                                         multiprocessing=True,
                                         start_date=start_date, end_date=end_date)


def main2024nov6():

    lats = (30, 80)
    lons = (-10, 50)
    min_date = dt.datetime(year=2023, month=3, day=30, tzinfo=dt.timezone.utc)
    max_date = dt.datetime(year=2023, month=4, day=5, tzinfo=dt.timezone.utc)
    source_site_hdf_directory_path = r"/home/vadymskipa/PhD_student/data/site_hdf/"
    source_los_hdf_directory = r"/home/vadymskipa/PhD_student/data/Europe_los_hdf/"
    los_hdfs_in_directory = os.listdir(source_los_hdf_directory)
    list_los_hdf_paths = []
    for los_hdf in los_hdfs_in_directory:
        temp_date = wlos.get_date_by_los_file_name(los_hdf)
        if temp_date < min_date:
            continue
        if temp_date > max_date:
            continue
        list_los_hdf_paths.append(os.path.join(source_los_hdf_directory, los_hdf))
    save_path = r"/home/vadymskipa/PhD_student/data/Europe_30-80__-10-50"
    save_los_tec_txt_file_for_region_from_los_files(list_los_hdf_paths, source_site_hdf_directory_path, save_path,
                                                    lats=lats, lons=lons, multiprocessing=True)


def main2024nov7():
    source_path = r"/home/vadymskipa/PhD_student/data/Europe_30-80__-10-50/los_tec_txt"
    save_path = r"/home/vadymskipa/PhD_student/data/Europe_30-80__-10-50/dtec_txt"
    params = LIST_PARAMS1
    min_date = dt.datetime(year=2023, month=3, day=30, tzinfo=dt.timezone.utc)
    max_date = dt.datetime(year=2023, month=4, day=5, tzinfo=dt.timezone.utc)
    get_dtec_from_los_tec_txt_for_lots_of_sites(source_path, save_path, params, min_date=min_date, max_date=max_date)


def main2024nov8():
    source_directories = [r"/home/vadymskipa/PhD_student/data/Europe_30-80__-10-50/dtec_txt/089-2023-03-30",
                          r"/home/vadymskipa/PhD_student/data/Europe_30-80__-10-50/dtec_txt/090-2023-03-31",
                          r"/home/vadymskipa/PhD_student/data/Europe_30-80__-10-50/dtec_txt/091-2023-04-01",
                          r"/home/vadymskipa/PhD_student/data/Europe_30-80__-10-50/dtec_txt/092-2023-04-02",
                          r"/home/vadymskipa/PhD_student/data/Europe_30-80__-10-50/dtec_txt/093-2023-04-03",
                          r"/home/vadymskipa/PhD_student/data/Europe_30-80__-10-50/dtec_txt/094-2023-04-04",
                          r"/home/vadymskipa/PhD_student/data/Europe_30-80__-10-50/dtec_txt/095-2023-04-05"]
    site_directory = r"/home/vadymskipa/PhD_student/data/site_hdf/"
    site_paths = wsite.get_site_file_paths_from_directory(site_directory)
    for source in source_directories:
        date = get_date_from_date_directory_name(os.path.basename(source))
        site_file = None
        for site_path in site_paths:
            if wsite.check_site_file_name_by_date(os.path.basename(site_path), date):
                site_file = site_path
                break
        save_sites_txt_for_site_directories(source, site_file, source)


def main2024nov9():
    source_dir = r"/home/vadymskipa/PhD_student/data/Europe_36-70__-10-20/dvtec_csv_blocks/all_sats"
    save_dir = r"/home/vadymskipa/PhD_student/data/Europe_36-70__-10-20/dvtec_csv_blocks/longitude_time_noobsmap"
    window = "Window_7200_Seconds"
    amplitude = 100
    time_step_sec = 120
    coord_name = "gdlon"

    entries = os.listdir(source_dir)
    list_date_dir = [dirname for dirname in entries if
                     (len(dirname) == 14 and os.path.isdir(os.path.join(source_dir, dirname)))]
    for date_dir in list_date_dir:
        source_path_1 = os.path.join(source_dir, date_dir)
        save_path_1 = get_directory_path(save_dir, date_dir)
        save_path_2 = get_directory_path(save_path_1, window)
        source_path_2 = os.path.join(source_path_1, f"{window}.hdf5")
        source_dataframe = pd.read_hdf(source_path_2, key="df")
        result_dataframe = source_dataframe.groupby(["timestamp", coord_name]).size().reset_index(name='noobs')
        temp_timestamp = source_dataframe.iloc[0].loc["timestamp"]
        temp_date = dt.datetime.fromtimestamp(temp_timestamp, tz=dt.timezone.utc)
        min_timestamp = dt.datetime(year=temp_date.year, month=temp_date.month, day=temp_date.day,
                                    tzinfo=dt.timezone.utc).timestamp()
        result_dataframe.loc[:, "new_timestamp"] = result_dataframe.loc[:, "timestamp"].apply(
            lambda x: min_timestamp + (x - min_timestamp) // time_step_sec * time_step_sec)
        result_dataframe = (result_dataframe.loc[:, ("new_timestamp", coord_name, "noobs")]
                            .groupby(["new_timestamp", coord_name]).sum().reset_index())
        result_dataframe.loc[:, "new_noobs"] = result_dataframe.loc[:, "noobs"] / (time_step_sec / 30)
        result_dataframe.drop(["noobs"], axis=1, inplace=True)
        result_dataframe.rename(columns={"new_noobs": "noobs", "new_timestamp": "timestamp"}, inplace=True)
        result_dataframe = add_datetime_column_to_df(result_dataframe)
        result = plot_time_coord_noobs(result_dataframe, save_path_2, coord_column=coord_name, amplitude=amplitude,
                                       time_step_sec=time_step_sec, coord_step=0.15)
        print(result, dt.datetime.now())


def plot_time_coord_noobs(time_coord_dataframe, save_directory, coord_column, amplitude, time_step_sec, coord_step):
    time_step = dt.timedelta(seconds=time_step_sec)

    min_datetime = time_coord_dataframe.loc[:, "datetime"].min()
    max_datetime = time_coord_dataframe.loc[:, "datetime"].max()
    start_datetime = dt.datetime(year=min_datetime.year, month=min_datetime.month, day=min_datetime.day,
                                 hour=min_datetime.hour, tzinfo=dt.timezone.utc)
    temp_datetime = start_datetime
    time_tick = dt.timedelta(hours=round((max_datetime - min_datetime).seconds / 3600 / 12))
    list_time_ticks = []
    while temp_datetime <= max_datetime:
        list_time_ticks.append(temp_datetime)
        temp_datetime += time_tick
    list_time_tick_name = [datetime.hour for datetime in list_time_ticks]
    if (max_datetime.minute == 0 and max_datetime.second == 0):
        time_limits = (start_datetime, dt.datetime(year=max_datetime.year, month=max_datetime.month,
                                                   day=max_datetime.day, hour=max_datetime.hour, tzinfo=dt.timezone.utc)
    )
    else:
        time_limits = (start_datetime, dt.datetime(year=max_datetime.year, month=max_datetime.month,
                                                   day=max_datetime.day, hour=max_datetime.hour,
                                                   tzinfo=dt.timezone.utc) + dt.timedelta(seconds=3600))

    min_coord = time_coord_dataframe.loc[:, coord_column].min()
    max_coord = time_coord_dataframe.loc[:, coord_column].max()
    start_coord = math.ceil(min_coord)
    temp_coord = start_coord
    coord_tick = round((max_coord - min_coord) / 6)
    list_coord_ticks = []
    while temp_coord <= max_coord:
        list_coord_ticks.append(temp_coord)
        temp_coord += coord_tick
    coord_limits = (int(min_coord), math.ceil(max_coord))

    fig = plt.figure(layout="tight", figsize = [12, 4.8])
    ax = fig.add_subplot(1, 1, 1)
    color_normalize = mplcolors.LogNorm(1, amplitude)
    colormap = plt.colormaps["viridis"]
    fig.colorbar(mplcm.ScalarMappable(norm=color_normalize, cmap=colormap), ax=ax)
    ax.set(xticks=list_time_ticks, xticklabels=list_time_tick_name, xlabel="time, hours", xlim=time_limits,
           yticks=list_coord_ticks, ylabel=coord_column, ylim=coord_limits)
    ax.grid(visible=True)
    ax.set_title("Number of observation")

    for index in time_coord_dataframe.index:
        lat = time_coord_dataframe.loc[index, coord_column]
        time = time_coord_dataframe.loc[index, "datetime"]
        color = colormap(color_normalize(time_coord_dataframe.loc[index, "noobs"]))
        list_x = [time, time, time + time_step, time + time_step, time]
        list_y = [lat, lat + coord_step, lat + coord_step, lat, lat]
        ax.fill(list_x, list_y, color=color)

    save_path = os.path.join(save_directory, f"{get_date_str(min_datetime)}.png")
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    return save_path


def main2024nov10():
    source_dir = r"/home/vadymskipa/PhD_student/data/Europe_36-70__-10-20/dvtec_csv_blocks/all_sats"
    save_dir = r"/home/vadymskipa/PhD_student/data/Europe_36-70__-10-20/dvtec_csv_blocks"
    window = "Window_3600_Seconds"
    amplitude = 0.5
    time_step_sec = 120
    list_pixel_width = [1, 3, 5, 9]
    list_another_coords = [10, 12, 14, 16]
    coord_name = "latitude"
    coord_step = 0.15
    if coord_name == "latitude":
        coord_column = "gdlat"
        another_coord_column = "gdlon"
        another_coord_name = "longitude"
    elif coord_name == "longitude":
        coord_column = "gdlon"
        another_coord_column = "gdlat"
        another_coord_name = "latitude"
    save_dir = get_directory_path(save_dir, f"{coord_name[0].upper()}{coord_name[1:]}_time_map")


    entries = os.listdir(source_dir)
    list_date_dir = [dirname for dirname in entries if
                     (len(dirname) == 14 and os.path.isdir(os.path.join(source_dir, dirname)))]
    for date_dir in list_date_dir:
        source_path_1 = os.path.join(source_dir, date_dir)
        save_path_1 = get_directory_path(save_dir, window)
        source_path_2 = os.path.join(source_path_1, f"{window}.hdf5")
        source_dataframe = pd.read_hdf(source_path_2, key="df")
        for pixel_width, another_coord in itertools.product(list_pixel_width, list_another_coords):
            selected_dataframe = source_dataframe.loc[source_dataframe.loc[:, another_coord_column] >=
                                                     (another_coord - pixel_width * coord_step / 2)]
            selected_dataframe = selected_dataframe.loc[selected_dataframe.loc[:, another_coord_column] <
                                                     (another_coord + pixel_width * coord_step / 2)]
            temp_timestamp = selected_dataframe.iloc[0].loc["timestamp"]
            temp_date = dt.datetime.fromtimestamp(temp_timestamp, tz=dt.timezone.utc)
            min_timestamp = dt.datetime(year=temp_date.year, month=temp_date.month, day=temp_date.day,
                                        tzinfo=dt.timezone.utc).timestamp()
            selected_dataframe.loc[:, "new_timestamp"] = selected_dataframe.loc[:, "timestamp"].apply(
                lambda x: min_timestamp + (x - min_timestamp) // time_step_sec * time_step_sec)
            selected_dataframe = (selected_dataframe.loc[:, ("new_timestamp", coord_column, "dvtec")]
                                .groupby(["new_timestamp", coord_column]).mean().reset_index())
            selected_dataframe.rename(columns={"new_timestamp": "timestamp"}, inplace=True)
            selected_dataframe = add_datetime_column_to_df(selected_dataframe)
            save_path_2 = get_directory_path(save_path_1, f"Amplitude_{amplitude}TECU")
            save_path_3 = get_directory_path(save_path_2,
                                             f"{another_coord_name[0].upper()}{another_coord_name[1:]}_{another_coord}")
            save_path_4 = get_directory_path(save_path_3, f"Pixel_Width_{pixel_width}")
            selected_dataframe.to_hdf(os.path.join(save_path_4, f"{get_date_str(temp_date)}.hdf"), key="df")
            result = plot_time_coord_dvtec_variations(selected_dataframe, save_path_4, coord_column=coord_column,
                                                      amplitude=amplitude, time_step_sec=time_step_sec, coord_step=0.15,
                                                      coordiante_name=coord_name)
            print(result, dt.datetime.now())


def plot_time_coord_dvtec_variations(time_coord_dataframe, save_directory, coord_column, amplitude, time_step_sec,
                                     coord_step, coordiante_name):
    time_step = dt.timedelta(seconds=time_step_sec)

    min_datetime = time_coord_dataframe.loc[:, "datetime"].min()
    max_datetime = time_coord_dataframe.loc[:, "datetime"].max()
    start_datetime = dt.datetime(year=min_datetime.year, month=min_datetime.month, day=min_datetime.day,
                                 hour=min_datetime.hour, tzinfo=dt.timezone.utc)
    temp_datetime = start_datetime
    time_tick = dt.timedelta(hours=round((max_datetime - min_datetime).seconds / 3600 / 12))
    list_time_ticks = []
    while temp_datetime <= max_datetime:
        list_time_ticks.append(temp_datetime)
        temp_datetime += time_tick
    list_time_tick_name = [datetime.hour for datetime in list_time_ticks]
    if (max_datetime.minute == 0 and max_datetime.second == 0):
        time_limits = (start_datetime, dt.datetime(year=max_datetime.year, month=max_datetime.month,
                                                   day=max_datetime.day, hour=max_datetime.hour, tzinfo=dt.timezone.utc)
    )
    else:
        time_limits = (start_datetime, dt.datetime(year=max_datetime.year, month=max_datetime.month,
                                                   day=max_datetime.day, hour=max_datetime.hour,
                                                   tzinfo=dt.timezone.utc) + dt.timedelta(seconds=3600))

    min_coord = time_coord_dataframe.loc[:, coord_column].min()
    max_coord = time_coord_dataframe.loc[:, coord_column].max()
    start_coord = math.ceil(min_coord)
    temp_coord = start_coord
    coord_tick = round((max_coord - min_coord) / 6)
    list_coord_ticks = []
    while temp_coord <= max_coord:
        list_coord_ticks.append(temp_coord)
        temp_coord += coord_tick
    coord_limits = (int(min_coord), math.ceil(max_coord))

    fig = plt.figure(layout="tight", figsize = [12, 4.8])
    ax = fig.add_subplot(1, 1, 1)
    color_normalize = mplcolors.Normalize(-amplitude, amplitude)
    colormap = plt.colormaps["viridis"]
    fig.colorbar(mplcm.ScalarMappable(norm=color_normalize, cmap=colormap), ax=ax)
    ax.set(xticks=list_time_ticks, xticklabels=list_time_tick_name, xlabel="time, hours", xlim=time_limits,
           yticks=list_coord_ticks, ylabel=f"{coordiante_name[0].upper()}{coordiante_name[1:]}", ylim=coord_limits)
    ax.grid(visible=True)
    ax.set_title("dTEC, TEC units")

    for index in time_coord_dataframe.index:
        lat = time_coord_dataframe.loc[index, coord_column]
        time = time_coord_dataframe.loc[index, "datetime"]
        color = colormap(color_normalize(time_coord_dataframe.loc[index, "dvtec"]))
        list_x = [time, time, time + time_step, time + time_step, time]
        list_y = [lat, lat + coord_step, lat + coord_step, lat, lat]
        ax.fill(list_x, list_y, color=color)

    save_path = os.path.join(save_directory, f"{get_date_str(min_datetime)}.png")
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    return save_path


def main2024nov10_mp():
    source_dir = r"/home/vadymskipa/PhD_student/data/Europe_36-70__-10-20/dvtec_csv_blocks/all_sats"
    save_dir = r"/home/vadymskipa/PhD_student/data/Europe_36-70__-10-20/dvtec_csv_blocks"
    window = "Window_7200_Seconds"
    amplitude = 0.5
    time_step_sec = 120
    list_pixel_width = [1, 3, 5, 9]
    # list_another_coords = [10, 12, 14, 16]
    # coord_name = "latitude"
    list_another_coords = [42, 46, 50, 56, 60]
    coord_name = "longitude"
    coord_step = 0.15
    if coord_name == "latitude":
        coord_column = "gdlat"
        another_coord_column = "gdlon"
        another_coord_name = "longitude"
    elif coord_name == "longitude":
        coord_column = "gdlon"
        another_coord_column = "gdlat"
        another_coord_name = "latitude"
    save_dir = get_directory_path(save_dir, f"{coord_name[0].upper()}{coord_name[1:]}_time_map")
    number_of_processes = nop = 4

    entries = os.listdir(source_dir)
    list_date_dir = [dirname for dirname in entries if
                     (len(dirname) == 14 and os.path.isdir(os.path.join(source_dir, dirname)))]
    for date_dir in list_date_dir:
        source_path_1 = os.path.join(source_dir, date_dir)
        save_path_1 = get_directory_path(save_dir, window)
        source_path_2 = os.path.join(source_path_1, f"{window}.hdf5")
        source_dataframe = pd.read_hdf(source_path_2, key="df")
        input_list = []

        for pixel_width, another_coord in itertools.product(list_pixel_width, list_another_coords):
            selected_dataframe = source_dataframe.loc[source_dataframe.loc[:, another_coord_column] >=
                                                      (another_coord - pixel_width * coord_step / 2)]
            selected_dataframe = selected_dataframe.loc[selected_dataframe.loc[:, another_coord_column] <
                                                        (another_coord + pixel_width * coord_step / 2)]
            temp_timestamp = selected_dataframe.iloc[0].loc["timestamp"]
            temp_date = dt.datetime.fromtimestamp(temp_timestamp, tz=dt.timezone.utc)
            min_timestamp = dt.datetime(year=temp_date.year, month=temp_date.month, day=temp_date.day,
                                        tzinfo=dt.timezone.utc).timestamp()
            selected_dataframe.loc[:, "new_timestamp"] = selected_dataframe.loc[:, "timestamp"].apply(
                lambda x: min_timestamp + (x - min_timestamp) // time_step_sec * time_step_sec)
            selected_dataframe = (selected_dataframe.loc[:, ("new_timestamp", coord_column, "dvtec")]
                                  .groupby(["new_timestamp", coord_column]).mean().reset_index())
            selected_dataframe.rename(columns={"new_timestamp": "timestamp"}, inplace=True)
            selected_dataframe = add_datetime_column_to_df(selected_dataframe)
            save_path_2 = get_directory_path(save_path_1, f"Amplitude_{amplitude}TECU")
            save_path_3 = get_directory_path(save_path_2,
                                             f"{another_coord_name[0].upper()}{another_coord_name[1:]}_{another_coord}")
            save_path_4 = get_directory_path(save_path_3, f"Pixel_Width_{pixel_width}")
            selected_dataframe.to_hdf(os.path.join(save_path_4, f"{get_date_str(temp_date)}.hdf"), key="df")
            input_dict = {"time_coord_dataframe": selected_dataframe, "save_directory": save_path_4,
                          "coord_column": coord_column, "amplitude": amplitude, "time_step_sec": time_step_sec,
                          "coord_step": 0.15, "coordiante_name": coord_name}
            input_list.append(input_dict)
        pool = multipool.Pool(nop)
        for result in pool.imap(main2024nov10_process, input_list):
            print(result, dt.datetime.now())


def main2024nov10_process(input_dict):
    return plot_time_coord_dvtec_variations(**input_dict)


def main2024nov11():
    source_dir_1 = r"/home/vadymskipa/PhD_student/data/123/los_txt"
    source_dir_2 = r"/home/vadymskipa/PhD_student/data/processed_data_for_receivers/2024/"

    list_sites = os.listdir(source_dir_1)
    dataframe_1 = pd.DataFrame()
    for site in list_sites:
        source_path_1 = os.path.join(source_dir_1, site)
        list_datedirs = os.listdir(source_path_1)
        temp_site_dataframe = pd.DataFrame()
        for datedir in list_datedirs:
            source_path_2 = os.path.join(source_path_1, datedir)
            list_sat_ids = os.listdir(source_path_2)
            temp_date_dataframe = pd.DataFrame()
            date = get_date_from_date_directory_name(datedir)
            for sat_id_file in list_sat_ids:
                source_path_3 = os.path.join(source_path_2, sat_id_file)
                temp_sat_dataframe = read_sat_file(source_path_3)
                temp_sat_dataframe.loc[:, "sat_id"] = int(sat_id_file[1:3])
                temp_date_dataframe = pd.concat([temp_date_dataframe, temp_sat_dataframe], ignore_index=True)
            temp_date_dataframe = add_timestamp_column_to_df(temp_date_dataframe, date)
            temp_site_dataframe = pd.concat([temp_site_dataframe, temp_date_dataframe], ignore_index=True)
        temp_site_dataframe.loc[:, "gps_site"] = site
        dataframe_1 = pd.concat([dataframe_1, temp_site_dataframe], ignore_index=True)
    dataframe_2 = pd.DataFrame()

    list_datefile = os.listdir(source_dir_2)
    for datefile in list_datefile:
        source_path_1 = os.path.join(source_dir_2, datefile)
        date = dt.datetime(year=int(datefile[9:13]), month=1, day=1, tzinfo=dt.timezone.utc)
        date_timestamp = date.timestamp()
        with h5py.File(source_path_1) as file:
            temp_date_dataframe = pd.DataFrame(file["data"][:])
        temp_date_dataframe.loc[:, "timestamp"] = round(temp_date_dataframe.loc[:, "time"] * 24 * 3600 + date_timestamp)
        dataframe_2 = pd.concat([dataframe_2, temp_date_dataframe], ignore_index=True)

    result_text = ""
    for site in np.unique(dataframe_1.loc[:, "gps_site"]):
        site_dataframe_1 = dataframe_1.loc[dataframe_1.loc[:, "gps_site"] == site]
        site_dataframe_2 = dataframe_2.loc[dataframe_2.loc[:, "site"] == site.lower().encode("ascii")]
        for sat_id in np.unique(site_dataframe_1.loc[:, "sat_id"]):
            sat_dataframe_1 = site_dataframe_1.loc[site_dataframe_1.loc[:, "sat_id"] == sat_id].reset_index()
            sat_dataframe_2 = site_dataframe_2.loc[site_dataframe_2.loc[:, "prn"] == sat_id]
            sat_dataframe_2 = sat_dataframe_2.loc[sat_dataframe_2.loc[:, "sat_type"] == 0]

            list_time_periods = get_time_periods(sat_dataframe_1.loc[:, "timestamp"], 300, 300)
            for start, finish in list_time_periods:
                temp_sat_dataframe_1 = sat_dataframe_1.loc[start:finish]
                merged_dataframe = pd.merge(temp_sat_dataframe_1.loc[:, ["timestamp", "los_tec"]],
                                            sat_dataframe_2.loc[:, ["timestamp", "los_tec"]],
                                            on='timestamp', how='inner', suffixes=("_1", "_2"))
                if merged_dataframe.empty:
                    result_text += (f"- | - | - | - | - sat {sat_id}, site {site}\n"
                          f"{dt.datetime.fromtimestamp(temp_sat_dataframe_1.iloc[0].loc['timestamp'], tz=dt.timezone.utc)}"
                          f"\n"
                          f"{dt.datetime.fromtimestamp(temp_sat_dataframe_1.iloc[-1].loc['timestamp'], tz=dt.timezone.utc)}"
                          f"\n"
                          f"empty\n")
                    continue
                merged_dataframe.loc[:, "diff"] = (merged_dataframe.loc[:, "los_tec_1"] -
                                                   merged_dataframe.loc[:, "los_tec_2"])
                mean = merged_dataframe.loc[:, "diff"].mean()
                std = merged_dataframe.loc[:, "diff"].std(ddof=0)
                result_text += (f"- | - | - | - | - sat {sat_id}, site {site}\n"
                      f"{dt.datetime.fromtimestamp(merged_dataframe.iloc[0].loc['timestamp'], tz=dt.timezone.utc)}\n"
                      f"{dt.datetime.fromtimestamp(merged_dataframe.iloc[-1].loc['timestamp'], tz=dt.timezone.utc)}\n"
                      f"mean = {mean:0=4},   std = {std:0=2}\n")
                start_date = dt.datetime.fromtimestamp(temp_sat_dataframe_1.iloc[0].loc['timestamp'], tz=dt.timezone.utc)
                plot_main2024nov11(r"/home/vadymskipa/PhD_student/data/processed_data_for_receivers/difference_plots_svg",
                                   f"Site{site}__sat{sat_id:0=2}__time{start_date.year:0=4}"
                                   f"{start_date.month:0=2}{start_date.day:0=2}_{start_date.hour:0=2}"
                                   f"{start_date.minute:0=2}{start_date.second:0=2}.svg", merged_dataframe,
                                   title=f"Site {site}, sat {sat_id:0=2}, mean = {mean:.4f},   std = {std:.4f}")
                print(f"Site{site}__sat{sat_id:0=2}__time{start_date.year:0=4}"
                                   f"{start_date.month:0=2}{start_date.day:0=2}_{start_date.hour:0=2}"
                                   f"{start_date.minute:0=2}{start_date.second:0=2}.txt")

    with open(os.path.join(r"/home/vadymskipa/PhD_student/data/", "result.txt"), "w") as file:
        file.write(result_text)


def plot_main2024nov11(save_path, name, dataframe, title=None):
    # title = name
    figure: fig.Figure = plt.figure(layout="tight", figsize=[16.0 * 160 / 300, 9.0 * 160 / 300])
    dataframe = add_datetime_column_to_df(dataframe)
    axes1: axs.Axes = figure.add_subplot(2, 1, 1)
    ytext1 = axes1.set_ylabel("STEC, TEC units")
    xtext1 = axes1.set_xlabel("Time")
    # time_array = dataframe.loc[:, "timestamp"] / 3600
    time_array = dataframe.loc[:, "datetime"]
    line1, = axes1.plot(time_array, dataframe["los_tec_1"], label="Data from GPS-TEC", linestyle=" ",
                        marker=".", color="blue", markeredgewidth=1, markersize=1.1)

    line2, = axes1.plot(time_array, dataframe["los_tec_2"], label="Data from partners",
                        linestyle=" ", marker=".", color="red", markeredgewidth=1, markersize=1.1)
    axes1.legend()
    axes2: axs.Axes = figure.add_subplot(2, 1, 2)
    line4, = axes2.plot(time_array, dataframe["diff"], linestyle=" ", marker=".",
                        markeredgewidth=1,
                        markersize=1,
                        label="Difference between outer data and GPS-TEC")
    axes2.legend()
    axes2.set_xlim(*axes1.get_xlim())
    axes2.grid(True)
    if title:
        figure.suptitle(title)
    plt.savefig(os.path.join(save_path, name), dpi=300)
    plt.close(figure)


def main2024nov12(save_dir = r"/home/vadymskipa/PhD_student/data/123/longitude_direction/hdf",
                  vertical_horizontal_ratio = 1.5, source_dir = r"/home/vadymskipa/PhD_student/data/123/dvtec_csv",
                  min_elm = 25):

    list_windows = []
    list_site = []
    list_date_dir = os.listdir(source_dir)
    source_dataframe = pd.DataFrame()
    for date_dir in list_date_dir:
        source_path_1 = os.path.join(source_dir, date_dir)
        date = get_date_from_date_directory_name(date_dir)
        list_site = os.listdir(source_path_1)
        date_dataframe = pd.DataFrame()
        for site in list_site:
            source_path_2 = os.path.join(source_path_1, site)
            list_windows = os.listdir(source_path_2)
            site_dataframe = pd.DataFrame()
            for window in list_windows:
                source_path_3 = os.path.join(source_path_2, window)
                list_sat_files = os.listdir(source_path_3)
                window_dataframe = pd.DataFrame()
                for sat_file in list_sat_files:
                    source_path_4 = os.path.join(source_path_3, sat_file)
                    sat_dataframe = pd.read_csv(source_path_4, index_col=0)
                    sat_id = int(sat_file[1:3])
                    sat_dataframe.loc[:, "sat_id"] = sat_id
                    window_dataframe = pd.concat([window_dataframe, sat_dataframe], ignore_index=True)
                window_dataframe.loc[:, "window"] = window
                site_dataframe = pd.concat([site_dataframe, window_dataframe], ignore_index=True)
            site_dataframe.loc[:, "gps_site"] = site
            date_dataframe = pd.concat([date_dataframe, site_dataframe], ignore_index=True)
        date_dataframe = add_timestamp_column_to_df(date_dataframe, date)
        source_dataframe = pd.concat([source_dataframe, date_dataframe], ignore_index=True)
    source_dataframe = source_dataframe.loc[source_dataframe.loc[:, "elm"] > min_elm]
    directional_dataframe = pd.DataFrame()
    for site in list_site:
        source_site_dataframe = source_dataframe.loc[source_dataframe.loc[:, "gps_site"] == site].reset_index(drop=True)
        for window in list_windows:
            source_window_dataframe = (source_site_dataframe.loc[source_site_dataframe.loc[:, "window"] == window].
                                       reset_index(drop=True))
            list_sat_ids = np.unique(source_window_dataframe.loc[:, "sat_id"])
            for sat_id in list_sat_ids:
                source_sat_dataframe = (source_window_dataframe.loc[source_window_dataframe.loc[:, "sat_id"] == sat_id].
                                       reset_index(drop=True))
                list_time_periods = get_time_periods(source_sat_dataframe.loc[:, "timestamp"], 30, 30)
                for start, end in list_time_periods:
                    source_period_dataframe = source_sat_dataframe.loc[start: end].reset_index(drop=True)
                    if len(source_period_dataframe) < 100:
                        continue
                    temp_directinal_dataframe = get_directional_part_of_observation(source_period_dataframe,
                                                                                    vertical_horizontal_ratio)
                    directional_dataframe = pd.concat([directional_dataframe, temp_directinal_dataframe],
                                                      ignore_index=True)
    directional_dataframe = add_datetime_column_to_df(directional_dataframe)
    for site in list_site:
        save_path_1 = get_directory_path(save_dir, site)
        directional_site_dataframe = directional_dataframe.loc[directional_dataframe.loc[:, "gps_site"] == site]
        for window in list_windows:
            save_path_2 = get_directory_path(save_path_1, window)
            directional_window_dataframe = directional_site_dataframe.loc[
                directional_site_dataframe.loc[:, "window"] == window]
            temp_date_series = directional_window_dataframe.loc[:, "datetime"].apply(
                lambda datetime: dt.datetime(year=datetime.year, month=datetime.month, day=datetime.day,
                                             tzinfo=dt.timezone.utc))
            list_dates = np.unique(temp_date_series)
            for date in list_dates:
                filename = get_date_str(date) + ".hdf5"
                directional_date_dataframe = directional_window_dataframe.loc[temp_date_series == date]
                directional_date_dataframe.to_hdf(os.path.join(save_path_2, filename), key="df")




def get_directional_part_of_observation(dataframe: pd.DataFrame, vertical_horizontal_ratio, time_diff=300,
                                        min_result_dataframe=900):
    diff_period = int(time_diff // (dataframe.iloc[1].loc["timestamp"] - dataframe.iloc[0].loc["timestamp"]))
    min_result_period = int(min_result_dataframe // (dataframe.iloc[1].loc["timestamp"] - dataframe.iloc[0].loc["timestamp"]))
    vert_diff = dataframe.loc[:, "gdlat"].diff(diff_period).abs()
    horiz_diff = (dataframe.loc[:, "gdlon"].diff(diff_period).abs() *
                  dataframe.loc[:, "gdlat"].apply(lambda lat: math.cos(math.radians(lat))))
    ratio_series = vert_diff / horiz_diff
    if vertical_horizontal_ratio > 1:
        condition_series = ratio_series > vertical_horizontal_ratio
    else:
        condition_series = ratio_series < vertical_horizontal_ratio
    condition_series = condition_series.astype(int)
    condition_diff = condition_series.diff()
    prestart_indexes = condition_diff.loc[condition_diff == 1].index
    preend_indexes = condition_diff.loc[condition_diff == -1].index
    result_dataframe = pd.DataFrame()
    for prestart, preend in itertools.zip_longest(prestart_indexes, preend_indexes, fillvalue=len(dataframe)):
        temp_dataframe = dataframe.loc[prestart - diff_period: preend - 1]
        if len(temp_dataframe) < min_result_period:
            continue
        result_dataframe = pd.concat([result_dataframe, temp_dataframe], ignore_index=True)
    if result_dataframe.empty:
        return None
    else:
        return result_dataframe


def main2024nov13(save_dir = r"/home/vadymskipa/PhD_student/data/123/longitude_direction/png",
    source_dir = r"/home/vadymskipa/PhD_student/data/123/longitude_direction/hdf"):
    list_site = os.listdir(source_dir)
    for site in list_site:
        source_path_1 = os.path.join(source_dir, site)
        save_path_1 = get_directory_path(save_dir, site)
        list_windows = os.listdir(source_path_1)
        for window in list_windows:
            source_path_2 = os.path.join(source_path_1, window)
            save_path_2 = get_directory_path(save_path_1, window)
            list_datefiles = os.listdir(source_path_2)
            for datefile in list_datefiles:
                source_path_3 = os.path.join(source_path_2, datefile)
                dataframe : pd.DataFrame = pd.read_hdf(source_path_3, key="df")
                new_dataframe = (dataframe.loc[:, ("datetime", "dvtec")]
                                 .groupby(["datetime"]).mean()).reset_index()
                noobs_series = ((dataframe.loc[:, ("datetime", "gps_site")]
                                 .groupby(["datetime"]).apply(lambda sites: len(sites))).to_frame(name="noobs").
                                reset_index())
                new_dataframe.loc[:, "noobs"] = noobs_series.loc[:, "noobs"]
                plot_directional_graph_main2024nov13(new_dataframe, save_path_2, datefile[:-4], min_noobs=1)
                print(site, window, datefile, dt.datetime.now())




def plot_directional_graph_main2024nov13(dataframe: pd.DataFrame, save_path, name, min_noobs=0):
    dataframe = dataframe.loc[dataframe.loc[:, "noobs"] >= min_noobs]
    figure: fig.Figure = plt.figure(layout="tight", figsize=[16.0 * 160 / 300, 9.0 * 160 / 300])
    axes1: axs.Axes = figure.add_subplot(2, 1, 1)
    ytext1 = axes1.set_ylabel("dTEC, TEC units")
    xtext1 = axes1.set_xlabel("Time, hrs")
    # time_array = dataframe.loc[:, "timestamp"] / 3600
    time_array = dataframe.loc[:, "datetime"]
    line1, = axes1.plot(time_array, dataframe["dvtec"], linestyle=" ",
                        marker=".", color="blue", markeredgewidth=1, markersize=1.1)
    axes1.set_xlim(time_array.iloc[0], time_array.iloc[-1])
    axes1.set_ylim(-1.0, 1.0)
    axes2: axs.Axes = figure.add_subplot(2, 1, 2)
    line4, = axes2.plot(time_array, dataframe["noobs"], linestyle=" ", marker=".",
                        markeredgewidth=1,
                        markersize=1,)
    axes2.set_xlim(*axes1.get_xlim())
    axes2.grid(True)
    plt.savefig(os.path.join(save_path, name + ".png"), dpi=300)
    plt.close(figure)


def main2024nov14():
    main2024nov12(save_dir = r"/home/vadymskipa/PhD_student/data/123/latitude_direction/hdf",
                  vertical_horizontal_ratio = 0.66)
    main2024nov13(save_dir=r"/home/vadymskipa/PhD_student/data/123/latitude_direction/png",
                  source_dir=r"/home/vadymskipa/PhD_student/data/123/latitude_direction/hdf")


def main2024nov15(save_dir = r"/home/vadymskipa/PhD_student/data/123/longitude_direction_2/hdf",
                  lat_limits=(-90, 90), lon_limits=(-180, 180),
                  source_dir = r"/home/vadymskipa/PhD_student/data/123/dvtec_csv",
                  min_elm = 30):

    list_windows = []
    list_site = []
    list_date_dir = os.listdir(source_dir)
    source_dataframe = pd.DataFrame()
    for date_dir in list_date_dir:
        source_path_1 = os.path.join(source_dir, date_dir)
        date = get_date_from_date_directory_name(date_dir)
        list_site = os.listdir(source_path_1)
        date_dataframe = pd.DataFrame()
        for site in list_site:
            source_path_2 = os.path.join(source_path_1, site)
            list_windows = os.listdir(source_path_2)
            site_dataframe = pd.DataFrame()
            for window in list_windows:
                source_path_3 = os.path.join(source_path_2, window)
                list_sat_files = os.listdir(source_path_3)
                window_dataframe = pd.DataFrame()
                for sat_file in list_sat_files:
                    source_path_4 = os.path.join(source_path_3, sat_file)
                    sat_dataframe = pd.read_csv(source_path_4, index_col=0)
                    sat_id = int(sat_file[1:3])
                    sat_dataframe.loc[:, "sat_id"] = sat_id
                    window_dataframe = pd.concat([window_dataframe, sat_dataframe], ignore_index=True)
                window_dataframe.loc[:, "window"] = window
                site_dataframe = pd.concat([site_dataframe, window_dataframe], ignore_index=True)
            site_dataframe.loc[:, "gps_site"] = site
            date_dataframe = pd.concat([date_dataframe, site_dataframe], ignore_index=True)
        date_dataframe = add_timestamp_column_to_df(date_dataframe, date)
        source_dataframe = pd.concat([source_dataframe, date_dataframe], ignore_index=True)
    source_dataframe = source_dataframe.loc[source_dataframe.loc[:, "elm"] > min_elm]
    directional_dataframe = pd.DataFrame()
    for site in list_site:
        source_site_dataframe = source_dataframe.loc[source_dataframe.loc[:, "gps_site"] == site].reset_index(drop=True)
        for window in list_windows:
            source_window_dataframe = (source_site_dataframe.loc[source_site_dataframe.loc[:, "window"] == window].
                                       reset_index(drop=True))
            list_sat_ids = np.unique(source_window_dataframe.loc[:, "sat_id"])
            for sat_id in list_sat_ids:
                source_sat_dataframe = (source_window_dataframe.loc[source_window_dataframe.loc[:, "sat_id"] == sat_id].
                                       reset_index(drop=True))
                list_time_periods = get_time_periods(source_sat_dataframe.loc[:, "timestamp"], 30, 30)
                for start, end in list_time_periods:
                    source_period_dataframe = source_sat_dataframe.loc[start: end].reset_index(drop=True)
                    if len(source_period_dataframe) < 100:
                        continue
                    temp_directinal_dataframe = get_directional_part_of_observation_2(source_period_dataframe,
                                          lat_limits=lat_limits, lon_limits=lon_limits)
                    directional_dataframe = pd.concat([directional_dataframe, temp_directinal_dataframe],
                                                      ignore_index=True)
    directional_dataframe = add_datetime_column_to_df(directional_dataframe)
    for site in list_site:
        save_path_1 = get_directory_path(save_dir, site)
        directional_site_dataframe = directional_dataframe.loc[directional_dataframe.loc[:, "gps_site"] == site]
        for window in list_windows:
            save_path_2 = get_directory_path(save_path_1, window)
            directional_window_dataframe = directional_site_dataframe.loc[
                directional_site_dataframe.loc[:, "window"] == window]
            temp_date_series = directional_window_dataframe.loc[:, "datetime"].apply(
                lambda datetime: dt.datetime(year=datetime.year, month=datetime.month, day=datetime.day,
                                             tzinfo=dt.timezone.utc))
            list_dates = np.unique(temp_date_series)
            for date in list_dates:
                filename = get_date_str(date) + ".hdf5"
                directional_date_dataframe = directional_window_dataframe.loc[temp_date_series == date]
                directional_date_dataframe.to_hdf(os.path.join(save_path_2, filename), key="df")




def get_directional_part_of_observation_2(dataframe: pd.DataFrame, duration=1800,
                                          lat_limits=(-90, 90), lon_limits=(-180, 180)):
    min_result_duration = int(duration // (dataframe.iloc[1].loc["timestamp"] - dataframe.iloc[0].loc["timestamp"]))
    lat_mask = np.logical_and(dataframe.loc[:, "gdlat"] >= lat_limits[0], dataframe.loc[:, "gdlat"] <= lat_limits[1])
    lon_mask = np.logical_and(dataframe.loc[:, "gdlon"] >= lon_limits[0], dataframe.loc[:, "gdlon"] <= lon_limits[1])
    mask = np.logical_and(lat_mask, lon_mask)
    result_dataframe = dataframe.loc[mask]
    if len(result_dataframe) < min_result_duration:
        return None
    else:
        return result_dataframe


def main2024nov16():
    main2024nov15(save_dir=r"/home/vadymskipa/PhD_student/data/123/longitude_direction_2/hdf",
                  lon_limits=(33, 39), min_elm=20,
                  source_dir=r"/home/vadymskipa/PhD_student/data/123/dvtec_csv")
    main2024nov15(save_dir=r"/home/vadymskipa/PhD_student/data/123/latitude_direction_2/hdf",
                  lat_limits=(48, 52), min_elm=20,
                  source_dir=r"/home/vadymskipa/PhD_student/data/123/dvtec_csv")
    main2024nov13(save_dir=r"/home/vadymskipa/PhD_student/data/123/longitude_direction_2/png",
                  source_dir=r"/home/vadymskipa/PhD_student/data/123/longitude_direction_2/hdf")
    main2024nov13(save_dir=r"/home/vadymskipa/PhD_student/data/123/latitude_direction_2/png",
                  source_dir=r"/home/vadymskipa/PhD_student/data/123/latitude_direction_2/hdf")


def main2024nov17():
    main2024nov15(save_dir=r"/home/vadymskipa/PhD_student/data/123/avg_dvtec/hdf", min_elm=30,
                  source_dir=r"/home/vadymskipa/PhD_student/data/123/dvtec_csv")
    main2024nov18(save_dir = r"/home/vadymskipa/PhD_student/data/123/avg_dvtec/png",
                  source_dir = r"/home/vadymskipa/PhD_student/data/123/avg_dvtec/hdf")


def main2024nov18(save_dir = r"/home/vadymskipa/PhD_student/data/123/longitude_direction/png",
    source_dir = r"/home/vadymskipa/PhD_student/data/123/longitude_direction/hdf"):
    list_site = os.listdir(source_dir)
    for site in list_site:
        source_path_1 = os.path.join(source_dir, site)
        save_path_1 = get_directory_path(save_dir, site)
        list_windows = os.listdir(source_path_1)
        for window in list_windows:
            source_path_2 = os.path.join(source_path_1, window)
            save_path_2 = get_directory_path(save_path_1, window)
            list_datefiles = os.listdir(source_path_2)
            for datefile in list_datefiles:
                source_path_3 = os.path.join(source_path_2, datefile)
                dataframe : pd.DataFrame = pd.read_hdf(source_path_3, key="df")
                new_dataframe = (dataframe.loc[:, ("datetime", "dvtec")]
                                 .groupby(["datetime"]).mean()).reset_index()
                noobs_series = ((dataframe.loc[:, ("datetime", "gps_site")]
                                 .groupby(["datetime"]).apply(lambda sites: len(sites))).to_frame(name="noobs").
                                reset_index())
                new_dataframe.loc[:, "noobs"] = noobs_series.loc[:, "noobs"]
                plot_avg_and_all_graph_main2024nov18(new_dataframe, dataframe, save_path_2, datefile[:-4], min_noobs=1)
                print(site, window, datefile, dt.datetime.now())




def plot_avg_and_all_graph_main2024nov18(dataframe: pd.DataFrame, all_obs_dataframe, save_path, name, min_noobs=0):
    dataframe = dataframe.loc[dataframe.loc[:, "noobs"] >= min_noobs]
    figure: fig.Figure = plt.figure(layout="tight", figsize=[16.0 * 160 / 300, 9.0 * 160 / 300])
    axes1: axs.Axes = figure.add_subplot(2, 1, 1)
    ytext1 = axes1.set_ylabel("dTEC, TEC units")
    xtext1 = axes1.set_xlabel("Time, hrs")
    # time_array = dataframe.loc[:, "timestamp"] / 3600
    time_array = dataframe.loc[:, "datetime"]
    line2, = axes1.plot(all_obs_dataframe.loc[:, "datetime"], all_obs_dataframe["dvtec"], linestyle=" ",
                        marker=".", color="blue", markeredgewidth=1, markersize=1.1)
    line1, = axes1.plot(time_array, dataframe["dvtec"], linestyle=" ",
                        marker=".", color="black", markeredgewidth=1, markersize=1.5)
    axes1.set_xlim(time_array.iloc[0], time_array.iloc[-1])
    axes1.set_ylim(-1.0, 1.0)
    axes2: axs.Axes = figure.add_subplot(2, 1, 2)
    line4, = axes2.plot(time_array, dataframe["noobs"], linestyle=" ", marker=".",
                        markeredgewidth=1,
                        markersize=1,)
    axes2.set_xlim(*axes1.get_xlim())
    axes2.grid(True)
    plt.savefig(os.path.join(save_path, name + ".png"), dpi=300)
    plt.close(figure)








def main2024nov():
    # main2024nov1()
    # main2024nov2()
    # main2024nov3()
    # main2024nov4()
    # main2024nov5()
    # main2024nov6()
    # main2024nov7()
    # main2024nov8()
    # main2024nov9()
    # main2024nov10()
    # main2024nov10_mp()
    # main2024nov11()
    # main2024nov12()
    # main2024nov13()
    # main2024nov14()
    # main2024nov16()
    main2024nov17()


def main2024dec1(save_dir = r"/home/vadymskipa/PhD_student/data/123",
                 source_dir = r"/home/vadymskipa/PhD_student/data/123",
                 min_elm = 30, speed_limits_kmph_list = (500,)):
    save_dir_name = "speed_limited_dvtec"
    save_path_1 = get_directory_path(save_dir, save_dir_name)
    source_path_1 = os.path.join(source_dir, "dvtec_csv")
    for speed_limit_kmph in speed_limits_kmph_list:
        save_path_2 = get_directory_path(save_path_1, f"speed_limit_{speed_limit_kmph}kmph")
        source_dataframe = read_dvtec_csv_dir(source_path_1)
        list_sites = np.unique(source_dataframe.loc[:, "gps_site"])
        for site in list_sites:
            site_source_dataframe = source_dataframe.loc[source_dataframe.loc[:, "gps_site"] == site]
            speed_limited_dataframe = processing_site_dvtec_dataframe_for_speed_limitation(site_source_dataframe,
                                                                                           speed_limit_kmph)
            if speed_limited_dataframe is None:
                continue
            save_site_dataframe_as_window_date_hdf_hierarchy(speed_limited_dataframe, save_path_2, site)
            print(f"speed_limit_{speed_limit_kmph}kmph", site, dt.datetime.now())





def read_dvtec_csv_dir(dvtec_csv_dir):
    list_date_dirs = os.listdir(dvtec_csv_dir)
    result_dataframe = pd.DataFrame()
    for date_dir in list_date_dirs:
        source_path_1 = os.path.join(dvtec_csv_dir, date_dir)
        date = get_date_from_date_directory_name(date_dir)
        list_site = os.listdir(source_path_1)
        date_dataframe = pd.DataFrame()
        for site in list_site:
            source_path_2 = os.path.join(source_path_1, site)
            list_windows = os.listdir(source_path_2)
            site_dataframe = pd.DataFrame()
            for window in list_windows:
                source_path_3 = os.path.join(source_path_2, window)
                list_sat_files = os.listdir(source_path_3)
                window_dataframe = pd.DataFrame()
                for sat_file in list_sat_files:
                    source_path_4 = os.path.join(source_path_3, sat_file)
                    sat_dataframe = pd.read_csv(source_path_4, index_col=0)
                    sat_id = int(sat_file[1:3])
                    sat_dataframe.loc[:, "sat_id"] = sat_id
                    window_dataframe = pd.concat([window_dataframe, sat_dataframe], ignore_index=True)
                window_dataframe.loc[:, "window"] = window
                site_dataframe = pd.concat([site_dataframe, window_dataframe], ignore_index=True)
            site_dataframe.loc[:, "gps_site"] = site
            date_dataframe = pd.concat([date_dataframe, site_dataframe], ignore_index=True)
        date_dataframe = add_timestamp_column_to_df(date_dataframe, date)
        date_dataframe = add_datetime_column_to_df(date_dataframe)
        result_dataframe = pd.concat([result_dataframe, date_dataframe], ignore_index=True)
    return result_dataframe


def processing_site_dvtec_dataframe_for_speed_limitation(site_dvtec_dataframe: pd.DataFrame, speed_limit_kmph):
    speed_limited_dataframe = pd.DataFrame()
    list_windows = np.unique(site_dvtec_dataframe.loc[:, "window"])
    for window in list_windows:
        source_window_dataframe = (site_dvtec_dataframe.loc[site_dvtec_dataframe.loc[:, "window"] == window].
                                   reset_index(drop=True))
        list_sat_ids = np.unique(source_window_dataframe.loc[:, "sat_id"])
        for sat_id in list_sat_ids:
            source_sat_dataframe = (source_window_dataframe.loc[source_window_dataframe.loc[:, "sat_id"] == sat_id].
                                    reset_index(drop=True))
            list_time_periods = get_time_periods(source_sat_dataframe.loc[:, "timestamp"], 30, 30)
            for start, end in list_time_periods:
                source_period_dataframe = source_sat_dataframe.loc[start: end].reset_index(drop=True)
                if len(source_period_dataframe) < 100:
                    continue
                temp_speed_limited_dataframe = processing_one_rec_sat_obs_for_speed_limitation(source_period_dataframe,
                                                                                    speed_limit_kmph=speed_limit_kmph)
                if temp_speed_limited_dataframe is None:
                    continue
                speed_limited_dataframe = pd.concat([speed_limited_dataframe, temp_speed_limited_dataframe],
                                                  ignore_index=True)
    if speed_limited_dataframe.empty:
        return None
    return speed_limited_dataframe

def processing_one_rec_sat_obs_for_speed_limitation(dataframe: pd.DataFrame, speed_limit_kmph, time_diff_s=600,
                                                    min_result_durration_s=1800):
    kmpd = 40000 / 360
    speed_limit_d = (speed_limit_kmph * time_diff_s / 3600) / kmpd
    diff_period = int(time_diff_s // (dataframe.iloc[1].loc["timestamp"] - dataframe.iloc[0].loc["timestamp"]))
    min_result_period = int(
        min_result_durration_s // (dataframe.iloc[1].loc["timestamp"] - dataframe.iloc[0].loc["timestamp"]))
    vert_diff = dataframe.loc[:, "gdlat"].diff(diff_period).abs()
    horiz_diff = (dataframe.loc[:, "gdlon"].diff(diff_period).abs() *
                  dataframe.loc[:, "gdlat"].apply(lambda lat: math.cos(math.radians(lat))))
    distance_diff = np.sqrt(vert_diff ** 2  + horiz_diff ** 2)
    condition_series = distance_diff < speed_limit_d
    condition_series = condition_series.astype(int)
    condition_diff = condition_series.diff()
    prestart_indexes = condition_diff.loc[condition_diff == 1].index
    preend_indexes = condition_diff.loc[condition_diff == -1].index
    result_dataframe = pd.DataFrame()
    for prestart, preend in itertools.zip_longest(prestart_indexes, preend_indexes, fillvalue=len(dataframe)):
        temp_dataframe = dataframe.loc[prestart - diff_period: preend - 1]
        if len(temp_dataframe) < min_result_period:
            continue
        result_dataframe = pd.concat([result_dataframe, temp_dataframe], ignore_index=True)
    if result_dataframe.empty:
        return None
    else:
        return result_dataframe


def save_site_dataframe_as_window_date_hdf_hierarchy(dataframe, save_directory, site):
    list_windows = np.unique(dataframe.loc[:, "window"])
    save_path_1 = get_directory_path(save_directory, site)
    for window in list_windows:
        save_path_2 = get_directory_path(save_path_1, window)
        window_dataframe = dataframe.loc[dataframe.loc[:, "window"] == window]
        temp_date_series = window_dataframe.loc[:, "datetime"].apply(
            lambda datetime: dt.datetime(year=datetime.year, month=datetime.month, day=datetime.day,
                                         tzinfo=dt.timezone.utc))
        list_dates = np.unique(temp_date_series)
        for date in list_dates:
            filename = get_date_str(date) + ".hdf5"
            date_dataframe = window_dataframe.loc[temp_date_series == date]
            date_dataframe.to_hdf(os.path.join(save_path_2, filename), key="df")


def main2024dec2(source_dir=r"/home/vadymskipa/PhD_student/data/123/speed_limited_dvtec"):
    save_dir_name = os.path.split(source_dir)[1] + "_png"
    save_path_0 = get_directory_path(os.path.split(source_dir)[0], save_dir_name)
    list_speed_limit_dirs = os.listdir(source_dir)
    for speed_limit_dir in list_speed_limit_dirs:
        save_path_1 = get_directory_path(save_path_0, speed_limit_dir)
        source_path_1 = os.path.join(source_dir, speed_limit_dir)
        list_site_dirs = os.listdir(source_path_1)
        for site_dir in list_site_dirs:
            save_path_2 = get_directory_path(save_path_1, site_dir)
            source_path_2 = os.path.join(source_path_1, site_dir)
            list_window_dirs = os.listdir(source_path_2)
            for window_dir in list_window_dirs:
                save_path_3 = get_directory_path(save_path_2, window_dir)
                source_path_3 = os.path.join(source_path_2, window_dir)
                list_date_hdf = os.listdir(source_path_3)
                for date_hdf in list_date_hdf:
                    source_path_4 = os.path.join(source_path_3, date_hdf)
                    filename = date_hdf[:-5] + ".png"
                    dataframe = pd.read_hdf(source_path_4, "df")
                    plot_day_dvtec(dataframe, save_path_3, filename)
                    print(speed_limit_dir, site_dir, window_dir, filename, dt.datetime.now())


def plot_day_dvtec(dataframe: pd.DataFrame, save_dir, save_filename):
    dataframe.sort_values(by="datetime", inplace=True)
    figure: fig.Figure = plt.figure(layout="tight", figsize=[16.0 * 160 / 300, 9.0 * 160 / 300])
    axes1: axs.Axes = figure.add_subplot(1, 1, 1)
    ytext1 = axes1.set_ylabel("dTEC, TEC units")
    xtext1 = axes1.set_xlabel("Time, hrs")
    # time_array = dataframe.loc[:, "timestamp"] / 3600
    time_array = dataframe.loc[:, "datetime"]
    date = dt.datetime(year=time_array.iloc[0].year, month=time_array.iloc[0].month, day=time_array.iloc[0].day,
                       tzinfo=dt.timezone.utc)
    line2, = axes1.plot(dataframe.loc[:, "datetime"], dataframe["dvtec"], linestyle="-",
                        marker="", color="blue", markeredgewidth=1, markersize=1.1)
    axes1.set_xlim(date, date + dt.timedelta(days=1))
    axes1.set_ylim(-1.0, 1.0)
    axes1.grid(True)
    plt.savefig(os.path.join(save_dir, save_filename), dpi=300)
    plt.close(figure)


def main2024dec3(save_dir = r"/home/vadymskipa/PhD_student/data/123",
                 source_dir = r"/home/vadymskipa/PhD_student/data/123",
                 min_elm = 30, relative=False):
    if relative:
        save_dir_name = "observation_difference_relative"
    else:
        save_dir_name = "observation_difference"
    save_path_1 = get_directory_path(save_dir, save_dir_name)
    source_path_1 = os.path.join(source_dir, "dvtec_csv")
    source_dataframe = read_dvtec_csv_dir(source_path_1)
    source_dataframe = source_dataframe.loc[source_dataframe.loc[:, "elm"] >= min_elm]
    list_sites = np.unique(source_dataframe.loc[:, "gps_site"])
    for site in list_sites:
        save_path_2 = get_directory_path(save_path_1, site)
        site_source_dataframe = source_dataframe.loc[source_dataframe.loc[:, "gps_site"] == site]
        list_windows = np.unique(site_source_dataframe.loc[:, "window"])
        for window in list_windows:
            save_path_3 = get_directory_path(save_path_2, window)
            window_source_dataframe = site_source_dataframe.loc[site_source_dataframe.loc[:, "window"] == window]
            if relative:
                window_source_dataframe = processing_dataframe_to_relative(window_source_dataframe)
            processed_dataframe = processing_dataframe_for_obs_difference(window_source_dataframe)
            processed_dataframe = add_datetime_column_to_df(processed_dataframe)
            temp_date_series = processed_dataframe.loc[:, "datetime"].apply(
                lambda datetime: dt.datetime(year=datetime.year, month=datetime.month, day=datetime.day,
                                             tzinfo=dt.timezone.utc))
            list_dates = np.unique(temp_date_series)
            for date in list_dates:
                filename = get_date_str(date) + ".hdf5"
                date_dataframe = processed_dataframe.loc[temp_date_series == date]
                date_dataframe.to_hdf(os.path.join(save_path_3, filename), key="df")


def processing_dataframe_to_relative(site_dvtec_dataframe: pd.DataFrame):
    processed_dataframe = pd.DataFrame()
    list_windows = np.unique(site_dvtec_dataframe.loc[:, "window"])
    for window in list_windows:
        source_window_dataframe = (site_dvtec_dataframe.loc[site_dvtec_dataframe.loc[:, "window"] == window].
                                   reset_index(drop=True))
        list_sat_ids = np.unique(source_window_dataframe.loc[:, "sat_id"])
        for sat_id in list_sat_ids:
            source_sat_dataframe = (source_window_dataframe.loc[source_window_dataframe.loc[:, "sat_id"] == sat_id].
                                    reset_index(drop=True))
            list_time_periods = get_time_periods(source_sat_dataframe.loc[:, "timestamp"], 30, 30)
            for start, end in list_time_periods:
                source_period_dataframe = source_sat_dataframe.loc[start: end].reset_index(drop=True)
                if len(source_period_dataframe) < 100:
                    continue
                source_period_dataframe.loc[:, "dvtec"] = (source_period_dataframe.loc[:, "dvtec"] /
                                                            source_period_dataframe.loc[:, "dvtec"].abs().max())
                processed_dataframe = pd.concat([processed_dataframe, source_period_dataframe],
                                                    ignore_index=True)
    if processed_dataframe.empty:
        return None
    return processed_dataframe



def processing_dataframe_for_obs_difference(dataframe: pd.DataFrame):
    result_dataframe = None
    list_sats = np.unique(dataframe.loc[:, "sat_id"])
    for sat1, sat2 in itertools.combinations(list_sats, 2):
        sat1_dataframe = dataframe.loc[dataframe.loc[:, "sat_id"] == sat1]
        sat2_dataframe = dataframe.loc[dataframe.loc[:, "sat_id"] == sat2]
        merged_dataframe = pd.merge(sat1_dataframe.loc[:, ["timestamp", "dtec", "dvtec", "azm", "elm", "gdlat", "gdlon", "sat_id"]],
                                    sat2_dataframe.loc[:, ["timestamp", "dtec", "dvtec", "azm", "elm", "gdlat", "gdlon", "sat_id"]],
                                    on='timestamp', how='inner', suffixes=("_1", "_2"))
        if merged_dataframe.empty:
            continue
        kmpd = 40000 / 360
        merged_dataframe.loc[:, "dvtec_difference"] = (
                (merged_dataframe.loc[:, "dvtec_1"] - merged_dataframe.loc[:, "dvtec_2"]).abs())
        merged_dataframe.loc[:, "gdlat_distance"] = (
                (merged_dataframe.loc[:, "gdlat_1"] - merged_dataframe.loc[:, "gdlat_2"]).abs() * kmpd)
        merged_dataframe.loc[:, "gdlon_distance"] = (
                (merged_dataframe.loc[:, "gdlon_1"] - merged_dataframe.loc[:, "gdlon_2"]).abs() *
                merged_dataframe.loc[:, "gdlat_1"].apply(lambda lat: math.cos(math.radians(lat))) * kmpd)
        merged_dataframe.loc[:, "distance"] = (
            np.sqrt(merged_dataframe.loc[:, "gdlat_distance"] ** 2 + merged_dataframe.loc[:, "gdlon_distance"] ** 2))
        result_dataframe = pd.concat([result_dataframe, merged_dataframe], ignore_index=True)
    return result_dataframe


def main2024dec4(save_dir = r"/home/vadymskipa/PhD_student/data/123",
                 source_dir = r"/home/vadymskipa/PhD_student/data/123", relative=False):
    source_dir_name = "observation_difference"
    save_dir_name = "observation_difference_png"
    if relative:
        source_dir_name = "observation_difference_relative"
        save_dir_name = "observation_difference_relative_png"
    source_path_1 = os.path.join(source_dir, source_dir_name)
    save_path_1 = get_directory_path(save_dir, save_dir_name)
    dict_save_params = {"latitude_distance": "gdlat_distance", "longitude_distance": "gdlon_distance",
                           "distance": "distance"}
    dict_save_path_1 = {key: get_directory_path(save_path_1, key) for key, value in dict_save_params.items()}
    list_sites = os.listdir(source_path_1)
    for site in list_sites:
        source_path_2 = os.path.join(source_path_1, site)
        dict_save_path_2 = {key: get_directory_path(save_path_1, site) for key, save_path_1 in dict_save_path_1.items()}
        list_windows = os.listdir(source_path_2)
        for window in list_windows:
            source_path_3 = os.path.join(source_path_2, window)
            dict_save_path_3 = {key: get_directory_path(save_path_2, window) for key, save_path_2 in dict_save_path_2.items()}
            list_files = os.listdir(source_path_3)
            for file in list_files:
                source_path_4 = os.path.join(source_path_3, file)
                new_file = file[:-5] + ".png"
                dataframe = pd.read_hdf(source_path_4, key="df")
                for key, value in dict_save_params.items():
                    plot_obs_differnce(dataframe, dict_save_path_3[key], new_file, key, value)


def plot_obs_differnce(dataframe: pd.DataFrame, save_dir, save_filename, parameter, column_name):
    figure: fig.Figure = plt.figure(layout="tight", figsize=[16.0 * 160 / 300, 9.0 * 160 / 300])
    axes1: axs.Axes = figure.add_subplot(1, 1, 1)
    ytext1 = axes1.set_ylabel(f"difference, TEC units")
    xtext1 = axes1.set_xlabel(f"{parameter}, km")
    line2, = axes1.plot(dataframe.loc[:, column_name], dataframe.loc[:, "dvtec_difference"], linestyle="",
                        marker=".", color="blue", markeredgewidth=1, markersize=1.1)
    max_distance = (dataframe.loc[:, column_name].max() // 100 + 1) * 100
    axes1.set_xlim(0, max_distance)
    axes1.set_ylim(0, 2.0)
    axes1.grid(True)
    plt.savefig(os.path.join(save_dir, save_filename), dpi=300)
    plt.close(figure)


def main2024dec5(save_dir = r"/home/vadymskipa/PhD_student/data/123",
                 source_dir = r"/home/vadymskipa/PhD_student/data/123",
                 min_elm = 30, oneside_local_window=5, number_of_extremums=3, min_dvtec=0.3):
    save_dir_name = "extremums"
    save_dir_name_1_5 = f"half_window_{oneside_local_window}"
    save_path_1 = get_directory_path(save_dir, save_dir_name)
    save_path_1_5 = get_directory_path(save_path_1, save_dir_name_1_5)
    source_path_1 = os.path.join(source_dir, "dvtec_csv")
    source_dataframe = read_dvtec_csv_dir(source_path_1)
    source_dataframe = source_dataframe.loc[source_dataframe.loc[:, "elm"] >= min_elm]
    list_sites = np.unique(source_dataframe.loc[:, "gps_site"])
    for site in list_sites:
        save_path_2 = get_directory_path(save_path_1_5, site)
        site_source_dataframe = source_dataframe.loc[source_dataframe.loc[:, "gps_site"] == site]
        list_windows = np.unique(site_source_dataframe.loc[:, "window"])
        for window in list_windows:
            save_path_3 = get_directory_path(save_path_2, window)
            window_source_dataframe = site_source_dataframe.loc[site_source_dataframe.loc[:, "window"] == window]
            temp_date_series = window_source_dataframe.loc[:, "datetime"].apply(
                lambda datetime: dt.datetime(year=datetime.year, month=datetime.month, day=datetime.day,
                                             tzinfo=dt.timezone.utc))
            list_dates = np.unique(temp_date_series)
            for date in list_dates:
                save_path_4_max = get_directory_path(save_path_3, "max")
                save_path_4_min = get_directory_path(save_path_3, "min")
                filename = get_date_str(date) + ".hdf5"
                date_dataframe = window_source_dataframe.loc[temp_date_series == date]
                tuple_extremum_dataframes = looking_for_extremums_in_site_dataframe(
                    date_dataframe, oneside_local_window=oneside_local_window, number_of_extremums=number_of_extremums,
                min_dvtec=min_dvtec)
                if tuple_extremum_dataframes[0] is not None:
                    tuple_extremum_dataframes[0].to_hdf(os.path.join(save_path_4_max, filename), key="df")
                if tuple_extremum_dataframes[1] is not None:
                    tuple_extremum_dataframes[1].to_hdf(os.path.join(save_path_4_min, filename), key="df")




def looking_for_extremums_in_site_dataframe(dataframe: pd.DataFrame, oneside_local_window=5, number_of_extremums=3,
                                            min_dvtec=0.3):
    max_dataframe = None
    min_dataframe = None
    list_sats = np.unique(dataframe.loc[:, "sat_id"])
    for sat in list_sats:
        sat_dataframe = dataframe.loc[dataframe.loc[:, "sat_id"] == sat]
        tuple_extremum_dataframes = looking_for_extremums_in_sat_dataframe(
            sat_dataframe, oneside_local_window=oneside_local_window, number_of_extremums=number_of_extremums,
            min_dvtec=min_dvtec)
        if tuple_extremum_dataframes[0] is not None:
            max_dataframe = pd.concat([max_dataframe, tuple_extremum_dataframes[0]], ignore_index=True)
        if tuple_extremum_dataframes[1] is not None:
            min_dataframe = pd.concat([min_dataframe, tuple_extremum_dataframes[1]], ignore_index=True)
    return max_dataframe, min_dataframe


def looking_for_extremums_in_sat_dataframe(dataframe: pd.DataFrame, oneside_local_window=5, number_of_extremums=3,
                                           min_dvtec=0.3):
    limited_local_max_dataframe = limited_local_min_dataframe = None
    shifted_dvtec_dataframes = pd.DataFrame()
    shifted_datetime_dataframes = pd.DataFrame()
    for i in range(-oneside_local_window, oneside_local_window + 1, 1):
        shifted_dvtec_dataframes.loc[: ,f"shift_{i}"] = dataframe.loc[:, "dvtec"].shift(i)
        shifted_datetime_dataframes.loc[: ,f"shift_{i}"] = dataframe.loc[:, "datetime"].shift(i)
    local_max_mask = dataframe.loc[:, "dvtec"] == shifted_dvtec_dataframes.max(axis=1)
    local_min_mask = dataframe.loc[:, "dvtec"] == shifted_dvtec_dataframes.min(axis=1)
    local_continuity_mask = dataframe.loc[:, "datetime"] == shifted_datetime_dataframes.mean(axis=1)
    local_max_dataframe = dataframe.loc[np.logical_and(local_max_mask, local_continuity_mask)]
    local_min_dataframe = dataframe.loc[np.logical_and(local_min_mask, local_continuity_mask)]
    local_max_dataframe = local_max_dataframe.loc[local_max_dataframe.loc[:, "dvtec"] >= min_dvtec]
    local_min_dataframe = local_min_dataframe.loc[local_min_dataframe.loc[:, "dvtec"] <= -min_dvtec]
    if not local_max_dataframe.empty:
        limited_local_max_dataframe = local_max_dataframe.nlargest(number_of_extremums, "dvtec", keep="all")
    if not local_min_dataframe.empty:
        limited_local_min_dataframe = local_min_dataframe.nsmallest(number_of_extremums, "dvtec", keep="all")
    return limited_local_max_dataframe, limited_local_min_dataframe


def main2024dec6(save_dir = r"/home/vadymskipa/PhD_student/data/123",
                source_dir = r"/home/vadymskipa/PhD_student/data/123", max_number_of_points_for_aprox=5):
    source_dir_name = "extremums"
    save_dir_name = "extremus_aprox_lines"
    source_path_1 = os.path.join(source_dir, source_dir_name)
    save_path_1 = get_directory_path(save_dir, save_dir_name)
    list_halfwindows = os.listdir(source_path_1)
    for halfwindow in list_halfwindows:
        source_path_2 = os.path.join(source_path_1, halfwindow)
        save_path_2 = get_directory_path(save_path_1, halfwindow)
        list_sites = os.listdir(source_path_2)
        for site in list_sites:
            source_path_3 = os.path.join(source_path_2, site)
            save_path_3 = get_directory_path(save_path_2, site)
            list_windows = os.listdir(source_path_3)
            for window in list_windows:
                source_path_4 = os.path.join(source_path_3, window)
                save_path_4 = get_directory_path(save_path_3, window)
                list_exrttypes = os.listdir(source_path_4)
                for extrtype in list_exrttypes:
                    source_path_5 = os.path.join(source_path_4, extrtype)
                    save_path_5 = get_directory_path(save_path_4, extrtype)
                    list_datefiles = os.listdir(source_path_5)
                    for datefile in list_datefiles:
                        source_path_6 = os.path.join(source_path_5, datefile)
                        save_path_6_lat = get_directory_path(save_path_5, "lat")
                        save_path_6_lon = get_directory_path(save_path_5, "lon")
                        source_dataframe = pd.read_hdf(source_path_6, key="df")
                        start = dt.datetime.now()
                        tuple_result = looking_for_top_aprx_results(
                            source_dataframe, max_number_of_points_for_aprox=max_number_of_points_for_aprox)
                        if tuple_result[0] is not None:
                            tuple_result[0].to_hdf(os.path.join(save_path_6_lat, datefile), key="df")
                        if tuple_result[1] is not None:
                            tuple_result[1].to_hdf(os.path.join(save_path_6_lon, datefile), key="df")
                        print(save_path_5, datefile, dt.datetime.now() - start)



def looking_for_top_aprx_results(dataframe: pd.DataFrame, max_number_of_points_for_aprox=5):
    first_datetime = dataframe.iloc[0].loc["datetime"]
    zero_timestamp = dt.datetime(year=first_datetime.year, month=first_datetime.month, day=first_datetime.day,
                                 tzinfo=dt.timezone.utc).timestamp()
    list_sats = np.unique(dataframe.loc[:, "sat_id"])
    dict_index_points = {}
    for sat in list_sats:
        dict_index_points[sat] = \
            dataframe.loc[dataframe.loc[:, "sat_id"] == sat].index.tolist()
    result_lat_dataframe = None
    result_lon_dataframe = None

    for number_of_points in range(3, max_number_of_points_for_aprox + 1, 1):
        if len(list_sats) < number_of_points:
            break
        for tuple_sats in itertools.combinations(list_sats, number_of_points):
            temp_list_indexes = [temp_list for sat, temp_list in dict_index_points.items() if sat in tuple_sats]
            for indexes in itertools.product(*temp_list_indexes):
                temp_dataframe = dataframe.loc[indexes, :]
                timestamp_values = np.array((temp_dataframe.loc[:, "timestamp"] - zero_timestamp) / 3600)
                if timestamp_values.max() - timestamp_values.min() > 2:
                    continue
                if number_of_points == 5:
                    pass
                lat_value = np.array(temp_dataframe.loc[:, "gdlat"])
                lon_value = np.array(temp_dataframe.loc[:, "gdlon"])
                M = timestamp_values[:, np.newaxis] ** [0, 1]
                p_lat, res_lat, rnk, s = lstsq(M, lat_value)
                p_lon, res_lon, rnk, s = lstsq(M, lon_value)
                temp_lat_dict = {"number_of_points": number_of_points,
                     "koef_0": p_lat[0] - zero_timestamp / 3600 * p_lat[1],
                     "koef_1": p_lat[1],
                     "delta": res_lat}
                for i in range(number_of_points):
                    temp_lat_dict[f"sat_id_{i}"] = tuple_sats[i]
                    temp_lat_dict[f"timestamp_{i}"] = temp_dataframe.iloc[i].loc["timestamp"]
                    temp_lat_dict[f"gd_lat_{i}"] = temp_dataframe.iloc[i].loc["gdlat"]
                try:
                    temp_lat_dataframe = pd.DataFrame(temp_lat_dict, index=[0,])
                except Exception as er:
                    continue
                result_lat_dataframe = pd.concat([result_lat_dataframe, temp_lat_dataframe], ignore_index=True)
                temp_lon_dict = {"number_of_points": number_of_points,
                                 "koef_0": p_lon[0] - zero_timestamp / 3600 * p_lon[1],
                                 "koef_1": p_lon[1],
                                 "delta": res_lon}
                for i in range(number_of_points):
                    temp_lon_dict[f"sat_id_{i}"] = tuple_sats[i]
                    temp_lon_dict[f"timestamp_{i}"] = temp_dataframe.iloc[i].loc["timestamp"]
                    temp_lon_dict[f"gd_lon_{i}"] = temp_dataframe.iloc[i].loc["gdlon"]
                try:
                    temp_lon_dataframe = pd.DataFrame(temp_lon_dict, index=[0,])
                except Exception as er:
                    continue
                result_lon_dataframe = pd.concat([result_lon_dataframe, temp_lon_dataframe], ignore_index=True)
    return result_lat_dataframe, result_lon_dataframe


def main2024dec7(save_dir = r"/home/vadymskipa/PhD_student/data/123",
                 source_dir = r"/home/vadymskipa/PhD_student/data/123",
                 number_of_lines=20):
    source_dir_name_1 = "extremums"
    source_dir_name_2 = "extremus_aprox_lines"
    save_dir_name = "extremus_aprox_lines_png"
    source_path_1_1 = os.path.join(source_dir, source_dir_name_1)
    source_path_2_1 = os.path.join(source_dir, source_dir_name_2)
    save_path_1 = get_directory_path(save_dir, save_dir_name)
    list_halfwindows = os.listdir(source_path_1_1)
    for halfwindow in list_halfwindows:
        source_path_1_2 = os.path.join(source_path_1_1, halfwindow)
        source_path_2_2 = os.path.join(source_path_2_1, halfwindow)
        save_path_2 = get_directory_path(save_path_1, halfwindow)
        list_sites = os.listdir(source_path_1_2)
        for site in list_sites:
            source_path_1_3 = os.path.join(source_path_1_2, site)
            source_path_2_3 = os.path.join(source_path_2_2, site)
            save_path_3 = get_directory_path(save_path_2, site)
            list_windows = os.listdir(source_path_1_3)
            for window in list_windows:
                source_path_1_4 = os.path.join(source_path_1_3, window)
                source_path_2_4 = os.path.join(source_path_2_3, window)
                save_path_4 = get_directory_path(save_path_3, window)
                list_exrttypes = os.listdir(source_path_1_4)
                for extrtype in list_exrttypes:
                    source_path_1_5 = os.path.join(source_path_1_4, extrtype)
                    source_path_2_5 = os.path.join(source_path_2_4, extrtype)
                    save_path_5 = get_directory_path(save_path_4, extrtype)
                    list_coords = os.listdir(source_path_2_5)
                    for coord in list_coords:
                        source_path_2_6 = os.path.join(source_path_2_5, coord)
                        save_path_6 = get_directory_path(save_path_5, coord)
                        list_datefiles = os.listdir(source_path_2_6)
                        for datefile in list_datefiles:
                            source_path_1_6 = os.path.join(source_path_1_5, datefile)
                            source_path_2_7 = os.path.join(source_path_2_6, datefile)
                            save_path_7 = get_directory_path(save_path_6, datefile[:-5])
                            dataframe_points = pd.read_hdf(source_path_1_6, key="df")
                            dataframe_lines = pd.read_hdf(source_path_2_7, key="df")
                            list_numbers_of_points = np.unique(dataframe_lines.loc[:, "number_of_points"])
                            for nop in list_numbers_of_points:
                                filename = f"number_of_points_{nop}.png"
                                temp_dataframe_lines = dataframe_lines.loc[dataframe_lines.loc[:, "number_of_points"] == nop]
                                temp_dataframe_lines = get_top_aprx_lines(temp_dataframe_lines, number_of_lines)
                                if coord == "lat":
                                    coord_name = "latitude"
                                    coord_column = "gdlat"
                                else:
                                    coord_name = "longitude"
                                    coord_column = "gdlon"
                                plot_aprox_lines(dataframe_points, temp_dataframe_lines, save_path_7, filename,
                                                 coord_column, coord_name)
                                print(save_path_7, filename)



def plot_aprox_lines(dataframe_points: pd.DataFrame, dataframe_lines: pd.DataFrame, save_dir, filename, coord_column,
                     coord_name):

    temp_datetime = dataframe_points.iloc[0].loc["datetime"]
    min_datetime = dt.datetime(year=temp_datetime.year, month=temp_datetime.month, day=temp_datetime.day,
                                 tzinfo=dt.timezone.utc)
    max_datetime = min_datetime + dt.timedelta(days=1)
    temp_datetime = min_datetime
    time_tick = dt.timedelta(hours=round((max_datetime - min_datetime).total_seconds() / 3600 / 12))
    if time_tick == 0:
        return None
    list_time_ticks = []
    while temp_datetime <= max_datetime:
        list_time_ticks.append(temp_datetime)
        temp_datetime += time_tick
    list_time_tick_name = [datetime.hour for datetime in list_time_ticks]
    time_limits = (min_datetime, max_datetime)

    min_coord = dataframe_points.loc[:, coord_column].min()
    max_coord = dataframe_points.loc[:, coord_column].max()
    start_coord = math.floor(min_coord)
    temp_coord = start_coord
    coord_tick = round((max_coord - min_coord) / 6)
    if coord_tick == 0:
        coord_tick = 1
    list_coord_ticks = []
    while temp_coord <= max_coord:
        list_coord_ticks.append(temp_coord)
        temp_coord += coord_tick
    list_coord_ticks.append(temp_coord)
    coord_limits = (int(min_coord), math.ceil(max_coord))

    fig = plt.figure(layout="tight", figsize=[12, 4.8])
    ax = fig.add_subplot(1, 1, 1)
    color_normalize = mplcolors.Normalize(0, 32)
    colormap = plt.colormaps["hsv"]
    fig.colorbar(mplcm.ScalarMappable(norm=color_normalize, cmap=colormap), ax=ax)
    ax.set(xticks=list_time_ticks, xticklabels=list_time_tick_name, xlabel="time, hours", xlim=time_limits,
           yticks=list_coord_ticks, ylabel=f"{coord_name[0].upper()}{coord_name[1:]}", ylim=coord_limits)
    ax.grid(visible=True)
    # ax.set_title("dTEC, TEC units")

    timestamp_columns = [column for column in dataframe_lines.columns if "timestamp" in column]
    for index in dataframe_lines.index:
        list_timestamps = [dataframe_lines.loc[index, column] for column in timestamp_columns]
        start = min(list_timestamps) - 900
        end = max(list_timestamps) + 900
        func = lambda x: dataframe_lines.loc[index, "koef_0"] + dataframe_lines.loc[index, "koef_1"] * (x / 3600)
        ax.plot((dt.datetime.fromtimestamp(start, tz=dt.timezone.utc),
                 dt.datetime.fromtimestamp(end, tz=dt.timezone.utc)), (func(start), func(end)),
                marker="", linestyle="-", linewidth=1, color="black")
    ax.scatter(dataframe_points.loc[:, "datetime"], dataframe_points.loc[:, coord_column], s=100,
               c=dataframe_points.loc[:, "sat_id"], marker=".", cmap=colormap, norm=color_normalize, linewidths=0)


    save_path = os.path.join(save_dir, filename)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    return save_path


def get_top_aprx_lines(dataframe: pd.DataFrame, number_of_lines):
    result_dataframe = dataframe.nsmallest(number_of_lines, "delta", keep="all")
    return result_dataframe


def main2024dec8(save_dir = r"/home/vadymskipa/PhD_student/data/123",
                source_dir = r"/home/vadymskipa/PhD_student/data/123",
                 naumber_of_lines=20):
    source_dir_name_1 = "extremums"
    source_dir_name_2 = "extremus_aprox_lines"
    save_dir_name = "extremums_png"
    source_path_1_1 = os.path.join(source_dir, source_dir_name_1)
    source_path_2_1 = os.path.join(source_dir, source_dir_name_2)
    save_path_1 = get_directory_path(save_dir, save_dir_name)
    list_halfwindows = os.listdir(source_path_1_1)
    for halfwindow in list_halfwindows:
        source_path_1_2 = os.path.join(source_path_1_1, halfwindow)
        source_path_2_2 = os.path.join(source_path_2_1, halfwindow)
        save_path_2 = get_directory_path(save_path_1, halfwindow)
        list_sites = os.listdir(source_path_1_2)
        for site in list_sites:
            source_path_1_3 = os.path.join(source_path_1_2, site)
            source_path_2_3 = os.path.join(source_path_2_2, site)
            save_path_3 = get_directory_path(save_path_2, site)
            list_windows = os.listdir(source_path_1_3)
            for window in list_windows:
                source_path_1_4 = os.path.join(source_path_1_3, window)
                source_path_2_4 = os.path.join(source_path_2_3, window)
                save_path_4 = get_directory_path(save_path_3, window)

                source_path_max_1_5 = os.path.join(source_path_1_4, "max")
                source_path_min_1_5 = os.path.join(source_path_1_4, "min")

                # source_path_2_5 = os.path.join(source_path_2_4, extrtype)
                # save_path_5 = get_directory_path(save_path_4, extrtype)
                list_coords = ["lat", "lon"]
                for coord in list_coords:
                #     # source_path_2_6 = os.path.join(source_path_2_5, coord)
                    save_path_6 = get_directory_path(save_path_4, coord)
                    list_datefiles = os.listdir(source_path_max_1_5)
                    for datefile in list_datefiles:
                        filename = f"{datefile[:-5]}.png"
                        source_path_max_1_6 = os.path.join(source_path_max_1_5, datefile)
                        source_path_min_1_6 = os.path.join(source_path_min_1_5, datefile)
                        # source_path_2_7 = os.path.join(source_path_2_6, datefile)
                        # save_path_7 = get_directory_path(save_path_6, datefile[:-5])
                        dataframe_points_max = pd.read_hdf(source_path_max_1_6, key="df")
                        dataframe_points_min = pd.read_hdf(source_path_min_1_6, key="df")
                        # dataframe_lines = pd.read_hdf(source_path_2_7, key="df")
                        # list_numbers_of_points = np.unique(dataframe_lines.loc[:, "number_of_points"])
                        # for nop in list_numbers_of_points:
                        #     filename = f"number_of_points_{nop}.png"
                        #     temp_dataframe_lines = dataframe_lines.loc[dataframe_lines.loc[:, "number_of_points"] == nop]
                        #     temp_dataframe_lines = get_top_aprx_lines(temp_dataframe_lines, naumber_of_lines)
                        if coord == "lat":
                            coord_name = "latitude"
                            coord_column = "gdlat"
                        else:
                            coord_name = "longitude"
                            coord_column = "gdlon"
                        plot_aprox_points(dataframe_points_max, dataframe_points_min, save_path_6, filename,
                                         coord_column, coord_name)
                        print(save_path_6, filename)


def plot_aprox_points(dataframe_points_max: pd.DataFrame, dataframe_points_min: pd.DataFrame, save_dir, filename,
                      coord_column, coord_name):

    temp_datetime_max = dataframe_points_max.iloc[0].loc["datetime"]
    temp_datetime_min = dataframe_points_min.iloc[0].loc["datetime"]
    min_datetime = min(dt.datetime(year=temp_datetime_max.year, month=temp_datetime_max.month, day=temp_datetime_max.day,
                                 tzinfo=dt.timezone.utc),
                       dt.datetime(year=temp_datetime_min.year, month=temp_datetime_min.month, day=temp_datetime_min.day,
                                   tzinfo=dt.timezone.utc)
                       )
    max_datetime = min_datetime + dt.timedelta(days=1)
    temp_datetime = min_datetime
    time_tick = dt.timedelta(hours=round((max_datetime - min_datetime).total_seconds() / 3600 / 12))
    if time_tick == 0:
        return None
    list_time_ticks = []
    while temp_datetime <= max_datetime:
        list_time_ticks.append(temp_datetime)
        temp_datetime += time_tick
    list_time_tick_name = [datetime.hour for datetime in list_time_ticks]
    time_limits = (min_datetime, max_datetime)

    min_coord = min(dataframe_points_max.loc[:, coord_column].min(), dataframe_points_min.loc[:, coord_column].min())
    max_coord = max(dataframe_points_max.loc[:, coord_column].max(), dataframe_points_min.loc[:, coord_column].max())
    start_coord = math.ceil(min_coord)
    temp_coord = start_coord
    coord_tick = round((max_coord - min_coord) / 6)
    if coord_tick == 0:
        coord_tick = 1
    list_coord_ticks = []
    while temp_coord <= max_coord:
        list_coord_ticks.append(temp_coord)
        temp_coord += coord_tick
    coord_limits = (int(min_coord), math.ceil(max_coord))

    fig = plt.figure(layout="tight", figsize=[12, 4.8])
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(dataframe_points_max.loc[:, "datetime"], dataframe_points_max.loc[:, coord_column], s=100,
               c="red", marker=".", linewidths=0)
    ax.scatter(dataframe_points_min.loc[:, "datetime"], dataframe_points_min.loc[:, coord_column], s=100,
               c="blue", marker=".", linewidths=0)
    ax.set(xticks=list_time_ticks, xticklabels=list_time_tick_name, xlabel="time, hours", xlim=time_limits,
           yticks=list_coord_ticks, ylabel=f"{coord_name[0].upper()}{coord_name[1:]}", ylim=coord_limits)
    ax.grid(visible=True)
    # ax.set_title("dTEC, TEC units")


    save_path = os.path.join(save_dir, filename)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    return save_path



def main2024dec():
    # main2024dec1(speed_limits_kmph_list=(180, 150, 120, 90))
    # main2024dec2()
    # main2024dec3(relative=True)
    # main2024dec4(relative=True)
    # main2024dec4()
    # main2024dec5(min_dvtec=0.5)
    # main2024dec6(max_number_of_points_for_aprox=5)
    # main2024dec7()
    # main2024dec5(min_dvtec=1, oneside_local_window=10, number_of_extremums=1)
    main2024dec6(max_number_of_points_for_aprox=5)
    main2024dec7(number_of_lines=5)
    # main2024dec8()


def main2025jan1():
    print(wfs.looking_for_all_inner_files_in_the_directory(
        r"/home/vadymskipa/PhD_student/data/Europe_36-70__-10-20/dvtec_csv_blocks/Latitude_time_map/"))
    print(wfs.looking_for_specific_inner_files_in_the_directory(
        r"/home/vadymskipa/PhD_student/data/Europe_36-70__-10-20/dvtec_csv_blocks/Latitude_time_map/", r"\d\d\d\-\d\d\d\d\-\d\d\-\d\d\.hdf"))


def main2025jan2():
    # source_los_directory_path = r"/home/vadymskipa/PhD_student/data/big_los_hdf/"
    # source_site_directory_path = r"/home/vadymskipa/PhD_student/data/site_hdf/"
    # save_region_directory_path = r"/home/vadymskipa/PhD_student/data/Europe_los_hdf/"
    # start_date = dt.datetime(year=2023, month=3, day=29, tzinfo=dt.timezone.utc)
    # end_date = dt.datetime(year=2023, month=3, day=29, tzinfo=dt.timezone.utc)
    # lf5py.save_region_los_files_from_directory(source_los_directory_path, source_site_directory_path,
    #                                            save_region_directory_path, region_name="europe", lats=(30, 80),
    #                                            lons=(-10, 50),
    #                                            repeatition_of_region_files=True,
    #                                            multiprocessing=True,
    #                                            start_date=start_date, end_date=end_date)
    # lats = (30, 80)
    # lons = (-10, 50)
    # min_date = dt.datetime(year=2023, month=3, day=29, tzinfo=dt.timezone.utc)
    # max_date = dt.datetime(year=2023, month=3, day=29, tzinfo=dt.timezone.utc)
    # source_site_hdf_directory_path = r"/home/vadymskipa/PhD_student/data/site_hdf/"
    # source_los_hdf_directory = r"/home/vadymskipa/PhD_student/data/Europe_los_hdf/"
    # los_hdfs_in_directory = os.listdir(source_los_hdf_directory)
    # list_los_hdf_paths = []
    # for los_hdf in los_hdfs_in_directory:
    #     temp_date = wlos.get_date_by_los_file_name(los_hdf)
    #     if temp_date < min_date:
    #         continue
    #     if temp_date > max_date:
    #         continue
    #     list_los_hdf_paths.append(os.path.join(source_los_hdf_directory, los_hdf))
    # save_path = r"/home/vadymskipa/PhD_student/data/Europe_30-80__-10-50"
    # save_los_tec_txt_file_for_region_from_los_files(list_los_hdf_paths, source_site_hdf_directory_path, save_path,
    #                                                 lats=lats, lons=lons, multiprocessing=True)

    source_path = r"/home/vadymskipa/PhD_student/data/Europe_30-80__-10-50/los_tec_txt"
    save_path = r"/home/vadymskipa/PhD_student/data/Europe_30-80__-10-50/dtec_txt"
    params = LIST_PARAMS1
    min_date = dt.datetime(year=2023, month=3, day=29, tzinfo=dt.timezone.utc)
    max_date = dt.datetime(year=2023, month=3, day=30, tzinfo=dt.timezone.utc)
    get_dtec_from_los_tec_txt_for_lots_of_sites(source_path, save_path, params, min_date=min_date,
                                                max_date=max_date)


def main2025jan3():
    source_los_directory_path = r"/home/vadymskipa/PhD_student/data/big_los_hdf/"
    source_site_directory_path = r"/home/vadymskipa/PhD_student/data/site_hdf/"
    save_region_directory_path = r"/home/vadymskipa/PhD_student/data/Europe_los_hdf/"
    start_date = dt.datetime(year=2023, month=4, day=6, tzinfo=dt.timezone.utc)
    end_date = dt.datetime(year=2023, month=4, day=9, tzinfo=dt.timezone.utc)
    lf5py.save_region_los_files_from_directory(source_los_directory_path, source_site_directory_path,
                                               save_region_directory_path, region_name="europe", lats=(30, 80),
                                               lons=(-10, 50),
                                               repeatition_of_region_files=True,
                                               multiprocessing=True,
                                               start_date=start_date, end_date=end_date)
    lats = (30, 80)
    lons = (-10, 50)
    min_date = dt.datetime(year=2023, month=4, day=5, tzinfo=dt.timezone.utc)
    max_date = dt.datetime(year=2023, month=4, day=9, tzinfo=dt.timezone.utc)
    source_site_hdf_directory_path = r"/home/vadymskipa/PhD_student/data/site_hdf/"
    source_los_hdf_directory = r"/home/vadymskipa/PhD_student/data/Europe_los_hdf/"
    los_hdfs_in_directory = os.listdir(source_los_hdf_directory)
    list_los_hdf_paths = []
    for los_hdf in los_hdfs_in_directory:
        temp_date = wlos.get_date_by_los_file_name(los_hdf)
        if temp_date < min_date:
            continue
        if temp_date > max_date:
            continue
        list_los_hdf_paths.append(os.path.join(source_los_hdf_directory, los_hdf))
    save_path = r"/home/vadymskipa/PhD_student/data/Europe_30-80__-10-50"
    save_los_tec_txt_file_for_region_from_los_files(list_los_hdf_paths, source_site_hdf_directory_path, save_path,
                                                    lats=lats, lons=lons, multiprocessing=True)

    source_path = r"/home/vadymskipa/PhD_student/data/Europe_30-80__-10-50/los_tec_txt"
    save_path = r"/home/vadymskipa/PhD_student/data/Europe_30-80__-10-50/dtec_txt"
    params = LIST_PARAMS1
    min_date = dt.datetime(year=2023, month=4, day=5, tzinfo=dt.timezone.utc)
    max_date = dt.datetime(year=2023, month=4, day=9, tzinfo=dt.timezone.utc)
    get_dtec_from_los_tec_txt_for_lots_of_sites(source_path, save_path, params, min_date=min_date,
                                                max_date=max_date)


def main2025feb1():
    source_directories = [r"/home/vadymskipa/PhD_student/data/Europe_30-80__-10-50/dtec_txt/088-2023-03-29",
                          r"/home/vadymskipa/PhD_student/data/Europe_30-80__-10-50/dtec_txt/096-2023-04-06",
                          r"/home/vadymskipa/PhD_student/data/Europe_30-80__-10-50/dtec_txt/097-2023-04-07",
                          r"/home/vadymskipa/PhD_student/data/Europe_30-80__-10-50/dtec_txt/098-2023-04-08",
                          r"/home/vadymskipa/PhD_student/data/Europe_30-80__-10-50/dtec_txt/099-2023-04-09"]
    site_directory = r"/home/vadymskipa/PhD_student/data/site_hdf/"
    site_paths = wsite.get_site_file_paths_from_directory(site_directory)
    for source in source_directories:
        date = get_date_from_date_directory_name(os.path.basename(source))
        site_file = None
        for site_path in site_paths:
            if wsite.check_site_file_name_by_date(os.path.basename(site_path), date):
                site_file = site_path
                break
        save_sites_txt_for_site_directories(source, site_file, source)




def main2025jan():
    # main2025jan1()
    # main2025jan2()
    # main2025jan3()
    pass


def main2025feb2(save_dir = r"/home/vadymskipa/PhD_student/data/123",
                 source_dir = r"/home/vadymskipa/PhD_student/data/123",
                 min_elm = 30, oneside_local_window=5, number_of_extremums=3, min_dvtec=0.3):
    save_dir_name = "extremums"
    save_dir_name_1_5 = f"half_window_{oneside_local_window}"
    save_path_1 = get_directory_path(save_dir, save_dir_name)
    save_path_1_5 = get_directory_path(save_path_1, save_dir_name_1_5)
    source_path_1 = os.path.join(source_dir, "dvtec_csv")
    source_dataframe = read_dvtec_csv_dir(source_path_1)
    source_dataframe = source_dataframe.loc[source_dataframe.loc[:, "elm"] >= min_elm]
    list_sites = np.unique(source_dataframe.loc[:, "gps_site"])
    for site in list_sites:
        save_path_2 = get_directory_path(save_path_1_5, site)
        site_source_dataframe = source_dataframe.loc[source_dataframe.loc[:, "gps_site"] == site]
        list_windows = np.unique(site_source_dataframe.loc[:, "window"])
        for window in list_windows:
            save_path_3 = get_directory_path(save_path_2, window)
            window_source_dataframe = site_source_dataframe.loc[site_source_dataframe.loc[:, "window"] == window]
            date = dt.datetime(year=2024, month=5, day=10, tzinfo=dt.timezone.utc)
            filename = get_date_str(date) + ".csv"
            date_dataframe = window_source_dataframe.loc[
                window_source_dataframe.loc[:, "datetime"] >=
                dt.datetime(year=2024, month=5, day=10, hour=20, tzinfo=dt.timezone.utc)]
            date_dataframe = date_dataframe.loc[
                date_dataframe.loc[:, "datetime"] <=
                dt.datetime(year=2024, month=5, day=11, tzinfo=dt.timezone.utc)]
            tuple_extremum_dataframes = looking_for_extremums_in_site_dataframe(
                date_dataframe, oneside_local_window=oneside_local_window, number_of_extremums=number_of_extremums,
            min_dvtec=min_dvtec)
            tuple_extremum_dataframes[0].loc[:, "ex"] = "max"
            tuple_extremum_dataframes[1].loc[:, "ex"] = "min"
            extremum_dataframe = pd.concat(tuple_extremum_dataframes, ignore_index=True)
            extremum_dataframe.sort_values(["sat_id", "datetime"], inplace=True)
            extremum_dataframe.reset_index(inplace=True)
            extremum_dataframe.to_csv(os.path.join(save_path_3, filename))


def main2025feb3():
    source_los_directory_path = r"/home/vadymskipa/PhD_student/data/big_los_hdf/"
    source_site_directory_path = r"/home/vadymskipa/PhD_student/data/site_hdf/"
    save_region_directory_path = r"/home/vadymskipa/PhD_student/data/Europe_los_hdf/"
    min_date = start_date = dt.datetime(year=2022, month=10, day=12, tzinfo=dt.timezone.utc)
    max_date = end_date = dt.datetime(year=2022, month=10, day=25, tzinfo=dt.timezone.utc)
    lf5py.save_region_los_files_from_directory(source_los_directory_path, source_site_directory_path,
                                               save_region_directory_path, region_name="europe", lats=(30, 80),
                                               lons=(-10, 50),
                                               repeatition_of_region_files=True,
                                               multiprocessing=True,
                                               start_date=start_date, end_date=end_date)
    lats = (30, 80)
    lons = (-10, 50)
    source_site_hdf_directory_path = r"/home/vadymskipa/PhD_student/data/site_hdf/"
    source_los_hdf_directory = r"/home/vadymskipa/PhD_student/data/Europe_los_hdf/"
    los_hdfs_in_directory = os.listdir(source_los_hdf_directory)
    list_los_hdf_paths = []
    for los_hdf in los_hdfs_in_directory:
        temp_date = wlos.get_date_by_los_file_name(los_hdf)
        if temp_date < min_date:
            continue
        if temp_date > max_date:
            continue
        list_los_hdf_paths.append(os.path.join(source_los_hdf_directory, los_hdf))
    save_path = r"/home/vadymskipa/PhD_student/data/Europe_30-80__-10-50"
    save_los_tec_txt_file_for_region_from_los_files(list_los_hdf_paths, source_site_hdf_directory_path, save_path,
                                                    lats=lats, lons=lons, multiprocessing=True)

    source_path = r"/home/vadymskipa/PhD_student/data/Europe_30-80__-10-50/los_tec_txt"
    save_path = r"/home/vadymskipa/PhD_student/data/Europe_30-80__-10-50/dtec_txt"
    params = LIST_PARAMS1
    get_dtec_from_los_tec_txt_for_lots_of_sites(source_path, save_path, params, min_date=min_date,
                                                max_date=max_date)


def main2025feb4():
    source_directory = r"/home/vadymskipa/PhD_student/data/Europe_30-80__-10-50/dtec_txt/"
    site_directory = r"/home/vadymskipa/PhD_student/data/site_hdf/"
    entries = os.listdir(source_directory)
    start_date = dt.datetime(year=2022, month=10, day=12, tzinfo=dt.timezone.utc)
    end_date = dt.datetime(year=2022, month=10, day=25, tzinfo=dt.timezone.utc)
    source_directories = [os.path.join(source_directory, entry) for entry in entries
                          if os.path.isdir(os.path.join(source_directory, entry)) and
                          get_date_from_date_directory_name(entry) <= end_date and
                          get_date_from_date_directory_name(entry) >= start_date]
    site_paths = wsite.get_site_file_paths_from_directory(site_directory)
    for source in source_directories:
        date = get_date_from_date_directory_name(os.path.basename(source))
        site_file = None
        for site_path in site_paths:
            if wsite.check_site_file_name_by_date(os.path.basename(site_path), date):
                site_file = site_path
                break
        save_sites_txt_for_site_directories(source, site_file, source)


def main2025feb5():
    source_los_directory_path = r"/run/media/vadymskipa/Elements SE/big_los_hdf/"
    source_site_directory_path = r"/home/vadymskipa/PhD_student/data/site_hdf/"
    save_region_directory_path = r"/home/vadymskipa/PhD_student/data/Europe_los_hdf/"
    min_date = start_date = dt.datetime(year=2023, month=3, day=28, tzinfo=dt.timezone.utc)
    max_date = end_date = dt.datetime(year=2023, month=4, day=10, tzinfo=dt.timezone.utc)
    lf5py.save_region_los_files_from_directory(source_los_directory_path, source_site_directory_path,
                                               save_region_directory_path, region_name="europe", lats=(30, 80),
                                               lons=(-10, 50),
                                               repeatition_of_region_files=True,
                                               multiprocessing=True,
                                               start_date=start_date, end_date=end_date)


def main2025feb6():
    source_los_directory_path = r"/home/vadymskipa/PhD_student/data/big_los_hdf/"
    source_site_directory_path = r"/home/vadymskipa/PhD_student/data/site_hdf/"
    save_region_directory_path = r"/home/vadymskipa/PhD_student/data/Europe_los_hdf/"
    min_date = start_date = dt.datetime(year=2023, month=3, day=28, tzinfo=dt.timezone.utc)
    max_date = end_date = dt.datetime(year=2023, month=4, day=10, tzinfo=dt.timezone.utc)
    lf5py.save_region_los_files_from_directory(source_los_directory_path, source_site_directory_path,
                                               save_region_directory_path, region_name="europe", lats=(30, 80),
                                               lons=(-10, 50),
                                               repeatition_of_region_files=True,
                                               multiprocessing=True,
                                               start_date=start_date, end_date=end_date)


def main2025feb7():
    min_date = start_date = dt.datetime(year=2023, month=3, day=28, tzinfo=dt.timezone.utc)
    max_date = end_date = dt.datetime(year=2023, month=4, day=10, tzinfo=dt.timezone.utc)
    lats = (30, 80)
    lons = (-10, 50)
    source_site_hdf_directory_path = r"/home/vadymskipa/PhD_student/data/site_hdf/"
    source_los_hdf_directory = r"/home/vadymskipa/PhD_student/data/Europe_los_hdf/"
    los_hdfs_in_directory = os.listdir(source_los_hdf_directory)
    list_los_hdf_paths = []
    for los_hdf in los_hdfs_in_directory:
        temp_date = wlos.get_date_by_los_file_name(los_hdf)
        if temp_date < min_date:
            continue
        if temp_date > max_date:
            continue
        list_los_hdf_paths.append(os.path.join(source_los_hdf_directory, los_hdf))
    save_path = r"/home/vadymskipa/PhD_student/data/Europe_30-80__-10-50"
    # save_los_tec_txt_file_for_region_from_los_files(list_los_hdf_paths, source_site_hdf_directory_path, save_path,
    #                                                 lats=lats, lons=lons, multiprocessing=True)

    source_path = r"/home/vadymskipa/PhD_student/data/Europe_30-80__-10-50/los_tec_txt"
    save_path = r"/home/vadymskipa/PhD_student/data/Europe_30-80__-10-50/dtec_txt"
    params = LIST_PARAMS1
    get_dtec_from_los_tec_txt_for_lots_of_sites(source_path, save_path, params, min_date=min_date,
                                                max_date=max_date)


def main2025feb8():
    source_directory = r"/home/vadymskipa/PhD_student/data/Europe_30-80__-10-50/dtec_txt/"
    site_directory = r"/home/vadymskipa/PhD_student/data/site_hdf/"
    entries = os.listdir(source_directory)
    start_date = dt.datetime(year=2023, month=3, day=28, tzinfo=dt.timezone.utc)
    end_date = dt.datetime(year=2023, month=4, day=10, tzinfo=dt.timezone.utc)
    source_directories = [os.path.join(source_directory, entry) for entry in entries
                          if os.path.isdir(os.path.join(source_directory, entry)) and
                          get_date_from_date_directory_name(entry) <= end_date and
                          get_date_from_date_directory_name(entry) >= start_date]
    site_paths = wsite.get_site_file_paths_from_directory(site_directory)
    for source in source_directories:
        date = get_date_from_date_directory_name(os.path.basename(source))
        site_file = None
        for site_path in site_paths:
            if wsite.check_site_file_name_by_date(os.path.basename(site_path), date):
                site_file = site_path
                break
        save_sites_txt_for_site_directories(source, site_file, source)


def main2025feb9():
    source_los_directory_path = r"/home/vadymskipa/PhD_student/data/big_los_hdf/"
    source_site_directory_path = r"/home/vadymskipa/PhD_student/data/site_hdf/"
    save_region_directory_path = r"/home/vadymskipa/PhD_student/data/Europe_los_hdf/"
    min_date = start_date = dt.datetime(year=2023, month=6, day=13, tzinfo=dt.timezone.utc)
    max_date = end_date = dt.datetime(year=2023, month=6, day=19, tzinfo=dt.timezone.utc)
    lf5py.save_region_los_files_from_directory(source_los_directory_path, source_site_directory_path,
                                               save_region_directory_path, region_name="europe", lats=(30, 80),
                                               lons=(-10, 50),
                                               repeatition_of_region_files=True,
                                               multiprocessing=True,
                                               start_date=start_date, end_date=end_date)
    lats = (30, 80)
    lons = (-10, 50)
    source_site_hdf_directory_path = r"/home/vadymskipa/PhD_student/data/site_hdf/"
    source_los_hdf_directory = r"/home/vadymskipa/PhD_student/data/Europe_los_hdf/"
    los_hdfs_in_directory = os.listdir(source_los_hdf_directory)
    list_los_hdf_paths = []
    for los_hdf in los_hdfs_in_directory:
        temp_date = wlos.get_date_by_los_file_name(los_hdf)
        if temp_date < min_date:
            continue
        if temp_date > max_date:
            continue
        list_los_hdf_paths.append(os.path.join(source_los_hdf_directory, los_hdf))
    save_path = r"/home/vadymskipa/PhD_student/data/Europe_30-80__-10-50"
    save_los_tec_txt_file_for_region_from_los_files(list_los_hdf_paths, source_site_hdf_directory_path, save_path,
                                                    lats=lats, lons=lons, multiprocessing=True)

    source_path = r"/home/vadymskipa/PhD_student/data/Europe_30-80__-10-50/los_tec_txt"
    save_path = r"/home/vadymskipa/PhD_student/data/Europe_30-80__-10-50/dtec_txt"
    params = LIST_PARAMS1
    get_dtec_from_los_tec_txt_for_lots_of_sites(source_path, save_path, params, min_date=min_date,
                                                max_date=max_date)


def main2025feb():
    # main2025feb1()
    # main2025feb2(save_dir = r"/home/vadymskipa/PhD_student/data/123/temp", min_dvtec=-2, number_of_extremums=100,
    #              oneside_local_window=30)
    # main2025feb3()
    # main2025feb4()
    # main2025feb5()
    # main2025feb6()
    # main2025feb7()
    # main2025feb8()
    main2025feb9()



if __name__ == "__main__":
    main2025feb()
    # main2025jan()
    # main2024dec()
    # main2024nov()
    # main2024oct()
    # main2024sep()
    # main2024aug()
    # main2024jul()
    # main2024jun()
    # main2024apr()
    # main2024apr8()
    # main2024apr9()
    # main2024apr7()
    # main2024apr6()
    # main2024apr5()
    # main2024apr4()
    # main2024apr3()
    # main2024apr2()
    # main2024apr1()
    # main2024feb6()
    # main2024feb5()
    # main2024feb4()
    # plot_graphs_for_site_directory(SOURCE_DIRECTORY_PATH11, SAVE_PATH11)
    # # get_dtec_from_los_tec_txt_1(SOURCE_DIRECTORY_PATH8_1, SOURCE_DIRECTORY_PATH11)
    # list1 = [r"/home/vadymskipa/Documents/PhD_student/data/Europe/los_20230605.europe.001.h5.hdf5",
    #          r"/home/vadymskipa/Documents/PhD_student/data/Europe/los_20230606.europe.001.h5.hdf5",
    #          r"/home/vadymskipa/Documents/PhD_student/data/Europe/los_20230607.europe.001.h5.hdf5",
    #          r"/home/vadymskipa/Documents/PhD_student/data/Europe/los_20230608.europe.001.h5.hdf5"]
    # lats = (44.38, 52.38)
    # lons = (22.14, 40.23)

    # save_los_tec_txt_file_for_region_from_los_files(list1, DIRECTORY_PATH1, SAVE_PATH13, lons=lons,
    #                                                 lats=lats)
    # get_dtec_from_los_tec_txt_many_sites2(SOURCE_PATH14, SAVE_PATH14, LIST_PARAMS1)
    # main2023jan1()
    # main2023jan2()
    # main2023jan3()
    # main2023jan4()
    # main2023jan5()
    # main2023jan6()
    # main2023jan7()
    # main2023jan8()
    # main2023jan9()
    # main2023jan11()
    # main2023jan10()
    # main2023jan12()
    # main2023jan13()
    # main2023jan14()
    # main2023jan15()
    # main2023jan16()
    # main2023jan17()
    # main2024jan18()
    # main2024feb1()
    # main2024feb2()
    # main2024feb3()
