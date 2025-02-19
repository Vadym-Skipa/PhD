import os
import pandas as pd
import numpy as np
import datetime as dt
from scipy.signal import savgol_filter
from typing import Dict
from typing import List
import matplotlib.pyplot as plt
import matplotlib.axes as axs
import matplotlib.figure as fig
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
import work2_1 as w21




def get_rinex3(dir_path):
    entries = os.listdir(dir_path)
    files_mo = [file for file in entries if (file[32:34] == "MO" and file[24:27] == "01H")]
    files_mn = [file for file in entries if (file[28:30] == "MN" and file[24:27] == "01H")]
    files_mo.sort(key=lambda x: int(x[12:23]))
    files_mn.sort(key=lambda x: int(x[12:23]))
    text_mo = ""
    with open(os.path.join(dir_path, files_mo[0]), "r") as file:
        text_mo = file.read()
    for index_mo in range(1, len(files_mo)):
        with open(os.path.join(dir_path, files_mo[index_mo]), "r") as file:
            for i in range(28):
                file.readline()
            for line in file:
                text_mo += line
    text_mn = ""
    with open(os.path.join(dir_path, files_mn[0]), "r") as file:
        text_mn = file.read()
    for index_mn in range(1, len(files_mn)):
        with open(os.path.join(dir_path, files_mn[index_mn]), "r") as file:
            for i in range(4):
                file.readline()
            for line in file:
                text_mn += line
    save_mo = files_mo[0][:24] + "01D" + files_mo[0][27:]
    save_mn = files_mn[0][:24] + "01D" + files_mn[0][27:]
    with open(os.path.join(dir_path, save_mo), "w") as file:
        file.write(text_mo)
    with open(os.path.join(dir_path, save_mn), "w") as file:
        file.write(text_mn)


def main1():
    source_dirs = [
        r"/home/vadymskipa/PhD_student/data/rinex/Stepanivka-2024/130(0509)/Base3",
        r"/home/vadymskipa/PhD_student/data/rinex/Stepanivka-2024/131(0510)/Base3",
        r"/home/vadymskipa/PhD_student/data/rinex/Stepanivka-2024/132(0511)/Base3",
        r"/home/vadymskipa/PhD_student/data/rinex/Stepanivka-2024/133(0512)/Base3",
        r"/home/vadymskipa/PhD_student/data/rinex/Stepanivka-2024/134(0513)/Base3",
    ]
    for dir in source_dirs:
        get_rinex3(dir)


def main2(file, date: dt.datetime, save_dir):
    data = rcmn.read_cmn_file_pd(file)
    timestamp1 = date.timestamp()
    data = data.assign(timestamp=timestamp1 + data["time"] * 3600)
    hour_list = [dt.datetime.fromtimestamp(data, tz=dt.timezone.utc).hour for data in data.loc[:, "timestamp"]]
    minute_list = [dt.datetime.fromtimestamp(data, tz=dt.timezone.utc).minute for data in data.loc[:, "timestamp"]]
    sec_list = [dt.datetime.fromtimestamp(data, tz=dt.timezone.utc).second for data in data.loc[:, "timestamp"]]
    data = data.assign(hour=hour_list,
                       min=minute_list,
                       sec=sec_list)
    data.rename(columns={"stec": "los_tec", "PRN": "sat_id", "azimuth": "azm", "elevation": "elm", "latitude": "gdlat", "longitude": "glon"}, inplace=True)

    save_path = w21.get_directory_path(save_dir, w21.get_date_str(date))
    sats = np.unique(data.loc[:, "sat_id"])
    for sat in sats:
        sat_data = data.loc[data.loc[:, "sat_id"] == sat]
        save_path_2 = os.path.join(save_path, f"G{sat:0=2}.txt")
        w21.save_los_tec_txt_from_dataframe(save_path_2, sat_data)




if __name__ == "__main__":
    source = [
        (r"/home/vadymskipa/PhD_student/data/123/Stepanivka-2024/130(0509)/Base3/Bas3130-2024-05-09.Cmn",
         dt.datetime(year=2024, month=5, day=9, tzinfo=dt.timezone.utc)),
        (r"/home/vadymskipa/PhD_student/data/123/Stepanivka-2024/131(0510)/Base3/Bas3131-2024-05-10.Cmn",
         dt.datetime(year=2024, month=5, day=10, tzinfo=dt.timezone.utc)),
        (r"/home/vadymskipa/PhD_student/data/123/Stepanivka-2024/132(0511)/Base3/Bas3132-2024-05-11.Cmn",
         dt.datetime(year=2024, month=5, day=11, tzinfo=dt.timezone.utc)),
        (r"/home/vadymskipa/PhD_student/data/123/Stepanivka-2024/133(0512)/Base3/Bas3133-2024-05-12.Cmn",
         dt.datetime(year=2024, month=5, day=12, tzinfo=dt.timezone.utc)),
        (r"/home/vadymskipa/PhD_student/data/123/Stepanivka-2024/134(0513)/Base3/Bas3134-2024-05-13.Cmn",
         dt.datetime(year=2024, month=5, day=13, tzinfo=dt.timezone.utc)),
        # (r"/home/vadymskipa/PhD_student/data/123/Zmiiv-2024/130/BASE130-2024-05-09.Cmn",
        #  dt.datetime(year=2024, month=5, day=9, tzinfo=dt.timezone.utc)),
        # (r"/home/vadymskipa/PhD_student/data/123/Zmiiv-2024/131/BASE131-2024-05-10.Cmn",
        #  dt.datetime(year=2024, month=5, day=10, tzinfo=dt.timezone.utc)),
        # (r"/home/vadymskipa/PhD_student/data/123/Zmiiv-2024/132/BASE132-2024-05-11.Cmn",
        #  dt.datetime(year=2024, month=5, day=11, tzinfo=dt.timezone.utc)),
        # (r"/home/vadymskipa/PhD_student/data/123/Zmiiv-2024/133/BASE133-2024-05-12.Cmn",
        #  dt.datetime(year=2024, month=5, day=12, tzinfo=dt.timezone.utc)),
        # (r"/home/vadymskipa/PhD_student/data/123/Zmiiv-2024/134/BASE134-2024-05-13.Cmn",
        #  dt.datetime(year=2024, month=5, day=13, tzinfo=dt.timezone.utc))
    ]
    save_dir = r"/home/vadymskipa/PhD_student/data/123/los_txt/Bas3"
    for s in source:
        main2(*s, save_dir)




