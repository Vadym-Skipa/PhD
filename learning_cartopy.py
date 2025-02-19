import cartopy.crs as ccrs
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.mpl.geoaxes as cgeoaxes
from cartopy.io.shapereader import Reader as creader
from shapely.geometry import Polygon
from shapely.geometry import Point
import cartopy.feature as cfeature
import os
import work2_1 as w21
import pandas as pd
import matplotlib.axes as axs
import matplotlib.figure as fig
import matplotlib.colors as mplcolors
import matplotlib.cm as mplcm
import matplotlib as mpl
import datetime as dt
import numpy as np
from collections import namedtuple
from typing import Dict
from matplotlib.transforms import offset_copy
import re
import work_site_file as wsite
import multiprocessing.pool as multipool
import math


SHAPEFILE = r"./geo/cntry02.shp"

def test1():
    ax: cgeoaxes.GeoAxes = plt.axes(projection=ccrs.LambertConformal())
    ax.coastlines()
    mypolygon = Polygon(shell=((-5, -10), (-5, 20), (15, 20), (15, -10), (-5, -10)))
    ax.add_geometries([mypolygon], crs=ccrs.PlateCarree(), color="red", alpha=0.3)
    # Save the plot by calling plt.savefig() BEFORE plt.show()
    # plt.savefig('coastlines.pdf')
    # plt.savefig('coastlines.png')
    print(ax.collections)
    plt.show()


def plot_city_lights():
    # Define resource for the NASA night-time illumination data.
    base_uri = 'https://map1c.vis.earthdata.nasa.gov/wmts-geo/wmts.cgi'
    layer_name = 'VIIRS_CityLights_2012'

    # Create a Cartopy crs for plain and rotated lat-lon projections.
    plain_crs = ccrs.PlateCarree()
    rotated_crs = ccrs.RotatedPole(pole_longitude=120.0, pole_latitude=45.0)

    fig = plt.figure()

    # Plot WMTS data in a specific region, over a plain lat-lon map.
    ax = fig.add_subplot(1, 2, 1, projection=plain_crs)
    ax.set_extent([-6, 3, 48, 58], crs=ccrs.PlateCarree())
    ax.coastlines(resolution='50m', color='yellow')
    ax.gridlines(color='lightgrey', linestyle='-')
    # Add WMTS imaging.
    ax.add_wmts(base_uri, layer_name=layer_name)

    # Plot WMTS data on a rotated map, over the same nominal region.
    ax = fig.add_subplot(1, 2, 2, projection=rotated_crs)
    ax.set_extent([-6, 3, 48, 58], crs=ccrs.PlateCarree())
    ax.coastlines(resolution='50m', color='yellow')
    ax.gridlines(color='lightgrey', linestyle='-')
    # Add WMTS imaging.
    ax.add_wmts(base_uri, layer_name=layer_name)

    plt.show()


def test2():
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.EquidistantConic(central_longitude=36, central_latitude=50))
    ax.set_extent([16, 56, 62, 40], crs=ccrs.PlateCarree())
    states_provinces = cfeature.NaturalEarthFeature(
        category='cultural',
        name='admin_0_boundary_lines_land',
        scale='50m',
        facecolor='none')

    ax.add_feature(cfeature.LAND)
    ax.add_feature(states_provinces, edgecolor='gray')
    ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)

    plt.show()


SOURCE_DIRECTORY1 = r"/home/vadymskipa/Documents/PhD_student/temp/SAVE/main1/TXT/057-2023-02-26/Window_3600_Seconds/"
SOURCE_DIRECTORY2 = r"/home/vadymskipa/Documents/PhD_student/temp/SAVE/main1/TXT/057-2023-02-26/Window_7200_Seconds/"


def read_date_directory_with_dtec_files(source_directory=SOURCE_DIRECTORY1):
    list_files = os.listdir(source_directory)
    dataframe = pd.DataFrame()
    for file in list_files:
        path = os.path.join(source_directory, file)
        dataframe = pd.concat([dataframe, w21.read_dtec_file(path)], ignore_index=True)
    max_lat = dataframe.loc[:, "gdlat"].max()
    min_lat = dataframe.loc[:, "gdlat"].min()
    max_lon = dataframe.loc[:, "gdlon"].max()
    min_lon = dataframe.loc[:, "gdlon"].min()
    print(f"lat: {min_lat} - {max_lat}\nlon: {min_lon} - {max_lon}")


def plot_dtec_graph(dataframe: pd.DataFrame, save_path=None, save_name=None, title=None):
    figure: fig.Figure = plt.figure(layout="tight", figsize=[16.0 * 120 / 300, 9.0 * 120 / 300])
    axes1: axs.Axes = figure.add_subplot(1, 1, 1)
    # line4, = axes1.plot(dataframe.loc[:, "timestamp"] / 3600, dataframe["diff"], linestyle="-", marker=" ",
    #                     markeredgewidth=0.5, markersize=1.5, linewidth=0.6)
    line4, = axes1.plot(dataframe.loc[:, "datetime"], dataframe["dtec"], linestyle="-", marker=" ",
                        markeredgewidth=0.5, markersize=1.5, linewidth=0.6)
    axes1.set_ylim(-1.1, 1.1)
    tick1: float = axes1.get_xticks()[0]
    print(tick1)
    tick_width = dt.timedelta(minutes=5)
    number_of_ticks = dt.timedelta(days=(axes1.get_xticks()[-1] - tick1)) // tick_width + 1
    ticks = [tick1 + (tick_width * i).days + (tick_width * i).seconds / (3600 * 24) for i in range(0, number_of_ticks)]
    axes1.set_xticks(ticks)
    axes1.grid(True)
    if title:
        figure.suptitle(title)
    plt.show()
    plt.close(figure)


def some_plot(source_directory=SOURCE_DIRECTORY1):
    list_files = os.listdir(source_directory)
    for file in list_files:
        path = os.path.join(source_directory, file)
        temp_dataframe: pd.DataFrame = w21.read_dtec_file(path)
        temp_dataframe = w21.add_timestamp_column_to_df(temp_dataframe, dt.datetime(year=2023, month=2, day=26,
                                                                                    tzinfo=dt.timezone.utc))
        temp_dataframe = w21.add_datetime_column_to_df(temp_dataframe)
        plot_dtec_graph(dataframe=temp_dataframe)


def filter_dataframe_by_min_elm(dataframe: pd.DataFrame, min_elm=30):
    mask = dataframe.loc[:, "elm"] >= min_elm
    dataframe = dataframe[mask].reset_index(drop=True)
    return dataframe


def get_start_timestamp(timestamp, period):
    start_date_datetime = dt.datetime.fromtimestamp(timestamp, tz=dt.timezone.utc)
    start_date_datetime = start_date_datetime.replace(hour=0, minute=0, second=0, microsecond=0)
    start_date_timestamp = start_date_datetime.timestamp()
    start_timestamp = (timestamp - start_date_timestamp) // period * period + start_date_timestamp
    return start_timestamp


def get_end_timestamp(timestamp, period):
    end_date_datetime = dt.datetime.fromtimestamp(timestamp, tz=dt.timezone.utc)
    end_date_datetime = end_date_datetime.replace(hour=0, minute=0, second=0, microsecond=0)
    end_date_timestamp = end_date_datetime.timestamp()
    end_timestamp = ((timestamp - end_date_timestamp) // period + 1) * period + end_date_timestamp
    return end_timestamp


def count_data_for_tick(dataframe: pd.DataFrame):
    data_dict = {}
    sat_id_array = np.unique(dataframe.loc[:, "sat_id"])
    for sat_id in sat_id_array:
        sat_id_dataframe: pd.DataFrame = dataframe.loc[dataframe.loc[:, "sat_id"] == sat_id]
        mean_dtec = sat_id_dataframe.loc[:, "dtec"].mean()
        mean_gdlat = sat_id_dataframe.loc[:, "gdlat"].mean()
        mean_gdlon = sat_id_dataframe.loc[:, "gdlon"].mean()
        number = len(sat_id_dataframe.index)
        Sat_tuple = namedtuple("Sat_id_tuple", ["mean_dtec", "mean_gdlon", "mean_gdlat", "number"])
        sat_tuple = Sat_tuple(mean_dtec, mean_gdlon, mean_gdlat, number)
        data_dict[sat_id] = sat_tuple
    return data_dict


def plot_sat_dtec(data_dict: Dict, title):
    fig: plt.Figure = plt.figure(layout="tight", figsize=[16.0 * 120 / 300, 9.0 * 120 / 300], dpi=300)
    fig.suptitle(title)
    ax: cgeoaxes.GeoAxes = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_extent([21, 51, 57, 35], crs=ccrs.PlateCarree())
    states_provinces = cfeature.NaturalEarthFeature(
        category='cultural',
        name='admin_0_boundary_lines_land',
        scale='50m',
        facecolor='none')

    ax.add_feature(cfeature.LAND)
    ax.add_feature(states_provinces, edgecolor='gray')
    ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)

    list_x = [sat_tuple.mean_gdlon for sat_tuple in data_dict.values()]
    list_y = [sat_tuple.mean_gdlat for sat_tuple in data_dict.values()]
    list_color = [sat_tuple.mean_dtec for sat_tuple in data_dict.values()]
    list_size = [sat_tuple.number for sat_tuple in data_dict.values()]
    list_sat_id = [sat_id for sat_id in data_dict.keys()]
    trans_offset = offset_copy(ax.transData, fig=fig, x=0.05, y=0.10, units='inches')
    color_normalize = mplcolors.Normalize(-0.6, 0.6)
    colormap = mpl.colormaps["inferno"]
    fig.colorbar(mplcm.ScalarMappable(norm=color_normalize, cmap=colormap), ax=ax)
    ax.scatter(x=list_x, y=list_y, s=list_size, c=list_color, transform=ccrs.PlateCarree(), cmap=colormap, norm=color_normalize)
    for index in range(len(list_x)):
        ax.text(list_x[index], list_y[index], list_sat_id[index], transform=trans_offset)
    # plt.show()
    return fig


SAVE_DIRECTORY_SOMEPLOT2_2 = r"/home/vadymskipa/Documents/PhD_student/temp/SAVE/some_plot2/Window_7200_Seconds"
SAVE_DIRECTORY_SOMEPLOT2 = r"/home/vadymskipa/Documents/PhD_student/temp/SAVE/some_plot2/Window_3600_Seconds"
def some_plot2(source_directory=SOURCE_DIRECTORY1, period=300, save_directory=SAVE_DIRECTORY_SOMEPLOT2):
    list_files = os.listdir(source_directory)
    dataframe = pd.DataFrame()
    for file in list_files:
        sat_id = int(os.path.splitext(file)[0][1:])
        path = os.path.join(source_directory, file)
        temp_dataframe: pd.DataFrame = w21.read_dtec_file(path)
        temp_dataframe = temp_dataframe.assign(sat_id=sat_id)
        dataframe = pd.concat([dataframe, temp_dataframe], ignore_index=True)
    dataframe = w21.add_timestamp_column_to_df(dataframe, dt.datetime(year=2023, month=2, day=26,
                                                                                tzinfo=dt.timezone.utc))
    dataframe = w21.add_datetime_column_to_df(dataframe)
    dataframe = filter_dataframe_by_min_elm(dataframe, 20)
    start_timestamp = get_start_timestamp(dataframe.loc[:, "timestamp"].min(), period)
    end_timestamp = get_end_timestamp(dataframe.loc[:, "timestamp"].max(), period)
    list_tick_timestamp = [timestamp for timestamp in range(int(start_timestamp), int(end_timestamp+1), period)]
    tick_dict = {}
    for tick_timestamp in list_tick_timestamp:
        mask = dataframe.loc[:, "timestamp"] >= tick_timestamp
        mask = np.logical_and(mask, dataframe.loc[:, "timestamp"] < (tick_timestamp + period))
        tick_dataframe = dataframe.loc[mask]
        tick_dict[tick_timestamp] = count_data_for_tick(tick_dataframe)
    for tick_timestamp, data_dict in tick_dict.items():
        tick_datetime = dt.datetime.fromtimestamp(tick_timestamp, tz=dt.timezone.utc)
        title = f"{tick_datetime.timetuple().tm_yday:0=3}-{tick_datetime.year}-{tick_datetime.month:0=2}-" \
                f"{tick_datetime.day:0=2}---{tick_datetime.hour:0=2}:{tick_datetime.minute:0=2}:" \
                f"{tick_datetime.second:0=2}"
        fig = plot_sat_dtec(data_dict, title)
        save_path = os.path.join(save_directory, title + ".png")
        fig.savefig(save_path)
        plt.close(fig)


SAVE_DIRECTORY_SOMEPLOT3 = r"/home/vadymskipa/Documents/PhD_student/temp/SAVE/some_plot3/Window_7200_Seconds"
def some_plot3():
    some_plot2(source_directory=SOURCE_DIRECTORY2, period=900, save_directory=SAVE_DIRECTORY_SOMEPLOT3)


def check_date_directory_name(dir_name):
    re_pattern = "^\d{3}-\d{4}-\d{2}-\d{2}$"
    if re.search(re_pattern, dir_name):
        return True
    return False


def check_window_directory_name(dir_name):
    re_pattern = "^Window_\d+_Seconds$"
    if re.search(re_pattern, dir_name):
        return True
    return False


SOURCE_DIRECTORY4 = r"/home/vadymskipa/Documents/PhD_student/temp/SAVE/main1/TXT"
SAVE_DIRECTORY_SOMEPLOT4 = r"/home/vadymskipa/Documents/PhD_student/temp/SAVE/some_plot4"
def some_plot4(source_directory=SOURCE_DIRECTORY4, period=300, save_directory=SAVE_DIRECTORY_SOMEPLOT4):
    list_entries = os.listdir(source_directory)
    list_directory_paths_0 = []
    for entry in list_entries:
        entry_path = os.path.join(source_directory, entry)
        if not os.path.isdir(entry_path):
            continue
        if check_date_directory_name(entry):
            list_directory_paths_0.append(entry_path)
    for directory_path_0 in list_directory_paths_0:
        save_directory_path_0 = os.path.join(save_directory, os.path.basename(directory_path_0))
        if not os.path.exists(save_directory_path_0):
            os.mkdir(save_directory_path_0)
        list_directory_paths_1 = []
        list_entries_1 = os.listdir(directory_path_0)
        for entry in list_entries_1:
            entry_path = os.path.join(directory_path_0, entry)
            if not os.path.isdir(entry_path):
                continue
            if check_window_directory_name(entry):
                list_directory_paths_1.append(entry_path)
        for directory_path_1 in list_directory_paths_1:
            save_directory_path_1 = os.path.join(save_directory_path_0, os.path.basename(directory_path_1))
            if not os.path.exists(save_directory_path_1):
                os.mkdir(save_directory_path_1)
            some_plot2(directory_path_1, period, save_directory_path_1)



def plot_diff_from_directory(source_directory, save_directory, name):
    list_entries = os.listdir(source_directory)
    for entry in list_entries:
        name_2 = name + "_" + os.path.splitext(entry)[0]
        entry_path = os.path.join(source_directory, entry)
        dataframe = w21.read_dtec_file(entry_path)
        temp_date = dt.datetime(year=int(name_2[4:8]), month=int(name_2[9:11]),
                                day=int(name_2[12:14]), tzinfo=dt.timezone.utc)
        dataframe = w21.add_timestamp_column_to_df(dataframe, temp_date)
        dataframe = w21.add_datetime_column_to_df(dataframe)
        w21.plot_difference_graph(save_directory, name_2, dataframe)

SAVE_DIRECTORY_PLOT_DIFF1 = r"/home/vadymskipa/Documents/PhD_student/temp/SAVE/plot_diff1"
def plot_diff1(source_directory=SOURCE_DIRECTORY4, save_directory=SAVE_DIRECTORY_PLOT_DIFF1):
    list_entries = os.listdir(source_directory)
    list_directory_paths_0 = []
    for entry in list_entries:
        entry_path = os.path.join(source_directory, entry)
        if not os.path.isdir(entry_path):
            continue
        if check_date_directory_name(entry):
            list_directory_paths_0.append(entry_path)
    for directory_path_0 in list_directory_paths_0:
        save_directory_path_0 = os.path.join(save_directory, os.path.basename(directory_path_0))
        name_0 = os.path.basename(directory_path_0)
        if not os.path.exists(save_directory_path_0):
            os.mkdir(save_directory_path_0)
        list_directory_paths_1 = []
        list_entries_1 = os.listdir(directory_path_0)
        for entry in list_entries_1:
            entry_path = os.path.join(directory_path_0, entry)
            if not os.path.isdir(entry_path):
                continue
            if check_window_directory_name(entry):
                list_directory_paths_1.append(entry_path)
        for directory_path_1 in list_directory_paths_1:
            save_directory_path_1 = os.path.join(save_directory_path_0, os.path.basename(directory_path_1))
            name_1 = name_0 + "_" + os.path.basename(directory_path_1)
            if not os.path.exists(save_directory_path_1):
                os.mkdir(save_directory_path_1)
            plot_diff_from_directory(directory_path_1, save_directory_path_1, name_1)


DEVIATION = 0.3
def plot_difference_graph2(save_path, name, dataframe, title=None):
    if not title:
        title = name
    figure: fig.Figure = plt.figure(layout="tight", figsize=[16.0 * 120 / 300, 9.0 * 120 / 300])
    axes1: axs.Axes = figure.add_subplot(1, 1, 1)
    # line4, = axes1.plot(dataframe.loc[:, "timestamp"] / 3600, dataframe["diff"], linestyle="-", marker=" ",
    #                     markeredgewidth=0.5, markersize=1.5, linewidth=0.6)
    temp_dataframe = dataframe.loc[dataframe.loc[:, "dtec"] > DEVIATION]
    line1, = axes1.plot(temp_dataframe.loc[:, "datetime"], temp_dataframe["dtec"], linestyle=" ",
                        marker=".", color="blue", markeredgewidth=0.8, markersize=0.9)
    temp_dataframe = dataframe.loc[dataframe.loc[:, "dtec"] < -DEVIATION]
    line2, = axes1.plot(temp_dataframe.loc[:, "datetime"], temp_dataframe["dtec"], linestyle=" ",
                        marker=".", color="red", markeredgewidth=0.8, markersize=0.9)
    temp_mask = np.logical_and(dataframe.loc[:, "dtec"] <= DEVIATION, dataframe.loc[:, "dtec"] >= -DEVIATION)
    temp_dataframe = dataframe.loc[temp_mask]
    line3, = axes1.plot(temp_dataframe.loc[:, "datetime"], temp_dataframe["dtec"], linestyle=" ",
                        marker=".", color="gray", markeredgewidth=0.5, markersize=0.6)
    start_datetime = dataframe.iloc[0].loc["datetime"].replace(hour=0, minute=0, second=0, microsecond=0)
    end_datetime = start_datetime + dt.timedelta(days=1)
    axes1.set_xlim(start_datetime, end_datetime)
    axes1.set_ylim(-1.5, 1.5)
    axes1.grid(True)
    if title:
        figure.suptitle(title)
    plt.savefig(os.path.join(save_path, name + ".png"), dpi=300)
    plt.close(figure)


def plot_diff_from_directory2(source_directory, save_directory, name):
    list_entries = os.listdir(source_directory)
    for entry in list_entries:
        name_2 = name + "_" + os.path.splitext(entry)[0]
        entry_path = os.path.join(source_directory, entry)
        dataframe = w21.read_dtec_file(entry_path)
        temp_date = dt.datetime(year=int(name_2[4:8]), month=int(name_2[9:11]),
                                day=int(name_2[12:14]), tzinfo=dt.timezone.utc)
        dataframe = w21.add_timestamp_column_to_df(dataframe, temp_date)
        dataframe = w21.add_datetime_column_to_df(dataframe)
        plot_difference_graph2(save_directory, name_2, dataframe)



SAVE_DIRECTORY_PLOT_DIFF2 = r"/home/vadymskipa/Documents/PhD_student/temp/SAVE/plot_diff2"
def plot_diff2(source_directory=SOURCE_DIRECTORY4, save_directory=SAVE_DIRECTORY_PLOT_DIFF2):
    list_entries = os.listdir(source_directory)
    list_directory_paths_0 = []
    for entry in list_entries:
        entry_path = os.path.join(source_directory, entry)
        if not os.path.isdir(entry_path):
            continue
        if check_date_directory_name(entry):
            list_directory_paths_0.append(entry_path)
    for directory_path_0 in list_directory_paths_0:
        save_directory_path_0 = os.path.join(save_directory, os.path.basename(directory_path_0))
        name_0 = os.path.basename(directory_path_0)
        if not os.path.exists(save_directory_path_0):
            os.mkdir(save_directory_path_0)
        list_directory_paths_1 = []
        list_entries_1 = os.listdir(directory_path_0)
        for entry in list_entries_1:
            entry_path = os.path.join(directory_path_0, entry)
            if not os.path.isdir(entry_path):
                continue
            if check_window_directory_name(entry):
                list_directory_paths_1.append(entry_path)
        for directory_path_1 in list_directory_paths_1:
            save_directory_path_1 = os.path.join(save_directory_path_0, os.path.basename(directory_path_1))
            name_1 = name_0 + "_" + os.path.basename(directory_path_1)
            if not os.path.exists(save_directory_path_1):
                os.mkdir(save_directory_path_1)
            plot_diff_from_directory2(directory_path_1, save_directory_path_1, name_1)


def print_ukraine_map():
    lats = (44.38, 52.38)
    lons = (22.14, 40.23)
    fig = plt.figure(layout="tight", figsize=(16.0 * 120 / 300, 9.0 * 120 / 300), dpi=300)
    projection1 = ccrs.PlateCarree()
    projection2 = ccrs.EquidistantConic(central_longitude=(lons[0] + lons[1])/2,
                                                                   central_latitude=(lats[0] + lats[1])/2)
    ax = fig.add_subplot(1, 1, 1, projection=projection1)
    ax.set_extent([lons[0] - 1, lons[1] + 1, lats[0] - 1, lats[1] + 1], crs=projection1)
    shape_feature = cfeature.ShapelyFeature(creader(SHAPEFILE).geometries(), projection1, facecolor='none')

    x_tics = list(range(int(lons[0]) + 1, int(lons[1]), 2))
    y_tics = list(range(int(lats[0]) + 1, int(lats[1]), 2))
    ax.add_feature(shape_feature, edgecolor='gray')
    ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, xlocs=x_tics, ylocs=y_tics)

    plt.savefig(r"/home/vadymskipa/Documents/PhD_student/temp/Ukraine_platecarree.jpg")


def print_sites_locations(sites: pd.DataFrame, lats, lons, save_path, print_names=True, lat_tics=2, lon_tics=2, radius=5):
    projection1 = ccrs.PlateCarree()
    projection2 = ccrs.EquidistantConic(central_longitude=(lons[0] + lons[1]) / 2,
                                        central_latitude=(lats[0] + lats[1]) / 2)
    fig = plt.figure(layout="tight", dpi=300)
    ax = fig.add_subplot(1, 1, 1, projection=projection2)
    ax.set_extent([lons[0] - 1, lons[1] + 1, lats[0] - 1, lats[1] + 1], crs=projection1)
    shape_feature = cfeature.ShapelyFeature(creader(SHAPEFILE).geometries(), projection1, facecolor='none')

    ax.add_feature(shape_feature, edgecolor='gray')
    # ax.gridlines(draw_labels=True, dms=True, x_inline=True, y_inline=True)

    list_x = [lon for lon in sites.loc[:, "gdlonr"]]
    list_y = [lat for lat in sites.loc[:, "gdlatr"]]
    try:
        list_sites = [site.decode("ascii") for site in sites.loc[:, "gps_site"]]
    except AttributeError:
        list_sites = [site for site in sites.loc[:, "gps_site"]]
    ax.scatter(x=list_x, y=list_y, transform=projection1, linewidths=0.0, pickradius=radius)
    if print_names:
        for index in range(len(list_x)):
            ax.text(list_x[index]+0.2, list_y[index]-0.1, list_sites[index], transform=projection1, size=8)
    x_tics = list(range(int(lons[0]), int(lons[1]), lon_tics))
    y_tics = list(range(int(lats[0]), int(lats[1]), lat_tics))
    # ax.set_xticks(x_tics)
    # ax.set_yticks(y_tics)
    ax.gridlines(xlocs=x_tics, ylocs=y_tics, draw_labels=True)

    plt.savefig(save_path, dpi=300)



def print_ukraine_sites():
    site_path = r"/home/vadymskipa/Documents/PhD_student/data/site_hdf/site_20230608.001.h5.hdf5"
    save_path = r"/home/vadymskipa/Documents/PhD_student/temp/Ukraine_equidistantconic_gps_sites_without_names.jpeg"
    lats = (44.38, 52.38)
    lons = (22.14, 40.23)

    dataframe_sites = wsite.get_sites_dataframe_by_coordinates(site_path, lons=lons, lats=lats)
    print_sites_locations(dataframe_sites, lats=lats, lons=lons, save_path=save_path, print_names=False)

def print_sites_from_sites_txt(sites_txt_path, save_path):
    sites_dataframe: pd.DataFrame = w21.read_sites_txt(sites_txt_path)
    min_lon = sites_dataframe.loc[:, "gdlonr"].min()
    max_lon = sites_dataframe.loc[:, "gdlonr"].max()
    min_lat = sites_dataframe.loc[:, "gdlatr"].min()
    max_lat = sites_dataframe.loc[:, "gdlatr"].max()
    print_sites_locations(sites_dataframe, lats=(min_lat - 2, max_lat + 2), lons=(min_lon - 2, max_lon + 2),
                          save_path=save_path, print_names=False, lat_tics=4, lon_tics=5, radius=3)


def main2023jan1():
    sites_txt_path = r"/home/vadymskipa/Documents/PhD_student/data/Europe_36-70__-10-20/dtec_txt/268-2020-09-24/Sites.txt"
    save_path = (r"/home/vadymskipa/Documents/PhD_student/data/Europe_36-70__-10-20/dtec_txt/268-2020-09-24/"
                 r"Sites-268-2020-09-24.jpeg")
    print_sites_from_sites_txt(sites_txt_path, save_path)


def main2023jan2():
    source_directories = [r"/home/vadymskipa/Documents/PhD_student/data/Europe_36-70__-10-20/dtec_txt/055-2023-02-24",
                          r"/home/vadymskipa/Documents/PhD_student/data/Europe_36-70__-10-20/dtec_txt/056-2023-02-25",
                          r"/home/vadymskipa/Documents/PhD_student/data/Europe_36-70__-10-20/dtec_txt/057-2023-02-26",
                          r"/home/vadymskipa/Documents/PhD_student/data/Europe_36-70__-10-20/dtec_txt/058-2023-02-27",
                          r"/home/vadymskipa/Documents/PhD_student/data/Europe_36-70__-10-20/dtec_txt/059-2023-02-28",
                          r"/home/vadymskipa/Documents/PhD_student/data/Europe_36-70__-10-20/dtec_txt/060-2023-03-01",
                          r"/home/vadymskipa/Documents/PhD_student/data/Europe_36-70__-10-20/dtec_txt/061-2023-03-02"]
    for source_directory in source_directories:
        sites_txt_path = os.path.join(source_directory, "Sites.txt")
        save_path = os.path.join(source_directory, "Sites.png")
        print_sites_from_sites_txt(sites_txt_path, save_path)


def main2023jan3():
    source_directories = [r"/home/vadymskipa/Documents/PhD_student/data/Europe_36-70__-10-20/dtec_txt/080-2023-03-21",
                          r"/home/vadymskipa/Documents/PhD_student/data/Europe_36-70__-10-20/dtec_txt/081-2023-03-22",
                          r"/home/vadymskipa/Documents/PhD_student/data/Europe_36-70__-10-20/dtec_txt/082-2023-03-23",
                          r"/home/vadymskipa/Documents/PhD_student/data/Europe_36-70__-10-20/dtec_txt/083-2023-03-24",
                          r"/home/vadymskipa/Documents/PhD_student/data/Europe_36-70__-10-20/dtec_txt/084-2023-03-25",
                          r"/home/vadymskipa/Documents/PhD_student/data/Europe_36-70__-10-20/dtec_txt/085-2023-03-26",
                          r"/home/vadymskipa/Documents/PhD_student/data/Europe_36-70__-10-20/dtec_txt/086-2023-03-27",
                          r"/home/vadymskipa/Documents/PhD_student/data/Europe_36-70__-10-20/dtec_txt/087-2023-03-28"]
    for source_directory in source_directories:
        sites_txt_path = os.path.join(source_directory, "Sites.txt")
        save_path = os.path.join(source_directory, "Sites.png")
        print_sites_from_sites_txt(sites_txt_path, save_path)


def main2023jan4():
    source_directories = [r"/home/vadymskipa/Documents/PhD_student/data/Europe_36-70__-10-20/dtec_txt/112-2023-04-22",
                          r"/home/vadymskipa/Documents/PhD_student/data/Europe_36-70__-10-20/dtec_txt/113-2023-04-23",
                          r"/home/vadymskipa/Documents/PhD_student/data/Europe_36-70__-10-20/dtec_txt/114-2023-04-24",
                          r"/home/vadymskipa/Documents/PhD_student/data/Europe_36-70__-10-20/dtec_txt/115-2023-04-25",
                          r"/home/vadymskipa/Documents/PhD_student/data/Europe_36-70__-10-20/dtec_txt/116-2023-04-26"]
    for source_directory in source_directories:
        sites_txt_path = os.path.join(source_directory, "Sites.txt")
        save_path = os.path.join(source_directory, "Sites.png")
        print_sites_from_sites_txt(sites_txt_path, save_path)


def main2024feb1():
    source_directories = [r"/home/vadymskipa/PhD_student/data/America_25-70__-125--65/dtec_txt/055-2023-02-24",
                          r"/home/vadymskipa/PhD_student/data/America_25-70__-125--65/dtec_txt/056-2023-02-25",
                          r"/home/vadymskipa/PhD_student/data/America_25-70__-125--65/dtec_txt/057-2023-02-26",
                          r"/home/vadymskipa/PhD_student/data/America_25-70__-125--65/dtec_txt/058-2023-02-27",
                          r"/home/vadymskipa/PhD_student/data/America_25-70__-125--65/dtec_txt/059-2023-02-28",
                          r"/home/vadymskipa/PhD_student/data/America_25-70__-125--65/dtec_txt/060-2023-03-01",
                          r"/home/vadymskipa/PhD_student/data/America_25-70__-125--65/dtec_txt/061-2023-03-02",
                          r"/home/vadymskipa/PhD_student/data/America_25-70__-125--65/dtec_txt/062-2023-03-03"]
    for source_directory in source_directories:
        sites_txt_path = os.path.join(source_directory, "Sites.txt")
        save_path = os.path.join(source_directory, "Sites.png")
        print_sites_from_sites_txt(sites_txt_path, save_path)


def main2024feb2():
    sites_dataframe = wsite.get_sites_dataframe(r"/home/vadymskipa/PhD_student/data/site_hdf/site_20230224.001.h5")
    # print_sites_locations(sites_dataframe, save_path=r"/home/vadymskipa/PhD_student/temp/sites1.png", lats=(-90, 90),
    #                       lons=(-180, 180))
    lats = (-90, 90)
    lons = (-180, 180)
    radius = 2
    lat_tics = 10
    lon_tics = 10
    save_path = r"/home/vadymskipa/PhD_student/temp/sites1.png"

    projection1 = ccrs.PlateCarree()
    # projection2 = ccrs.EquidistantConic(central_longitude=(lons[0] + lons[1]) / 2,
    #                                     central_latitude=(lats[0] + lats[1]) / 2)
    fig = plt.figure(layout="tight", dpi=300)
    ax = fig.add_subplot(1, 1, 1, projection=projection1)
    ax.set_extent([lons[0] - 1, lons[1] + 1, lats[0] - 1, lats[1] + 1], crs=projection1)
    shape_feature = cfeature.ShapelyFeature(creader(SHAPEFILE).geometries(), projection1, facecolor='none')

    ax.add_feature(shape_feature, edgecolor='gray')
    # ax.gridlines(draw_labels=True, dms=True, x_inline=True, y_inline=True)

    list_x = [lon for lon in sites_dataframe.loc[:, "gdlonr"]]
    list_y = [lat for lat in sites_dataframe.loc[:, "gdlatr"]]
    try:
        list_sites = [site.decode("ascii") for site in sites_dataframe.loc[:, "gps_site"]]
    except AttributeError:
        list_sites = [site for site in sites_dataframe.loc[:, "gps_site"]]
    ax.scatter(x=list_x, y=list_y, transform=projection1, linewidths=0.0, pickradius=radius)
    # if print_names:
    #     for index in range(len(list_x)):
    #         ax.text(list_x[index] + 0.2, list_y[index] - 0.1, list_sites[index], transform=projection1, size=8)
    x_tics = list(range(int(lons[0]), int(lons[1]), lon_tics))
    y_tics = list(range(int(lats[0]), int(lats[1]), lat_tics))
    # ax.set_xticks(x_tics)
    # ax.set_yticks(y_tics)
    ax.gridlines(xlocs=x_tics, ylocs=y_tics, draw_labels=True)

    plt.savefig(save_path, dpi=300)


def main2024feb3():
    source_directories = [r"/home/vadymskipa/PhD_student/data/South_Africa_-20--35__15-35/dtec_txt/055-2023-02-24",
                          r"/home/vadymskipa/PhD_student/data/South_Africa_-20--35__15-35/dtec_txt/056-2023-02-25",
                          r"/home/vadymskipa/PhD_student/data/South_Africa_-20--35__15-35/dtec_txt/057-2023-02-26",
                          r"/home/vadymskipa/PhD_student/data/South_Africa_-20--35__15-35/dtec_txt/058-2023-02-27",
                          r"/home/vadymskipa/PhD_student/data/South_Africa_-20--35__15-35/dtec_txt/059-2023-02-28",
                          r"/home/vadymskipa/PhD_student/data/South_Africa_-20--35__15-35/dtec_txt/060-2023-03-01",
                          r"/home/vadymskipa/PhD_student/data/South_Africa_-20--35__15-35/dtec_txt/061-2023-03-02",
                          r"/home/vadymskipa/PhD_student/data/South_Africa_-20--35__15-35/dtec_txt/062-2023-03-03"]
    for source_directory in source_directories:
        sites_txt_path = os.path.join(source_directory, "Sites.txt")
        save_path = os.path.join(source_directory, "Sites.png")
        print_sites_from_sites_txt(sites_txt_path, save_path)


def main2024apr1():
    source_directories = [r"/home/vadymskipa/PhD_student/data/Europe_30-80__-10-50/dtec_txt/268-2020-09-24"]
    for source_directory in source_directories:
        sites_txt_path = os.path.join(source_directory, "Sites.txt")
        save_path = os.path.join(source_directory, "Sites.png")
        print_sites_from_sites_txt(sites_txt_path, save_path)

def main2024jun1():
    source_directories = [r"/home/vadymskipa/PhD_student/data/Europe_36-70__-10-20/dtec_txt/064-2016-03-04",
                          r"/home/vadymskipa/PhD_student/data/Europe_36-70__-10-20/dtec_txt/065-2016-03-05",
                          r"/home/vadymskipa/PhD_student/data/Europe_36-70__-10-20/dtec_txt/066-2016-03-06",
                          r"/home/vadymskipa/PhD_student/data/Europe_36-70__-10-20/dtec_txt/067-2016-03-07",
                          r"/home/vadymskipa/PhD_student/data/Europe_36-70__-10-20/dtec_txt/068-2016-03-08",
                          r"/home/vadymskipa/PhD_student/data/Europe_36-70__-10-20/dtec_txt/069-2016-03-09",
                          r"/home/vadymskipa/PhD_student/data/Europe_36-70__-10-20/dtec_txt/070-2016-03-10",
                          r"/home/vadymskipa/PhD_student/data/Europe_36-70__-10-20/dtec_txt/071-2016-03-11"]
    site_directory = r"/home/vadymskipa/PhD_student/data/site_hdf/"
    for source_directory in source_directories:
        sites_txt_path = os.path.join(source_directory, "Sites.txt")
        save_path = os.path.join(source_directory, "Sites.png")
        print_sites_from_sites_txt(sites_txt_path, save_path)


def main2024jun2():
    source_directories = [r"/home/vadymskipa/PhD_student/data/Europe_36-70__-10-20/dtec_txt/243-2016-08-30",
                          r"/home/vadymskipa/PhD_student/data/Europe_36-70__-10-20/dtec_txt/244-2016-08-31",
                          r"/home/vadymskipa/PhD_student/data/Europe_36-70__-10-20/dtec_txt/245-2016-09-01",
                          r"/home/vadymskipa/PhD_student/data/Europe_36-70__-10-20/dtec_txt/246-2016-09-02",
                          r"/home/vadymskipa/PhD_student/data/Europe_36-70__-10-20/dtec_txt/247-2016-09-03",
                          r"/home/vadymskipa/PhD_student/data/Europe_36-70__-10-20/dtec_txt/248-2016-09-04"]
    site_directory = r"/home/vadymskipa/PhD_student/data/site_hdf/"
    for source_directory in source_directories:
        sites_txt_path = os.path.join(source_directory, "Sites.txt")
        save_path = os.path.join(source_directory, "Sites.png")
        print_sites_from_sites_txt(sites_txt_path, save_path)


def print_sites_from_sites_txt_new(sites_txt_path, save_path):
    sites_dataframe: pd.DataFrame = w21.read_sites_txt(sites_txt_path)
    min_lon = sites_dataframe.loc[:, "gdlonr"].min()
    max_lon = sites_dataframe.loc[:, "gdlonr"].max()
    min_lat = sites_dataframe.loc[:, "gdlatr"].min()
    max_lat = sites_dataframe.loc[:, "gdlatr"].max()
    print_sites_locations_new(sites_dataframe, lats=(min_lat - 2, max_lat + 2), lons=(min_lon - 2, max_lon + 2),
                          save_path=save_path, print_names=False, lat_tics=4, lon_tics=5, radius=3)


def print_sites_locations_new(sites: pd.DataFrame, lats, lons, save_path, print_names=True, lat_tics=2, lon_tics=2, radius=5):
    projection1 = ccrs.PlateCarree()
    projection2 = ccrs.EquidistantConic(central_longitude=(lons[0] + lons[1]) / 2,
                                        central_latitude=(lats[0] + lats[1]) / 2)
    fig = plt.figure(layout="tight", dpi=300)
    ax = fig.add_subplot(1, 1, 1, projection=projection2)
    ax.set_extent([lons[0] - 1, lons[1] + 1, lats[0] - 1, lats[1] + 1], crs=projection1)
    shape_feature = cfeature.ShapelyFeature(creader(SHAPEFILE).geometries(), projection1, facecolor='none')

    ax.add_feature(shape_feature, edgecolor='gray')
    # ax.gridlines(draw_labels=True, dms=True, x_inline=True, y_inline=True)

    list_x = [lon for lon in sites.loc[:, "gdlonr"]]
    list_y = [lat for lat in sites.loc[:, "gdlatr"]]
    list_xy = []
    for i in range(len(list_x)):
        list_xy.append((list_x[i], list_y[i]))
    try:
        list_sites = [site.decode("ascii") for site in sites.loc[:, "gps_site"]]
    except AttributeError:
        list_sites = [site for site in sites.loc[:, "gps_site"]]
    ax.plot(list_x, list_y, linestyle=" ", marker=".", markersize=2, color="black", transform=projection1)

    # ax.plot(array_x, array_y)
    # for i in range(len(list_x)):
    #     ax.plot(list_x[i], list_y[i], marker=".", markersize=2, color="black", transform=projection1)
    if print_names:
        for index in range(len(list_x)):
            ax.text(list_x[index]+0.2, list_y[index]-0.1, list_sites[index], transform=projection1, size=8)
    x_tics = list(range(int(lons[0]), int(lons[1]), lon_tics))
    y_tics = list(range(int(lats[0]), int(lats[1]), lat_tics))
    # ax.set_xticks(x_tics)
    # ax.set_yticks(y_tics)
    ax.gridlines(xlocs=x_tics, ylocs=y_tics, draw_labels=True)

    plt.savefig(save_path, dpi=300)


def main2024jul2():
    dataframe = w21.main2024jul3()
    source_directory = r"/home/vadymskipa/PhD_student/data/Europe_30-80__-10-50/dvtec_altitude1/dvtec_txt_alt_400/268-2020-09-24"
    site_directory = r"/home/vadymskipa/PhD_student/data/site_hdf/"
    sites_txt_path = os.path.join(source_directory, "Sites.txt")
    sites_dataframe: pd.DataFrame = w21.read_sites_txt(sites_txt_path)
    min_lon = sites_dataframe.loc[:, "gdlonr"].min()
    max_lon = sites_dataframe.loc[:, "gdlonr"].max()
    min_lat = sites_dataframe.loc[:, "gdlatr"].min()
    max_lat = sites_dataframe.loc[:, "gdlatr"].max()
    lons = (min_lon, max_lon)
    lats = (min_lat, max_lat)
    lat_tics = 4
    lon_tics = 5
    min_date = dt.datetime(year=2020, month=9, day=24, hour=9, minute=30, tzinfo=dt.timezone.utc)
    max_date = dt.datetime(year=2020, month=9, day=24, hour=12, minute=0, tzinfo=dt.timezone.utc)
    min_date_ts = min_date.timestamp()
    max_date_ts = max_date.timestamp()
    period_of_averaging = 300
    color_list = ["green", "red", "cyan", "magenta", "yellow", "black"]


    for timestamp in range(int(min_date_ts), int(max_date_ts), period_of_averaging):
        date = dt.datetime.fromtimestamp(timestamp=timestamp)
        save_path = os.path.join(source_directory, f"{date.hour:0=2}{date.minute:0=2}.png")
        timestamp_dataframe = dataframe.loc[dataframe.loc[:, "timestamp"] == timestamp]
        sat_list = np.unique(timestamp_dataframe.loc[:, "sat_id"])
        gps_site_list = np.unique(timestamp_dataframe.loc[:, "gps_site"])


        projection1 = ccrs.PlateCarree()
        projection2 = ccrs.EquidistantConic(central_longitude=(lons[0] + lons[1]) / 2,
                                            central_latitude=(lats[0] + lats[1]) / 2)
        fig = plt.figure(layout="tight", dpi=300)
        ax = fig.add_subplot(1, 1, 1, projection=projection2)
        ax.set_extent([lons[0] - 1, lons[1] + 1, lats[0] - 1, lats[1] + 1], crs=projection1)
        shape_feature = cfeature.ShapelyFeature(creader(SHAPEFILE).geometries(), projection1, facecolor='none')

        ax.add_feature(shape_feature, edgecolor='gray')
        # ax.gridlines(draw_labels=True, dms=True, x_inline=True, y_inline=True)

        list_x = [lon for lon in sites_dataframe.loc[:, "gdlonr"]]
        list_y = [lat for lat in sites_dataframe.loc[:, "gdlatr"]]

        ax.plot(list_x, list_y, linestyle=" ", marker=".", markersize=2, color="black", transform=projection1)
        for gps_site in gps_site_list:
            gps_site_dataframe = timestamp_dataframe.loc[timestamp_dataframe.loc[:, "gps_site"] == gps_site]
            for i in range(len(sat_list)):
                color_i = i % 6
                try:
                    sat_series = gps_site_dataframe.loc[gps_site_dataframe.loc[:, "sat_id"] == sat_list[i]].iloc[0]
                    lon_d = 2 * sat_series.loc["gdlonr"] - sat_series.loc["gdlon"]
                    lat_d = 2 * sat_series.loc["gdlatr"] - sat_series.loc["gdlat"]
                    ax.plot((sat_series.loc["gdlonr"], lon_d), (sat_series.loc["gdlatr"], lat_d), linestyle="-",
                            marker=".", markersize=2, color=color_list[color_i], transform=projection1, linewidth=1.5)
                except Exception as er:
                    print(er, sat_list[i], "\n", gps_site_dataframe.loc[:, "sat_id"])


        # ax.plot(array_x, array_y)
        # for i in range(len(list_x)):
        #     ax.plot(list_x[i], list_y[i], marker=".", markersize=2, color="black", transform=projection1)
        x_tics = list(range(int(lons[0]), int(lons[1]), lon_tics))
        y_tics = list(range(int(lats[0]), int(lats[1]), lat_tics))
        # ax.set_xticks(x_tics)
        # ax.set_yticks(y_tics)
        ax.gridlines(xlocs=x_tics, ylocs=y_tics, draw_labels=True)

        plt.savefig(save_path, dpi=300)




def main2024jul():
    main2024jul2()


def main2024jul1():
    source_directories = [r"/home/vadymskipa/PhD_student/data/Europe_30-80__-10-50/dvtec_altitude/dvtec_txt_alt_400/268-2020-09-24"]
    site_directory = r"/home/vadymskipa/PhD_student/data/site_hdf/"
    for source_directory in source_directories:
        sites_txt_path = os.path.join(source_directory, "Sites.txt")
        save_path = os.path.join(source_directory, "Sites.png")
        print_sites_from_sites_txt_new(sites_txt_path, save_path)


def main2024sep1():
    save_dir = r"/home/vadymskipa/PhD_student/data/Europe_36-70__-10-20/dvtec_csv_5min/maps"
    source_dirs = [r"/home/vadymskipa/PhD_student/data/Europe_36-70__-10-20/dvtec_csv_5min/130-2024-05-09",
                   r"/home/vadymskipa/PhD_student/data/Europe_36-70__-10-20/dvtec_csv_5min/131-2024-05-10",
                   r"/home/vadymskipa/PhD_student/data/Europe_36-70__-10-20/dvtec_csv_5min/132-2024-05-11",
                   r"/home/vadymskipa/PhD_student/data/Europe_36-70__-10-20/dvtec_csv_5min/133-2024-05-12",
                   r"/home/vadymskipa/PhD_student/data/Europe_36-70__-10-20/dvtec_csv_5min/134-2024-05-13"]
    

    for source_dir in source_dirs:
        date = w21.get_date_from_date_directory_name(os.path.basename(source_dir))
        save_dir_1 = w21.get_directory_path(save_dir, w21.get_date_str(date))
        sites_txt_path = os.path.join(source_dir, "Sites.txt")
        sites_dataframe: pd.DataFrame = w21.read_sites_txt(sites_txt_path)
        min_lon = sites_dataframe.loc[:, "gdlonr"].min()
        max_lon = sites_dataframe.loc[:, "gdlonr"].max()
        min_lat = sites_dataframe.loc[:, "gdlatr"].min()
        max_lat = sites_dataframe.loc[:, "gdlatr"].max()
        lons = (min_lon, max_lon)
        lats = (min_lat, max_lat)
        lat_tics = 4
        lon_tics = 5
        min_date_ts = int(date.timestamp())
        max_date_ts = min_date_ts + 24 * 3600
        period_of_averaging = 300
        timestamp_list = []
        for timestamp in range(int(min_date_ts), int(max_date_ts), period_of_averaging):
            timestamp_list.append(timestamp)
        entries = os.listdir(source_dir)
        sats = [f"G{i:0=2}" for i in range(1, 33)]
        sites = []
        for entry in entries:
            if os.path.isdir(os.path.join(source_dir, entry)):
                sites.append(entry)
        for sat in sats:
            save_dir_2 = w21.get_directory_path(save_dir_1, sat)
            sat_dataframe = pd.DataFrame()
            for site in sites:
                source_dir_1 = os.path.join(source_dir, site, "Window_7200_Seconds")
                read_path = os.path.join(source_dir_1, sat + ".csv")
                try:
                    site_dataframe = pd.read_csv(read_path)
                except FileNotFoundError as er:
                    print(er)
                    continue
                site_dataframe.insert(0, "gps_site", site)
                sat_dataframe = pd.concat((sat_dataframe, site_dataframe), ignore_index=True)
            sat_dataframe = w21.add_timestamp_column_to_df(sat_dataframe, date)

            for timestamp in timestamp_list:
                time = dt.datetime.fromtimestamp(timestamp, tz=dt.timezone.utc)
                hour = time.hour
                minute = time.minute
                filename = f"{hour:0=2}{minute:0=2}{sat}.png"
                timestamp_dataframe :pd.DataFrame = sat_dataframe.loc[sat_dataframe.loc[:, "timestamp"] == timestamp]
                if timestamp_dataframe.empty:
                    continue

                projection1 = ccrs.PlateCarree()
                projection2 = ccrs.EquidistantConic(central_longitude=(lons[0] + lons[1]) / 2,
                                                    central_latitude=(lats[0] + lats[1]) / 2)
                fig = plt.figure(layout="tight", dpi=300)
                ax = fig.add_subplot(1, 1, 1, projection=projection2)
                ax.set_extent([lons[0] - 1, lons[1] + 1, lats[0] - 1, lats[1] + 1], crs=projection1)
                shape_feature = cfeature.ShapelyFeature(creader(SHAPEFILE).geometries(), projection1, facecolor='none')

                ax.add_feature(shape_feature, edgecolor='gray')
                # ax.gridlines(draw_labels=True, dms=True, x_inline=True, y_inline=True)

                list_x = [lon for lon in sites_dataframe.loc[:, "gdlonr"]]
                list_y = [lat for lat in sites_dataframe.loc[:, "gdlatr"]]

                ax.plot(list_x, list_y, linestyle=" ", marker=".", markersize=1.5, color="black", transform=projection1)
                color_normalize = mplcolors.Normalize(-0.2, 0.2)
                colormap = plt.colormaps["inferno"]
                fig.colorbar(mplcm.ScalarMappable(norm=color_normalize, cmap=colormap), ax=ax)
                for i in timestamp_dataframe.index:
                    ax.plot(timestamp_dataframe.loc[i, "gdlon"], timestamp_dataframe.loc[i, "gdlat"],
                            color=colormap(timestamp_dataframe.loc[i, "dvtec"]), linestyle=" ", marker=".", markersize=2,
                            transform=projection1)

                # ax.plot(array_x, array_y)
                # for i in range(len(list_x)):
                #     ax.plot(list_x[i], list_y[i], marker=".", markersize=2, color="black", transform=projection1)
                x_tics = list(range(int(lons[0]), int(lons[1]), lon_tics))
                y_tics = list(range(int(lats[0]), int(lats[1]), lat_tics))
                # ax.set_xticks(x_tics)
                # ax.set_yticks(y_tics)
                ax.gridlines(xlocs=x_tics, ylocs=y_tics, draw_labels=True)

                save_path = os.path.join(save_dir_2, filename)
                plt.savefig(save_path, dpi=300)
                plt.close(fig)
                print(filename, dt.datetime.now())


def main2024sep2():
    save_dir = r"/home/vadymskipa/PhD_student/data/Europe_36-70__-10-20/dvtec_csv_5min/maps"
    source_dirs = [r"/home/vadymskipa/PhD_student/data/Europe_36-70__-10-20/dvtec_csv_5min/130-2024-05-09",
                   r"/home/vadymskipa/PhD_student/data/Europe_36-70__-10-20/dvtec_csv_5min/131-2024-05-10",
                   r"/home/vadymskipa/PhD_student/data/Europe_36-70__-10-20/dvtec_csv_5min/132-2024-05-11",
                   r"/home/vadymskipa/PhD_student/data/Europe_36-70__-10-20/dvtec_csv_5min/133-2024-05-12",
                   r"/home/vadymskipa/PhD_student/data/Europe_36-70__-10-20/dvtec_csv_5min/134-2024-05-13"]

    for source_dir in source_dirs:
        sites_txt_path = os.path.join(source_dir, "Sites.txt")
        sites_dataframe: pd.DataFrame = w21.read_sites_txt(sites_txt_path)
        date = w21.get_date_from_date_directory_name(os.path.basename(source_dir))
        save_dir_1 = w21.get_directory_path(save_dir, w21.get_date_str(date))
        min_date_ts = int(date.timestamp())
        max_date_ts = min_date_ts + 24 * 3600
        period_of_averaging = 300
        timestamp_list = []
        for timestamp in range(int(min_date_ts), int(max_date_ts), period_of_averaging):
            timestamp_list.append(timestamp)
        entries = os.listdir(source_dir)
        sats = [f"G{i:0=2}" for i in range(2, 33)]
        sites = []
        for entry in entries:
            if os.path.isdir(os.path.join(source_dir, entry)):
                sites.append(entry)
        for sat in sats:
            save_dir_2 = w21.get_directory_path(save_dir_1, sat)
            sat_dataframe = pd.DataFrame()
            for site in sites:
                source_dir_1 = os.path.join(source_dir, site, "Window_7200_Seconds")
                read_path = os.path.join(source_dir_1, sat + ".csv")
                try:
                    site_dataframe = pd.read_csv(read_path)
                except FileNotFoundError as er:
                    print(er)
                    continue
                site_dataframe.insert(0, "gps_site", site)
                sat_dataframe = pd.concat((sat_dataframe, site_dataframe), ignore_index=True)
            sat_dataframe = w21.add_timestamp_column_to_df(sat_dataframe, date)

            mp_list = []
            for timestamp in timestamp_list:
                timestamp_dataframe: pd.DataFrame = sat_dataframe.loc[sat_dataframe.loc[:, "timestamp"] == timestamp]
                if timestamp_dataframe.empty:
                    continue
                time = dt.datetime.fromtimestamp(timestamp, tz=dt.timezone.utc)
                mp_list.append((time, sites_dataframe, sat, timestamp_dataframe))
            pool = multipool.Pool(7)
            for filename, fig in pool.imap(main2024sep2_p, mp_list):
                plt.figure(fig)
                save_path = os.path.join(save_dir_2, filename)
                plt.savefig(save_path, dpi=300)
                plt.close(fig)
                print(filename, dt.datetime.now())






def main2024sep2_p(data):
    time, sites_dataframe, sat, timestamp_dataframe = data
    min_lon = sites_dataframe.loc[:, "gdlonr"].min()
    max_lon = sites_dataframe.loc[:, "gdlonr"].max()
    min_lat = sites_dataframe.loc[:, "gdlatr"].min()
    max_lat = sites_dataframe.loc[:, "gdlatr"].max()
    lons = (min_lon, max_lon)
    lats = (min_lat, max_lat)
    lat_tics = 4
    lon_tics = 5
    # time = dt.datetime.fromtimestamp(timestamp, tz=dt.timezone.utc)
    hour = time.hour
    minute = time.minute
    filename = f"{hour:0=2}{minute:0=2}{sat}.png"
    # timestamp_dataframe: pd.DataFrame = sat_dataframe.loc[sat_dataframe.loc[:, "timestamp"] == timestamp]
    if timestamp_dataframe.empty:
        return None

    projection1 = ccrs.PlateCarree()
    projection2 = ccrs.EquidistantConic(central_longitude=(lons[0] + lons[1]) / 2,
                                        central_latitude=(lats[0] + lats[1]) / 2)
    fig = plt.figure(layout="tight", dpi=300)
    ax = fig.add_subplot(1, 1, 1, projection=projection2)
    ax.set_extent([lons[0] - 1, lons[1] + 1, lats[0] - 1, lats[1] + 1], crs=projection1)
    shape_feature = cfeature.ShapelyFeature(creader(SHAPEFILE).geometries(), projection1, facecolor='none')

    ax.add_feature(shape_feature, edgecolor='gray')
    # ax.gridlines(draw_labels=True, dms=True, x_inline=True, y_inline=True)

    list_x = [lon for lon in sites_dataframe.loc[:, "gdlonr"]]
    list_y = [lat for lat in sites_dataframe.loc[:, "gdlatr"]]

    ax.plot(list_x, list_y, linestyle=" ", marker=".", markersize=1, color="black", transform=projection1)
    color_normalize = mplcolors.Normalize(-1.0, 1.0)
    colormap = plt.colormaps["Spectral"]
    fig.colorbar(mplcm.ScalarMappable(norm=color_normalize, cmap=colormap), ax=ax)
    for i in timestamp_dataframe.index:
        ax.plot(timestamp_dataframe.loc[i, "gdlon"], timestamp_dataframe.loc[i, "gdlat"],
                color=colormap(timestamp_dataframe.loc[i, "dvtec"]), linestyle=" ", marker=".",
                markersize=2,
                transform=projection1)

    # ax.plot(array_x, array_y)
    # for i in range(len(list_x)):
    #     ax.plot(list_x[i], list_y[i], marker=".", markersize=2, color="black", transform=projection1)
    x_tics = list(range(int(lons[0]), int(lons[1]), lon_tics))
    y_tics = list(range(int(lats[0]), int(lats[1]), lat_tics))
    # ax.set_xticks(x_tics)
    # ax.set_yticks(y_tics)
    ax.gridlines(xlocs=x_tics, ylocs=y_tics, draw_labels=True)
    return filename, fig


def main2024sep3():
    save_dir = r"/home/vadymskipa/PhD_student/data/Europe_36-70__-10-20/dvtec_csv_5min/maps"
    source_dirs = [
        # r"/home/vadymskipa/PhD_student/data/Europe_36-70__-10-20/dvtec_csv_5min/130-2024-05-09",
        # r"/home/vadymskipa/PhD_student/data/Europe_36-70__-10-20/dvtec_csv_5min/131-2024-05-10",
        r"/home/vadymskipa/PhD_student/data/Europe_36-70__-10-20/dvtec_csv_5min/132-2024-05-11",
        # r"/home/vadymskipa/PhD_student/data/Europe_36-70__-10-20/dvtec_csv_5min/133-2024-05-12",
        # r"/home/vadymskipa/PhD_student/data/Europe_36-70__-10-20/dvtec_csv_5min/134-2024-05-13"
    ]
    window = "Window_3600_Seconds"

    for source_dir in source_dirs:
        sites_txt_path = os.path.join(source_dir, "Sites.txt")
        sites_dataframe: pd.DataFrame = w21.read_sites_txt(sites_txt_path)
        date = w21.get_date_from_date_directory_name(os.path.basename(source_dir))
        save_dir_1 = w21.get_directory_path(save_dir, w21.get_date_str(date))
        min_date_ts = int(date.timestamp())
        max_date_ts = min_date_ts + 24 * 3600
        period_of_averaging = 300
        timestamp_list = []
        for timestamp in range(int(min_date_ts), int(max_date_ts), period_of_averaging):
            timestamp_list.append(timestamp)
        entries = os.listdir(source_dir)
        sats = [f"G{i:0=2}" for i in range(18, 33)]
        sites = []
        for entry in entries:
            if os.path.isdir(os.path.join(source_dir, entry)):
                sites.append(entry)
        for sat in sats:
            save_dir_1_5 = w21.get_directory_path(save_dir_1, window)
            save_dir_2 = w21.get_directory_path(save_dir_1_5, sat)
            sat_dataframe = pd.DataFrame()
            for site in sites:
                source_dir_1 = os.path.join(source_dir, site, window)
                read_path = os.path.join(source_dir_1, sat + ".csv")
                try:
                    site_dataframe = pd.read_csv(read_path)
                except FileNotFoundError as er:
                    print(er)
                    continue
                site_dataframe.insert(0, "gps_site", site)
                sat_dataframe = pd.concat((sat_dataframe, site_dataframe), ignore_index=True)
            sat_dataframe = w21.add_timestamp_column_to_df(sat_dataframe, date)

            mp_list = []
            for timestamp in timestamp_list:
                timestamp_dataframe: pd.DataFrame = sat_dataframe.loc[sat_dataframe.loc[:, "timestamp"] == timestamp]
                if timestamp_dataframe.empty:
                    continue
                time = dt.datetime.fromtimestamp(timestamp, tz=dt.timezone.utc)
                mp_list.append((time, sites_dataframe, sat, timestamp_dataframe, save_dir_2))
            pool = multipool.Pool(7)
            for filename in pool.imap(main2024sep3_p, mp_list):
                print(filename, dt.datetime.now())






def main2024sep3_p(data):
    time, sites_dataframe, sat, timestamp_dataframe, save_dir_2 = data
    timestamp_dataframe = timestamp_dataframe.loc[timestamp_dataframe.loc[:, "elm"] >= 30]
    min_lon = sites_dataframe.loc[:, "gdlonr"].min()
    max_lon = sites_dataframe.loc[:, "gdlonr"].max()
    min_lat = sites_dataframe.loc[:, "gdlatr"].min()
    max_lat = sites_dataframe.loc[:, "gdlatr"].max()
    lons = (min_lon, max_lon)
    lats = (min_lat, max_lat)
    lat_tics = 4
    lon_tics = 5
    # time = dt.datetime.fromtimestamp(timestamp, tz=dt.timezone.utc)
    hour = time.hour
    minute = time.minute
    filename = f"{hour:0=2}{minute:0=2}{sat}.png"
    # timestamp_dataframe: pd.DataFrame = sat_dataframe.loc[sat_dataframe.loc[:, "timestamp"] == timestamp]
    if timestamp_dataframe.empty:
        return None

    projection1 = ccrs.PlateCarree()
    projection2 = ccrs.EquidistantConic(central_longitude=(lons[0] + lons[1]) / 2,
                                        central_latitude=(lats[0] + lats[1]) / 2)
    fig = plt.figure(layout="tight", dpi=300)
    ax = fig.add_subplot(1, 1, 1, projection=projection2)
    ax.set_extent([lons[0] - 1, lons[1] + 1, lats[0] - 1, lats[1] + 1], crs=projection1)
    shape_feature = cfeature.ShapelyFeature(creader(SHAPEFILE).geometries(), projection1, facecolor='none')

    ax.add_feature(shape_feature, edgecolor='gray')
    # ax.gridlines(draw_labels=True, dms=True, x_inline=True, y_inline=True)

    list_x = [lon for lon in sites_dataframe.loc[:, "gdlonr"]]
    list_y = [lat for lat in sites_dataframe.loc[:, "gdlatr"]]

    ax.plot(list_x, list_y, linestyle=" ", marker=".", markersize=1, color="black", transform=projection1)
    color_normalize = mplcolors.Normalize(-1.0, 1.0)
    colormap = plt.colormaps["Spectral"]
    fig.colorbar(mplcm.ScalarMappable(norm=color_normalize, cmap=colormap), ax=ax)
    ax.scatter(timestamp_dataframe.loc[:, "gdlon"], timestamp_dataframe.loc[:, "gdlat"],
               c=timestamp_dataframe.loc[:, "dvtec"], marker=".", cmap=colormap, norm=color_normalize, linewidths=0,
               transform=projection1)
    # for i in timestamp_dataframe.index:
    #     ax.plot(timestamp_dataframe.loc[i, "gdlon"], timestamp_dataframe.loc[i, "gdlat"],
    #             color=colormap(timestamp_dataframe.loc[i, "dvtec"]), linestyle=" ", marker=".",
    #             markersize=2,
    #             transform=projection1)

    # ax.plot(array_x, array_y)
    # for i in range(len(list_x)):
    #     ax.plot(list_x[i], list_y[i], marker=".", markersize=2, color="black", transform=projection1)
    x_tics = list(range(int(lons[0]), int(lons[1]), lon_tics))
    y_tics = list(range(int(lats[0]), int(lats[1]), lat_tics))
    # ax.set_xticks(x_tics)
    # ax.set_yticks(y_tics)
    ax.gridlines(xlocs=x_tics, ylocs=y_tics, draw_labels=True)
    save_path = os.path.join(save_dir_2, filename)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    return filename


def main2024sep4():
    # save_dir = r"/home/vadymskipa/PhD_student/data/Europe_36-70__-10-20/dvtec_csv_5min/maps"
    # source_dirs = [r"/home/vadymskipa/PhD_student/data/Europe_36-70__-10-20/dvtec_csv_5min/130-2024-05-09",
    #                r"/home/vadymskipa/PhD_student/data/Europe_36-70__-10-20/dvtec_csv_5min/131-2024-05-10",
    #                r"/home/vadymskipa/PhD_student/data/Europe_36-70__-10-20/dvtec_csv_5min/132-2024-05-11",
    #                r"/home/vadymskipa/PhD_student/data/Europe_36-70__-10-20/dvtec_csv_5min/133-2024-05-12",
    #                r"/home/vadymskipa/PhD_student/data/Europe_36-70__-10-20/dvtec_csv_5min/134-2024-05-13"]
    save_dir = r"/home/vadymskipa/PhD_student/data/123/dvtec_csv_5min/maps"
    source_dirs = [r"/home/vadymskipa/PhD_student/data/123/dvtec_csv_5min/130-2024-05-09",
                   r"/home/vadymskipa/PhD_student/data/123/dvtec_csv_5min/131-2024-05-10",
                   r"/home/vadymskipa/PhD_student/data/123/dvtec_csv_5min/132-2024-05-11",
                   r"/home/vadymskipa/PhD_student/data/123/dvtec_csv_5min/133-2024-05-12",
                   r"/home/vadymskipa/PhD_student/data/123/dvtec_csv_5min/134-2024-05-13"]
    window = "Window_3600_Seconds"

    for source_dir in source_dirs:
        sites_txt_path = os.path.join(source_dir, "Sites.txt")
        sites_dataframe: pd.DataFrame = w21.read_sites_txt(sites_txt_path)
        date = w21.get_date_from_date_directory_name(os.path.basename(source_dir))
        save_dir_1 = w21.get_directory_path(save_dir, w21.get_date_str(date))
        save_dir_1_5 = w21.get_directory_path(save_dir_1, window)
        save_dir_2 = w21.get_directory_path(save_dir_1_5, "all_sats")
        min_date_ts = int(date.timestamp())
        max_date_ts = min_date_ts + 24 * 3600
        period_of_averaging = 300
        timestamp_list = []
        for timestamp in range(int(min_date_ts), int(max_date_ts), period_of_averaging):
            timestamp_list.append(timestamp)
        dataframe = pd.DataFrame()
        entries = os.listdir(source_dir)
        sites = []
        for entry in entries:
            if os.path.isdir(os.path.join(source_dir, entry)):
                sites.append(entry)
        for site in sites:
            site_dataframe = pd.DataFrame()
            source_dir_2 = os.path.join(source_dir, site, window)
            entries_2 = os.listdir(source_dir_2)
            sats = [text[:3] for text in entries_2]
            for sat in sats:
                read_path = os.path.join(source_dir_2, sat + ".csv")
                try:
                    sat_dataframe = pd.read_csv(read_path)
                except FileNotFoundError as er:
                    print(er)
                    continue
                sat_dataframe.insert(0, "sat_id", int(sat[1:]))
                site_dataframe = pd.concat((site_dataframe, sat_dataframe), ignore_index=True)
            site_dataframe.insert(0, "gps_site", site)
            dataframe = pd.concat((dataframe, site_dataframe), ignore_index=True)
        dataframe = w21.add_timestamp_column_to_df(dataframe, date)

        mp_list = []
        for timestamp in timestamp_list:
            timestamp_dataframe: pd.DataFrame = dataframe.loc[dataframe.loc[:, "timestamp"] == timestamp]
            if timestamp_dataframe.empty:
                continue
            time = dt.datetime.fromtimestamp(timestamp, tz=dt.timezone.utc)
            mp_list.append((time, sites_dataframe, timestamp_dataframe, save_dir_2))
        pool = multipool.Pool(7)
        for filename in pool.imap(main2024sep4_p, mp_list):
            print(filename, dt.datetime.now())






def main2024sep4_p(data):
    time, sites_dataframe, timestamp_dataframe, save_dir_2 = data
    # min_lon = sites_dataframe.loc[:, "gdlonr"].min()
    # max_lon = sites_dataframe.loc[:, "gdlonr"].max()
    # min_lat = sites_dataframe.loc[:, "gdlatr"].min()
    # max_lat = sites_dataframe.loc[:, "gdlatr"].max()
    # lons = (min_lon, max_lon)
    # lats = (min_lat, max_lat)
    # lat_tics = 4
    # lon_tics = 5
    lons = (15, 55)
    lats = (35, 65)
    lat_tics = 2
    lon_tics = 3
    # time = dt.datetime.fromtimestamp(timestamp, tz=dt.timezone.utc)
    hour = time.hour
    minute = time.minute
    filename = f"{hour:0=2}{minute:0=2}all_sats.png"
    # timestamp_dataframe: pd.DataFrame = sat_dataframe.loc[sat_dataframe.loc[:, "timestamp"] == timestamp]
    if timestamp_dataframe.empty:
        return None

    projection1 = ccrs.PlateCarree()
    projection2 = ccrs.EquidistantConic(central_longitude=(lons[0] + lons[1]) / 2,
                                        central_latitude=(lats[0] + lats[1]) / 2)
    fig = plt.figure(layout="tight", dpi=300)
    ax = fig.add_subplot(1, 1, 1, projection=projection2)
    ax.set_extent([lons[0] - 1, lons[1] + 1, lats[0] - 1, lats[1] + 1], crs=projection1)
    shape_feature = cfeature.ShapelyFeature(creader(SHAPEFILE).geometries(), projection1, facecolor='none')

    ax.add_feature(shape_feature, edgecolor='gray')
    # ax.gridlines(draw_labels=True, dms=True, x_inline=True, y_inline=True)

    list_x = [lon for lon in sites_dataframe.loc[:, "gdlonr"]]
    list_y = [lat for lat in sites_dataframe.loc[:, "gdlatr"]]

    ax.plot(list_x, list_y, linestyle=" ", marker=".", markersize=1, color="black", transform=projection1)
    color_normalize = mplcolors.Normalize(-1.0, 1.0)
    colormap = plt.colormaps["Spectral"]
    fig.colorbar(mplcm.ScalarMappable(norm=color_normalize, cmap=colormap), ax=ax)
    ax.scatter(timestamp_dataframe.loc[:, "gdlon"], timestamp_dataframe.loc[:, "gdlat"],
               c=timestamp_dataframe.loc[:, "dvtec"], marker = ".", cmap=colormap, norm=color_normalize, linewidths=0,
               transform=projection1)
    # for i in timestamp_dataframe.index:
    #     ax.plot(timestamp_dataframe.loc[i, "gdlon"], timestamp_dataframe.loc[i, "gdlat"],
    #             color=colormap(timestamp_dataframe.loc[i, "dvtec"]), linestyle=" ", marker=".",
    #             markersize=1,
    #             transform=projection1)

    # ax.plot(array_x, array_y)
    # for i in range(len(list_x)):
    #     ax.plot(list_x[i], list_y[i], marker=".", markersize=2, color="black", transform=projection1)
    x_tics = list(range(int(lons[0]), int(lons[1]), lon_tics))
    y_tics = list(range(int(lats[0]), int(lats[1]), lat_tics))
    # ax.set_xticks(x_tics)
    # ax.set_yticks(y_tics)
    ax.gridlines(xlocs=x_tics, ylocs=y_tics, draw_labels=True)
    save_path = os.path.join(save_dir_2, filename)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    return filename


def main2024sep5():
    save_dir = r"/home/vadymskipa/PhD_student/data/123/dvtec_site_sat_observation_map"
    source_dir = r"/home/vadymskipa/PhD_student/data/123/dvtec_site_sat_observation/"
    windows = os.listdir(source_dir)
    for window in windows:
        source_dir_1 = os.path.join(source_dir, window)
        sats = os.listdir(source_dir_1)
        for sat in sats:
            source_dir_2 = os.path.join(source_dir_1, sat)
            sites = os.listdir(source_dir_2)
            for site in sites:
                source_dir_3 = os.path.join(source_dir_2, site)
                files = os.listdir(source_dir_3)
                for file in files:
                    source_path = os.path.join(source_dir_3, file)
                    dataframe = pd.read_csv(source_path)
                    save_dir_1 = w21.get_directory_path(save_dir, window)
                    save_dir_2 = w21.get_directory_path(save_dir_1, sat)
                    save_dir_3 = w21.get_directory_path(save_dir_2, site)
                    filename = f"{file[:-3]}png"
                    result = save_site_sat_observation_map(dataframe, save_dir_3, filename)
                    print(result)





def save_site_sat_observation_map(dataframe, save_dir, filename, lons = (15, 55), lats = (35, 65), lat_tics = 2,
    lon_tics = 3, site_coordinates=None, dtec_max=1.0, dtec_min=-1.0, min_elm=0):
    if min_elm:
        dataframe = dataframe.loc[dataframe.loc[:, "elm"] >= min_elm]
    projection1 = ccrs.PlateCarree()
    projection2 = ccrs.EquidistantConic(central_longitude=(lons[0] + lons[1]) / 2,
                                        central_latitude=(lats[0] + lats[1]) / 2)
    fig = plt.figure(layout="tight", dpi=300)
    ax = fig.add_subplot(1, 1, 1, projection=projection2)
    ax.set_extent([lons[0] - 1, lons[1] + 1, lats[0] - 1, lats[1] + 1], crs=projection1)
    shape_feature = cfeature.ShapelyFeature(creader(SHAPEFILE).geometries(), projection1, facecolor='none')

    ax.add_feature(shape_feature, edgecolor='gray')
    # ax.gridlines(draw_labels=True, dms=True, x_inline=True, y_inline=True)

    if site_coordinates:
        lat, lon = site_coordinates
        ax.plot(lon, lat, linestyle=" ", marker=".", markersize=1, color="black", transform=projection1)


    color_normalize = mplcolors.Normalize(dtec_min, dtec_max)
    colormap = plt.colormaps["Spectral"]
    fig.colorbar(mplcm.ScalarMappable(norm=color_normalize, cmap=colormap), ax=ax)
    ax.scatter(dataframe.loc[:, "gdlon"], dataframe.loc[:, "gdlat"],
               c=dataframe.loc[:, "dvtec"], marker=".", cmap=colormap, norm=color_normalize, linewidths=0,
               transform=projection1)

    x_tics = list(range(int(lons[0]), int(lons[1]), lon_tics))
    y_tics = list(range(int(lats[0]), int(lats[1]), lat_tics))
    ax.gridlines(xlocs=x_tics, ylocs=y_tics, draw_labels=True)
    save_path = os.path.join(save_dir, filename)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    return filename


def main2024sep5_mp():
    save_dir = r"/home/vadymskipa/PhD_student/data/123/dvtec_site_sat_observation_map"
    source_dir = r"/home/vadymskipa/PhD_student/data/123/dvtec_site_sat_observation/"
    windows = os.listdir(source_dir)
    for window in windows:
        source_dir_1 = os.path.join(source_dir, window)
        sats = os.listdir(source_dir_1)
        for sat in sats:
            source_dir_2 = os.path.join(source_dir_1, sat)
            sites = os.listdir(source_dir_2)
            for site in sites:
                source_dir_3 = os.path.join(source_dir_2, site)
                files = os.listdir(source_dir_3)

                input_list = []
                for file in files:
                    source_path = os.path.join(source_dir_3, file)
                    dataframe = pd.read_csv(source_path)
                    save_dir_1 = w21.get_directory_path(save_dir, window)
                    save_dir_2 = w21.get_directory_path(save_dir_1, sat)
                    save_dir_3 = w21.get_directory_path(save_dir_2, site)
                    filename = f"{file[:-3]}png"
                    input_list.append((dataframe, save_dir_3, filename))

                pool = multipool.Pool(7)
                for filename in pool.imap(save_site_sat_observation_map_process, input_list):
                    print(filename, dt.datetime.now())


def save_site_sat_observation_map_process(data_tuple):
    res = save_site_sat_observation_map(*data_tuple, min_elm=30)
    return res


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
        date = w21.get_date_from_date_directory_name(datedir)
        source_path_1 = os.path.join(source_path, datedir, window)
        save_path_1 = w21.get_directory_path(save_path, datedir)
        save_path_2 = w21.get_directory_path(save_path_1, window)
        save_path_3 = w21.get_directory_path(save_path_2, f"Amplitude_{ampl}_tecu")
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
                                            hour=int(time_file[:2]), minute=int(time_file[2:4]))
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
        for timestamp in list_timestamp:
            timestamp_dataframe = date_dataframe.loc[date_dataframe.loc[:, "timestamp"] == timestamp]
            timestamp_datetime = dt.datetime.fromtimestamp(timestamp, tz=dt.timezone.utc)
            filename = f"{timestamp_datetime.hour:0=2}{timestamp_datetime.minute:0=2}.png"
            save_blocks_map(timestamp_dataframe, save_path_3, filename, dtec_max=ampl, dtec_min=-ampl,
                            blocks_lat_tick=0.15, blocks_lon_tick=0.15, lons=(-10, 20), lats=(36, 70))
            print(f"{date.year}-{date.month}-{date.day}", window, filename, dt.datetime.now())



# def save_blocks_map(dataframe, save_dir, filename, lons = (15, 55), lats = (35, 65), lat_tics = 2,
#     lon_tics = 3, dtec_max=1.0, dtec_min=-1.0, blocks_lat_tick=1.0, blocks_lon_tick=1.0, nos_min=0):
#     projection1 = ccrs.PlateCarree()
#     projection2 = ccrs.EquidistantConic(central_longitude=(lons[0] + lons[1]) / 2,
#                                         central_latitude=(lats[0] + lats[1]) / 2)
#     fig = plt.figure(layout="tight", dpi=300)
#     ax = fig.add_subplot(1, 1, 1, projection=projection2)
#     ax.set_extent([lons[0] - 1, lons[1] + 1, lats[0] - 1, lats[1] + 1], crs=projection1)
#     shape_feature = cfeature.ShapelyFeature(creader(SHAPEFILE).geometries(), projection1, facecolor='none')
#
#     ax.add_feature(shape_feature, edgecolor='gray')
#
#
#
#     color_normalize = mplcolors.Normalize(dtec_min, dtec_max)
#     colormap = plt.colormaps["Spectral"]
#     fig.colorbar(mplcm.ScalarMappable(norm=color_normalize, cmap=colormap), ax=ax)
#
#     start = dt.datetime.now()
#     # dataframe.loc[:, "dvtec_sum"] = dataframe.loc[:, "dvtec"] * dataframe.loc[:, "nos"]
#     # dataframe = dataframe.assign(dvtec_sum=lambda x: x["dvtec"] * x["nos"])
#     # coord_dataframe = dataframe.groupby(["gdlat", "gdlon"]).size().reset_index()
#     # list_data_lat = np.unique(dataframe.loc[:, "gdlat"])
#     for data_lat in list_data_lat:
#         lat_dataframe = dataframe.loc[dataframe.loc[:, "gdlat"] == data_lat]
#         list_data_lon = np.unique(lat_dataframe.loc[:, "gdlon"])
#         for data_lon in list_data_lon:
#     # for index in coord_dataframe.index:
#         data_lon = coord_dataframe.loc[index, "gdlon"]
#         data_lat = coord_dataframe.loc[index, "gdlat"]
#         list_x = [data_lon, data_lon, data_lon + blocks_lon_tick, data_lon + blocks_lon_tick, data_lon]
#         list_y = [data_lat, data_lat + blocks_lat_tick, data_lat + blocks_lat_tick, data_lat, data_lat]
#         mask = np.logical_and(dataframe.loc[:, "gdlat"] == data_lat, dataframe.loc[:, "gdlon"] == data_lon)
#         temp_dataframe = dataframe.loc[mask]
#         nos_sum = temp_dataframe.loc[:, "nos"].sum()
#         dvtec = temp_dataframe.loc[:, "dvtec_sum"].sum() / nos_sum
#         color = colormap(dvtec)
#         if nos_sum >= nos_min:
#             ax.fill(list_x, list_y, color, transform=projection1)
#             # list_x = [data_lon, data_lon, data_lon + blocks_lon_tick, data_lon + blocks_lon_tick, data_lon]
#             # list_y = [data_lat, data_lat + blocks_lat_tick, data_lat + blocks_lat_tick, data_lat, data_lat]
#             # lon_dataframe = lat_dataframe.loc[lat_dataframe.loc[:, "gdlon"] == data_lon]
#             # dvtec_sum = 0
#             # nos_sum = 0
#             # for index in lon_dataframe.index:
#             #     nos = lon_dataframe.loc[index, "nos"]
#             #     dvtec = lon_dataframe.loc[index, "dvtec"]
#             #     nos_sum += nos
#             #     dvtec_sum += dvtec * nos
#             # dvtec = dvtec_sum / nos_sum
#             # nos_sum = lon_dataframe.loc[:, "nos"].sum()
#             # dvtec = lon_dataframe.loc[:, "dvtec_sum"].sum() / nos_sum
#             # color = colormap(dvtec)
#             # if nos_sum >= nos_min:
#             #     ax.fill(list_x, list_y, color, transform=projection1)
#     print(f"calculating continued for {dt.datetime.now() - start}")
#
#     x_tics = list(range(int(lons[0]), int(lons[1]), lon_tics))
#     y_tics = list(range(int(lats[0]), int(lats[1]), lat_tics))
#     ax.gridlines(xlocs=x_tics, ylocs=y_tics, draw_labels=True)
#     save_path = os.path.join(save_dir, filename)
#     fig.savefig(save_path, dpi=300)
#     plt.close(fig)
#     return filename


def main2024oct2():
    source_path = r"/home/vadymskipa/PhD_student/data/Europe_36-70__-10-20/dvtec_csv_5min/blocks/all_sats_smooth"
    save_path = r"/home/vadymskipa/PhD_student/data/Europe_36-70__-10-20/dvtec_csv_5min/blocks_maps/blocks_maps_smooth"
    window = "Window_3600_Seconds"
    lats = (36, 70)
    lons = (-10, 20)
    block_lat_tick = 0.15
    block_lon_tick = 0.15
    ampl=4
    list_lats = [round((lats[0] + block_lat_tick * i), 2) for i in range(math.ceil((lats[1] - lats[0]) / block_lat_tick))]
    list_lons = [round((lons[0] + block_lon_tick * i), 2) for i in range(math.ceil((lons[1] - lons[0]) / block_lon_tick))]
    stock_blocks_dataframe = pd.DataFrame({"lats": list_lats}, columns=["lats", *list_lons])
    list_block_boundaries_dataframe = create_blocks_boundaries_dataframe(stock_blocks_dataframe,
                                                                         blocks_lat_tick=block_lat_tick,
                                                                         blocks_lon_tick=block_lat_tick)

    entries_1 = os.listdir(source_path)
    list_datedir = [datedir for datedir in entries_1 if len(datedir) == 14]
    for datedir in list_datedir:
        date = w21.get_date_from_date_directory_name(datedir)
        if date <= dt.datetime(year=2024, month=5, day=9, tzinfo=dt.timezone.utc):
            continue
        if date >= dt.datetime(year=2024, month=5, day=12, tzinfo=dt.timezone.utc):
            continue
        source_path_1 = os.path.join(source_path, datedir, window)
        save_path_1 = w21.get_directory_path(save_path, datedir)
        save_path_2 = w21.get_directory_path(save_path_1, window)
        save_path_3 = w21.get_directory_path(save_path_2, f"Amplitude_{ampl}_tecu")
        list_timefile = os.listdir(source_path_1)
        dict_time_dataframe = {}
        for timefile in list_timefile:
            source_path_2 = os.path.join(source_path_1, timefile)
            temp_dataframe: pd.DataFrame = pd.read_csv(source_path_2)
            dict_time_dataframe[timefile] = temp_dataframe
        # input_list = [(dataframe, save_path_3, f"{filename[:4]}.png", lons, lats, ampl, list_block_boundaries_dataframe)
        #               for filename, dataframe in dict_time_dataframe.items()]
        # pool = multipool.Pool(1)
        # for filename in pool.imap(save_blocks_table_map_proccess_2, input_list):
        #     print(f"{filename} done! {dt.datetime.now()}")
        input_list = [(dataframe, save_path_3, f"{filename[:4]}.png", lons, lats, ampl, block_lat_tick, block_lon_tick,)
                      for filename, dataframe in dict_time_dataframe.items()]
        pool = multipool.Pool(8)
        for filename in pool.imap(save_blocks_table_map_proccess, input_list):
            print(f"{date} --- {filename} done! --- {dt.datetime.now()}")






def save_blocks_table_map(dataframe, save_dir, filename, block_lat_tick, block_lon_tick,
                          lons = (15, 55), lats = (35, 65), lat_tics = 2, lon_tics = 3,
                          amplitude = 1.0):
    projection1 = ccrs.PlateCarree()
    projection2 = ccrs.EquidistantConic(central_longitude=(lons[0] + lons[1]) / 2,
                                        central_latitude=(lats[0] + lats[1]) / 2)
    fig = plt.figure(layout="tight", dpi=300)
    ax = fig.add_subplot(1, 1, 1, projection=projection2)
    ax.set_extent([lons[0] - 1, lons[1] + 1, lats[0] - 1, lats[1] + 1], crs=projection1)
    shape_feature = cfeature.ShapelyFeature(creader(SHAPEFILE).geometries(), projection1, facecolor='none')

    ax.add_feature(shape_feature, edgecolor='gray')



    color_normalize = mplcolors.Normalize(-amplitude, amplitude)
    colormap = plt.colormaps["Spectral"]
    fig.colorbar(mplcm.ScalarMappable(norm=color_normalize, cmap=colormap), ax=ax)


    dataframe.drop([dataframe.columns[0]], axis=1, inplace=True)
    i = 0
    for index in dataframe.index:
        for i_column in range(1, len(dataframe.columns), 1):
            data_lon = float(dataframe.columns[i_column])
            data_lat = dataframe.loc[index, "lats"]
            list_x = [data_lon, data_lon, data_lon + block_lon_tick, data_lon + block_lon_tick, data_lon]
            list_y = [data_lat, data_lat + block_lat_tick, data_lat + block_lat_tick, data_lat, data_lat]
            dvtec = dataframe.iloc[index, i_column]
            color = colormap(color_normalize(dvtec))
            if pd.isna(dvtec):
                continue
            ax.fill(list_x, list_y, color=color, transform=projection1)

    x_tics = list(range(int(lons[0]), int(lons[1]), lon_tics))
    y_tics = list(range(int(lats[0]), int(lats[1]), lat_tics))
    ax.gridlines(xlocs=x_tics, ylocs=y_tics, draw_labels=True)
    save_path = os.path.join(save_dir, filename)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    return filename


def save_blocks_table_map_2(dataframe, save_dir, filename, list_block_boundaries_dataframe,
                          lons = (15, 55), lats = (35, 65), lat_tics = 2, lon_tics = 3,
                          amplitude = 1.0):
    projection1 = ccrs.PlateCarree()
    projection2 = ccrs.EquidistantConic(central_longitude=(lons[0] + lons[1]) / 2,
                                        central_latitude=(lats[0] + lats[1]) / 2)
    fig = plt.figure(layout="tight", dpi=300)
    ax = fig.add_subplot(1, 1, 1, projection=projection2)
    ax.set_extent([lons[0] - 1, lons[1] + 1, lats[0] - 1, lats[1] + 1], crs=projection1)
    shape_feature = cfeature.ShapelyFeature(creader(SHAPEFILE).geometries(), projection1, facecolor='none')

    ax.add_feature(shape_feature, edgecolor='gray')



    color_normalize = mplcolors.Normalize(-amplitude, amplitude)
    colormap = plt.colormaps["Spectral"]
    fig.colorbar(mplcm.ScalarMappable(norm=color_normalize, cmap=colormap), ax=ax)


    dataframe.drop([dataframe.columns[0]], axis=1, inplace=True)
    i = 0
    for index in dataframe.index:
        for i_column in range(1, len(dataframe.columns), 1):
            left = list_block_boundaries_dataframe[2].iloc[index, i_column]
            right = list_block_boundaries_dataframe[3].iloc[index, i_column]
            top = list_block_boundaries_dataframe[0].iloc[index, i_column]
            bottom = list_block_boundaries_dataframe[1].iloc[index, i_column]
            list_x = [left, left, right, right, left]
            list_y = [bottom, top, top, bottom, bottom]
            dvtec = dataframe.iloc[index, i_column]
            color = colormap(color_normalize(dvtec))
            if pd.isna(dvtec):
                continue
            ax.fill(list_x, list_y, color=color, transform=projection1)

    x_tics = list(range(int(lons[0]), int(lons[1]), lon_tics))
    y_tics = list(range(int(lats[0]), int(lats[1]), lat_tics))
    ax.gridlines(xlocs=x_tics, ylocs=y_tics, draw_labels=True)
    save_path = os.path.join(save_dir, filename)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    return filename


def save_blocks_table_map_proccess(tup):
    dataframe, save_dir, filename, lons, lats, amplitude, block_lat_tick, block_lon_tick, = tup
    return save_blocks_table_map(dataframe=dataframe, save_dir=save_dir, filename=filename, lons=lons, lats=lats,
                                 amplitude=amplitude, block_lat_tick=block_lat_tick, block_lon_tick=block_lon_tick)


def save_blocks_table_map_proccess_2(tup):
    dataframe, save_dir, filename, lons, lats, amplitude, list_block_boundaries_dataframe = tup
    return save_blocks_table_map_2(dataframe=dataframe, save_dir=save_dir, filename=filename, lons=lons, lats=lats,
                                 amplitude=amplitude, list_block_boundaries_dataframe=list_block_boundaries_dataframe)


def create_blocks_boundaries_dataframe(stock_blocks_dataframe, blocks_lat_tick, blocks_lon_tick):
    left_dataframe: pd.DataFrame = stock_blocks_dataframe.copy()
    right_dataframe: pd.DataFrame = stock_blocks_dataframe.copy()
    top_dataframe: pd.DataFrame = stock_blocks_dataframe.copy()
    bottom_dataframe: pd.DataFrame = stock_blocks_dataframe.copy()

    for i_index in range(len(stock_blocks_dataframe.index)):
        lat = left_dataframe.iloc[i_index].loc["lats"]
        for i_column in range(1, len(stock_blocks_dataframe.columns), 1):
            top_dataframe.iloc[i_index, i_column] = lat + blocks_lat_tick
            bottom_dataframe.iloc[i_index, i_column] = lat
            left_dataframe.iloc[i_index, i_column] = stock_blocks_dataframe.columns[i_column]
            right_dataframe.iloc[i_index, i_column] = stock_blocks_dataframe.columns[i_column] + blocks_lon_tick

    return top_dataframe, bottom_dataframe, left_dataframe, right_dataframe


def main2024oct3():
    source_directories = [r"/home/vadymskipa/PhD_student/data/Japan_20-50__130-160/dtec_txt/055-2023-02-24",
                          r"/home/vadymskipa/PhD_student/data/Japan_20-50__130-160/dtec_txt/056-2023-02-25",
                          r"/home/vadymskipa/PhD_student/data/Japan_20-50__130-160/dtec_txt/057-2023-02-26",
                          r"/home/vadymskipa/PhD_student/data/Japan_20-50__130-160/dtec_txt/058-2023-02-27",
                          r"/home/vadymskipa/PhD_student/data/Japan_20-50__130-160/dtec_txt/059-2023-02-28",
                          r"/home/vadymskipa/PhD_student/data/Japan_20-50__130-160/dtec_txt/060-2023-03-01",
                          r"/home/vadymskipa/PhD_student/data/Japan_20-50__130-160/dtec_txt/061-2023-03-02",
                          r"/home/vadymskipa/PhD_student/data/Japan_20-50__130-160/dtec_txt/062-2023-03-03"]
    site_directory = r"/home/vadymskipa/PhD_student/data/site_hdf/"
    for source_directory in source_directories:
        sites_txt_path = os.path.join(source_directory, "Sites.txt")
        save_path = os.path.join(source_directory, "Sites.png")
        print_sites_from_sites_txt_new(sites_txt_path, save_path)



def main2024oct():
    # main2024oct1()
    # main2024oct2()
    main2024oct3()


def main2024nov1():
    source_directories = [r"/home/vadymskipa/PhD_student/data/Europe_30-80__-10-50/dtec_txt/089-2023-03-30",
                          r"/home/vadymskipa/PhD_student/data/Europe_30-80__-10-50/dtec_txt/090-2023-03-31",
                          r"/home/vadymskipa/PhD_student/data/Europe_30-80__-10-50/dtec_txt/091-2023-04-01",
                          r"/home/vadymskipa/PhD_student/data/Europe_30-80__-10-50/dtec_txt/092-2023-04-02",
                          r"/home/vadymskipa/PhD_student/data/Europe_30-80__-10-50/dtec_txt/093-2023-04-03",
                          r"/home/vadymskipa/PhD_student/data/Europe_30-80__-10-50/dtec_txt/094-2023-04-04",
                          r"/home/vadymskipa/PhD_student/data/Europe_30-80__-10-50/dtec_txt/095-2023-04-05"]
    for source_directory in source_directories:
        sites_txt_path = os.path.join(source_directory, "Sites.txt")
        save_path = os.path.join(source_directory, "Sites.png")
        print_sites_from_sites_txt_new(sites_txt_path, save_path)


def main2024nov2(save_dir = r"/home/vadymskipa/PhD_student/data/123/longitude_direction/png_1",
    source_dir = r"/home/vadymskipa/PhD_student/data/123/longitude_direction/hdf"):


    list_site = os.listdir(source_dir)
    for site in list_site:
        source_path_1 = os.path.join(source_dir, site)
        save_path_1 = w21.get_directory_path(save_dir, site)
        list_windows = os.listdir(source_path_1)
        for window in list_windows:
            source_path_2 = os.path.join(source_path_1, window)
            save_path_2 = w21.get_directory_path(save_path_1, window)
            list_datefiles = os.listdir(source_path_2)
            input_list = []
            for datefile in list_datefiles:
                source_path_3 = os.path.join(source_path_2, datefile)
                dataframe: pd.DataFrame = pd.read_hdf(source_path_3, key="df")
                # plot_main2024nov2(dataframe, save_path_2, datefile)
                # print(site, window, datefile[:-4], dt.datetime.now())
                input_list.append((dataframe, save_path_2, datefile[: -5]))

            pool = multipool.Pool(8)
            for filename in pool.imap(main2024nov2_procces, input_list):
                print(site, window, filename, dt.datetime.now())
            pool.close()


def main2024nov2_procces(tup):
    return plot_main2024nov2(*tup)


def plot_main2024nov2(dataframe, save_dir, filename, lons = (15, 55), lats = (35, 65), lat_ticks = 2, lon_ticks=3):
    projection1 = ccrs.PlateCarree()
    projection2 = ccrs.EquidistantConic(central_longitude=(lons[0] + lons[1]) / 2,
                                        central_latitude=(lats[0] + lats[1]) / 2)
    fig = plt.figure(layout="tight", dpi=300)
    ax = fig.add_subplot(1, 1, 1, projection=projection2)
    ax.set_extent([lons[0] - 1, lons[1] + 1, lats[0] - 1, lats[1] + 1], crs=projection1)
    shape_feature = cfeature.ShapelyFeature(creader(SHAPEFILE).geometries(), projection1, facecolor='none')

    ax.add_feature(shape_feature, edgecolor='gray')
    # ax.gridlines(draw_labels=True, dms=True, x_inline=True, y_inline=True)


    lat = dataframe.loc[:, "gdlat"]
    lon = dataframe.loc[:, "gdlon"]

    ax.plot(lon, lat, linestyle=" ", marker=".", markersize=1, color="black", transform=projection1)


    x_tics = list(range(int(lons[0]), int(lons[1]), lon_ticks))
    y_tics = list(range(int(lats[0]), int(lats[1]), lat_ticks))
    ax.gridlines(xlocs=x_tics, ylocs=y_tics, draw_labels=True)
    save_path = os.path.join(save_dir, filename + ".png")
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    return filename


def main2024nov3():
    main2024nov2(save_dir=r"/home/vadymskipa/PhD_student/data/123/longitude_direction/png_1",
                 source_dir=r"/home/vadymskipa/PhD_student/data/123/longitude_direction/hdf")
    main2024nov2(save_dir=r"/home/vadymskipa/PhD_student/data/123/latitude_direction/png_1",
                 source_dir=r"/home/vadymskipa/PhD_student/data/123/latitude_direction/hdf")


def main2024nov4():
    main2024nov2(save_dir=r"/home/vadymskipa/PhD_student/data/123/longitude_direction_2/png_1",
                 source_dir=r"/home/vadymskipa/PhD_student/data/123/longitude_direction_2/hdf")
    main2024nov2(save_dir=r"/home/vadymskipa/PhD_student/data/123/latitude_direction_2/png_1",
                 source_dir=r"/home/vadymskipa/PhD_student/data/123/latitude_direction_2/hdf")


def main2024nov5():
    main2024nov2(save_dir=r"/home/vadymskipa/PhD_student/data/123/avg_dvtec/png_1",
                 source_dir=r"/home/vadymskipa/PhD_student/data/123/avg_dvtec/hdf")




def main2024nov():
    # main2024nov1()
    # main2024nov2()
    # main2024nov3()
    # main2024nov4()
    main2024nov5(

    )

def main2025feb1():
    source_directories = [r"/home/vadymskipa/PhD_student/data/Europe_30-80__-10-50/dtec_txt/088-2023-03-29",
                          r"/home/vadymskipa/PhD_student/data/Europe_30-80__-10-50/dtec_txt/096-2023-04-06",
                          r"/home/vadymskipa/PhD_student/data/Europe_30-80__-10-50/dtec_txt/097-2023-04-07",
                          r"/home/vadymskipa/PhD_student/data/Europe_30-80__-10-50/dtec_txt/098-2023-04-08",
                          r"/home/vadymskipa/PhD_student/data/Europe_30-80__-10-50/dtec_txt/099-2023-04-09"]
    for source_directory in source_directories:
        sites_txt_path = os.path.join(source_directory, "Sites.txt")
        save_path = os.path.join(source_directory, "Sites.png")
        print_sites_from_sites_txt_new(sites_txt_path, save_path)


def main2025feb2():
    source_directory = r"/home/vadymskipa/PhD_student/data/Europe_30-80__-10-50/dtec_txt/"
    entries = os.listdir(source_directory)
    start_date = dt.datetime(year=2022, month=10, day=12, tzinfo=dt.timezone.utc)
    end_date = dt.datetime(year=2022, month=10, day=25, tzinfo=dt.timezone.utc)
    source_directories = [os.path.join(source_directory, entry) for entry in entries
                          if os.path.isdir(os.path.join(source_directory, entry)) and
                          w21.get_date_from_date_directory_name(entry) <= end_date and
                          w21.get_date_from_date_directory_name(entry) >= start_date]
    for source_directory in source_directories:
        sites_txt_path = os.path.join(source_directory, "Sites.txt")
        save_path = os.path.join(source_directory, "Sites.png")
        print_sites_from_sites_txt_new(sites_txt_path, save_path)


def main2025feb3():
    source_directory = r"/home/vadymskipa/PhD_student/data/Europe_30-80__-10-50/dtec_txt/"
    entries = os.listdir(source_directory)
    start_date = dt.datetime(year=2023, month=3, day=28, tzinfo=dt.timezone.utc)
    end_date = dt.datetime(year=2023, month=4, day=10, tzinfo=dt.timezone.utc)
    source_directories = [os.path.join(source_directory, entry) for entry in entries
                          if os.path.isdir(os.path.join(source_directory, entry)) and
                          w21.get_date_from_date_directory_name(entry) <= end_date and
                          w21.get_date_from_date_directory_name(entry) >= start_date]
    for source_directory in source_directories:
        sites_txt_path = os.path.join(source_directory, "Sites.txt")
        save_path = os.path.join(source_directory, "Sites.png")
        print_sites_from_sites_txt_new(sites_txt_path, save_path)


if __name__ == "__main__":
    main2025feb3()
    # main2025feb2()
    # main2025feb1()
    # main2024nov()
    # main2024oct()
    # main2024sep5_mp()
    # main2024sep5()
    # main2024sep4()
    # main2024sep3()
    # main2024sep2()
    # main2024sep1()
    # main2024jul()
    # main2024jun2()
    # main2024jun1()
    # main2024apr1()
    # main2024feb3()
    # main2023jan1()
    # main2023jan2()
    # main2023jan3()
    # main2023jan4()
    # main2024feb1()
    # main2024feb2()