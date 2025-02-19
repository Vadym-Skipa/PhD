import h5py
import work_site_file as wsite
import numpy as np
import pandas as pd
import datetime as dt
import os
import re
import multiprocessing as mp
import multiprocessing.pool as multipool


STEP_SITES_CONSTANT = 100
NUMBER_OF_CORES = mp.cpu_count()


def get_data_for_one_site_GPS(path: str, site: str):
    """

    @param path:
    @param site:
    @return:
    """
    with h5py.File(path, "r") as file:
        arr_of_sites = file["Data"]["Table Layout"]["gps_site"]
        np_site = site.encode("ascii")
        indices_for_site = arr_of_sites == np_site
        data_for_site = file["Data"]["Table Layout"][indices_for_site]
    GPS = "GPS     ".encode("ascii")
    arr_of_GNSS = data_for_site["gnss_type"]
    indices_of_GPS = arr_of_GNSS == GPS
    data_outcome = data_for_site[indices_of_GPS]
    return data_outcome


def get_data_for_one_site_GPS_pd(path: str, site: str):
    data_outcome = pd.DataFrame(get_data_for_one_site_GPS(path, site))
    return data_outcome


def get_data_by_indices(path: str, indices):
    with h5py.File(path, "r") as file:
        data = file["Data"]["Table Layout"][indices]
    return data


def get_data_by_indices_pd(path: str, indices):
    data_outcome = pd.DataFrame(get_data_by_indices(path, indices))
    return data_outcome


def get_data_by_indices_GPS(path: str, indices):
    data = get_data_by_indices(path, indices)
    GPS = "GPS     ".encode("ascii")
    arr_of_GNSS = data["gnss_type"]
    indices_of_GPS = arr_of_GNSS == GPS
    data_outcome = data[indices_of_GPS]
    return data_outcome


def get_data_by_indices_GPS_pd(path: str, indices):
    data_outcome = pd.DataFrame(get_data_by_indices_GPS(path, indices))
    return data_outcome


def get_indices_for_site(path, site: str):
    file = h5py.File(path, "r")
    arr_of_sites = file["Data"]["Table Layout"]["gps_site"]
    file.close()
    np_site = site.encode("ascii")
    indices_for_site = arr_of_sites == np_site
    return indices_for_site


def get_indices_for_sites(path, sites):
    file = h5py.File(path, "r")
    arr_of_sites = file["Data"]["Table Layout"]["gps_site"]
    file.close()
    indices = np.full(len(arr_of_sites), False)
    for site in sites:
        temp_indices = arr_of_sites == site.encode("ascii")
        indices = np.logical_or(indices, temp_indices)
    return indices


def get_site_array(path):
    with h5py.File(path, "r") as file:
        arr_of_sites = file["Data"]["Table Layout"]["gps_site"]
    return arr_of_sites


def get_indices_for_sites_from_site_array(site_array, sites, multiprocessing=False):
    if multiprocessing:
        return get_indices_for_sites_from_site_array_multiprocessing(site_array=site_array, sites=sites)
    indices = np.full(len(site_array), False)
    for site in sites:
        temp_indices = site_array == site.encode("ascii")
        indices = np.logical_or(indices, temp_indices)
    return indices


def get_indices_for_sites_from_site_array_task(temp_tuple):
    site_array, sites = temp_tuple
    indices = get_indices_for_sites_from_site_array(site_array, sites)
    return indices



def get_indices_for_sites_from_site_array_multiprocessing(site_array: np.ndarray, sites,
                                                          number_of_processes=NUMBER_OF_CORES - 1):
    # size_of_array = site_array.size * site_array.itemsize
    # sites_encode = [el.encode("ascii") for el in sites]
    # wanted_sites_array = np.array(sites_encode)
    # size_of_wanted_sites_array = wanted_sites_array.size * wanted_sites_array.itemsize
    # sahred_memory_wanted_sites = shmem.SharedMemory(name="SharedWantedSitesArray", create=True,
    #                                                 size=size_of_wanted_sites_array)
    # shared_wanted_sites_array = np.ndarray(wanted_sites_array.shape, wanted_sites_array.dtype,
    #                                        buffer=sahred_memory_wanted_sites.buf)
    # shared_wanted_sites_array[:] = wanted_sites_array[:]
    # shared_memory_site_array = shmem.SharedMemory(name="SharedSiteArray", create=True, size=size_of_array)
    # shared_site_array = np.ndarray(site_array.shape, site_array.dtype, buffer=shared_memory_site_array.buf)
    # shared_site_array[:] = site_array
    if number_of_processes == 0:
        number_of_processes = 1
    pool = multipool.Pool(number_of_processes)
    len_of_piece = 1000000
    if len(site_array) > len_of_piece:
        list_of_indexes_mp = [(el, el + len_of_piece) for el in range(0, len(site_array) - len_of_piece, len_of_piece)]
        list_of_indexes_mp.append((list_of_indexes_mp[-1][1], len(site_array)))
    else:
        list_of_indexes_mp = [(0, len(site_array))]
    list_of_input = [(site_array[start:end], sites) for (start, end) in list_of_indexes_mp]
    temp_list = []
    result_array = np.array(temp_list, dtype="bool")
    for small_array in pool.imap(get_indices_for_sites_from_site_array_task, list_of_input):
        result_array = np.concatenate((result_array, small_array))
    pool.close()
    return result_array





def _copy_metadata_group_from_source_to_save(source, save):
    file_read = h5py.File(source, "r")
    file_save = h5py.File(save, "a")
    if "Metadata" in file_read:
        file_read["Metadata"].copy(file_read["Metadata"], file_save)
    file_read.close()
    file_save.close()


def _create_dst_table_layout_grp_data_similar_to_source(source, save):
    file_read = h5py.File(source, "r")
    file_save = h5py.File(save, "a")
    if "Data" in file_read and "Table Layout" in file_read["Data"]:
        dtype_dst = file_read["Data"]["Table Layout"].dtype
        chunks_dst = file_read["Data"]["Table Layout"].chunks
        max_shape_dst = file_read["Data"]["Table Layout"].shape
        compression = file_read["Data"]["Table Layout"].compression
        compression_opts = file_read["Data"]["Table Layout"].compression_opts
        group_data = file_save.create_group("Data")
        dataset_table_layout = group_data.create_dataset("Table Layout", maxshape=max_shape_dst, chunks=chunks_dst,
                                                         dtype=dtype_dst, shape=(0,), compression=compression,
                                                         compression_opts=compression_opts)
    file_read.close()
    file_save.close()


def add_data_to_dst_table_layout(data, file_path):
    file = h5py.File(file_path, "a")
    dst_table_layout: h5py.Dataset = file["Data"]["Table Layout"]
    len_dst = dst_table_layout.len()
    len_data = len(data)
    dst_table_layout.resize(len_dst + len_data, axis=0)
    dst_table_layout[len_dst:len_dst+len_data] = data
    file.close()


def save_region_los_file(los_path, site_path, save_path, lons, lats, step_of_sites=STEP_SITES_CONSTANT,
                         multiprocessing=False):
    sites_dataframe = wsite.get_sites_dataframe_by_coordinates(site_path, lons=lons, lats=lats)
    sites = [np_site.decode("ascii") for np_site in sites_dataframe.loc[:, "gps_site"]]
    site_array = get_site_array(los_path)
    _copy_metadata_group_from_source_to_save(los_path, save_path)
    _create_dst_table_layout_grp_data_similar_to_source(los_path, save_path)
    for start_index in range(0, len(sites), step_of_sites):
        end_index = start_index + step_of_sites
        if end_index > len(sites):
            end_index = len(sites)
        temp_indices = get_indices_for_sites_from_site_array(site_array, sites[start_index:end_index],
                                                             multiprocessing=multiprocessing)
        temp_data = get_data_by_indices(los_path, temp_indices)
        add_data_to_dst_table_layout(temp_data, save_path)


def check_los_file_name_by_date(file_name, date: dt.date):
    date_str = f"{date.year:0=4}{date.month:0=2}{date.day:0=2}"
    pattern = f"^los_{date_str}\S*(\.h5|\.hdf5)$"
    if re.search(pattern, file_name):
        return True
    return False


def get_date_by_los_file_name(file_name):
    year = int(file_name[4:8])
    month = int(file_name[8:10])
    day = int(file_name[10:12])
    date = dt.datetime(year=year, month=month, day=day, tzinfo=dt.timezone.utc)
    return date


def check_los_file_name(file_name):
    if re.search("^los_(\d{8})\S*(\.h5|\.hdf5)$", file_name):
        return True
    return False


def get_region_los_file_name(source_los_file_name, region_name):
    region_los_file_name = f"{source_los_file_name[0:12]}.{region_name}{source_los_file_name[12:]}"
    return region_los_file_name


def create_site_file_from_los(los_path, save_directory_path, step=40):
    start =dt.datetime.now()
    site_array = get_site_array(los_path)
    # site_array_unicode =
    unique_site_array = np.unique(site_array)
    gdlatr_list = []
    gdlonr_list = []
    for i in range(0, len(unique_site_array), step):
        print(unique_site_array[i], dt.datetime.now() - start)
        sites = [site.decode("ascii") for site in unique_site_array[i:i + step]]
        indices = get_indices_for_sites_from_site_array(site_array, sites, multiprocessing=True)
        sites_data: pd.DataFrame = get_data_by_indices_pd(los_path, indices)
        for site in unique_site_array[i:i + step]:
            site_dataframe = sites_data.loc[sites_data.loc[:, "gps_site"] == site]
            gdlatr = site_dataframe.iloc[0].loc["gdlatr"]
            gdlonr = site_dataframe.iloc[0].loc["gdlonr"]
            gdlatr_list.append(gdlatr)
            gdlonr_list.append(gdlonr)
    site_filename = f"site_{os.path.basename(los_path)[4:12]}.hdf5"
    save_path = os.path.join(save_directory_path, site_filename)
    print("___save")
    with h5py.File(save_path, "x") as file:
        group_data = file.create_group("Data")
        dst_dtype = np.dtype([("gps_site", "S4"), ('gdlatr', '<f8'), ('gdlonr', '<f8')])
        dataset_table_layout = group_data.create_dataset("Table Layout", shape=(len(unique_site_array),), dtype=dst_dtype)
        dataset_table_layout["gps_site"] = unique_site_array
        dataset_table_layout["gdlatr"] = gdlatr_list
        dataset_table_layout["gdlonr"] = gdlonr_list


