import h5py
import numpy as np
import work_los_file as wlos
import read_los_file as rlos
import os
import datetime as dt
import work_site_file as wsite
import re


test_file = r"C:\Users\lisov\Downloads\gps200204g.002.hdf5"

check_file = r"/home/vadymskipa/Downloads/site_20200314.001.h5"
check_file2 = r"/home/vadymskipa/Downloads/site_20211010.001.h5"


def test1():
    with h5py.File(test_file, "r") as file:
        print(file.keys())

def test2():
    with h5py.File(test_file, "r") as file:
        print(file["Data"])
        print(file["Data"].keys())

def test3():
    with h5py.File(test_file, "r") as file:
        print(file["Data"])
        print(file["Data"].keys())
        print(file["Data"]["Table Layout"])

def test4():
    with h5py.File(test_file, "r") as file:
        test_dst = file["Data"]["Table Layout"]
        print(test_dst[0]["tec"])

def check_site_in_site_file(check_site, check_site_file):
    file = h5py.File(check_site_file, "r")
    site_arr = file["Data"]["Table Layout"]["gps_site"]
    sites = np.unique(site_arr)
    file.close()
    res = False
    np_check_site = np.array([check_site], dtype="S4")[0]
    sites_list = list(sites)
    if np_check_site in sites_list:
        res = True
    return res

def print_all_sites(check_site_file):
    file = h5py.File(check_site_file, "r")
    site_arr = file["Data"]["Table Layout"]["gps_site"]
    sites = np.unique(site_arr)
    file.close()
    sites_list = list(sites)
    print(sites_list)
    print(sites.dtype)


def print_groups_of_hdf5file(path):
    file = h5py.File(path, "r")
    for group in file:
        print(group)
    file.close()


def print_datasets_of_group_data_of_hdf5file1(path):
    file = h5py.File(path, "r")
    grp_data: h5py.Group = file["Data"]
    print(grp_data.keys())
    file.close()

def copy_metadata_group_from_read_hdf5_to_save_hdf5(path_read, path_save):
    file_read = h5py.File(path_read, "r")
    file_save = h5py.File(path_save, "a")
    if "Metadata" in file_read:
        file_read["Metadata"].copy(file_read["Metadata"], file_save)
    file_read.close()
    file_save.close()


def get_type_of_dst_table_layout_grp_data_hdf5_file(path):
    file = h5py.File(path, "r")
    dst_dtype = file["Data"]["Table Layout"].dtype
    file.close()
    return dst_dtype


def get_chunk_size_of_dst_table_layout_grp_data_hdf5_file(path):
    file = h5py.File(path, "r")
    dst_chunk = file["Data"]["Table Layout"].chunks
    file.close()
    return dst_chunk


def try__add_data_to_dst_table_layout(source_path, end_path):
    source_file = h5py.File(source_path, "r")
    data = source_file["Data"]["Table Layout"][1:6]
    source_file.close()
    wlos.add_data_to_dst_table_layout(data, end_path)


def try__create_hdf5_for_one_site(source_path, save_path, site_name, save_parameter="directory"):
    if save_parameter == "directory":
        source_los_file_name = os.path.basename(source_path)
        save_file_name = f"{source_los_file_name[0:12]}.{site_name}{source_los_file_name[12:]}"
        save_path = os.path.join(save_path, save_file_name)
    elif save_parameter == "file":
        pass
    else:
        raise ValueError("save_parameter must be 'directory' or 'file'")
    print(f"Start looking for indices ---- {dt.datetime.now()}")
    indices = rlos.get_indices_for_site(source_path, site_name)
    print(f"End looking for indices ------ {dt.datetime.now()}"
          f"Start reading data -----------")
    data = rlos.get_data_by_indices_GPS(source_path, indices)
    print(f"End reading data ------------- {dt.datetime.now()}"
          f"Start making copy ------------")
    wlos._copy_metadata_group_from_source_to_save(source_path, save_path)
    wlos._create_dst_table_layout_grp_data_similar_to_source(source_path, save_path)
    wlos.add_data_to_dst_table_layout(data, save_path)
    print(f"End making copy -------------- {dt.datetime.now()}")

def try__add_data_to_dst_table_layout2(source_path, end_path, site_name):
    print(f"Start looking for indices ---- {dt.datetime.now()}")
    indices = rlos.get_indices_for_site(source_path, site_name)
    print(f"End looking for indices ------ {dt.datetime.now()}",
          f"Number of True indices is {np.count_nonzero(indices)}",
          f"Start reading data -----------", sep="\n")
    data = rlos.get_data_by_indices_GPS(source_path, indices)
    print(f"End reading data ------------- {dt.datetime.now()}",
          len(data),
          f"Start making copy ------------", sep="\n")
    wlos.add_data_to_dst_table_layout(data, end_path)
    print(f"End making copy -------------- {dt.datetime.now()}")


def comparison1():
    path1 = r"/home/vadymskipa/Documents/PhD_student/data/data1/los_20230224.001.h5.hdf5"
    path2 = r"/home/vadymskipa/Documents/PhD_student/data/data1/los_20230225.001.h5.hdf5"
    path3 = r"/home/vadymskipa/Documents/PhD_student/data/data1/los_20230226.001.h5.hdf5"
    path4 = r"/home/vadymskipa/Documents/PhD_student/data/data1/los_20230227.001.h5.hdf5"
    path5 = r"/home/vadymskipa/Documents/PhD_student/data/data1/los_20230228.001.h5.hdf5"
    site = "krrs"
    start = dt.datetime.now()
    print(f"Start 'with file' ---- {start}")
    print("---Reading indices ---- ", end="")
    with h5py.File(path1, "r") as file:
        arr_of_sites = file["Data"]["Table Layout"]["gps_site"]
        file.close()
        np_site = site.encode("ascii")
        indices = arr_of_sites == np_site
    print(f"{dt.datetime.now() - start}")
    print("---Reading data ------- ", end="")
    with h5py.File(path1, "r") as file:
        data = file["Data"]["Table Layout"][indices]
    print(f"{dt.datetime.now() - start}")
    indices = None
    data = None
    start = dt.datetime.now()
    print(f"Start 'File-close' --- {start}")
    print("---Reading indices ---- ", end="")
    file = h5py.File(path1, "r")
    arr_of_sites = file["Data"]["Table Layout"]["gps_site"]
    file.close()
    np_site = site.encode("ascii")
    indices = arr_of_sites == np_site
    file.close()
    print(f"{dt.datetime.now() - start}")
    print("---Reading data ------- ", end="")
    file = h5py.File(path1, "r")
    data = file["Data"]["Table Layout"][indices]
    file.close()
    print(f"{dt.datetime.now() - start}")
    indices = None
    data = None

    start = dt.datetime.now()
    print(f"Start 'File-close' --- {start}")
    print("---Reading indices ---- ", end="")
    file = h5py.File(path2, "r")
    arr_of_sites = file["Data"]["Table Layout"]["gps_site"]
    file.close()
    np_site = site.encode("ascii")
    indices = arr_of_sites == np_site
    file.close()
    print(f"{dt.datetime.now() - start}")
    print("---Reading data ------- ", end="")
    file = h5py.File(path2, "r")
    data = file["Data"]["Table Layout"][indices]
    file.close()
    print(f"{dt.datetime.now() - start}")
    indices = None
    data = None
    start = dt.datetime.now()
    print(f"Start 'with file' ---- {start}")
    print("---Reading indices ---- ", end="")
    with h5py.File(path2, "r") as file:
        arr_of_sites = file["Data"]["Table Layout"]["gps_site"]
        file.close()
        np_site = site.encode("ascii")
        indices = arr_of_sites == np_site
    print(f"{dt.datetime.now() - start}")
    print("---Reading data ------- ", end="")
    with h5py.File(path2, "r") as file:
        data = file["Data"]["Table Layout"][indices]
    print(f"{dt.datetime.now() - start}")
    indices = None
    data = None

    start = dt.datetime.now()
    print(f"Start 'File-close' --- {start}")
    print("---Reading indices ---- ", end="")
    file = h5py.File(path3, "r")
    arr_of_sites = file["Data"]["Table Layout"]["gps_site"]
    file.close()
    np_site = site.encode("ascii")
    indices = arr_of_sites == np_site
    file.close()
    print(f"{dt.datetime.now() - start}")
    print("---Reading data ------- ", end="")
    file = h5py.File(path3, "r")
    data = file["Data"]["Table Layout"][indices]
    file.close()
    print(f"{dt.datetime.now() - start}")
    indices = None
    data = None
    start = dt.datetime.now()
    print(f"Start 'with file' ---- {start}")
    print("---Reading indices ---- ", end="")
    with h5py.File(path3, "r") as file:
        arr_of_sites = file["Data"]["Table Layout"]["gps_site"]
        file.close()
        np_site = site.encode("ascii")
        indices = arr_of_sites == np_site
    print(f"{dt.datetime.now() - start}")
    print("---Reading data ------- ", end="")
    with h5py.File(path3, "r") as file:
        data = file["Data"]["Table Layout"][indices]
    print(f"{dt.datetime.now() - start}")
    indices = None
    data = None

    start = dt.datetime.now()
    print(f"Start 'with file' ---- {start}")
    print("---Reading indices ---- ", end="")
    with h5py.File(path4, "r") as file:
        arr_of_sites = file["Data"]["Table Layout"]["gps_site"]
        file.close()
        np_site = site.encode("ascii")
        indices = arr_of_sites == np_site
    print(f"{dt.datetime.now() - start}")
    print("---Reading data ------- ", end="")
    with h5py.File(path4, "r") as file:
        data = file["Data"]["Table Layout"][indices]
    print(f"{dt.datetime.now() - start}")
    indices = None
    data = None
    start = dt.datetime.now()
    print(f"Start 'File-close' --- {start}")
    print("---Reading indices ---- ", end="")
    file = h5py.File(path4, "r")
    arr_of_sites = file["Data"]["Table Layout"]["gps_site"]
    file.close()
    np_site = site.encode("ascii")
    indices = arr_of_sites == np_site
    file.close()
    print(f"{dt.datetime.now() - start}")
    print("---Reading data ------- ", end="")
    file = h5py.File(path4, "r")
    data = file["Data"]["Table Layout"][indices]
    file.close()
    print(f"{dt.datetime.now() - start}")
    indices = None
    data = None


def comparison2():
    path1 = r"/home/vadymskipa/Documents/PhD_student/data/data1/los_20230224.001.h5.hdf5"
    path2 = r"/home/vadymskipa/Documents/PhD_student/data/data1/los_20230225.001.h5.hdf5"
    path3 = r"/home/vadymskipa/Documents/PhD_student/data/data1/los_20230226.001.h5.hdf5"
    path4 = r"/home/vadymskipa/Documents/PhD_student/data/data1/los_20230227.001.h5.hdf5"
    path5 = r"/home/vadymskipa/Documents/PhD_student/data/data1/los_20230228.001.h5.hdf5"
    site = "krrs"
    start = dt.datetime.now()
    print(f"Start 'with file' ----- {start}")
    print("---Reading indices ---- ", end="")
    with h5py.File(path1, "r") as file:
        arr_of_sites = file["Data"]["Table Layout"]["gps_site"]
        file.close()
        np_site = site.encode("ascii")
        indices = arr_of_sites == np_site
    print(f"{dt.datetime.now() - start}")
    print("---Reading data ------- ", end="")
    with h5py.File(path1, "r") as file:
        data = file["Data"]["Table Layout"][indices]
    print(f"{dt.datetime.now() - start}")
    del indices
    del data
    del arr_of_sites
    start = dt.datetime.now()
    print(f"Start function -------- {start}")
    print("---Reading indices ---- ", end="")
    indices = rlos.get_indices_for_site(path1, site)
    print(f"{dt.datetime.now() - start}")
    print("---Reading data ------- ", end="")
    data = rlos.get_data_by_indices_GPS(path1, indices)
    print(f"{dt.datetime.now() - start}")
    del indices
    del data

    start = dt.datetime.now()
    print(f"Start function -------- {start}")
    print("---Reading indices ---- ", end="")
    indices = rlos.get_indices_for_site(path1, site)
    print(f"{dt.datetime.now() - start}")
    print("---Reading data ------- ", end="")
    data = rlos.get_data_by_indices_GPS(path1, indices)
    print(f"{dt.datetime.now() - start}")
    del indices
    del data
    start = dt.datetime.now()
    print(f"Start 'with file' ----- {start}")
    print("---Reading indices ---- ", end="")
    with h5py.File(path1, "r") as file:
        arr_of_sites = file["Data"]["Table Layout"]["gps_site"]
        file.close()
        np_site = site.encode("ascii")
        indices = arr_of_sites == np_site
    print(f"{dt.datetime.now() - start}")
    print("---Reading data ------- ", end="")
    with h5py.File(path1, "r") as file:
        data = file["Data"]["Table Layout"][indices]
    print(f"{dt.datetime.now() - start}")
    del indices
    del data
    del arr_of_sites

    start = dt.datetime.now()
    print(f"Start 'with file' ----- {start}")
    print("---Reading indices ---- ", end="")
    with h5py.File(path1, "r") as file:
        arr_of_sites = file["Data"]["Table Layout"]["gps_site"]
        file.close()
        np_site = site.encode("ascii")
        indices = arr_of_sites == np_site
    print(f"{dt.datetime.now() - start}")
    print("---Reading data ------- ", end="")
    with h5py.File(path1, "r") as file:
        data = file["Data"]["Table Layout"][indices]
    print(f"{dt.datetime.now() - start}")
    del indices
    del data
    del arr_of_sites
    start = dt.datetime.now()
    print(f"Start function -------- {start}")
    print("---Reading indices ---- ", end="")
    indices = rlos.get_indices_for_site(path1, site)
    print(f"{dt.datetime.now() - start}")
    print("---Reading data ------- ", end="")
    data = rlos.get_data_by_indices_GPS(path1, indices)
    print(f"{dt.datetime.now() - start}")
    del indices
    del data


def try__get_indices_for_sites(path_los, path_site, lats, lons):
    print(dt.datetime.now())
    sites_dataframe = wsite.get_sites_dataframe_by_coordinates(path_site, lons=lons, lats=lats)
    print(dt.datetime.now())
    np_sites = sites_dataframe.loc[:, "gps_site"]
    sites = []
    for np_site in np_sites:
        sites.append(np_site.decode("ascii"))
    print(sites)
    indices = rlos.get_indices_for_sites(path_los, sites)
    print(dt.datetime.now())
    print(f"{indices.sum()}/{len(indices)}")


def save_europe_los_file(los_path, site_path, save_path, save_parameter="directory"):
    start = dt.datetime.now()
    print(start)
    lats = (20, 80)
    lons = (-5, 50)
    length_of_text = 35
    if save_parameter == "directory":
        source_los_file_name = os.path.basename(los_path)
        save_file_name = f"{source_los_file_name[0:12]}.europe{source_los_file_name[12:]}"
        save_path = os.path.join(save_path, save_file_name)
    elif save_parameter == "file":
        pass
    else:
        raise ValueError("save_parameter must be 'directory' or 'file'")
    sites_dataframe = wsite.get_sites_dataframe_by_coordinates(site_path, lons=lons, lats=lats)
    np_sites = sites_dataframe.loc[:, "gps_site"]
    sites = []
    for np_site in np_sites:
        sites.append(np_site.decode("ascii"))
    print(f"{'Reading site array':-<{length_of_text}s}", end="")
    site_array = rlos.get_site_array(los_path)
    print(f"{dt.datetime.now() - start}")
    temp_start = dt.datetime.now()
    print(f"{'Creating new file':-<{length_of_text}s}", end="")
    wlos._copy_metadata_group_from_source_to_save(los_path, save_path)
    wlos._create_dst_table_layout_grp_data_similar_to_source(los_path, save_path)
    print(f"{dt.datetime.now() - temp_start}")
    step_of_sites = 100
    for start_index in range(0, len(sites), step_of_sites):
        end_index = start_index + step_of_sites
        if end_index > len(sites):
            end_index = len(sites)
        temp_text = f"Work with {start_index}-{end_index} sites /{len(sites)}"
        temp_start = dt.datetime.now()
        print(f"{temp_text:-<{length_of_text}s}{temp_start}")
        print(f"{'':<5s}{'Processing indices':-<{length_of_text-5}s}", end="")
        temp_indices = rlos.get_indices_for_sites_from_site_array(site_array, sites[start_index:end_index])
        print(f"{dt.datetime.now() - temp_start}")
        temp_start = dt.datetime.now()
        print(f"{'':5s}{'Reading data':-<{length_of_text-5}s}", end="")
        temp_data = wlos.get_data_by_indices(los_path, temp_indices)
        print(f"{dt.datetime.now() - temp_start}")
        temp_start = dt.datetime.now()
        print(f"{'':5s}{'Saving data':-<{length_of_text-5}s}", end="")
        wlos.add_data_to_dst_table_layout(temp_data, save_path)
        print(f"{dt.datetime.now() - temp_start}")
    print(f"The duration is {dt.datetime.now() - start}")
    print(dt.datetime.now())


def save_europe_los_file_multiprocessing(los_path, site_path, save_path, save_parameter="directory"):
    start = dt.datetime.now()
    print(start)
    lats = (20, 80)
    lons = (-5, 50)
    length_of_text = 35
    if save_parameter == "directory":
        source_los_file_name = os.path.basename(los_path)
        save_file_name = f"{source_los_file_name[0:12]}.europe{source_los_file_name[12:]}"
        save_path = os.path.join(save_path, save_file_name)
    elif save_parameter == "file":
        pass
    else:
        raise ValueError("save_parameter must be 'directory' or 'file'")
    sites_dataframe = wsite.get_sites_dataframe_by_coordinates(site_path, lons=lons, lats=lats)
    np_sites = sites_dataframe.loc[:, "gps_site"]
    sites = []
    for np_site in np_sites:
        sites.append(np_site.decode("ascii"))
    print(f"{'Reading site array':-<{length_of_text}s}", end="")
    site_array = rlos.get_site_array(los_path)
    print(f"{dt.datetime.now() - start}")
    temp_start = dt.datetime.now()
    print(f"{'Creating new file':-<{length_of_text}s}", end="")
    wlos._copy_metadata_group_from_source_to_save(los_path, save_path)
    wlos._create_dst_table_layout_grp_data_similar_to_source(los_path, save_path)
    print(f"{dt.datetime.now() - temp_start}")
    step_of_sites = 100
    for start_index in range(0, len(sites), step_of_sites):
        end_index = start_index + step_of_sites
        if end_index > len(sites):
            end_index = len(sites)
        temp_text = f"Work with {start_index}-{end_index} sites /{len(sites)}"
        temp_start = dt.datetime.now()
        print(f"{temp_text:-<{length_of_text}s}{temp_start}")
        print(f"{'':<5s}{'Processing indices':-<{length_of_text-5}s}", end="")
        temp_indices = wlos.get_indices_for_sites_from_site_array_multiprocessing(site_array.view(),
                                                                                  sites[start_index:end_index],
                                                                                  number_of_processes=7)
        print(f"{dt.datetime.now() - temp_start}")
        temp_start = dt.datetime.now()
        print(f"{'':5s}{'Reading data':-<{length_of_text-5}s}", end="")
        temp_data = wlos.get_data_by_indices(los_path, temp_indices)
        print(f"{dt.datetime.now() - temp_start}")
        temp_start = dt.datetime.now()
        print(f"{'':5s}{'Saving data':-<{length_of_text-5}s}", end="")
        wlos.add_data_to_dst_table_layout(temp_data, save_path)
        print(f"{dt.datetime.now() - temp_start}")
    print(f"The duration is {dt.datetime.now() - start}")
    print(dt.datetime.now())


def save_region_los_files_from_directory(source_los_directory_path, source_site_directory_path,
                                         save_region_directory_path, region_name, lats, lons,
                                         repeatition_of_region_files=False, multiprocessing=False,
                                         start_date=dt.datetime.min, end_date=dt.datetime.max):
    list_of_objects_in_los_directory = os.listdir(source_los_directory_path)
    list_los_file_names = [el for el in list_of_objects_in_los_directory
                           if (os.path.isfile(os.path.join(source_los_directory_path, el)) and
                               wlos.check_los_file_name(el))]
    list_of_objects_in_site_directory = os.listdir(source_site_directory_path)
    list_site_file_names = [el for el in list_of_objects_in_site_directory
                            if (os.path.isfile(os.path.join(source_site_directory_path, el)) and
                                wsite.check_site_file_name(el))]
    list_of_dates = []
    if repeatition_of_region_files:
        list_of_objects_in_save_region_directory = os.listdir(save_region_directory_path)
        list_save_regions_file_names = [el for el in list_of_objects_in_save_region_directory
                                if (os.path.isfile(os.path.join(save_region_directory_path, el)) and
                                    wlos.check_los_file_name(el))]
        list_of_dates = [wlos.get_date_by_los_file_name(el) for el in list_save_regions_file_names
                             if (region_name in el)]
    for los_file_name in list_los_file_names:
        print(f"working with {los_file_name} --- {dt.datetime.now()}")
        date = wlos.get_date_by_los_file_name(los_file_name)
        if repeatition_of_region_files and (date in list_of_dates):
            continue
        if not (start_date <= date <= end_date):
            continue
        site_file_name = None
        for site_file_name_check in list_site_file_names:
            if wsite.check_site_file_name_by_date(site_file_name_check, date):
                site_file_name = site_file_name_check
        if not site_file_name:
            continue
        los_file_path = os.path.join(source_los_directory_path, los_file_name)
        site_file_path = os.path.join(source_site_directory_path, site_file_name)
        save_fie_name = wlos.get_region_los_file_name(los_file_name, region_name)
        save_file_path = os.path.join(save_region_directory_path, save_fie_name)
        wlos.save_region_los_file(los_file_path, site_file_path, save_file_path, lats=lats, lons=lons,
                                  multiprocessing=multiprocessing)


def save_europe_los_files_from_directory(source_los_directory_path, source_site_directory_path,
                                         save_region_directory_path, repeatition_of_region_files=False,
                                         multiprocessing=False):
    save_region_los_files_from_directory(source_los_directory_path, source_site_directory_path,
                                         save_region_directory_path, region_name="europe", lats=(30, 80),
                                         lons=(-10, 50),
                                         repeatition_of_region_files=repeatition_of_region_files,
                                         multiprocessing=multiprocessing)


def try_finding_los_and_site_for_the_same_date(source_los_directory_path, source_site_directory_path):
    list_of_objects_in_los_directory = os.listdir(source_los_directory_path)
    list_los_file_names = [el for el in list_of_objects_in_los_directory
                           if (os.path.isfile(os.path.join(source_los_directory_path, el)) and
                               wlos.check_los_file_name(el))]
    list_of_objects_in_site_directory = os.listdir(source_site_directory_path)
    list_site_file_names = [el for el in list_of_objects_in_site_directory
                            if (os.path.isfile(os.path.join(source_site_directory_path, el)) and
                                wsite.check_site_file_name(el))]
    result_list = []
    for los_file_name in list_los_file_names:
        date = wlos.get_date_by_los_file_name(los_file_name)
        site_file_name = None
        for site_file_name_check in list_site_file_names:
            if wsite.check_site_file_name_by_date(site_file_name_check, date):
                site_file_name = site_file_name_check
        if not site_file_name:
            continue
        result_list.append((los_file_name, site_file_name))
    print(result_list)


def try_list_of_region_dates(region_directory_path, region_name):
    list_of_objects_in_save_region_directory = os.listdir(region_directory_path)
    list_save_regions_file_names = [el for el in list_of_objects_in_save_region_directory
                                    if (os.path.isfile(os.path.join(region_directory_path, el)) and
                                        wlos.check_los_file_name(el))]
    list_of_dates = [wlos.get_date_by_los_file_name(el) for el in list_save_regions_file_names if (region_name in el)]
    print(list_of_dates)


def get_quantity_of_the_site_and_the_sat_appearing(site, los_file_path, sat_id):
    data = wlos.get_data_for_one_site_GPS_pd(los_file_path, site)
    sub_data = data.loc[data.loc[:, "sat_id"] == sat_id]
    return len(sub_data)


def print_compress_method(los_files_path):
    with h5py.File(los_files_path, "r") as file:
        compression = file["Data"]["Table Layout"].compression
        compression_opts = file["Data"]["Table Layout"].compression_opts
        print(f"compression={compression}, compression_opts={compression_opts}")


def main2024feb1():
    source_los_directory_path = r"/home/vadymskipa/HDD1/big_los_hdf/"
    source_site_directory_path = r"/home/vadymskipa/PhD_student/data/site_hdf/"
    save_region_directory_path = r"/home/vadymskipa/PhD_student/data/America_25-70__-125--65/los_hdf/"
    start_date = dt.datetime(year=2023, month=2, day=24, tzinfo=dt.timezone.utc)
    end_date = dt.datetime(year=2023, month=3, day=3, tzinfo=dt.timezone.utc)
    save_region_los_files_from_directory(source_los_directory_path, source_site_directory_path,
                                         save_region_directory_path, region_name="north_america", lats=(25, 70),
                                         lons=(-125, -65),
                                         repeatition_of_region_files=True,
                                         multiprocessing=True,
                                         start_date=start_date, end_date=end_date)


def main2024feb2():
    source_los_directory_path = r"/home/vadymskipa/HDD1/big_los_hdf/"
    source_site_directory_path = r"/home/vadymskipa/PhD_student/data/site_hdf/"
    save_region_directory_path = r"/home/vadymskipa/PhD_student/data/South_Africa_-20--35__15-35/los_hdf/"
    start_date = dt.datetime(year=2023, month=2, day=24, tzinfo=dt.timezone.utc)
    end_date = dt.datetime(year=2023, month=3, day=3, tzinfo=dt.timezone.utc)
    save_region_los_files_from_directory(source_los_directory_path, source_site_directory_path,
                                         save_region_directory_path, region_name="south_africa", lats=(-35, -20),
                                         lons=(15, 35),
                                         repeatition_of_region_files=True,
                                         multiprocessing=True,
                                         start_date=start_date, end_date=end_date)


def main2024jun():
    # main2024jun1()
    # main2024jun2()
    main2024jun3()


def main2024jun1():
    source_los_directory_path = r"/home/vadymskipa/HDD1/big_los_hdf/"
    source_site_directory_path = r"/home/vadymskipa/PhD_student/data/site_hdf/"
    save_region_directory_path = r"/home/vadymskipa/PhD_student/data/Europe_los_hdf/"
    start_date = dt.datetime(year=2016, month=3, day=4, tzinfo=dt.timezone.utc)
    end_date = dt.datetime(year=2016, month=3, day=11, tzinfo=dt.timezone.utc)
    save_region_los_files_from_directory(source_los_directory_path, source_site_directory_path,
                                         save_region_directory_path, region_name="europe", lats=(30, 80),
                                         lons=(-10, 50),
                                         repeatition_of_region_files=True,
                                         multiprocessing=True,
                                         start_date=start_date, end_date=end_date)


def main2024jun2():
    with h5py.File(r"/home/vadymskipa/1.hdf5", "x") as file:
        group_data = file.create_group("Data")
        dst_dtype = np.dtype([("gps_site", "S4"), ('gdlatr', '<f8'), ('gdlonr', '<f8')])
        dataset_table_layout = group_data.create_dataset("Table Layout", shape=(2,), dtype=dst_dtype)
        dataset_table_layout["gps_site"] = ["asdf", "ghjk"]
        dataset_table_layout["gdlatr"] = [2.5, 3.5]
        dataset_table_layout["gdlonr"] = [2.6, 3.4]


def main2024jun3():
    source_los_directory_path = r"/home/vadymskipa/HDD2/big_los_hdf/"
    source_site_directory_path = r"/home/vadymskipa/PhD_student/data/site_hdf/"
    save_region_directory_path = r"/home/vadymskipa/PhD_student/data/Europe_los_hdf/"
    start_date = dt.datetime(year=2016, month=8, day=30, tzinfo=dt.timezone.utc)
    end_date = dt.datetime(year=2016, month=9, day=4, tzinfo=dt.timezone.utc)
    save_region_los_files_from_directory(source_los_directory_path, source_site_directory_path,
                                         save_region_directory_path, region_name="europe", lats=(30, 80),
                                         lons=(-10, 50),
                                         repeatition_of_region_files=True,
                                         multiprocessing=True,
                                         start_date=start_date, end_date=end_date)


def main2024aug1():
    source_los_directory_path = r"/home/vadymskipa/HDD1/big_los_hdf/"
    source_site_directory_path = r"/home/vadymskipa/PhD_student/data/site_hdf/"
    save_region_directory_path = r"/home/vadymskipa/PhD_student/data/Europe_los_hdf/"
    start_date = dt.datetime(year=2024, month=5, day=9, tzinfo=dt.timezone.utc)
    end_date = dt.datetime(year=2024, month=5, day=13, tzinfo=dt.timezone.utc)
    save_region_los_files_from_directory(source_los_directory_path, source_site_directory_path,
                                         save_region_directory_path, region_name="europe", lats=(30, 80),
                                         lons=(-10, 50),
                                         repeatition_of_region_files=True,
                                         multiprocessing=True,
                                         start_date=start_date, end_date=end_date)


def main2024oct1():
    source_los_directory_path = r"/home/vadymskipa/HDD1/big_los_hdf/"
    source_site_directory_path = r"/home/vadymskipa/PhD_student/data/site_hdf/"
    save_region_directory_path = r"/home/vadymskipa/PhD_student/data/Japan_los_hdf/"
    start_date = dt.datetime(year=2023, month=3, day=3, tzinfo=dt.timezone.utc)
    end_date = dt.datetime(year=2023, month=3, day=3, tzinfo=dt.timezone.utc)
    save_region_los_files_from_directory(source_los_directory_path, source_site_directory_path,
                                         save_region_directory_path, region_name="japan", lats=(20, 50),
                                         lons=(130, 160),
                                         repeatition_of_region_files=True,
                                         multiprocessing=True,
                                         start_date=start_date, end_date=end_date)



if __name__ == "__main__":
    main2024oct1()
    # main2024aug1()
    # main2024jun()
    # path = r"/home/vadymskipa/Documents/PhD_student/data/data1/site_20230226.001.hdf5"
    # path2 = r"/tmp/site_20230226.001.h5.hdf5"
    # path3 = r"/home/vadymskipa/Documents/PhD_student/data/data1/los_20230224.001.h5.hdf5"
    # path4 = r"/home/vadymskipa/Documents/PhD_student/data/smaller_data/new_los.hdf5"
    # path5 = r"/home/vadymskipa/Documents/PhD_student/data/smaller_data/"
    # path6 = r"/home/vadymskipa/Documents/PhD_student/data/smaller_data/los_20230224.krrs.001.h5.hdf5"
    # path7 = r"/home/vadymskipa/Documents/PhD_student/data/data1/los_20230607.001.h5.hdf5"
    # path8 = r"/home/vadymskipa/Documents/PhD_student/data/data1/site_20230607.001.h5.hdf5"
    # path9 = r"/home/vadymskipa/Documents/PhD_student/data/Europe/"
    # path10 = r"/home/vadymskipa/Documents/PhD_student/data/data1/"
    # path11 = r"/home/vadymskipa/Documents/PhD_student/data/Europe/los_20230225.europe.001.h5.hdf5"
    # path_site_hdf = r"/home/vadymskipa/Documents/PhD_student/data/site_hdf/"
    # path_los_hdf = r"/home/vadymskipa/Documents/PhD_student/data/big_los_hdf/"
    # path_europe_los_hdf = r"/home/vadymskipa/Documents/PhD_student/data/Europe_los_hdf/"
    # save_europe_los_files_from_directory(path_los_hdf, path_site_hdf, path_europe_los_hdf,
    #                                      repeatition_of_region_files=True, multiprocessing=True)
    # path12 = r"/home/vadymskipa/Documents/PhD_student/data/big_los_hdf/los_20200924.001.h5"
    # path13 = r"/home/vadymskipa/Documents/PhD_student/data/big_los_hdf/los_20221117.001.h5.hdf5"
    # try__add_data_to_dst_table_layout2(path3, path6, "fra2")
    # try__get_indices_for_sites(path7, path8, (20, 80), (-5, 50))
    # save_europe_los_file_multiprocessing(path7, path8, path5)
    # save_europe_los_files_from_directory(path10, path10, path9)
    # try_finding_los_and_site_for_the_same_date(path10, path10)
    # try_list_of_region_dates(path9, "europe")
    # save_europe_los_files_from_directory(path10, path10, path9, repeatition_of_region_files=True, multiprocessing=True)
    # path12 = r"/home/vadymskipa/Documents/PhD_student/data/Europe/los_20230606.europe.001.h5.hdf5"
    # path13 = r"/home/vadymskipa/Documents/PhD_student/data/data1/los_20230606.001.h5.hdf5"
    # print(get_quantity_of_the_site_and_the_sat_appearing("krrs", path13, 15))
    # print_compress_method(path12)
    # print_compress_method(path13)
    # main2024feb1()
    # main2024feb2()
    # print(get_type_of_dst_table_layout_grp_data_hdf5_file(r"/home/vadymskipa/HDD1/big_los_hdf/los_20160304.001.h5.hdf5"))