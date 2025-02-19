import numpy as np 
import h5py
import pandas as pd


def get_data_for_one_site_GPS(path: str, site: str):
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


def get_data_by_indices_GPS(path: str, indices):
    with h5py.File(path, "r") as file:
        data = file["Data"]["Table Layout"][indices]
    GPS = "GPS     ".encode("ascii")
    arr_of_GNSS = data["gnss_type"]
    indices_of_GPS = arr_of_GNSS == GPS
    data_outcome = data[indices_of_GPS]
    return data_outcome


def get_data_by_indeces_GPS_pd(path: str, indices):
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


def get_indices_for_sites_from_site_array(site_array, sites):
    indices = np.full(len(site_array), False)
    for site in sites:
        temp_indices = site_array == site.encode("ascii")
        indices = np.logical_or(indices, temp_indices)
    return indices


