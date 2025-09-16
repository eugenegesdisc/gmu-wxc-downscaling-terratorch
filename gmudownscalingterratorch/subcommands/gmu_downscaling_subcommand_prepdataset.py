"""
    A subcommand plugin for preparing datasets for downscaling.
"""
import os
import yaml
import logging
from dateutil import parser as dtu_parser
import datetime as py_dt
from gmudownscalingterratorch.datasets.downscaling import (
    GmuMerra2ToPrism4km)

logger = logging.getLogger(__name__)


def add_cli_arg(the_main_parsers):
    """
        A subcommand plugin.
    """
    #print("Add argument here - ", the_main_parsers)
    the_subcommand = the_main_parsers.add_parser(
        "prepdataset",
        description="prepdataset - "
        "prepare a subset of dataset for downscaling tests ",
        help="Download and extract a set of data for testing with downscaling deep learning.")
    the_subcommand.add_argument(
        '-i', '--input-times', nargs="+", type=str,
        required=True,
        dest="input_times",
        help="List of input-times. Each input-time can be iso time string or a string of "
        "paired time string separated by ';'. For example, '2020-01-01;2020-01-02 23:59:59', "
        "'2020-01-03'")
    the_subcommand.add_argument(
        '-r', '--root-dir', type=str,
        required=True,
        dest="root_dir",
        help="Root directory for data. Downloaded data will be stored at root-dir/download.")
    the_subcommand.add_argument(
        '-n', '--name-of-sample', type=str,
        required=True,
        dest="name_of_sample",
        help="Name for the sample set. Extracted file will be stored at root-dir/{name-of-sample}.")
    the_subcommand.add_argument(
        '--auth-config-file', type=str,
        dest="auth_config_file",
        default="../cfg/.gmu_downscaling_earthdata_login.cfg",
        help="Authtentication configuration file.")
    the_subcommand.add_argument(
        '--download', action="store_true", required=False, default=False,
        help='Enable download')
    the_subcommand.add_argument(
        '--extract', action="store_true", required=False, default=False,
        help='Enable extract')
    the_subcommand.add_argument(
        '--convert', action="store_true", required=False, default=False,
        help='Enable convert (to GeoTIFF)')
    the_subcommand.add_argument(
        '--overwrite', action="store_true", required=False, default=False,
        help='Enforce overwrite - on download and/or extract')



def _load_earthdata_authentication(
        earthdata_login_config:str):
    username=None
    password=None
    edl_token=None
    with open(earthdata_login_config, 'r') as file:
        data = yaml.safe_load(file)
        if "edl_token" in data:
            edl_token = data["edl_token"]
        if "username" in data:
            username = data["username"]
        if "password" in data:
            password = data["password"]
    return username, password, edl_token

def _pre_download_and_extract_data(
        data_root_path:str,
        data_periods:list[str],
        target_data_folder:str,
        download:bool=True,
        extract:bool=True,
        convert:bool=True,
        overwrite:bool=True,
        username:str=None,
        password:str=None,
        edl_token:str=None,
        ):
    input_data_periods = list()
    for a_str in data_periods:
        if ";" in a_str:
            a_strs = a_str.split(";")
            if len(a_strs) != 2:
                logger.warning("time period malformatted: '{a_str}'")
            start_str = a_strs[0].strip()
            end_str = a_strs[1].strip()
            input_data_periods.append((start_str, end_str))
        else:
            #if one date string, expand it to cover the whole day
            date_format = "%Y%m%d %H:%M:%S"
            c_date = dtu_parser.isoparse(a_str)
            start_time = py_dt.datetime(
                year=c_date.year,
                month=c_date.month,
                day=c_date.day)
            end_time = py_dt.datetime(
                year=c_date.year,
                month=c_date.month,
                day=c_date.day,
                hour=23,
                minute=59,
                second=59
                )
            start_str = py_dt.datetime.strftime(start_time,date_format)
            end_str = py_dt.datetime.strftime(end_time,date_format)
            input_data_periods.append((start_str,end_str))

    m2t1nxflx_paths  = os.path.join(
        data_root_path,"m2t1nxflx")
    m2t1nxflx_down_dir =  os.path.join(
        data_root_path,"m2t1nxflx",
        "download")
    m2i1nxasm_paths =  os.path.join(
        data_root_path,"m2i1nxasm")
    m2i1nxasm_down_dir =  os.path.join(
        data_root_path,"m2i1nxasm",
        "download")
    m2t1nxlnd_paths =  os.path.join(
        data_root_path,"m2t1nxlnd")
    m2t1nxlnd_down_dir =  os.path.join(
        data_root_path,"m2t1nxlnd",
        "download")
    m2t1nxrad_paths =  os.path.join(
        data_root_path,"m2t1nxrad")
    m2t1nxrad_down_dir =  os.path.join(
        data_root_path,"m2t1nxrad",
        "download")
    m2c0nxctm_paths =  os.path.join(
        data_root_path,"m2c0nxctm")
    m2c0nxctm_down_dir =  os.path.join(
        data_root_path,"m2c0nxctm",
        "download")
    m2i3nvasm_paths =  os.path.join(
        data_root_path,"m2i3nvasm")
    m2i3nvasm_down_dir =  os.path.join(
        data_root_path,"m2i3nvasm",
        "download")
    climate_surface_paths= os.path.join(
        data_root_path,"climatology",
        "surface")
    climate_surface_down_dir= os.path.join(
        data_root_path,"climatology",
        "download")
    climate_vertical_paths= os.path.join(
        data_root_path,"climatology3d",
        "vertical")
    climate_vertical_down_dir= os.path.join(
        data_root_path,"climatology3d",
        "download")
    prism_paths =  os.path.join(
        data_root_path,"dataprism4km")
    prism_down_dir =  os.path.join(
        data_root_path,"dataprism4km",
        "download")
    prism_static_paths =  os.path.join(
        data_root_path,"staticprism4km")
    prism_static_down_dir =  os.path.join(
        data_root_path,"staticprism4km",
        "download")



    for tp in input_data_periods:
        logger.info("Processing period - ", tp)
        ret_data = GmuMerra2ToPrism4km(
            start_date=tp[0],
            end_date=tp[1],
            download=download,
            extract=extract,
            convert=convert,
            predict_mode=False,
            overwrite=overwrite,
            use_climatology=True,
            use_hr_auxiliary=True,
            username=username,
            password=password,
            edl_token=edl_token,
            m2t1nxflx_paths = os.path.join(
                m2t1nxflx_paths,
                target_data_folder),
            m2t1nxflx_down_dir = m2t1nxflx_down_dir,
            m2i1nxasm_paths = os.path.join(
                m2i1nxasm_paths,
                target_data_folder),
            m2i1nxasm_down_dir = m2i1nxasm_down_dir,
            m2t1nxlnd_paths = os.path.join(
                m2t1nxlnd_paths,
                target_data_folder),
            m2t1nxlnd_down_dir = m2t1nxlnd_down_dir,
            m2t1nxrad_paths = os.path.join(
                m2t1nxrad_paths,
                target_data_folder),
            m2t1nxrad_down_dir = m2t1nxrad_down_dir,
            m2c0nxctm_paths = os.path.join(
                m2c0nxctm_paths,
                target_data_folder),
            m2c0nxctm_down_dir = m2c0nxctm_down_dir,
            m2i3nvasm_paths = os.path.join(
                m2i3nvasm_paths,
                target_data_folder),
            m2i3nvasm_down_dir = m2i3nvasm_down_dir,
            climate_surface_paths = climate_surface_paths,
            climate_surface_down_dir = climate_surface_down_dir,
            climate_vertical_paths = climate_vertical_paths,
            climate_vertical_down_dir = climate_vertical_down_dir,
            prism_paths = os.path.join(
                prism_paths,
                target_data_folder),
            prism_down_dir = prism_down_dir,
            prism_static_paths = os.path.join(
                prism_static_paths,
                target_data_folder),
            prism_static_down_dir = prism_static_down_dir,
            position_encoding="fourier"
        )
    return True



def process(the_args):
    """
        A subcommand plugin - process.
    """
    input_times = the_args.input_times
    root_dir = the_args.root_dir
    name_of_sample = the_args.name_of_sample
    download = the_args.download
    extract = the_args.extract
    convert = the_args.convert
    overwrite = the_args.overwrite
    auth_config_file = the_args.auth_config_file
    username, password, edl_token = _load_earthdata_authentication(
        earthdata_login_config=auth_config_file)

    _pre_download_and_extract_data(
            data_root_path=root_dir,
            data_periods=input_times,
            target_data_folder=name_of_sample,
            download=download,
            extract=extract,
            convert=convert,
            overwrite=overwrite,
            username=username,
            password=password,
            edl_token=edl_token,
            )
