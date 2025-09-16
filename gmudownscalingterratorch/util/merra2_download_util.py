import os
import re
import requests
import logging
import datetime as py_dt
import xarray as xr
#import dateutil.relativedelta as dt_relativedelta
import dateutil.parser as dtu_parser
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import (
    HfHubHTTPError,
    OfflineModeIsEnabled)

from .gdaldataset_util import (
    gdal_get_subdatasets,
    gdal_export_3d_subdataset_to_separate_geotiff,
    gdal_export_subdataset_to_separate_geotiff,
    gdal_export_const_subdataset_to_separate_geotiff,
    gdal_export_subdatasets_to_separate_geotiffs_with_fixes,
    gdal_export_3d_subdatasets_to_separate_geotiffs_with_fixes
)

logger = logging.getLogger(__name__)

class SessionWithHeaderRedirection(requests.Session):
    AUTH_HOST = 'urs.earthdata.nasa.gov'

    def __init__(self,  username, password, host=None):
        if host:
            self.AUTH_HOST = host
        super().__init__()
        self.auth = (username, password)


   # Overrides from the library to keep headers when redirected to or from
   # the NASA auth host.
    def rebuild_auth(self, prepared_request, response):
        headers = prepared_request.headers
        url = prepared_request.url

        if 'Authorization' in headers:
            original_parsed = requests.utils.urlparse(response.request.url)
            redirect_parsed = requests.utils.urlparse(url)

            if (original_parsed.hostname != redirect_parsed.hostname) and \
                    redirect_parsed.hostname != self.AUTH_HOST and \
                    original_parsed.hostname != self.AUTH_HOST:
                del headers['Authorization']
        return
    

class TokenAuthSessionWithRedirection(requests.Session):
    """
        Token session with redirection support.
    """
    def __init__(self, token):
        """
            Initialize a TokenAuthSessionWithRedirection class

            Args:
                token: the authentication token
        """
        super().__init__()
        self.token = token

    def prepare_request(self, request:requests.Request):
        """
            Prepare the request.

            Args:
                request:  The request to be modified with addition of token
                    to produce a PreparedRequest.
        """
        request.headers['Authorization'] = f'Bearer {self.token}'
        return super().prepare_request(request)
    
    def rebuild_auth(self,
                     prepared_request: requests.PreparedRequest,
                     response:requests.Response):
        """
            Re-build the reques after redirect.

            Args:
                prepared_request: The PreparedRequest to be re-assigned with
                    stored token.

        """
        prepared_request.headers['Authorization'] = f'Bearer {self.token}'
        return

def _download_binary_file_with_authentication(
        remote_url:str,
        local_filepath:str,
        username:str=None,
        password:str=None,
        edl_token:str=None)->bool:
    """
        Download a file at url to a local file

        Args:
            remote_url: The remote url address
            local_filepath: The file path for the local file
            username: User name if authentication is done with username/password
            password: Password paired to the username for authentication
            edl_token: An authenticated token - it has priority than username/password
        
        Returns:
            If True, succeeded. If False, some error occurred.
    """
    if edl_token:
        session = TokenAuthSessionWithRedirection(token=edl_token)
    else:
        session = SessionWithHeaderRedirection(
            username=username,
            password=password)
    try:
        response = session.get(
            url=remote_url,
            stream=True,
            timeout=300)
        response.raise_for_status()

        chunk_size = 1024 * 1024

        with open(local_filepath, "wb") as file:
            for chunk in response.iter_content(
                chunk_size=chunk_size):
                file.write(chunk)
        return True
    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed: {e}")
    except IOError as e:
        logger.error(f"IOError occurred: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
    return False


def download_data_merra2_climatology_from_hugging_with_template(
        repo_id:str,
        hf_resource_temp:str,
        start_date:str,
        end_date:str,
        target_dir:str,
        overwrite:bool=False,
        )->int:
    """
        Download data from the MERRA-2 climatology data to 
        search between [start_date, end_date] in days. All 3-hourly data
        of each day will be downloaded and/or extracted.

        Args:
            repo_id: a repository id in the Hugging Face 
            resource_temp: a path string which may contain template
                variable - {doy}, {hod}
            start_date: a start datetime string in ISO 8601 format.
            end_date: an end datetime string in ISO 8601 format.
            target_dir: a target local directory as the destination
                folder.
            overwrite: If True, exiting files in the destination folder
                will be overwritten.

        Returns:
            Number of data downloaded. If negative value, it will be the
            number of errors encountered.
    """
    n_down = 0
    n_error = 0
    start_datetime = dtu_parser.isoparse(start_date)
    end_datetime = dtu_parser.isoparse(end_date)

    n_days = (end_datetime - start_datetime).days + 1
    if n_days >= 365:
        n_days = 366
        start_datetime = py_dt.datetime(
            year=2020, month=1, day=1)
        end_datetime = py_dt.datetime(
            year=2020, month=12, day=31,
            hour=23, minute=59, second=59)

    logger.info("To download %d days of data with template=%s",
                n_days, hf_resource_temp)
    
    # making target directory if necessary
    os.makedirs(target_dir, exist_ok=True)

    hour3_delta = py_dt.timedelta(hours=3)
    for nd in range(n_days):
        ndelta = py_dt.timedelta(days=nd)
        c_day = start_datetime+ndelta
        day_of_year = c_day.timetuple().tm_yday
        doy_str = f"{day_of_year:03d}"
        # set it to start to 00:00:00 of the day
        _start_time = py_dt.datetime(
            year=c_day.year,
            month=c_day.month,
            day=c_day.day,
            hour=0,
            minute=0,
            second=0)
        for hh in range(8):
            hour_of_day = _start_time.hour
            hod_str = f"{hour_of_day:02d}"
            hb_file_path = hf_resource_temp.replace(
                "{doy}",doy_str).replace(
                    "{hod}", hod_str)
            #filename = hb_file_path #os.path.basename(hb_file_path)
            tgt_full_path = _form_local_file_name_download_from_hf(
                hf_file_path=hb_file_path,
                target_local_dir=target_dir)
            #os.path.join(target_dir, filename)
            if (overwrite or not os.path.exists(tgt_full_path)):
                if day_of_year == 366:
                    if not _download_climate_365_and_001_average_to_366(
                        repo_id=repo_id,
                        hour_of_day=hour_of_day,
                        hf_resource_temp=hf_resource_temp,
                        target_366_hour_nc=tgt_full_path,
                        target_dir=target_dir,
                        overwrite=overwrite):
                        n_error -= 1
                else:
                    if not _download_one_3hour_climatology_data(
                        repo_id=repo_id,
                        hb_file_path=hb_file_path,
                        target_dir=target_dir):
                        n_error -= 1

            _start_time += hour3_delta

    if n_error < 0:
        return n_error
    else:
        return n_down

def _form_local_file_name_download_from_hf(
        hf_file_path:str,
        target_local_dir:str)->str:
    if "/" in hf_file_path:
        hf_paths = hf_file_path.split("/")
        tgt_full_path = os.path.join(target_local_dir,*hf_paths)
        return tgt_full_path
    tgt_full_path = os.path.join(target_local_dir,hf_file_path)
    return tgt_full_path
    
def _download_climate_365_and_001_average_to_366(
    repo_id:str,
    hour_of_day:int,
    hf_resource_temp:str,
    target_dir:str,
    target_366_hour_nc:str,
    overwrite:bool=False
    )->bool:

    hod_str = f"{hour_of_day:02d}"
    # day-365
    doy_str_365 = f"365"
    hb_file_path = hf_resource_temp.replace(
        "{doy}",doy_str_365).replace(
            "{hod}", hod_str)
    #filename = os.path.basename(hb_file_path)
    tgt_full_path_365 = _form_local_file_name_download_from_hf(
        hf_file_path=hb_file_path,
        target_local_dir=target_dir)
    #os.path.join(target_dir, filename)
    if (overwrite or not os.path.exists(tgt_full_path_365)):
        if not _download_one_3hour_climatology_data(
            repo_id=repo_id,
            hb_file_path=hb_file_path,
            target_dir=target_dir):
            return False
    # day - 001    

    doy_str_001 = f"001"
    hb_file_path = hf_resource_temp.replace(
        "{doy}",doy_str_001).replace(
            "{hod}", hod_str)
    #filename = os.path.basename(hb_file_path)
    tgt_full_path_001 = _form_local_file_name_download_from_hf(
        hf_file_path=hb_file_path,
        target_local_dir=target_dir)
    #= os.path.join(target_dir, filename)
    if (overwrite or not os.path.exists(tgt_full_path_001)):
        if not _download_one_3hour_climatology_data(
            repo_id=repo_id,
            hb_file_path=hb_file_path,
            target_dir=target_dir):
            return False
    
    # combine/aggregate by average and save
    ds1 = xr.open_dataset(tgt_full_path_001)
    ds2 = xr.open_dataset(tgt_full_path_365)
    ds = (ds1+ds2)/2
    ds.to_netcdf(target_366_hour_nc)
    return True



def _download_one_3hour_climatology_data(
        repo_id:str,
        hb_file_path:str,
        target_dir:str
        )->bool:
    try:
        hf_hub_download(
            repo_id=repo_id,
            filename=hb_file_path,
            local_dir=target_dir)
        return True
    except HfHubHTTPError as err:
        logger.error("HF-Download failed on %s: %s",
                        hb_file_path, err)
    except OfflineModeIsEnabled as err:
        logger.error("Data %s in offline mode. Try again later: %s",
                        hb_file_path, err)
    except Exception as err:
        logger.error("Failed at downloading %s: %s",
                        hb_file_path, err)
    return False


def download_data_merra2_with_template(
        url_temp:str,
        start_date:str,
        end_date:str,
        target_dir:str,
        overwrite:bool=False,
        username:str="",
        password:str="",
        edl_token:str="",
        )->int:
    """
        Download data from the GESDISC repo using a template url to 
        search between [start_date, end_date].

        Args:
            url_temp: a url string template which may contain template
                variable - {year}, {month}, {date}
            start_date: a start datetime string in ISO 8601 format.
            end_date: an end datetime string in ISO 8601 format.
            target_dir: a target local directory as the destination
                folder.
            overwrite: If True, exiting files in the destination folder
                will be overwritten.
            username: User name if authetication is donw with username/password pair.
            password: Password paired to the username for authentication.
            edl_token: An authenticated token - it has priority over username/password

        Returns:
            Number of data downloaded. If negative value, it will be the
            number of errors encountered.
    """
    n_down = 0
    n_error = 0
    start_datetime = dtu_parser.isoparse(start_date)
    end_datetime = dtu_parser.isoparse(end_date)

    n_days = (end_datetime-start_datetime).days+1
    logger.info("To download %d days (between %s and %s) of data with template=%s",
                n_days, start_date, end_date, url_temp)
    # making target directory if necessary
    os.makedirs(target_dir, exist_ok=True)

    for nd in range(n_days):
        ndelta = py_dt.timedelta(days=nd)
        c_day = start_datetime+ndelta
        year_str = f"{c_day.year}"
        month_str = f"{c_day.month:02}"
        day_str = f"{c_day.day:02}"
        url_temp2 = _get_updated_merra2_template(
            year=c_day.year, old_template=url_temp)
        url_str = url_temp2.replace(
            "{year}", year_str).replace(
                "{month}", month_str).replace(
                    "{day}", day_str)
        filename = url_str.rsplit("/", 1)[1]
        target_full_name = os.path.join(target_dir, filename)

        if ((not os.path.exists(target_full_name))
             or overwrite):
            if _download_binary_file_with_authentication(
                remote_url=url_str,
                local_filepath=target_full_name,
                username=username,
                password=password,
                edl_token=edl_token):
                n_down +=1
            else:
                n_error -= 1

    if n_error < 0:
        return n_error
    else:
        return n_down


def extract_subdatasets_band_from_downloaded_const_merra2_with_template(
        filename_temp:str,
        start_date:str,
        end_date:str,
        source_dir:str,
        target_dir:str,
        subdatasets:list[str]|None = None,
        overwrite:bool=False,
        )->int:
    """
        Download data from the GESDISC repo using a template url to 
        search between [start_date, end_date].

        Args:
            filename_temp: a filename template string that may
                contain template variable - {year}, {month},
                {day}
            start_date: a start datetime string in ISO 8601 format.
            end_date: an end datetime string in ISO 8601 format.
            source_dir: the source directory holding downloaded
                MERRA 2 data.
            target_dir: a target local directory as the destination
                folder.
            subdatasets: a list of subdatasets.
            overwrite: If True, exiting files in the destination folder
                will be overwritten.

        Returns:
            Number of data downloaded. If negative value, it will be the
            number of errors encountered.
    """
    n_extract = 0
    n_error = 0
    start_datetime = dtu_parser.isoparse(start_date)
    end_datetime = dtu_parser.isoparse(end_date)

    n_days = (end_datetime - start_datetime).days + 1
    logger.info("To extract %d days of data with template=%s in source dir %s",
                n_days, filename_temp,
                source_dir)
    
    # making target directory if necessary
    os.makedirs(target_dir, exist_ok=True)

    for nd in range(n_days):
        ndelta = py_dt.timedelta(days=nd)
        c_day = start_datetime+ndelta
        year_str = f"{c_day.year}"
        month_str = f"{c_day.month:02}"
        day_str = f"{c_day.day:02}"
        filename_temp2 = _get_updated_merra2_template(
            year=c_day.year, old_template=filename_temp)
        filename = filename_temp2.replace(
            "{year}", year_str).replace(
                "{month}", month_str).replace(
                    "{day}", day_str)
        source_full_name = os.path.join(source_dir, filename)
        if not os.path.exists(source_full_name):
            logger.warning("File '%s' is not downloaded",
                           source_full_name)
            continue
        all_subdatasets = gdal_get_subdatasets(
            filename=source_full_name)
        if not subdatasets:
            subdatasets = [k for k in all_subdatasets]
        target_basename = os.path.splitext(
            os.path.basename(source_full_name))[0]
        # extract bands as separate files
        for key in subdatasets:
            subdataset_name = all_subdatasets[key][0]
            if gdal_export_const_subdataset_to_separate_geotiff(
                subdataset_name=subdataset_name,
                elem_name=key,
                target_basename=target_basename,
                target_dir=target_dir,
                overwrite=overwrite):
                n_extract += 1
            else:
                n_error -= 1


    if n_error < 0:
        return n_error
    else:
        return n_extract

def convert_string_datetime_to_2020_climatology_datetime(
        dt_dateitme_str:str, out_date_format:str="%Y%m%dT%H%M%S")->str:
    c_dt = dtu_parser.isoparse(dt_dateitme_str)
    a_dt = convert_datetime_to_2020_climatology_datetime(c_dt)
    ret_dt_str = py_dt.datetime.strftime(a_dt,out_date_format)
    return ret_dt_str

def convert_datetime_to_2020_climatology_datetime(
        dt_datetime:py_dt.datetime)->py_dt.datetime:
    year_2020_day1 = py_dt.datetime(year=2020, month=1, day=1)
    c_dt_day1 = py_dt.datetime(year=dt_datetime.year, month=1, day=1)
    tdelta = year_2020_day1 - c_dt_day1
    adjust_dt = dt_datetime + tdelta
    return adjust_dt

def extract_subdatasets_band_data_merra2_climatology_from_hugging_with_template(
        hf_resource_temp:str,
        start_date:str,
        end_date:str,
        source_dir:str,
        target_dir:str,
        subdatasets:list[str]|None = None,
        overwrite:bool=False,
        date_format:str="%Y%m%dT%H%M%S",
        )->int:
    """
        Download data from the GESDISC repo using a template url to 
        search between [start_date, end_date].

        Args:
            filename_temp: a filename template string that may
                contain template variable - {doy}, {hod}
            start_date: a start datetime string in ISO 8601 format.
            end_date: an end datetime string in ISO 8601 format.
            source_dir: the source directory holding downloaded
                MERRA 2 data.
            target_dir: a target local directory as the destination
                folder.
            subdatasets: a list of subdatasets.
            overwrite: If True, exiting files in the destination folder
                will be overwritten.

        Returns:
            Number of data downloaded. If negative value, it will be the
            number of errors encountered.
    """
    n_extract = 0
    n_error = 0
    start_datetime = dtu_parser.isoparse(start_date)
    end_datetime = dtu_parser.isoparse(end_date)

    n_days = (end_datetime - start_datetime).days+1
    if n_days >= 365:
        n_days = 366
        start_datetime = py_dt.datetime(
            year=2020, month=1, day=1)
        end_datetime = py_dt.datetime(
            year=2020, month=12, day=31,
            hour=23, minute=59, second=59)
    logger.info("To extract %d days of data with template=%s in source dir %s",
                n_days, hf_resource_temp,
                source_dir)
    
    # making target directory if necessary
    os.makedirs(target_dir, exist_ok=True)

    year_2020_day1 = py_dt.datetime(year=2020, month=1, day=1)
    hour3_delta = py_dt.timedelta(hours=3)
    for nd in range(n_days):
        ndelta = py_dt.timedelta(days=nd)
        c_day = start_datetime+ndelta
        day_of_year = c_day.timetuple().tm_yday
        doy_str = f"{day_of_year:03d}"
        # set it to start to 00:00:00 of the day
        _start_time = py_dt.datetime(
            year=c_day.year,
            month=c_day.month,
            day=c_day.day,
            hour=0,
            minute=0,
            second=0)
        year_2020_time_start = year_2020_day1 + py_dt.timedelta(days=(day_of_year-1))
        for hh in range(8):
            hour_of_day = _start_time.hour
            hod_str = f"{hour_of_day:02d}"
            hb_file_path = hf_resource_temp.replace(
                "{doy}",doy_str).replace(
                    "{hod}", hod_str)
            #filename = hb_file_path #os.path.basename(hb_file_path)
            src_full_path = _form_local_file_name_download_from_hf(
                hf_file_path=hb_file_path,
                target_local_dir=source_dir)
            if not os.path.exists(src_full_path):
                logger.warning("File '%s' is not downloaded",
                            src_full_path)
                continue
            if not subdatasets:
                all_subdatasets = gdal_get_subdatasets(
                    filename=src_full_path)
                subdatasets = [k for k in all_subdatasets]

            filename = os.path.basename(src_full_path)
            base_filename = os.path.splitext(filename)[0]
            #_start_hour = year_2020_time_start.hour
            year_2020_end = year_2020_time_start+hour3_delta-py_dt.timedelta(seconds=1)
            y2020_start_str = py_dt.datetime.strftime(year_2020_time_start,date_format)
            y2020_stop_str = py_dt.datetime.strftime(year_2020_end,date_format)
            tgt_filepath_prefix = (
                f"{base_filename}"
                f"_sd_")
            tgt_filepath_suffix = (
                f"_"
                f"{y2020_start_str}"
                f"_to_"
                f"{y2020_stop_str}"
                f".tif"
                )
            a_extracts, a_errors = gdal_export_subdatasets_to_separate_geotiffs_with_fixes(
                src_filepath=src_full_path,
                target_prefix=tgt_filepath_prefix,
                target_suffix=tgt_filepath_suffix,
                target_dir=target_dir,
                elem_names=subdatasets,
                overwrite=overwrite)
            n_error += a_errors
            n_extract += a_extracts

            _start_time += hour3_delta
            year_2020_time_start += hour3_delta


    if n_error < 0:
        return n_error
    else:
        return n_extract


def extract_subdatasets_band_from_downloaded_merra2_with_template(
        filename_temp:str,
        start_date:str,
        end_date:str,
        source_dir:str,
        target_dir:str,
        subdatasets:list[str]|None = None,
        overwrite:bool=False,
        )->int:
    """
        Download data from the GESDISC repo using a template url to 
        search between [start_date, end_date].

        Args:
            filename_temp: a filename template string that may
                contain template variable - {year}, {month},
                {day}
            start_date: a start datetime string in ISO 8601 format.
            end_date: an end datetime string in ISO 8601 format.
            source_dir: the source directory holding downloaded
                MERRA 2 data.
            target_dir: a target local directory as the destination
                folder.
            subdatasets: a list of subdatasets.
            overwrite: If True, exiting files in the destination folder
                will be overwritten.

        Returns:
            Number of data downloaded. If negative value, it will be the
            number of errors encountered.
    """
    n_extract = 0
    n_error = 0
    start_datetime = dtu_parser.isoparse(start_date)
    end_datetime = dtu_parser.isoparse(end_date)

    n_days = (end_datetime - start_datetime).days+1
    logger.info("To extract %d days of data with template=%s in source dir %s",
                n_days, filename_temp,
                source_dir)
    
    # making target directory if necessary
    os.makedirs(target_dir, exist_ok=True)

    for nd in range(n_days):
        ndelta = py_dt.timedelta(days=nd)
        c_day = start_datetime+ndelta
        year_str = f"{c_day.year}"
        month_str = f"{c_day.month:02}"
        day_str = f"{c_day.day:02}"
        filename_temp2 = _get_updated_merra2_template(
            year=c_day.year, old_template=filename_temp)
        filename = filename_temp2.replace(
            "{year}", year_str).replace(
                "{month}", month_str).replace(
                    "{day}", day_str)
        source_full_name = os.path.join(source_dir, filename)
        if not os.path.exists(source_full_name):
            logger.warning("File '%s' is not downloaded",
                           source_full_name)
            continue
        all_subdatasets = gdal_get_subdatasets(
            filename=source_full_name)
        if not subdatasets:
            subdatasets = [k for k in all_subdatasets]
        target_basename = os.path.splitext(
            os.path.basename(source_full_name))[0]
        # extract bands as separate files
        for key in subdatasets:
            subdataset_name = all_subdatasets[key][0]
            if gdal_export_subdataset_to_separate_geotiff(
                subdataset_name=subdataset_name,
                elem_name=key,
                target_basename=target_basename,
                target_dir=target_dir,
                overwrite=overwrite):
                n_extract += 1
            else:
                n_error -= 1


    if n_error < 0:
        return n_error
    else:
        return n_extract



def _get_updated_merra2_template(
        year:int,
        old_template:str)->str:
    """
        Get the proper template by replacing {YYYY} parameter
        Current range: {"before 1991": "100", "1992 to 2000": "200",
                        "2001-2010": "300", "after 2011": "400"

        Args:
            year: The year to determine the digit.
            old_template: Previous version of template with {YYYY} parameter.

        Returns:
            An updated template.
    """
    if year < 1992:
        return old_template.replace("{YYYY}", "1")
    elif year < 2001:
        return old_template.replace("{YYYY}", "2")
    elif year < 2011:
        return old_template.replace("{YYYY}", "3")
    return old_template.replace("{YYYY}", "4")


def extract_3d_subdatasets_band_from_downloaded_merra2_with_template(
        filename_temp:str,
        start_date:str,
        end_date:str,
        source_dir:str,
        target_dir:str,
        subdatasets:list[str]|None = None,
        levels:list[float] | None = None,
        overwrite:bool=False,
        )->int:
    """
        Download data from the GESDISC repo using a template url to 
        search between [start_date, end_date].

        Args:
            filename_temp: a filename template string that may
                contain template variable - {year}, {month},
                {day}
            start_date: a start datetime string in ISO 8601 format.
            end_date: an end datetime string in ISO 8601 format.
            source_dir: the source directory holding downloaded
                MERRA 2 data.
            target_dir: a target local directory as the destination
                folder.
            subdatasets: a list of subdatasets.
            overwrite: If True, exiting files in the destination folder
                will be overwritten.

        Returns:
            Number of data downloaded. If negative value, it will be the
            number of errors encountered.
    """
    n_extract = 0
    n_error = 0
    start_datetime = dtu_parser.isoparse(start_date)
    end_datetime = dtu_parser.isoparse(end_date)

    n_days = (end_datetime - start_datetime).days+1
    logger.info("To extract %d days of data with template=%s in source dir %s",
                n_days, filename_temp,
                source_dir)
    
    # making target directory if necessary
    os.makedirs(target_dir, exist_ok=True)

    for nd in range(n_days):
        ndelta = py_dt.timedelta(days=nd)
        c_day = start_datetime+ndelta
        year_str = f"{c_day.year}"
        month_str = f"{c_day.month:02}"
        day_str = f"{c_day.day:02}"
        filename_temp2 = _get_updated_merra2_template(
            year=c_day.year, old_template=filename_temp)
        filename = filename_temp2.replace(
            "{year}", year_str).replace(
                "{month}", month_str).replace(
                    "{day}", day_str)
        source_full_name = os.path.join(source_dir, filename)
        if not os.path.exists(source_full_name):
            logger.warning("File '%s' is not downloaded",
                           source_full_name)
            continue
        all_subdatasets = gdal_get_subdatasets(
            filename=source_full_name)
        if not subdatasets:
            subdatasets = [k for k in all_subdatasets]
        target_basename = os.path.splitext(
            os.path.basename(source_full_name))[0]
        # extract bands as separate files
        for key in subdatasets:
            subdataset_name = all_subdatasets[key][0]
            if gdal_export_3d_subdataset_to_separate_geotiff(
                subdataset_name=subdataset_name,
                elem_name=key,
                levels=levels,
                target_basename=target_basename,
                target_dir=target_dir,
                overwrite=overwrite):
                n_extract += 1
            else:
                n_error -= 1


    if n_error < 0:
        return n_error
    else:
        return n_extract


def extract_3d_subdatasets_band_from_merra2_climatology_from_hugging_with_template(
        hf_resource_temp:str,
        start_date:str,
        end_date:str,
        source_dir:str,
        target_dir:str,
        subdatasets:list[str]|None = None,
        levels:list[float]|None = None,
        overwrite:bool=False,
        date_format:str="%Y%m%dT%H%M%S"
        )->int:
    """
        Download data from the GESDISC repo using a template url to 
        search between [start_date, end_date].

        Args:
            filename_temp: a filename template string that may
                contain template variable - {year}, {month},
                {day}
            start_date: a start datetime string in ISO 8601 format.
            end_date: an end datetime string in ISO 8601 format.
            source_dir: the source directory holding downloaded
                MERRA 2 data.
            target_dir: a target local directory as the destination
                folder.
            subdatasets: a list of subdatasets.
            overwrite: If True, exiting files in the destination folder
                will be overwritten.

        Returns:
            Number of data downloaded. If negative value, it will be the
            number of errors encountered.
    """
    n_extract = 0
    n_error = 0
    start_datetime = dtu_parser.isoparse(start_date)
    end_datetime = dtu_parser.isoparse(end_date)

    n_days = (end_datetime - start_datetime).days+1
    if n_days >= 365:
        n_days = 366
        start_datetime = py_dt.datetime(
            year=2020, month=1, day=1)
        end_datetime = py_dt.datetime(
            year=2020, month=12, day=31,
            hour=23, minute=59, second=59)
    logger.info("To extract %d days of data with template=%s in source dir %s",
                n_days, hf_resource_temp,
                source_dir)
    
    # making target directory if necessary
    os.makedirs(target_dir, exist_ok=True)

    year_2020_day1 = py_dt.datetime(year=2020, month=1, day=1)
    hour3_delta = py_dt.timedelta(hours=3)
    for nd in range(n_days):
        ndelta = py_dt.timedelta(days=nd)
        c_day = start_datetime+ndelta
        day_of_year = c_day.timetuple().tm_yday
        doy_str = f"{day_of_year:03d}"
        # set it to start to 00:00:00 of the day
        _start_time = py_dt.datetime(
            year=c_day.year,
            month=c_day.month,
            day=c_day.day,
            hour=0,
            minute=0,
            second=0)
        year_2020_time_start = year_2020_day1 + py_dt.timedelta(days=(day_of_year-1))
        for hh in range(8):
            hour_of_day = _start_time.hour
            hod_str = f"{hour_of_day:02d}"
            hb_file_path = hf_resource_temp.replace(
                "{doy}",doy_str).replace(
                    "{hod}", hod_str)
            #filename = hb_file_path #os.path.basename(hb_file_path)
            src_full_path = _form_local_file_name_download_from_hf(
                hf_file_path=hb_file_path,
                target_local_dir=source_dir)
            if not os.path.exists(src_full_path):
                logger.warning("File '%s' is not downloaded",
                            src_full_path)
                continue
            if not subdatasets:
                all_subdatasets = gdal_get_subdatasets(
                    filename=src_full_path)
                subdatasets = [k for k in all_subdatasets]

            filename = os.path.basename(src_full_path)
            base_filename = os.path.splitext(filename)[0]
            #_start_hour = year_2020_time_start.hour
            year_2020_end = year_2020_time_start+hour3_delta-py_dt.timedelta(seconds=1)
            y2020_start_str = py_dt.datetime.strftime(year_2020_time_start,date_format)
            y2020_stop_str = py_dt.datetime.strftime(year_2020_end,date_format)
            tgt_filepath_prefix = (
                f"{base_filename}"
                f"_sd_")
            tgt_filepath_suffix = (
                f"_"
                f"{y2020_start_str}"
                f"_to_"
                f"{y2020_stop_str}"
                f".tif"
                )
            a_extracts, a_errors = gdal_export_3d_subdatasets_to_separate_geotiffs_with_fixes(
                src_filepath=src_full_path,
                target_prefix=tgt_filepath_prefix,
                target_suffix=tgt_filepath_suffix,
                target_dir=target_dir,
                elem_names=subdatasets,
                levels=levels,
                overwrite=overwrite)
            n_error += a_errors
            n_extract += a_extracts

            _start_time += hour3_delta
            year_2020_time_start += hour3_delta

    if n_error < 0:
        return n_error
    else:
        return n_extract

