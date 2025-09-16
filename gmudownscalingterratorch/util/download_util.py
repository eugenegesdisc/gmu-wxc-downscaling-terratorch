import os
import re
import requests
import logging
import dateutil.parser
import datetime
import zipfile
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

def _extract_single_file_from_zip_file(zip_path:str, file_to_extract:str, extract_path="."):
    """
        Extracts a single file from a ZIP archive
    """
    try:
        with zipfile.ZipFile(zip_path,"r") as zip_ref:
            zip_ref.extract(file_to_extract, extract_path)
        logger.info(f"Successfully etracted '{file_to_extract}' from '{zip_path}'")
    except FileNotFoundError:
        logger.error(f"Error in ZIP file '{zip_path}' not found.")
        return
    except KeyError:
        logger.error("File '{file_to_extract}' not found in '{zip_path}")
        return
    except Exception as e:
        logger.error(f"An error occurred: {e}")

def _find_files_in_zip_file(
        zip_path:str,
        prefix:str="",
        suffix:str="")->list[str]:
    ret_list = list()
    try:
        with zipfile.ZipFile(zip_path,"r") as zip_ref:
            for file_name in zip_ref.namelist():
                if (file_name.endswith(suffix)
                    and file_name.startswith(prefix)):
                    ret_list.append(file_name)
    except FileNotFoundError:
        logger.error(f"Error in ZIP file '{zip_path}' not found.")
        return
    except KeyError:
        logger.error("File '{file_to_extract}' not found in '{zip_path}")
        return
    except Exception as e:
        logger.error(f"An error occurred: {e}")
    return ret_list

def _download_binary_file(
        url:str,
        filepath:str):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status() # Rise HTTPError for bad responses (4xx or 5xx)

        with open(filepath, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
    except requests.exceptions.RequestException as e:
        logger.error(f"REquest failed {e}")
    except IOError as e:
        logger.error(f"IO error occurred: {e}")
    except Exception as e:
        logger.error(f"An unexcpeted error occurred: {e}")

def _get_service_url_for_prism(
        region:str="us",
        res:str="800m",
        element:str="ppt",
        date:str="19810101",
        format:str=None)->str:
    if (format in ["nc","asc","bil"]):
        ret_str = f"https://services.nacse.org/prism/data/get/{region}/{res}/{element}/{date}?format={format}"
    else:
        ret_str = f"https://services.nacse.org/prism/data/get/{region}/{res}/{element}/{date}"
    return ret_str

def _get_bulk_download_url_for_prism(
        region:str="us",
        resolution:str="800m",
        element:str="ppt",
        period:str="daily",
        year:str="1981",
        format:str="tif")->str:
    if format == "bil":
        ret_str = f"https://data.prism.oregonstate.edu/{period}/{element}/{year}"
    else:
        ret_str = f"https://data.prism.oregonstate.edu/time_series/{region}/an/{resolution}/{element}/{period}/{year}"
    return ret_str

def _extract_links_from_a_webpage(
        url:str,
        suffix:str=".zip",
        prefix:str="")->list[str]:
    ret_list=list()
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        links = soup.find_all("a")
        _suffix = ""
        if suffix:
            _suffix = suffix
        _prefix = ""
        if prefix:
            _prefix = prefix

        for link in links:
            href = link.get("href")
            if (href and
                href.endswith(_suffix) and
                href.startswith(_prefix) and
                (href not in ret_list)):
                ret_list.append(href)
        return ret_list
    except requests.exceptions.RequestException as e:
        logger.error(f"Error featching URL: {e}")
        return ret_list
    except Exception as e:
        logger.error(f"An error occurred: {e}")
    return ret_list    

def download_file_use_prism_service(
        targetpath:str=".",
        region:str="us",
        resolution:str="800m",
        element:str="ppt",
        date:str="19800101",
        format:str=None,
        overwrite:bool=False):
    zip_url = _get_service_url_for_prism(
        region=region,
        res=resolution,
        element=element,
        date=date,
        format=format)
    zipfile = f"prism_{element}_{region}_{resolution}_{date}.zip"
    local_path_base = targetpath
    local_path_full = os.path.join(local_path_base, zipfile)
    if (os.path.exists(local_path_full)
        and (not overwrite)):
        logger.info(f"Target {local_path_full} exists. Not downloaded FROM {zip_url}")
    else:
        logger.info(f"Download FROM {zip_url} TO {local_path_full}")
        _download_binary_file(url=zip_url,filepath=local_path_full)
    print(_find_files_in_zip_file(local_path_full))

def _filter_filenames_by_date_range(
        filename_list:list[str],
        date_reg:str=None,
        date_format:str=None,
        iso_date_begin:str=None,
        iso_date_end:str=None)->list[str]:
    if not date_reg:
        return filename_list
    if ((not iso_date_begin) and
        (not iso_date_end)):
        return filename_list
    date_begin = dateutil.parser.isoparse(iso_date_begin)
    date_end = dateutil.parser.isoparse(iso_date_end)

def _extract_date_from_filename(filename, date_pattern):
    """
    Extracts a date from a filename using a regular expression with a named group.

    Args:
        filename (str): The name of the file.
        date_pattern (str): The regular expression pattern to match the date, 
                           including a named group for the date 
                           (e.g., r"(?P<date>\\d{8})").

    Returns:
        str: The extracted date string, or None if no match is found.
    """
    match = re.search(date_pattern, filename)
    if match:
        return match.group("date")
    else:
        return None
def _convert_string_to_datetime(
    date_str:str,
    date_format:str=None):
    """
        if date_format is given, datetime will be converted using strptime.
        Otherwise, assumeing iso format. and using dateutil.parser.isoparser.
    """
    if not date_str:
        return None
    if date_format:
        return datetime.datetime.strptime(date_str, date_format)
    return dateutil.parser.isoparse(date_str)

def _filter_filenames_after_a_date(
        filename_list:list[str],
        date_reg:str=None,
        date_format:str=None,
        date_begin:datetime.datetime=None):
    if not date_begin:
        return filename_list
    ret_list = list()
    for fname in filename_list:
        dt = _convert_string_to_datetime(
            date_str=_extract_date_from_filename(
                filename=fname,
                date_pattern=date_reg),
            date_format=date_format)
        if (dt and
            dt < date_begin):
            continue
        ret_list.append(fname)
    return ret_list


def _filter_filenames_before_a_date(
        filename_list:list[str],
        date_reg:str=None,
        date_format:str=None,
        date_end:datetime.datetime=None):
    if not date_end:
        return filename_list
    ret_list = list()
    for fname in filename_list:
        dt = _convert_string_to_datetime(
            date_str=_extract_date_from_filename(
                filename=fname,
                date_pattern=date_reg),
            date_format=date_format)
        if (dt and
            dt > date_end):
            continue
        ret_list.append(fname)
    return ret_list



def bulk_download_files_from_prism(
        region:str="us",
        resolution:str="800m",
        element:str="ppt",
        period="daily",
        year="1981",
        format:str=None,
        prefix:str=None,
        suffix:str=None,
        startindex:int=0,
        count:int=-1,
        targetpath=".",
        date_reg:str=None,
        date_format:str=None,
        date_filter_begin:str=None,
        date_filter_end:str=None,
        overwrite:bool=False):
    blk_url = _get_bulk_download_url_for_prism(
        region=region,
        resolution=resolution,
        element=element,
        period=period,
        year=year,
        format=format)
    links = _extract_links_from_a_webpage(
        url=blk_url,
        prefix=prefix,
        suffix=suffix)
    links = _filter_filenames_before_a_date(
        filename_list=links,
        date_reg=date_reg,
        date_format=date_format,
        date_end=_convert_string_to_datetime(date_filter_end))
    
    links = _filter_filenames_after_a_date(
        filename_list=links,
        date_reg=date_reg,
        date_format=date_format,
        date_begin=_convert_string_to_datetime(date_filter_begin))

    max_ret = len(links)
    start_idx = 0
    if startindex > 0:
        start_idx = startindex
        max_ret -= startindex
    if count > 0:
        max_ret = count
    
    base_url = blk_url
    if not base_url.endswith("/"):
        base_url = base_url + "/"
    local_path_base = targetpath
    _loop_idx = 0
    for zipfile in links:
        if _loop_idx >= start_idx:
            zf_url = base_url + zipfile
            local_path_full = os.path.join(
                local_path_base,
                zipfile)
            if (os.path.exists(local_path_full)
                and (not overwrite)):
                logger.info(f"Target {local_path_full} exists. Not downloaded FROM {zf_url}")
            else:
                f_path = os.path.abspath(local_path_full)
                logger.info(f"Download FROM {zf_url} TO {local_path_full}")
                _download_binary_file(
                    url=zf_url,
                    filepath=local_path_full)
            for in_file in _find_files_in_zip_file(local_path_full):
                the_full_path = os.path.join(local_path_base, in_file)
                if (os.path.exists(the_full_path)
                    and (not overwrite)):
                    logger.info(f"Target {the_full_path} exists. Not extracted FROM {local_path_full}")
                else:
                    _extract_single_file_from_zip_file(
                        local_path_full,in_file,local_path_base)
        _loop_idx += 1
        if ((_loop_idx - start_idx) >= max_ret):
            break

def prism_bulk_download_files(
        region:str="us",
        resolution:str="800m",
        element:str="ppt",
        period="daily",
        format:str=None,
        prefix:str=None,
        suffix:str=None,
        startindex:int=0,
        count:int=-1,
        targetpath=".",
        date_reg:str=None,
        date_format:str=None,
        date_filter_begin:str=None,
        date_filter_end:str=None,
        overwrite:bool=False):
    """
        blk_url should be a template with "{year}" to be replaced.
        If in one year, blk_url should not have {year} in the template.
    """
    if (date_filter_begin and date_filter_end):
        start_date = dateutil.parser.isoparse(date_filter_begin)
        end_date = dateutil.parser.isoparse(date_filter_end)
        years = _calculate_years_between_dates(
            start_date=start_date, end_date=end_date)
        year_str = f"{start_date.year}"
        if years == 0:
            bulk_download_files_from_prism(
                region=region,
                resolution=resolution,
                element=element,
                period=period,
                year=year_str,
                format=format,
                prefix=prefix,
                suffix=suffix,
                startindex=startindex,
                count=count,
                targetpath=targetpath,
                date_reg=date_reg,
                date_format=date_format,
                date_filter_begin=date_filter_begin,
                date_filter_end=date_filter_end,
                overwrite=overwrite)
            return
        else:
            year_idx = start_date.year
            start_date_idx = start_date
            for yr in range(years):
                cyear = year_idx+yr
                cyear_str = f"{cyear}"
                if yr == 0:
                    bulk_download_files_from_prism(
                        region=region,
                        resolution=resolution,
                        element=element,
                        period=period,
                        year=cyear_str,
                        format=format,
                        prefix=prefix,
                        suffix=suffix,
                        startindex=startindex,
                        count=count,
                        targetpath=targetpath,
                        date_reg=date_reg,
                        date_format=date_format,
                        date_filter_begin=date_filter_begin,
                        date_filter_end=None,
                        overwrite=overwrite)
                else:
                    bulk_download_files_from_prism(
                        region=region,
                        resolution=resolution,
                        element=element,
                        period=period,
                        year=cyear_str,
                        format=format,
                        prefix=prefix,
                        suffix=suffix,
                        startindex=startindex,
                        count=count,
                        targetpath=targetpath,
                        date_reg=date_reg,
                        date_format=date_format,
                        date_filter_begin=None,
                        date_filter_end=None,
                        overwrite=overwrite)
            cyear = end_date.year
            cyear_str = f"{cyear}"
            bulk_download_files_from_prism(
                region=region,
                resolution=resolution,
                element=element,
                period=period,
                year=cyear_str,
                format=format,
                prefix=prefix,
                suffix=suffix,
                startindex=startindex,
                count=count,
                targetpath=targetpath,
                date_reg=date_reg,
                date_format=date_format,
                date_filter_begin=None,
                date_filter_end=date_filter_end,
                overwrite=overwrite)


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
    def __init__(self, token):
        super().__init__()
        self.token = token

    def prepare_request(self, request):
        request.headers['Authorization'] = f'Bearer {self.token}'
        return super().prepare_request(request)
    
    def rebuild_auth(self,
                     prepared_request: requests.PreparedRequest,
                     response):
        prepared_request.headers['Authorization'] = f'Bearer {self.token}'
        return

def _download_binary_file_with_authentication(
        url:str,
        filepath:str,
        username:str=None,
        password:str=None,
        edl_token:str=None)->bool:
    if edl_token:
        session = TokenAuthSessionWithRedirection(token=edl_token)
    else:
        session = SessionWithHeaderRedirection(
            username=username,
            password=password)
    try:
        response = session.get(url, stream=True)
        response.raise_for_status()

        with open(filepath, "wb") as file:
            for chunk in response.iter_content(
                chunk_size=8192):
                file.write(chunk)
        return True
    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed: {e}")
        return False
    except IOError as e:
        logger.error(f"IOError occurred: {e}")
        return False
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        return False


def bulk_download_files_on_a_page(
        blk_url:str,
        username:str=None,
        password:str=None,
        edl_token:str=None,
        prefix:str=None,
        suffix:str=None,
        startindex:int=0,
        count:int=-1,
        targetpath=".",
        date_reg:str=None,
        date_format:str=None,
        date_filter_begin:str=None,
        date_filter_end:str=None,
        overwrite:bool=False):
    _prefix = ""
    if prefix:
        _prefix = prefix
    _suffix = ""
    if suffix:
        _suffix = suffix
    
    links = _extract_links_from_a_webpage(
        url=blk_url,
        prefix=_prefix,
        suffix=_suffix)
    links = _filter_filenames_before_a_date(
        filename_list=links,
        date_reg=date_reg,
        date_format=date_format,
        date_end=_convert_string_to_datetime(date_filter_end))
    
    links = _filter_filenames_after_a_date(
        filename_list=links,
        date_reg=date_reg,
        date_format=date_format,
        date_end=_convert_string_to_datetime(date_filter_begin))

    max_ret = len(links)
    start_idx = 0
    if startindex > 0:
        start_idx = startindex
        max_ret -= startindex
    if count > 0:
        max_ret = count
    
    bsae_url = blk_url
    if not base_url.endswith("/"):
        base_url = base_url + "/"
    local_path_base = targetpath
    _loop_idx = 0
    for zipfile in links:
        if _loop_idx >= start_idx:
            zf_url = base_url + zipfile
            local_path_full = os.path.join(
                local_path_base,
                zipfile)
            if (os.path.exists(local_path_full)
                and (not overwrite)):
                logger.info(f"Target {local_path_full} exists. Not downloaded FROM {zf_url}")
            else:
                logger.info(f"Download FROM {zf_url} TO {local_path_full}")
                if (((not username) or (not password))
                    and (not edl_token)): 
                    _download_binary_file(
                        url=zf_url,
                        filepath=local_path_full)
                else:
                    _download_binary_file_with_authentication(
                        url=zf_url,
                        filepath=local_path_full,
                        username=username,
                        password=password,
                        edl_token=edl_token)
        _loop_idx += 1
        if ((_loop_idx - start_idx) >= max_ret):
            break

def years_bulk_download_files(
        blk_url:str,
        username:str=None,
        password:str=None,
        edl_token:str=None,
        prefix:str=None,
        suffix:str=None,
        startindex:int=0,
        count:int=-1,
        targetpath=".",
        date_reg:str=None,
        date_format:str=None,
        date_filter_begin:str=None,
        date_filter_end:str=None,
        overwrite:bool=False):
    """
        blk_url should be a template with "{year}" to be replaced.
        If in one year, blk_url should not have {year} in the template.
    """
    if (date_filter_begin and date_filter_end):
        start_date = dateutil.parser.isoparse(date_filter_begin)
        end_date = dateutil.parser.isoparse(date_filter_end)
        years = _calculate_years_between_dates(
            start_date=start_date, end_date=end_date)
        if years == 0:
            bulk_download_files_on_a_page(
                blk_url=blk_url,
                username=username,
                password=password,
                edl_token=edl_token,
                prefix=prefix,
                suffix=suffix,
                startindex=startindex,
                count=count,
                targetpath=targetpath,
                date_reg=date_reg,
                date_format=date_format,
                date_filter_begin=date_filter_begin,
                date_filter_end=date_filter_end,
                overwrite=overwrite)
            return
        else:
            year_idx = start_date.year
            start_date_idx = start_date
            for yr in range(years):
                new_blk_url = blk_url
                cyear = year_idx+yr
                modified_blk_url = new_blk_url.replace("{year}", f"{cyear}")
                if yr == 0:
                    bulk_download_files_on_a_page(
                        blk_url=modified_blk_url,
                        username=username,
                        password=password,
                        edl_token=edl_token,
                        prefix=prefix,
                        suffix=suffix,
                        startindex=startindex,
                        count=count,
                        targetpath=targetpath,
                        date_reg=date_reg,
                        date_format=date_format,
                        date_filter_begin=date_filter_begin,
                        date_filter_end=None,
                        overwrite=overwrite)                
                else:
                    bulk_download_files_on_a_page(
                        blk_url=modified_blk_url,
                        username=username,
                        password=password,
                        edl_token=edl_token,
                        prefix=prefix,
                        suffix=suffix,
                        startindex=startindex,
                        count=count,
                        targetpath=targetpath,
                        date_reg=date_reg,
                        date_format=date_format,
                        date_filter_begin=None,
                        date_filter_end=None,
                        overwrite=overwrite)                
            new_blk_url = blk_url
            cyear = end_date.year
            modified_blk_url = new_blk_url.replace("{year}", f"{cyear}")
            bulk_download_files_on_a_page(
                blk_url=modified_blk_url,
                username=username,
                password=password,
                edl_token=edl_token,
                prefix=prefix,
                suffix=suffix,
                startindex=startindex,
                count=count,
                targetpath=targetpath,
                date_reg=date_reg,
                date_format=date_format,
                date_filter_begin=None,
                date_filter_end=date_filter_end,
                overwrite=overwrite)

    
def _calculate_years_between_dates(start_date, end_date):
    """
    Calculates the number of years between two dates.

    Args:
        start_date (date): The starting date.
        end_date (date): The ending date.

    Returns:
        int: The number of years between the two dates.
    """
    years = end_date.year - start_date.year
    #if (end_date.month, end_date.day) < (start_date.month, start_date.day):
    #    years -= 1
    return years