import os
import re
#import dateutil.relativedelta
import requests
import logging
import datetime
#import dateutil.relativedelta
import dateutil.parser
import zipfile

logger = logging.getLogger(__name__)

def _extract_single_file_from_zip_file(
        zip_file_path:str,
        file_to_extract:str,
        target_dir:str=".")->bool:
    """
        Extract a single file from a zip file.

        Args:
            zip_file_path: the full filename path to a zip file.
            file_to_extract: the file with path to be extracted from
                a zip file.
            target_dir: A target directory for the extracted file.
        
        Returns:
            A boolean value indicates the status of the extraction. Farse could be 
            caused by several failures - file not found, no element found in zip file.
    """
    try:
        with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
            zip_ref.extract(file_to_extract,target_dir)
        return True
    except FileNotFoundError:
        logger.error("Zip file '%s' not found.", zip_file_path, exc_info=True)
    except KeyError:
        logger.error("Key '%s' is not found in zip file %s.",
                     file_to_extract, zip_file_path, exc_info=True)
    except Exception as e:
        logger.error("An unexpteced eorr: %s", e, exc_info=True)
    return False

def _string_endswith_regex(
        textstr:str, patterns:str|list[str])->bool:
    """
        Match any of the regular expressions in a text string from the end.

        Args:
            textstr: The text to be searched.
            patterns: An regular expression or a list of regular 
                expressions.

        Returns:
            True - any match found. False - No match pattern found.
    """
    if patterns is None:
        patterns = ""

    if isinstance(patterns, list):
        reg_ex = ""
        for p in patterns:
            reg_ex = reg_ex + '|' + p + '$'
        if len(reg_ex) > 1:
            reg_ex = reg_ex[1:]
    else:
        reg_ex = patterns + "$"
    
    return bool(re.search(pattern=reg_ex,string=textstr))

def _string_startswith_regex(
        textstr:str, patterns:str|list[str])->bool:
    """
        Match any of the regular expressions in a text string from the front.

        Args:
            textstr: The text to be searched.
            patterns: A regular expression or a list of regular 
                expressions.

        Returns:
            True - any match found. False - No match pattern found.
    """
    if patterns is None:
        patterns = ""

    if isinstance(patterns, list):
        reg_ex = ""
        for p in patterns:
            reg_ex = reg_ex + '|^' + p
        if len(reg_ex) > 1:
            reg_ex = reg_ex[1:]
    else:
        reg_ex = "^" + patterns
    
    return bool(re.search(pattern=reg_ex,string=textstr))

def _find_files_in_zip_file(
        zip_file_path:str,
        prefix_reg_patterns:str|list[str]="",
        suffix_reg_patterns:str|list[str]="")->list[str]|None:
    """
        List all files match any of the regular expressions at the end
        or at the beginning.

        Args:
            zip_file_path: the full filename path for zip file.
            prefix_reg_patterns: a regular expression or a list of regular
                expressions to be matched at the beginning.
            suffix_reg_patterns: a regular expression or a list of regular'
                expressions to be matched at the end.
        
        Returns:
            True - any match found. False - none of the expressions 
            matches.
    """
    try:
        ret_list = list()
        with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
            for file_name in zip_ref.namelist():
                if ( _string_startswith_regex(
                    textstr=file_name, patterns=prefix_reg_patterns) and
                    _string_endswith_regex(
                        textstr=file_name, patterns=suffix_reg_patterns)):
                    ret_list.append(file_name)
        return ret_list
    except FileNotFoundError:
        logger.error("Zip file '%s' not found.", zip_file_path, exc_info=True)
    except KeyError:
        logger.error("In zip file '%s': no match found with "
                     "prefix '%s' and suffix '%s'",
                     zip_file_path, prefix_reg_patterns, 
                     suffix_reg_patterns,
                     exc_info=True)
    except Exception as e:
        logger.error("An unexpected error: %s",e, exc_info=True)

def _download_binary_file(
        url_str:str,
        target_file_path:str)->bool:
    """
        Download a binary file.

        Args:
            url_str: The source URL
            target_file_path: The full path name for the destination
                file.
        
        Returns:
            If True, successed. If False, an error occurred.
    """
    try:
        response = requests.get(url_str, stream=True)
        response.raise_for_status()
        chunk_size = 1024*1024
        with open(target_file_path, "wb") as file:
            for chunk in response.iter_content(
                chunk_size=chunk_size):
                file.write(chunk)
        return True
    except requests.exceptions.RequestException as e:
        logger.error("Request failed: %s", e)
    except IOError as e:
        logger.error("IO error occurred: %s", e)
    except Exception as e:
        logger.error("An unexpected error ocurred: %s", e)
    return False

def download_data_prism_with_template(
        url_temp:str,
        start_date:str,
        end_date:str,
        target_dir:str,
        overwrite:bool=False,
        )->int:
    """
        Download data from the PRISM repo using a template url to 
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

        Returns:
            Number of data downloaded. If negative value, it will be the
            number of errors encountered.
    """
    n_down = 0
    n_error = 0
    start_datetime = dateutil.parser.isoparse(start_date)
    end_datetime = dateutil.parser.isoparse(end_date)

    n_days = (end_datetime - start_datetime).days+1
    logger.info("To download %d days of data with template=%s",
                n_days, url_temp)
    
    # making target directory if necessary
    os.makedirs(target_dir, exist_ok=True)

    for nd in range(n_days):
        ndelta = datetime.timedelta(days=nd)
        c_day = start_datetime+ndelta
        year_str = f"{c_day.year}"
        month_str = f"{c_day.month:02}"
        day_str = f"{c_day.day:02}"
        url_str = url_temp.replace(
            "{year}", year_str).replace(
                "{month}", month_str).replace(
                    "{day}", day_str)
        filename = url_str.rsplit("/", 1)[1]
        target_full_name = os.path.join(target_dir, filename)

        if ((not os.path.exists(target_full_name))
             or overwrite):
            if _download_binary_file(
                url_str=url_str,
                target_file_path=target_full_name):
                n_down +=1
            else:
                n_error -= 1

    if n_error < 0:
        return n_error
    return n_down

def extract_files_from_downloaded_prism_with_template(
        filename_temp:str,
        start_date:str,
        end_date:str,
        source_dir:str,
        target_dir:str,
        endings:list[str]|None=None,
        overwrite:bool=False,
        )->int:
    """
        Extract data from the downloaded PRISM data files
        matching a template between [start_date, end_date]

        Args:
            filename_temp: a filename template string that may
                contain template variable - {year}, {month},
                {day}/
            start_date: a start datetime string in ISO 8601 format.
            end_date: an end datetime string in ISO 8601 format.
            source_dir: the source directory holding downloaded
                PRISM data in zip.
            target_dir: the target directory to hold the extracted
                files.
            endings: a list of endings for the files to be extracted
                from the zip files.
            overwrite: If True, existing files in the target directory
                will be overwritten.
        
        Returns:
            Number of files extracted. If negative value, it indicates
            the number of errors encountered.
    """
    n_extract = 0
    n_error = 0
    start_datetime = dateutil.parser.isoparse(start_date)
    end_datetime = dateutil.parser.isoparse(end_date)

    n_days = (end_datetime - start_datetime).days+1
    logger.info("To extract %d days of data with template=%s",
                n_days, filename_temp)
    
    # making target directory if necessary
    os.makedirs(target_dir, exist_ok=True)

    for nd in range(n_days):
        ndelta = datetime.timedelta(days=nd)
        c_day = start_datetime+ndelta
        year_str = f"{c_day.year}"
        month_str = f"{c_day.month:02}"
        day_str = f"{c_day.day:02}"
        filename_str = filename_temp.replace(
            "{year}", year_str).replace(
                "{month}", month_str).replace(
                    "{day}", day_str)
        full_name_in_download = os.path.join(source_dir, filename_str)

        if not os.path.exists(full_name_in_download):
            logger.warning("File '%s' is not downloaded into '%s'",
                        full_name_in_download, source_dir)
            continue

        # filename without extension
        #filename_base = os.path.splitext(filename_str)[0]

        extract_files = _find_files_in_zip_file(
            zip_file_path=full_name_in_download,
            suffix_reg_patterns=endings)
        
        # extract files
        for efile in extract_files:
            target_full_path = os.path.join(target_dir, efile)
            if ((not os.path.exists(target_full_path))
                or overwrite):
                if _extract_single_file_from_zip_file(
                    zip_file_path=full_name_in_download,
                    file_to_extract=efile,
                    target_dir=target_dir):
                    n_extract += 1
                else:
                    n_error -= 1
        
    if n_error < 0:
        return n_error
    return n_extract

