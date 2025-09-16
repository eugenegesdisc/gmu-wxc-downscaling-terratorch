"""Test routines in gmudownscalingterratorch.util
    prism_download_util
"""
import logging
import pytest

from gmudownscalingterratorch.util.decorators import (
    decorator_prism_exceptions_to_logger,
    decorator_timeit_to_logger
)

from gmudownscalingterratorch.util import (
    download_data_prism_with_template,
    extract_files_from_downloaded_prism_with_template)

logger = logging.getLogger(__name__)

@decorator_timeit_to_logger(logger=logger)
@decorator_prism_exceptions_to_logger(logger=logger)
def deco_test_download_data_prism_with_template():
    """
    Download - (8.6mb, 9mb)
    https://data.prism.oregonstate.edu/time_series/us/an/800m/ppt/daily/1981/prism_ppt_us_30s_19810101.zip
    https://data.prism.oregonstate.edu/time_series/us/an/800m/ppt/daily/1981/prism_ppt_us_30s_19810102.zip
    """
    tt = download_data_prism_with_template(
        url_temp=("https://data.prism.oregonstate.edu/"
                  "time_series/us/an/800m/ppt/daily/{year}/"
                  "prism_ppt_us_30s_{year}{month}{day}.zip"),
        start_date="1981-01-01",
        end_date="1981-01-02",
        target_dir="../exp/data4test/download")
    if tt >= 0:
        return True
    return False

@pytest.mark.option1
def test_download_data_prism_with_template():
    result = deco_test_download_data_prism_with_template()
    assert result


@decorator_timeit_to_logger(logger=logger)
@decorator_prism_exceptions_to_logger(logger=logger)
def deco_test_extract_files_from_downloaded_prism_with_template():
    """
    Extract files with extensions [.tif, .prj]
        prism_ppt_us_30s_19810101.zip
        prism_ppt_us_30s_19810102.zip
    """
    tt = extract_files_from_downloaded_prism_with_template(
        filename_temp="prism_ppt_us_30s_{year}{month}{day}.zip",
        start_date="1981-01-01",
        end_date="1981-01-02",
        source_dir="../exp/data4test/download",
        target_dir="../exp/data4test/prism",
        endings=[r"\.tif",r"\.prj"])

    if tt >= 0:
        return True
    return False

@pytest.mark.option1
def test_extract_files_from_downloaded_prism_with_template():
    result = deco_test_extract_files_from_downloaded_prism_with_template()
    assert result
