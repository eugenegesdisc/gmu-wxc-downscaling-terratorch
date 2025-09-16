"""
    Module to test Merra 2 data downloading routine
"""
import logging
import pytest

from gmudownscalingterratorch.util.decorators import(
    decorator_merra2_exceptions_to_logger,
    decorator_timeit_to_logger
)
from gmudownscalingterratorch.util import (
    download_data_merra2_with_template,
    extract_subdatasets_band_from_downloaded_merra2_with_template
    )

logger = logging.getLogger(__name__)


@decorator_timeit_to_logger(logger=logger)
@decorator_merra2_exceptions_to_logger(logger=logger)
def deco_test_download_data_merra2_with_template()->bool:
    """
    Download - (197mb)
https://goldsmr4.gesdisc.eosdis.nasa.gov/data/MERRA2/M2I1NXASM.5.12.4/1981/01/MERRA2_100.inst1_2d_asm_Nx.19810101.nc4

    """
    tt = download_data_merra2_with_template(
        url_temp=("https://goldsmr4.gesdisc.eosdis.nasa.gov/"
                  "data/MERRA2/M2I1NXASM.5.12.4/{year}/{month}/"
                  "MERRA2_100.inst1_2d_asm_Nx.{year}{month}{day}.nc4"),
        start_date="1981-01-01",
        end_date="1981-01-03",
        target_dir="../exp/data4test/download",
        username="earthdata-username",
        password="earthdata-password")
    if tt >= 0:
        return True
    return False

@pytest.mark.option1
def test_download_data_merra2_with_template_decorator():
    """
    Test download merra2 data from earthdata
    """
    result = deco_test_download_data_merra2_with_template()
    assert result


@decorator_timeit_to_logger(logger=logger)
@decorator_merra2_exceptions_to_logger(logger=logger)
def deco_test_extract_subdatasets_band_from_downloaded_merra2_with_template()->bool:
    """
    Test extrracting bands as seaparate GeoTIFF files
    """
    tt = extract_subdatasets_band_from_downloaded_merra2_with_template(
        filename_temp="MERRA2_100.inst1_2d_asm_Nx.{year}{month}{day}.nc4",
        start_date="1981-01-01",
        end_date="1981-01-02",
        source_dir="../exp/data4test/download",
        target_dir="../exp/data4test/merra2",
        subdatasets=["T2M"]
        )
    if tt >= 0:
        return True
    return False

@pytest.mark.option1
def test_extract_subdatasets_band_from_downloaded_merra2_with_template():
    """
    Test extract_subdatasets_band_from_downloaded_merra2_with_template
    """
    result = deco_test_extract_subdatasets_band_from_downloaded_merra2_with_template()
    assert result
