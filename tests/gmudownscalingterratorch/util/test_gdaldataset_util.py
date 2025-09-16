import logging
import pytest
from gmudownscalingterratorch.util import(
    gdal_get_subdatasets,
    gdal_get_subdataset_timebands,
    gdal_export_subdataset_to_separate_geotiff
)
from gmudownscalingterratorch.util.decorators import (
    decorator_timeit_to_logger
)

logger = logging.getLogger(__name__)

@pytest.mark.option1
def test_gdal_get_subdatasets():
    tt = gdal_get_subdatasets(
        filename="../exp/data4test/download/MERRA2_100.inst1_2d_asm_Nx.19810102.nc4"
    )
    logger.info("subdatasets %s",
                tt)
    assert True

@pytest.mark.option1
def test_gdal_get_subdataset_timebands():
    tt = gdal_get_subdataset_timebands(
        subdataset_name='NETCDF:"../exp/data4test/download/MERRA2_100.inst1_2d_asm_Nx.19810102.nc4":V50M')
    logger.info("subdatasets %s",
                tt)
    assert True

@pytest.mark.option1
def test_gdal_export_subdataset_to_seaparate_geotiff():
    tt = gdal_export_subdataset_to_separate_geotiff(
        subdataset_name='NETCDF:"../exp/data4test/download/MERRA2_100.inst1_2d_asm_Nx.19810102.nc4":T2M',
        elem_name="T2M",
        target_basename="MERRA2_100.inst1_2d_asm_Nx.19810102",
        target_dir="../exp/data4test/merra2",
        overwrite=True)
    
    assert True