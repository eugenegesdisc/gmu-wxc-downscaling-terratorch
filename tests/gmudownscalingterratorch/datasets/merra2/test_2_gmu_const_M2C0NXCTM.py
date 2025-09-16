"""
Module for testing GmuMerra2M2CONXCTM dataset
"""
import logging
import pytest
from dateutil import parser as dtu_parser
from torchgeo.datasets import BoundingBox

from gmudownscalingterratorch.datasets.merra2 import (
    GmuMerra2M2C0NXCTM)

from gmudownscalingterratorch.util.decorators import (
    decorator_merra2_exceptions_to_logger,
    decorator_timeit_to_logger)

logger = logging.getLogger(__name__)

class DecoTest2GmuMerra2M2C0NXCTM:
    """
    Test GmuMerra2M2C0NXCTM dataset
    """
    @decorator_timeit_to_logger(logger=logger)
    @decorator_merra2_exceptions_to_logger(logger=logger)
    def deco_test_download(self)->bool:
        """
        Test download
        """
        ds = GmuMerra2M2C0NXCTM(
            paths="../exp/m2c0nxctm/merra2",
            download=True,
            username="earthdata-username",
            password="earthdata-password",
            target_download_dir="../exp/m2c0nxctm/download")
        if not ds:
            return False
        return True

    @decorator_timeit_to_logger(logger=logger)
    @decorator_merra2_exceptions_to_logger(logger=logger)
    def deco_test_extract(self)->bool:
        """
        Test extract
        """
        ds = GmuMerra2M2C0NXCTM(
            paths="../exp/m2c0nxctm/merra2",
            download=False,
            target_download_dir="../exp/m2c0nxctm/download",
            #bands=["T2M"],
            overwrite=False,
            extract=True)
        if not ds:
            return False
        return True

    @decorator_timeit_to_logger(logger=logger)
    @decorator_merra2_exceptions_to_logger(logger=logger)
    def deco_test_super_init(self)->bool:
        """
        Test super().__init__
        """
        ds = GmuMerra2M2C0NXCTM(
            paths="../exp/m2c0nxctm/merra2",
            download=False,
            target_download_dir="../exp/m2c0nxctm/download",
            #bands=["T2M"],
            extract=False)
        if not ds:
            return False
        return True

    @decorator_timeit_to_logger(logger=logger)
    @decorator_merra2_exceptions_to_logger(logger=logger)
    def deco_test_getitem(self)->bool:
        """
        Test super().__getitem__
        """
        ds = GmuMerra2M2C0NXCTM(
            paths="../exp/m2c0nxctm/merra2",
            download=False,
            target_download_dir="../exp/m2c0nxctm/download",
            bands=["FRACI"],
            extract=False)
        # non-extended boundingbox may miss one pixels
        data = ds.__getitem__(
            BoundingBox(-180, 180, -90, 90,
                        dtu_parser.isoparse("2000-12-31 00:00:00").timestamp(),
                        dtu_parser.isoparse("2000-12-31 00:59:59").timestamp()))
        print("keys=", data.keys())
        print("image-shape:", data["image"].shape)
        print(data["image"][0, :, :]) # month january

        ds = GmuMerra2M2C0NXCTM(
            paths="../exp/m2c0nxctm/merra2",
            download=False,
            target_download_dir="../exp/m2c0nxctm/download",
            extract=False)
        print("resolution=", ds.res)
        # to get the full coverage of the given area
        # using extended bounding box - 
        # bbx.minx-ds.res[0]/2.0, bbx.maxx+ds.res[0]/2.0,
        # bbx.miny-ds.res[1]/2.0, bbx.maxy+ds.res[1]/2.0
        # gdal use center to represent the pixel
        data2 = ds.__getitem__(
            BoundingBox(-180-(ds.res[0]/2), 180+(ds.res[0]/2),
                        -90-(ds.res[1]/2), 90+(ds.res[1]/2),
                        dtu_parser.isoparse("2000-12-31 01:00:00").timestamp(),
                        dtu_parser.isoparse("2000-12-31 01:59:59").timestamp()))
        print("data2-image-shape:", data2["image"].shape)
        fraci_idx = ds.all_bands.index("FRACI")
        print(data2["image"][
            fraci_idx*12+0, # 12 months, index start from 0
            :,:]) #month january

        if not data:
            return False
        return True

    @decorator_timeit_to_logger(logger=logger)
    @decorator_merra2_exceptions_to_logger(logger=logger)
    def deco_test_get_len(self)->bool:
        """
        Test super().__init__
        """
        ds = GmuMerra2M2C0NXCTM(
            paths="../exp/m2c0nxctm/merra2",
            download=False,
            target_download_dir="../exp/m2c0nxctm/download",
            extract=False)
        print("len=", ds.__len__(), "number of bands=", len(ds.all_bands))
        if ds.__len__() < 1:
            return False
        return True

class Test2GmuMerra2M2C0NXCTM:
    """
    Test GmuMerra2M2C0NXCTM
    """
    @pytest.mark.option1
    def test_download(self):
        """
            Test download
        """
        dec_cls = DecoTest2GmuMerra2M2C0NXCTM()
        result = dec_cls.deco_test_download()
        assert result

    @pytest.mark.option1
    def test_extract(self):
        """
        #    Test download
        """
        dec_cls = DecoTest2GmuMerra2M2C0NXCTM()
        result = dec_cls.deco_test_extract()
        assert result

    @pytest.mark.option1
    def test_super_init(self):
        """
            Test super().__init__
        """
        dec_cls = DecoTest2GmuMerra2M2C0NXCTM()
        result = dec_cls.deco_test_super_init()
        assert result

    @pytest.mark.option1
    def test_get_len(self):
        """
            Test super().__init__
        """
        dec_cls = DecoTest2GmuMerra2M2C0NXCTM()
        result = dec_cls.deco_test_get_len()
        assert result


    @pytest.mark.option1
    def test_getitem(self):
        """
            Test super().__getitem__()
        """
        dec_cls = DecoTest2GmuMerra2M2C0NXCTM()
        result = dec_cls.deco_test_getitem()
        assert result
