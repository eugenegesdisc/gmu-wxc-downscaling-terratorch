"""
Module for testing GmuMerra2M2I1NXLND dataset
"""
import logging
import pytest
from dateutil import parser as dtu_parser
from torchgeo.datasets import BoundingBox

from gmudownscalingterratorch.datasets.merra2 import (
    GmuMerra2M2T1NXLND
    )

from gmudownscalingterratorch.util.decorators import (
    decorator_merra2_exceptions_to_logger,
    decorator_timeit_to_logger)

logger = logging.getLogger(__name__)

class DecoTest2GmuMerra2M2T1NXLND:
    """
    Test GmuMerra2M2T1NXLND dataset
    """
    @decorator_timeit_to_logger(logger=logger)
    @decorator_merra2_exceptions_to_logger(logger=logger)
    def deco_test_download(self)->bool:
        """
        Test download
        """
        ds = GmuMerra2M2T1NXLND(
            paths="../exp/m2t1nxlnd/merra2",
            download=True,
            start_date="2000-12-31",
            end_date="2001-01-01",
            username="earthdata-username",
            password="earthdata-password",
            target_download_dir="../exp/m2t1nxlnd/download")
        if not ds:
            return False
        return True

    @decorator_timeit_to_logger(logger=logger)
    @decorator_merra2_exceptions_to_logger(logger=logger)
    def deco_test_extract(self)->bool:
        """
        Test extract
        """
        ds = GmuMerra2M2T1NXLND(
            paths="../exp/m2t1nxlnd/merra2",
            download=False,
            start_date="2000-12-31",
            end_date="2001-01-01",
            target_download_dir="../exp/m2t1nxlnd/download",
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
        ds = GmuMerra2M2T1NXLND(
            paths="../exp/m2t1nxlnd/merra2",
            download=False,
            start_date="2000-12-31",
            end_date="2001-01-01",
            target_download_dir="../exp/m2t1nxlnd/download",
            #bands=["T2M"],
            extract=False)
        if not ds:
            return False
        return True

    @decorator_timeit_to_logger(logger=logger)
    @decorator_merra2_exceptions_to_logger(logger=logger)
    def deco_test_get_len(self)->bool:
        """
        Test super().__len__
        """
        ds = GmuMerra2M2T1NXLND(
            paths="../exp/m2t1nxlnd/merra2",
            download=False,
            start_date="2000-12-31",
            end_date="2001-01-01",
            target_download_dir="../exp/m2t1nxlnd/download",
            #bands=["T2M"],
            extract=False)
        print("size=", ds.__len__()/len(ds.all_bands), "number of bands:", len(ds.all_bands))
        if ds.__len__() < 1:
            return False
        return True


    @decorator_timeit_to_logger(logger=logger)
    @decorator_merra2_exceptions_to_logger(logger=logger)
    def deco_test_getitem(self)->bool:
        """
        Test super().__getitem__
        """
        ds = GmuMerra2M2T1NXLND(
            paths="../exp/m2t1nxlnd/merra2",
            download=False,
            start_date="2000-12-31",
            end_date="2001-01-01",
            target_download_dir="../exp/m2t1nxlnd/download",
            bands=["LAI"],
            extract=False)
        data = ds.__getitem__(
            BoundingBox(-180, 180, -90, 90,
                        dtu_parser.isoparse("2000-12-31 00:00:00").timestamp(),
                        dtu_parser.isoparse("2000-12-31 00:59:59").timestamp()))
        print("keys=", data.keys())
        print("image-shape:", data["image"].shape)
        print(data["image"])

        ds = GmuMerra2M2T1NXLND(
            paths="../exp/m2t1nxlnd/merra2",
            download=False,
            start_date="2000-12-31",
            end_date="2001-01-01",
            target_download_dir="../exp/m2t1nxlnd/download",
            #bands=["LAI"],
            extract=False)
        data2 = ds.__getitem__(
            BoundingBox(-180, 180, -90, 90,
                        dtu_parser.isoparse("2000-12-31 01:00:00").timestamp(),
                        dtu_parser.isoparse("2000-12-31 01:59:59").timestamp()))
        print("data2-image-shape:", data2["image"].shape)
        print(data2["image"][ds.all_bands.index("LAI"),:,:])

        if not data:
            return False
        return True


class Test2GmuMerra2M2T1NXLND:
    """
    Test GmuMerra2M2I1NXASM
    """
    @pytest.mark.option1
    def test_download(self):
        """
            Test download
        """
        dec_cls = DecoTest2GmuMerra2M2T1NXLND()
        result = dec_cls.deco_test_download()
        assert result

    @pytest.mark.option1
    def test_extract(self):
        """
        #    Test download
        """
        dec_cls = DecoTest2GmuMerra2M2T1NXLND()
        result = dec_cls.deco_test_extract()
        assert result

    @pytest.mark.option1
    def test_super_init(self):
        """
            Test super().__init__
        """
        dec_cls = DecoTest2GmuMerra2M2T1NXLND()
        result = dec_cls.deco_test_super_init()
        assert result

    @pytest.mark.option1
    def test_get_len(self):
        """
            Test super().__len__()
        """
        dec_cls = DecoTest2GmuMerra2M2T1NXLND()
        result = dec_cls.deco_test_get_len()
        assert result


    @pytest.mark.option1
    def test_getitem(self):
        """
            Test super().__getitem__()
        """
        dec_cls = DecoTest2GmuMerra2M2T1NXLND()
        result = dec_cls.deco_test_getitem()
        assert result
