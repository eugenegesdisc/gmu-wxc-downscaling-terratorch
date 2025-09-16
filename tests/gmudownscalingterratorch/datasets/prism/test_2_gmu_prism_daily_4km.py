"""
Test datasets for PRISM climate data
"""
import pytest
import logging
from dateutil import parser as dtu_parser
from torchgeo.datasets import BoundingBox

logger = logging.getLogger(__name__)

from gmudownscalingterratorch.util.decorators import (
    decorator_prism_exceptions_to_logger,
    decorator_timeit_to_logger
)
from gmudownscalingterratorch.datasets.prism import GmuPrismDaily4km


class DecoTest2GmuPrismDaily4km:
    """
    Decorable class: Test PRISM datasets
    Not discoverable by pytest due to custom decorator
    """
    @decorator_timeit_to_logger(logger=logger)
    @decorator_prism_exceptions_to_logger(logger=logger)
    def deco_test_download(self)->bool:
        ds = GmuPrismDaily4km(
            paths="../exp/dataprism4km/prism",
            #bands=["pppttt","ppt","tmax"], # this with purposedly error band 'pppttt'
            download=True,
            target_download_dir="../exp/dataprism4km/download",
            start_date="2000-12-31",
            end_date="2001-01-01",
        )

        if not ds:
            return False
        return True

    @decorator_timeit_to_logger(logger=logger)
    @decorator_prism_exceptions_to_logger(logger=logger)
    def deco_test_extract(self):
        ds = GmuPrismDaily4km(
            paths="../exp/dataprism4km/prism",
            #bands=["ppt","tmax"],
            download=False,
            target_download_dir="../exp/dataprism4km/download",
            start_date="2000-12-31",
            end_date="2001-01-01",
            extract=True,
            endings=[r"\.tif", r"\.prj"]
        )

        if not ds:
            return False
        return True

    @decorator_timeit_to_logger(logger=logger)
    @decorator_prism_exceptions_to_logger(logger=logger)
    def deco_test_super_init(self):
        ds = GmuPrismDaily4km(
            paths="../exp/dataprism4km/prism",
            bands=["ppt","tmax"],
            download=False,
            target_download_dir="../exp/dataprism4km/download",
            start_date="2000-12-31",
            end_date="2001-01-01",
            extract=False
        )

        if not ds:
            return False
        return True

    @decorator_timeit_to_logger(logger=logger)
    @decorator_prism_exceptions_to_logger(logger=logger)
    def deco_test_get_len(self)->bool:
        """
        Test super().__init__
        """
        ds = GmuPrismDaily4km(
            paths="../exp/dataprism4km/prism",
            download=False,
            target_download_dir="../exp/dataprism4km/prism",
            extract=False)
        print("len=", ds.__len__())
        print("number of bands=", len(ds.all_bands))
        print("number of days=", ds.__len__()/len(ds.all_bands))
        if ds.__len__() < 1:
            return False
        return True

    @decorator_timeit_to_logger(logger=logger)
    @decorator_prism_exceptions_to_logger(logger=logger)
    def deco_test_getitem(self)->bool:
        """
        Test super().__getitem__
        """
        ds = GmuPrismDaily4km(
            paths="../exp/dataprism4km/prism",
            download=False,
            target_download_dir="../exp/dataprism4km/download",
            bands=["tmean"],
            extract=False)
        # non-extended boundingbox may miss one pixels
        data = ds.__getitem__(
            BoundingBox(-180, 180, -90, 90,
                        dtu_parser.isoparse("2000-12-31 00:00:00").timestamp(),
                        dtu_parser.isoparse("2000-12-31 23:59:59").timestamp()))
        print("keys=", data.keys())
        print("image-shape:", data["image"].shape)
        print(data["image"][0, 300:310, 300:310]) # mon

        ds = GmuPrismDaily4km(
            paths="../exp/dataprism4km/prism",
            download=False,
            target_download_dir="../exp/dataprism4km/download",
            extract=False)
        print("resolution=", ds.res)
        # to get the full coverage of the given area
        # using extended bounding box - 
        # bbx.minx-ds.res[0]/2.0, bbx.maxx+ds.res[0]/2.0,
        # bbx.miny-ds.res[1]/2.0, bbx.maxy+ds.res[1]/2.0
        # gdal use center to represent the pixel
        data2 = ds.__getitem__(
            BoundingBox(-125-(ds.res[0]/2), -66.5+(ds.res[0]/2),
                        24.08333333-(ds.res[1]/2), 49.91666666+(ds.res[1]/2),
                        dtu_parser.isoparse("2001-01-01 00:00:00").timestamp(),
                        dtu_parser.isoparse("2001-01-01 01:59:59").timestamp()))
        print("data2-image-shape:", data2["image"].shape)
        tmean_idx = ds.all_bands.index("tmin")
        print(data2["image"][
            tmean_idx, 
            300:310,300:310]) #month january

        if not data:
            return False
        return True

    @decorator_timeit_to_logger(logger=logger)
    @decorator_prism_exceptions_to_logger(logger=logger)
    def deco_test_getitem_whole(self)->bool:
        """
        Test super().__getitem__
        """
        ds = GmuPrismDaily4km(
            paths="../exp/dataprism4km/prism",
            download=False,
            target_download_dir="../exp/dataprism4km/download",
            bands=["tmean"],
            extract=False)
        # non-extended boundingbox may miss one pixels
        print("bounds=", ds.bounds)
        data = ds.__getitem__(
            BoundingBox(ds.bounds.minx, ds.bounds.maxx, ds.bounds.miny, ds.bounds.maxy,
                        dtu_parser.isoparse("2001-01-01 00:00:00").timestamp(),
                        dtu_parser.isoparse("2001-01-01 23:59:59").timestamp()))
        print("keys=", data.keys())
        print("image-shape:", data["image"].shape)
        print(data["image"][0, 300:310, 300:310]) # mon

        ds = GmuPrismDaily4km(
            paths="../exp/dataprism4km/prism",
            download=False,
            target_download_dir="../exp/dataprism4km/download",
            extract=False)
        print("resolution=", ds.res)
        # to get the full coverage of the given area
        # using extended bounding box - 
        # bbx.minx-ds.res[0]/2.0, bbx.maxx+ds.res[0]/2.0,
        # bbx.miny-ds.res[1]/2.0, bbx.maxy+ds.res[1]/2.0
        # gdal use center to represent the pixel
        data2 = ds.__getitem__(
            BoundingBox(-125-(ds.res[0]/2), -66.5+(ds.res[0]/2),
                        24.08333333-(ds.res[1]/2), 49.91666666+(ds.res[1]/2),
                        dtu_parser.isoparse("2001-01-01 00:00:00").timestamp(),
                        dtu_parser.isoparse("2001-01-01 01:59:59").timestamp()))
        print("data2-image-shape:", data2["image"].shape)
        tmean_idx = ds.all_bands.index("tmean")
        print(data2["image"][
            tmean_idx, 
            300:310,300:310]) #month january

        if not data:
            return False
        return True

#@pytest.mark.option1
class TestGmuPrismDaily4km:
    """
    Test PRISM datasets
    """
    @pytest.mark.option1
    def test_download(self):
        dec_cls = DecoTest2GmuPrismDaily4km()
        result = dec_cls.deco_test_download()
        assert result

    @pytest.mark.option1
    def test_extract(self):
        dec_cls = DecoTest2GmuPrismDaily4km()
        result = dec_cls.deco_test_extract()
        assert result

    @pytest.mark.option1
    def test_super_init(self):
        dec_cls = DecoTest2GmuPrismDaily4km()
        result = dec_cls.deco_test_super_init()
        assert result


    @pytest.mark.option1
    def test_get_len(self):
        """
            Test super().__len__
        """
        dec_cls = DecoTest2GmuPrismDaily4km()
        result = dec_cls.deco_test_get_len()
        assert result


    @pytest.mark.option1
    def test_getitem(self):
        """
            Test super().__getitem__()
        """
        dec_cls = DecoTest2GmuPrismDaily4km()
        result = dec_cls.deco_test_getitem()
        assert result


    @pytest.mark.option1
    def test_getitem_whole(self):
        """
            Test super().__getitem__()
        """
        dec_cls = DecoTest2GmuPrismDaily4km()
        result = dec_cls.deco_test_getitem_whole()
        assert result
