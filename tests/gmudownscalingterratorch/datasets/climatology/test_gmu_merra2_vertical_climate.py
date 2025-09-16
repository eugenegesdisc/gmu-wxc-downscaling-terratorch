"""
Module for testing GmuMerra2VerticalClimate dataset
"""
import logging
import pytest
from dateutil import parser as dtu_parser
from torchgeo.datasets.utils import BoundingBox

from gmudownscalingterratorch.datasets.climatology import (
    GmuMerra2VerticalClimate)

from gmudownscalingterratorch.util.decorators import (
    decorator_merra2_exceptions_to_logger,
    decorator_timeit_to_logger)

from gmudownscalingterratorch.util import (
    convert_datetime_to_2020_climatology_datetime,
    convert_string_datetime_to_2020_climatology_datetime
)


logger = logging.getLogger(__name__)

class DecoTest2GmuMerra2VerticalClimate:
    """
    Test GmuMerra2VerticalClimate dataset
    """
    @decorator_timeit_to_logger(logger=logger)
    @decorator_merra2_exceptions_to_logger(logger=logger)
    def deco_test_download(self)->bool:
        """
        Test download
        """
        ds = GmuMerra2VerticalClimate(
            paths="../exp/climatology3d/vertical",
            download=True,
            start_date="2000-12-31",
            end_date="2001-01-01",
            target_download_dir="../exp/climatology3d/download")
        if not ds:
            return False
        return True

    @decorator_timeit_to_logger(logger=logger)
    @decorator_merra2_exceptions_to_logger(logger=logger)
    def deco_test_extract(self)->bool:
        """
        Test extract
        """
        ds = GmuMerra2VerticalClimate(
            paths="../exp/climatology3d/vertical",
            download=False,
            start_date="2000-12-31",
            end_date="2001-01-01",
            target_download_dir="../exp/climatology3d/download",
            #bands=["T"],
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
        ds = GmuMerra2VerticalClimate(
            paths="../exp/climatology3d/vertical",
            download=False,
            start_date="2000-12-31",
            end_date="2001-01-01",
            target_download_dir="../exp/climatology3d/download",
            #bands=["T2M"],
            extract=False)
        if not ds:
            return False
        return True

    @decorator_timeit_to_logger(logger=logger)
    @decorator_merra2_exceptions_to_logger(logger=logger)
    def deco_test_get_len(self)->bool:
        """
        Test super().__init__
        """
        ds = GmuMerra2VerticalClimate(
            paths="../exp/climatology3d/vertical",
            download=False,
            start_date="2000-12-31",
            end_date="2001-01-01",
            target_download_dir="../exp/climatology3d/download",
            #bands=["T2M"],
            extract=False)
        print("The size is number of files or temporal repeats (in 3 hour intervals)")
        print("size=", ds.__len__()/(len(ds.all_bands)),
              "number of bands:", len(ds.all_bands),
              "number of levels:", len(ds.all_levels))
        if ds.__len__() < 1:
            return False
        return True


    @decorator_timeit_to_logger(logger=logger)
    @decorator_merra2_exceptions_to_logger(logger=logger)
    def deco_test_getitem(self)->bool:
        """
        Test super().__getitem__
        """
        start_date="2000-12-31"
        end_date="2001-01-01"
        bbx_mint_1="2000-12-31 00:00:00"
        bbx_maxt_1="2000-12-31 02:59:59"
        bbx_mint_2="2000-12-31 02:00:00"
        bbx_maxt_2="2000-12-31 05:59:59"

        ad_bbx_mint_1_str = convert_string_datetime_to_2020_climatology_datetime(
            dt_dateitme_str=bbx_mint_1,
            out_date_format="%Y%m%d %H:%M:%S")
        ad_bbx_maxt_1_str = convert_string_datetime_to_2020_climatology_datetime(
            dt_dateitme_str=bbx_maxt_1,
            out_date_format="%Y%m%d %H:%M:%S")
        ad_bbx_mint_2_str = convert_string_datetime_to_2020_climatology_datetime(
            dt_dateitme_str=bbx_mint_2,
            out_date_format="%Y%m%d %H:%M:%S")
        ad_bbx_maxt_2_str = convert_string_datetime_to_2020_climatology_datetime(
            dt_dateitme_str=bbx_maxt_2,
            out_date_format="%Y%m%d %H:%M:%S")

        ds = GmuMerra2VerticalClimate(
            paths="../exp/climatology3d/vertical",
            download=False,
            start_date="2000-12-31",
            end_date="2001-01-01",
            target_download_dir="../exp/climatology3d/download",
            bands=["T"],
            extract=False)
        # time-slice: Needs to make sure ad_bbx_mint < ad_bbx_maxt
        # This should work for this test since each retrieval will be on time step (3-hourly data)
        data = ds.__getitem__(
            BoundingBox(-180, 180, -90, 90,
                        dtu_parser.isoparse(ad_bbx_mint_1_str).timestamp(),
                        dtu_parser.isoparse(ad_bbx_maxt_1_str).timestamp()))
        print("keys=", data.keys())
        print("image-shape:", data["image"].shape)
        print(data["image"][ds.all_levels.index(44), :, :])



        ds = GmuMerra2VerticalClimate(
            paths="../exp/climatology3d/vertical",
            download=False,
            start_date="2000-12-31",
            end_date="2001-01-01",
            target_download_dir="../exp/climatology3d/download",
            #bands=["T"],
            extract=False)
        it_idx = ds.all_bands.index("T")*len(ds.all_levels)+ds.all_levels.index(44)
        data2 = ds.__getitem__(
            BoundingBox(-180, 180, -90, 90,
                        dtu_parser.isoparse(ad_bbx_mint_2_str).timestamp(),
                        dtu_parser.isoparse(ad_bbx_maxt_2_str).timestamp()))
        print("data2-image-shape:", data2["image"].shape)
        print(data2["image"][it_idx,:,:])

        if not data:
            return False
        return True

    @decorator_timeit_to_logger(logger=logger)
    @decorator_merra2_exceptions_to_logger(logger=logger)
    def deco_test_getitem_subset(self)->bool:
        """
        Test super().__getitem__
        """
        start_date="2000-12-31"
        end_date="2001-01-01"
        bbx_mint_1="2000-12-31 00:00:00"
        bbx_maxt_1="2000-12-31 02:59:59"
        bbx_mint_2="2000-12-31 02:00:00"
        bbx_maxt_2="2000-12-31 05:59:59"

        ad_bbx_mint_1_str = convert_string_datetime_to_2020_climatology_datetime(
            dt_dateitme_str=bbx_mint_1,
            out_date_format="%Y%m%d %H:%M:%S")
        ad_bbx_maxt_1_str = convert_string_datetime_to_2020_climatology_datetime(
            dt_dateitme_str=bbx_maxt_1,
            out_date_format="%Y%m%d %H:%M:%S")
        ad_bbx_mint_2_str = convert_string_datetime_to_2020_climatology_datetime(
            dt_dateitme_str=bbx_mint_2,
            out_date_format="%Y%m%d %H:%M:%S")
        ad_bbx_maxt_2_str = convert_string_datetime_to_2020_climatology_datetime(
            dt_dateitme_str=bbx_maxt_2,
            out_date_format="%Y%m%d %H:%M:%S")

        ds = GmuMerra2VerticalClimate(
            paths="../exp/climatology3d/vertical",
            download=False,
            start_date="2000-12-31",
            end_date="2001-01-01",
            target_download_dir="../exp/climatology3d/download",
            bands=["T"],
            extract=False)
        data = ds.__getitem__(
            BoundingBox(-124.67, -66.95, 25.84, 49.38,
                        dtu_parser.isoparse(ad_bbx_mint_1_str).timestamp(),
                        dtu_parser.isoparse(ad_bbx_maxt_1_str).timestamp()))
        print("keys=", data.keys())
        print("image-shape:", data["image"].shape)
        print(data["image"][ds.all_levels.index(44), :, :])



        ds = GmuMerra2VerticalClimate(
            paths="../exp/climatology3d/vertical",
            download=False,
            start_date="2000-12-31",
            end_date="2001-01-01",
            target_download_dir="../exp/climatology3d/download",
            #bands=["T"],
            extract=False)
        it_idx = ds.all_bands.index("T")*len(ds.all_levels)+ds.all_levels.index(44)
        data2 = ds.__getitem__(
            BoundingBox(-124.67, -66.95, 25.84, 49.38,
                        dtu_parser.isoparse(ad_bbx_mint_2_str).timestamp(),
                        dtu_parser.isoparse(ad_bbx_maxt_2_str).timestamp()))
        print("data2-image-shape:", data2["image"].shape)
        print(data2["image"][it_idx,:,:])

        if not data:
            return False
        return True

    @decorator_timeit_to_logger(logger=logger)
    @decorator_merra2_exceptions_to_logger(logger=logger)
    def deco_test_getitem_subset_time(self)->bool:
        """
        Test super().__getitem__
        """
        start_date="2000-12-31"
        end_date="2001-01-01"
        bbx_mint_1="2000-12-31 00:00:00"
        bbx_maxt_1="2000-12-31 02:59:59"
        bbx_mint_2="2000-12-31 02:00:00"
        bbx_maxt_2="2000-12-31 05:59:59"

        ad_bbx_mint_1_str = convert_string_datetime_to_2020_climatology_datetime(
            dt_dateitme_str=bbx_mint_1,
            out_date_format="%Y%m%d %H:%M:%S")
        ad_bbx_maxt_1_str = convert_string_datetime_to_2020_climatology_datetime(
            dt_dateitme_str=bbx_maxt_1,
            out_date_format="%Y%m%d %H:%M:%S")
        ad_bbx_mint_2_str = convert_string_datetime_to_2020_climatology_datetime(
            dt_dateitme_str=bbx_mint_2,
            out_date_format="%Y%m%d %H:%M:%S")
        ad_bbx_maxt_2_str = convert_string_datetime_to_2020_climatology_datetime(
            dt_dateitme_str=bbx_maxt_2,
            out_date_format="%Y%m%d %H:%M:%S")
        ds = GmuMerra2VerticalClimate(
            paths="../exp/climatology3d/vertical",
            download=False,
            start_date="2000-12-31",
            end_date="2001-01-01",
            target_download_dir="../exp/climatology3d/download",
            bands=["T"],
            extract=False)
        data = ds.__getitem__(
            BoundingBox(-124.67, -66.95, 25.84, 49.38,
                        dtu_parser.isoparse(ad_bbx_mint_1_str).timestamp(),
                        dtu_parser.isoparse(ad_bbx_maxt_1_str).timestamp()))
        print("keys=", data.keys())
        print("image-shape:", data["image"].shape)
        print(data["image"][ds.all_levels.index(44), :, :])



        ds = GmuMerra2VerticalClimate(
            paths="../exp/climatology3d/vertical",
            download=False,
            start_date="2000-12-31",
            end_date="2001-01-01",
            target_download_dir="../exp/climatology3d/download",
            #bands=["T"],
            extract=False)
        it_idx = ds.all_bands.index("T")*len(ds.all_levels)+ds.all_levels.index(44)
        data2 = ds.__getitem__(
            BoundingBox(-124.67, -66.95, 25.84, 49.38,
                        dtu_parser.isoparse(ad_bbx_mint_2_str).timestamp(),
                        dtu_parser.isoparse(ad_bbx_maxt_2_str).timestamp()))
        print("data2-image-shape:", data2["image"].shape)
        print(data2["image"][it_idx,:,:])

        if not data:
            return False
        return True


class Test2GmuMerra2VerticalClimate:
    """
    Test GmuMerra2VerticalClimate
    """
    @pytest.mark.option1
    def test_download(self):
        """
            Test download
        """
        dec_cls = DecoTest2GmuMerra2VerticalClimate()
        result = dec_cls.deco_test_download()
        assert result

    @pytest.mark.option1
    def test_extract(self):
        """
        #    Test download
        """
        dec_cls = DecoTest2GmuMerra2VerticalClimate()
        result = dec_cls.deco_test_extract()
        assert result

    @pytest.mark.option1
    def test_super_init(self):
        """
            Test super().__init__
        """
        dec_cls = DecoTest2GmuMerra2VerticalClimate()
        result = dec_cls.deco_test_super_init()
        assert result

    @pytest.mark.option1
    def test_get_len(self):
        """
            Test super().__init__
        """
        dec_cls = DecoTest2GmuMerra2VerticalClimate()
        result = dec_cls.deco_test_get_len()
        assert result


    @pytest.mark.option1
    def test_getitem(self):
        """
            Test super().__getitem__
        """
        dec_cls = DecoTest2GmuMerra2VerticalClimate()
        result = dec_cls.deco_test_getitem()
        assert result


    @pytest.mark.option1
    def test_getitem_subset(self):
        """
            Test super().__getitem__ with spatial subset
        """
        dec_cls = DecoTest2GmuMerra2VerticalClimate()
        result = dec_cls.deco_test_getitem_subset()
        assert result


    @pytest.mark.option1
    def test_getitem_subset_time(self):
        """
            Test super().__getitem__ with spatial subset and specific timestamp
        """
        dec_cls = DecoTest2GmuMerra2VerticalClimate()
        result = dec_cls.deco_test_getitem_subset_time()
        assert result
