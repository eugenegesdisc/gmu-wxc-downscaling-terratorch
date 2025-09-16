"""
Module for testing GmuMerra2M2I3NVASM dataset
"""
import logging
import pytest
from dateutil import parser as dtu_parser
from torchgeo.datasets.utils import BoundingBox

from gmudownscalingterratorch.datasets.merra2 import (
    GmuMerra2M2I3NVASM)

from gmudownscalingterratorch.util.decorators import (
    decorator_merra2_exceptions_to_logger,
    decorator_timeit_to_logger)

logger = logging.getLogger(__name__)

class DecoTest2GmuMerra2M2I3NVASM:
    """
    Test GmuMerra2M2I3NVASM dataset
    """
    @decorator_timeit_to_logger(logger=logger)
    @decorator_merra2_exceptions_to_logger(logger=logger)
    def deco_test_download(self)->bool:
        """
        Test download
        """
        ds = GmuMerra2M2I3NVASM(
            paths="../exp/m2i3nvasm/merra2",
            download=True,
            start_date="2000-12-31",
            end_date="2001-01-01",
            username="earthdata-username",
            password="earthdata-password",
            target_download_dir="../exp/m2i3nvasm/download")
        if not ds:
            return False
        return True

    @decorator_timeit_to_logger(logger=logger)
    @decorator_merra2_exceptions_to_logger(logger=logger)
    def deco_test_extract(self)->bool:
        """
        Test extract
        """
        ds = GmuMerra2M2I3NVASM(
            paths="../exp/m2i3nvasm/merra2",
            download=False,
            start_date="2000-12-31",
            end_date="2001-01-01",
            target_download_dir="../exp/m2i3nvasm/download",
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
        ds = GmuMerra2M2I3NVASM(
            paths="../exp/m2i3nvasm/merra2",
            download=False,
            start_date="2000-12-31",
            end_date="2001-01-01",
            target_download_dir="../exp/m2i3nvasm/download",
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
        ds = GmuMerra2M2I3NVASM(
            paths="../exp/m2i3nvasm/merra2",
            download=False,
            start_date="2000-12-31",
            end_date="2001-01-01",
            target_download_dir="../exp/m2i3nvasm/download",
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
        Test super().__init__
        """
        ds = GmuMerra2M2I3NVASM(
            paths="../exp/m2i3nvasm/merra2",
            download=False,
            start_date="2000-12-31",
            end_date="2001-01-01",
            target_download_dir="../exp/m2i3nvasm/download",
            bands=["T"],
            extract=False)
        data = ds.__getitem__(
            BoundingBox(-180, 180, -90, 90,
                        dtu_parser.isoparse("2000-12-31 00:00:00").timestamp(),
                        dtu_parser.isoparse("2000-12-31 02:59:59").timestamp()))
        print("keys=", data.keys())
        print("image-shape:", data["image"].shape)
        print(data["image"][ds.all_levels.index(33), :, :])



        ds = GmuMerra2M2I3NVASM(
            paths="../exp/m2i3nvasm/merra2",
            download=False,
            start_date="2000-12-31",
            end_date="2001-01-01",
            target_download_dir="../exp/m2i3nvasm/download",
            #bands=["T"],
            extract=False)
        it_idx = ds.all_bands.index("T")
        data2 = ds.__getitem__(
            BoundingBox(-180, 180, -90, 90,
                        dtu_parser.isoparse("2000-12-31 02:00:00").timestamp(),
                        dtu_parser.isoparse("2000-12-31 05:59:59").timestamp()))
        print("data2-image-shape:", data2["image"].shape)
        print(data2["image"][(it_idx*72+33),:,:])

        if not data:
            return False
        return True

    @decorator_timeit_to_logger(logger=logger)
    @decorator_merra2_exceptions_to_logger(logger=logger)
    def deco_test_getitem_subset(self)->bool:
        """
        Test super().__init__
        """
        ds = GmuMerra2M2I3NVASM(
            paths="../exp/m2i3nvasm/merra2",
            download=False,
            start_date="2000-12-31",
            end_date="2001-01-01",
            target_download_dir="../exp/m2i3nvasm/download",
            bands=["T"],
            extract=False)
        data = ds.__getitem__(
            BoundingBox(-124.67, -66.95, 25.84, 49.38,
                        dtu_parser.isoparse("2000-12-31 00:00:00").timestamp(),
                        dtu_parser.isoparse("2000-12-31 02:59:59").timestamp()))
        print("keys=", data.keys())
        print("image-shape:", data["image"].shape)
        print(data["image"][ds.all_levels.index(33), :, :])



        ds = GmuMerra2M2I3NVASM(
            paths="../exp/m2i3nvasm/merra2",
            download=False,
            start_date="2000-12-31",
            end_date="2001-01-01",
            target_download_dir="../exp/m2i3nvasm/download",
            #bands=["T"],
            extract=False)
        #it_idx = ds.all_bands.index("T")
        it_idx = ds.all_bands.index("T")*len(ds.all_levels)+ds.all_levels.index(33)

        data2 = ds.__getitem__(
            BoundingBox(-124.67, -66.95, 25.84, 49.38,
                        dtu_parser.isoparse("2000-12-31 02:00:00").timestamp(),
                        dtu_parser.isoparse("2000-12-31 05:59:59").timestamp()))
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
        ds = GmuMerra2M2I3NVASM(
            paths="../exp/m2i3nvasm/merra2",
            download=False,
            start_date="2000-12-31",
            end_date="2001-01-01",
            target_download_dir="../exp/m2i3nvasm/download",
            bands=["T"],
            extract=False)
        data = ds.__getitem__(
            BoundingBox(-124.67, -66.95, 25.84, 49.38,
                        dtu_parser.isoparse("2000-12-31 03:00:00").timestamp(),
                        dtu_parser.isoparse("2000-12-31 03:00:00").timestamp()))
        print("keys=", data.keys())
        print("image-shape:", data["image"].shape)
        print(data["image"][ds.all_levels.index(33), :, :])



        ds = GmuMerra2M2I3NVASM(
            paths="../exp/m2i3nvasm/merra2",
            download=False,
            start_date="2000-12-31",
            end_date="2001-01-01",
            target_download_dir="../exp/m2i3nvasm/download",
            #bands=["T"],
            extract=False)
        it_idx = ds.all_bands.index("T")*len(ds.all_levels)+ds.all_levels.index(33)
        data2 = ds.__getitem__(
            BoundingBox(-124.67, -66.95, 25.84, 49.38,
                        dtu_parser.isoparse("2000-12-31 06:00:00").timestamp(),
                        dtu_parser.isoparse("2000-12-31 06:00:00").timestamp()))
        print("data2-image-shape:", data2["image"].shape)
        print(data2["image"][it_idx,:,:])

        if not data:
            return False
        return True


    @decorator_timeit_to_logger(logger=logger)
    @decorator_merra2_exceptions_to_logger(logger=logger)
    def deco_test_extract_subset_14level(self)->bool:
        """
        Test extract with submset of levels
        """
        ds = GmuMerra2M2I3NVASM(
            paths="../exp/m2i3nvasm/merra2_14",
            download=False,
            start_date="2000-12-31",
            end_date="2001-01-01",
            levels=[72,71,68,63,56,53,51,48,45,44,43,41,39,34],
            target_download_dir="../exp/m2i3nvasm/download",
            #bands=["T2M"],
            overwrite=False,
            extract=True)
        if not ds:
            return False
        return True

    @decorator_timeit_to_logger(logger=logger)
    @decorator_merra2_exceptions_to_logger(logger=logger)
    def deco_test_super_init_14level(self)->bool:
        """
        Test super().__init__
        """
        ds = GmuMerra2M2I3NVASM(
            paths="../exp/m2i3nvasm/merra2_14",
            download=False,
            start_date="2000-12-31",
            end_date="2001-01-01",
            levels=[72,71,68,63,56,53,51,48,45,44,43,41,39,34],
            target_download_dir="../exp/m2i3nvasm/download",
            #bands=["T2M"],
            extract=False)
        if not ds:
            return False
        return True


    @decorator_timeit_to_logger(logger=logger)
    @decorator_merra2_exceptions_to_logger(logger=logger)
    def deco_test_get_len_14level(self)->bool:
        """
        Test super().__len__
        """
        ds = GmuMerra2M2I3NVASM(
            paths="../exp/m2i3nvasm/merra2_14",
            download=False,
            start_date="2000-12-31",
            end_date="2001-01-01",
            levels=[72,71,68,63,56,53,51,48,45,44,43,41,39,34],
            target_download_dir="../exp/m2i3nvasm/download",
            #bands=["T2M"],
            extract=False)
        print("size=", ds.__len__()/len(ds.all_bands), "number of bands:", len(ds.all_bands))
        if ds.__len__() < 1:
            return False
        return True


    @decorator_timeit_to_logger(logger=logger)
    @decorator_merra2_exceptions_to_logger(logger=logger)
    def deco_test_getitem_14level(self)->bool:
        """
        Test super().__getitem__
        """
        ds = GmuMerra2M2I3NVASM(
            paths="../exp/m2i3nvasm/merra2_14",
            download=False,
            start_date="2000-12-31",
            end_date="2001-01-01",
            levels=[72,71,68,63,56,53,51,48,45,44,43,41,39,34],
            target_download_dir="../exp/m2i3nvasm/download",
            bands=["T"],
            extract=False)
        data = ds.__getitem__(
            BoundingBox(-180, 180, -90, 90,
                        dtu_parser.isoparse("2000-12-31 00:00:00").timestamp(),
                        dtu_parser.isoparse("2000-12-31 02:59:59").timestamp()))
        print("keys=", data.keys())
        print("image-shape:", data["image"].shape)
        print(data["image"][ds.all_levels.index(43), :, :])
        if data:
            return True
        else:
            return False


    @decorator_timeit_to_logger(logger=logger)
    @decorator_merra2_exceptions_to_logger(logger=logger)
    def deco_test_getitem_subset_14level(self)->bool:
        """
        Test super().__init__
        """
        ds = GmuMerra2M2I3NVASM(
            paths="../exp/m2i3nvasm/merra2_14",
            download=False,
            start_date="2000-12-31",
            end_date="2001-01-01",
            levels=[72,71,68,63,56,53,51,48,45,44,43,41,39,34],
            target_download_dir="../exp/m2i3nvasm/download",
            bands=["T"],
            extract=False)
        data = ds.__getitem__(
            BoundingBox(-124.67, -66.95, 25.84, 49.38,
                        dtu_parser.isoparse("2000-12-31 00:00:00").timestamp(),
                        dtu_parser.isoparse("2000-12-31 02:59:59").timestamp()))
        print("keys=", data.keys())
        print("image-shape:", data["image"].shape)
        print(data["image"][ds.all_levels.index(43), :, :])



        ds = GmuMerra2M2I3NVASM(
            paths="../exp/m2i3nvasm/merra2",
            download=False,
            start_date="2000-12-31",
            end_date="2001-01-01",
            levels=[72,71,68,63,56,53,51,48,45,44,43,41,39,34],
            target_download_dir="../exp/m2i3nvasm/download",
            #bands=["T"],
            extract=False)
        #it_idx = ds.all_bands.index("T")
        it_idx = ds.all_bands.index("T")*len(ds.all_levels)+ds.all_levels.index(43)

        data2 = ds.__getitem__(
            BoundingBox(-124.67, -66.95, 25.84, 49.38,
                        dtu_parser.isoparse("2000-12-31 02:00:00").timestamp(),
                        dtu_parser.isoparse("2000-12-31 05:59:59").timestamp()))
        print("data2-image-shape:", data2["image"].shape)
        print(data2["image"][it_idx,:,:])

        if not data:
            return False
        return True
    
    @decorator_timeit_to_logger(logger=logger)
    @decorator_merra2_exceptions_to_logger(logger=logger)
    def deco_test_getitem_subset_time_14level(self)->bool:
        """
        Test super().__getitem__
        """
        ds = GmuMerra2M2I3NVASM(
            paths="../exp/m2i3nvasm/merra2_14",
            download=False,
            start_date="2000-12-31",
            end_date="2001-01-01",
            levels=[72,71,68,63,56,53,51,48,45,44,43,41,39,34],
            target_download_dir="../exp/m2i3nvasm/download",
            bands=["T"],
            extract=False)
        data = ds.__getitem__(
            BoundingBox(-124.67, -66.95, 25.84, 49.38,
                        dtu_parser.isoparse("2000-12-31 03:00:00").timestamp(),
                        dtu_parser.isoparse("2000-12-31 03:00:00").timestamp()))
        print("keys=", data.keys())
        print("image-shape:", data["image"].shape)
        print(data["image"][ds.all_levels.index(43), :, :])



        ds = GmuMerra2M2I3NVASM(
            paths="../exp/m2i3nvasm/merra2",
            download=False,
            start_date="2000-12-31",
            end_date="2001-01-01",
            levels=[72,71,68,63,56,53,51,48,45,44,43,41,39,34],
            target_download_dir="../exp/m2i3nvasm/download",
            #bands=["T"],
            extract=False)
        it_idx = ds.all_bands.index("T")*len(ds.all_levels)+ds.all_levels.index(43)
        data2 = ds.__getitem__(
            BoundingBox(-124.67, -66.95, 25.84, 49.38,
                        dtu_parser.isoparse("2000-12-31 06:00:00").timestamp(),
                        dtu_parser.isoparse("2000-12-31 06:00:00").timestamp()))
        print("data2-image-shape:", data2["image"].shape)
        print(data2["image"][it_idx,:,:])

        if not data:
            return False
        return True


class Test2GmuMerra2M2I3NVASM:
    """
    Test GmuMerra2M2I3NVASM
    """
    @pytest.mark.option1
    def test_download(self):
        """
            Test download
        """
        dec_cls = DecoTest2GmuMerra2M2I3NVASM()
        result = dec_cls.deco_test_download()
        assert result

    @pytest.mark.option1
    def test_extract(self):
        """
        #    Test extract
        """
        dec_cls = DecoTest2GmuMerra2M2I3NVASM()
        result = dec_cls.deco_test_extract()
        assert result

    @pytest.mark.option1
    def test_super_init(self):
        """
            Test super().__init__
        """
        dec_cls = DecoTest2GmuMerra2M2I3NVASM()
        result = dec_cls.deco_test_super_init()
        assert result

    @pytest.mark.option1
    def test_get_len(self):
        """
            Test super().__init__
        """
        dec_cls = DecoTest2GmuMerra2M2I3NVASM()
        result = dec_cls.deco_test_get_len()
        assert result


    @pytest.mark.option1
    def test_getitem(self):
        """
            Test super().__getitem__
        """
        dec_cls = DecoTest2GmuMerra2M2I3NVASM()
        result = dec_cls.deco_test_getitem()
        assert result


    @pytest.mark.option1
    def test_getitem_subset(self):
        """
            Test super().__getitem__ with spatial subset
        """
        dec_cls = DecoTest2GmuMerra2M2I3NVASM()
        result = dec_cls.deco_test_getitem_subset()
        assert result


    @pytest.mark.option1
    def test_getitem_subset_time(self):
        """
            Test super().__getitem__ with spatial subset and specific timestamp
        """
        dec_cls = DecoTest2GmuMerra2M2I3NVASM()
        result = dec_cls.deco_test_getitem_subset_time()
        assert result


    #------------------
    @pytest.mark.option1
    def test_extract_subset_14level(self):
        """
        #    Test extract
        """
        dec_cls = DecoTest2GmuMerra2M2I3NVASM()
        result = dec_cls.deco_test_extract_subset_14level()
        assert result

    @pytest.mark.option1
    def test_super_init_14level(self):
        """
            Test super().__init__
        """
        dec_cls = DecoTest2GmuMerra2M2I3NVASM()
        result = dec_cls.deco_test_super_init_14level()
        assert result

    @pytest.mark.option1
    def test_get_len_14level(self):
        """
            Test super().__init__
        """
        dec_cls = DecoTest2GmuMerra2M2I3NVASM()
        result = dec_cls.deco_test_get_len_14level()
        assert result


    @pytest.mark.option1
    def test_getitem_14level(self):
        """
            Test super().__getitem__
        """
        dec_cls = DecoTest2GmuMerra2M2I3NVASM()
        result = dec_cls.deco_test_getitem_14level()
        assert result


    @pytest.mark.option1
    def test_getitem_subset_14level(self):
        """
            Test super().__getitem__ with spatial subset
        """
        dec_cls = DecoTest2GmuMerra2M2I3NVASM()
        result = dec_cls.deco_test_getitem_subset_14level()
        assert result


    @pytest.mark.option1
    def test_getitem_subset_time_14level(self):
        """
            Test super().__getitem__ with spatial subset and specific timestamp
        """
        dec_cls = DecoTest2GmuMerra2M2I3NVASM()
        result = dec_cls.deco_test_getitem_subset_time_14level()
        assert result
