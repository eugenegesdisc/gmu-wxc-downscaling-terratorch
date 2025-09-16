"""
    Module for testing GmuPrismDaily800m dataset
"""
import resource
import inspect
import logging
import pytest

from gmudownscalingterratorch.util.decorators import (
    decorator_prism_exceptions_to_logger,
    decorator_timeit_to_logger
)
from gmudownscalingterratorch.datasets.prism import (
    GmuPrismDaily800m,
    PrismDownloadError,
    PrismUnknownElement,
    PrismZipfileExtractError)

logger = logging.getLogger(__name__)

class DecorableGmuPrismDaily800m:
    """
        Decorable class: Testing GmuPrismDaily800m dataset
    """
    @decorator_timeit_to_logger(logger=logger)
    @decorator_prism_exceptions_to_logger(logger=logger)
    def test_download(self)->bool:
        """
        Download data
        """
        ds = GmuPrismDaily800m(
            paths="../exp/prismdata/prism",
            bands=["ppt","tmax"],
            download=True,
            target_download_dir="../exp/prismdata/download",
            start_date="1981-01-01",
            end_date="1981-01-02",
        )

        if not ds:
            return False
        return True


    @decorator_timeit_to_logger(logger=logger)
    @decorator_prism_exceptions_to_logger(logger=logger)
    def test_extract(self):
        """
            Extract data
        """
        ds = GmuPrismDaily800m(
            paths="../exp/prismdata/prism",
            bands=["ppt","tmax"],
            download=False,
            target_download_dir="../exp/prismdata/download",
            start_date="1981-01-01",
            end_date="1981-01-02",
            extract=True,
            endings=[r"\.tif", r"\.prj"]
        )

        if not ds:
            return False
        return True

    @decorator_timeit_to_logger(logger=logger)
    @decorator_prism_exceptions_to_logger(logger=logger)
    def test_super_init(self):
        ds = GmuPrismDaily800m(
            paths="../exp/prismdata/prism",
            bands=["ppt","tmax"],
            download=False,
            target_download_dir="../exp/prismdata/download",
            start_date="1981-01-01",
            end_date="1981-01-02",
            extract=False
        )

        if not ds:
            return False
        return True

@pytest.mark.option1
class TestGmuPrismDaily800m:
    """
        Testing GmuPrismDaily800m dataset
    """
    #@pytest.mark.option1
    def test_download(self):
        """
        Download data
        """
        dec_cls = DecorableGmuPrismDaily800m()
        result = dec_cls.test_download()
        assert result

    #@pytest.mark.option1
    def test_extract(self):
        """
            Extract data
        """
        dec_cls = DecorableGmuPrismDaily800m()
        result = dec_cls.test_extract()
        assert result

    #@pytest.mark.option1
    def test_super_init(self):
        """
            Init RasterDataset - super()
        """
        dec_cls = DecorableGmuPrismDaily800m()
        result = dec_cls.test_super_init()
        assert result
