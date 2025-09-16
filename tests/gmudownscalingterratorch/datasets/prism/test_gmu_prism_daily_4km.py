"""
Test datasets for PRISM climate data
"""
import inspect
import resource
import pytest
import logging

logger = logging.getLogger(__name__)

from gmudownscalingterratorch.util.decorators import (
    decorator_prism_exceptions_to_logger,
    decorator_timeit_to_logger
)
from gmudownscalingterratorch.datasets.prism import GmuPrismDaily4km


class DecorableGmuPrismDaily4km:
    """
    Decorable class: Test PRISM datasets
    Not discoverable by pytest due to custom decorator
    """
    @decorator_timeit_to_logger(logger=logger)
    @decorator_prism_exceptions_to_logger(logger=logger)
    def test_download(self)->bool:
        ds = GmuPrismDaily4km(
            paths="../exp/prismdata/prism",
            bands=["pppttt","ppt","tmax"],
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
        ds = GmuPrismDaily4km(
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
        ds = GmuPrismDaily4km(
            paths="../exp/prismdata/prism",
            bands=["ppt","tmax"],
            download=False,
            target_download_dir="../exp/prismdata/download",
            start_date="1981-01-01",
            end_date="1981-01-01",
            extract=False
        )

        if not ds:
            return False
        return True

#@pytest.mark.option1
class TestGmuPrismDaily4km:
    """
    Test PRISM datasets
    """
    @pytest.mark.option1
    def test_download(self):
        dec_cls = DecorableGmuPrismDaily4km()
        result = dec_cls.test_download()
        assert result

    @pytest.mark.option1
    def test_extract(self):
        dec_cls = DecorableGmuPrismDaily4km()
        result = dec_cls.test_extract()
        assert result

    @pytest.mark.option1
    def test_super_init(self):
        dec_cls = DecorableGmuPrismDaily4km()
        result = dec_cls.test_super_init()
        assert result
