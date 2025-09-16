"""Module for GmuPrismDaily800m dataset"""
from typing import Any
from collections.abc import Callable, Iterable, Sequence
from pyproj import CRS
from torchgeo.datasets.utils import (
    Path
)

from torchgeo.datasets import RasterDataset
from gmudownscalingterratorch.util import(
    download_data_prism_with_template,
    extract_files_from_downloaded_prism_with_template
)
from .gmu_prism_exceptions import (
    PrismDownloadError, PrismUnknownElement,PrismZipfileExtractError
)

        
class GmuPrismDaily800m(RasterDataset):
    """
        PRISM Climate Date at 800m or 30s resolution
        original data: prism_ppt_us_30s_20200101.zip
    """
    filename_glob = "prism_*_30s_*.tif"
    filename_regex = (r"prism_(?P<band>[a-zA-Z0-9]+)_"
                      r"(?P<region>[a-z]+)_(?P<resolution>[a-z0-9]+)_"
                      r"(?P<date>\d{8}).tif")
    date_format = "%Y%m%d"
    is_image = True
    separate_files = True
    all_bands = ("ppt", "tdmean", "tmax", "tmean", "tmin",
                 "vpdmax", "vpdmin")

    PRISM800MPPT_BASE_URL_TEMPLATE=(
        "https://data.prism.oregonstate.edu/time_series/us/an/800m/"
        "ppt/daily/{year}/prism_ppt_us_30s_{year}{month}{day}.zip")
    PRISM800MTDMEAN_BASE_URL_TEMPLATE=(
        "https://data.prism.oregonstate.edu/time_series/us/an/800m/"
        "tdmean/daily/{year}/prism_tdmean_us_30s_{year}{month}{day}.zip")
    PRISM800MTMAX_BASE_URL_TEMPLATE=(
        "https://data.prism.oregonstate.edu/time_series/us/an/800m/"
        "tmax/daily/{year}/prism_tmax_us_30s_{year}{month}{day}.zip")
    PRISM800MTMEAN_BASE_URL_TEMPLATE=(
        "https://data.prism.oregonstate.edu/time_series/us/an/800m/"
        "tmean/daily/{year}/prism_tmean_us_30s_{year}{month}{day}.zip")
    PRISM800MTMIN_BASE_URL_TEMPLATE=(
        "https://data.prism.oregonstate.edu/time_series/us/an/800m/"
        "tmin/daily/{year}/prism_tmin_us_30s_{year}{month}{day}.zip")
    PRISM800MVPDMAX_BASE_URL_TEMPLATE=(
        "https://data.prism.oregonstate.edu/time_series/us/an/800m/"
        "vpdmax/daily/{year}/prism_vpdmax_us_30s_{year}{month}{day}.zip")
    PRISM800MVPDMIN_BASE_URL_TEMPLATE=(
        "https://data.prism.oregonstate.edu/time_series/us/an/800m/"
        "vpdmin/daily/{year}/prism_vpdmin_us_30s_{year}{month}{day}.zip")

    def __init__(
            self,
            paths: Path | Iterable[Path] = "prism",
            crs: CRS | None = "EPSG:4326",
            res: float | tuple[float, float] | None = (0.0083333,0.0083333),
            bands: Sequence[str] | None = None,
            transforms: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
            cache: bool = True,

            download:bool=False,
            target_download_dir:str = "download",
            start_date:str=None,
            end_date:str=None,
            extract:bool = False,
            endings:list[str]|None = None,
            overwrite:bool=False):
        """
            Initialize a new GmuPrismDaily800m instance.

            Args:
                paths: one or more roo directories to search or files to load
                crs: :term:`coordinate resference system (CRS)` to warp to
                    (defautls to the CRS of the first file found)
                res: resolution of the datazest in units of CRS
                    (defaults to the resolution of the first file found)
                bands: bands to return (defaults to all bands)
                transforms: a function/tgransform that takes an input smaple
                    and returns a transformed version.
                cache: if True, intermidiate processed data will be cached.
                
                download: if True, data between [start_date, end_date] will be
                    downloaded to target_download_dir from PRISM cliate data repo.
                target_download_dir: one directory to hold the downloaded data (i.e.
                    zip files directly downloaded from PRISM cliate data repo)
                start_date: a start datetime string in ISO 8601 format
                end_date: an end datetime string in ISO 8601 format.
                extract: if True, selected files with endings will be extracted from 
                    the zip files and saved to paths - shich should be one directory
                    string.
                endings: A list of endings in regular expression to be included the files
                    to be extracted from the zip files.
                overwrite: if True, either download or extract operation will be forced
                    to overwrite the data in the destination directory when files exist
                    there.
            
            Raises:
                AssertionError: If *bands* are invalid.
                DatasetNotFoundError: If dataset is not found.
        """
        if bands:
            self.all_bands = list()
            for band in bands:
                self.all_bands.append(band)
        if download:            
            for elem in self.all_bands:
                temp = self._get_template(elem)
                status = download_data_prism_with_template(
                    url_temp=temp,
                    start_date=start_date,
                    end_date=end_date,
                    target_dir=target_download_dir,
                    overwrite=overwrite)
                if status < 0:
                    raise PrismDownloadError(
                        nerrors=status,
                        url_temp=temp,
                        target_path=target_download_dir)

        if extract:
            for elem in self.all_bands:
                temp = self._get_template(elem)
                filename_temp = temp.rsplit("/",1)[1]
                status = extract_files_from_downloaded_prism_with_template(
                    filename_temp=filename_temp,
                    start_date=start_date,
                    end_date=end_date,
                    source_dir=target_download_dir,
                    target_dir=paths,
                    endings=endings,
                    overwrite=overwrite)
                if status < 0:
                    raise PrismZipfileExtractError(
                        nerrors=start_date,
                        filename_temp=filename_temp,
                        zip_source_path=target_download_dir)

        self.paths = paths   
        super().__init__(
            paths=self.paths, crs=crs,
            res=res, bands=bands,
            transforms=transforms,
            cache=cache)

    def _get_template(
            self, temp_attr:str)->str:
        """
            Get the applicable template.

            Args:
                temp_str: short name of the element.

            Returns:
                A template string.
        """
        if temp_attr == "ppt":
            return self.PRISM800MPPT_BASE_URL_TEMPLATE
        elif temp_attr == "tdmean":
            return self.PRISM800MTDMEAN_BASE_URL_TEMPLATE
        elif temp_attr == "tmax":
            return self.PRISM800MTMAX_BASE_URL_TEMPLATE
        elif temp_attr == "tmin":
            return self.PRISM800MTMIN_BASE_URL_TEMPLATE
        elif temp_attr == "tmean":
            return self.PRISM800MTMEAN_BASE_URL_TEMPLATE
        elif temp_attr == "vpdmax":
            return self.PRISM800MVPDMAX_BASE_URL_TEMPLATE
        elif temp_attr == "vpdmin":
            return self.PRISM800MVPDMIN_BASE_URL_TEMPLATE
        else:
            raise PrismUnknownElement(elem=temp_attr)