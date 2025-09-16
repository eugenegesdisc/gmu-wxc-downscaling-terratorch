"""Module for GmuPrismStatic800m dataset"""
from typing import Any
from collections.abc import Callable, Iterable, Sequence
from pyproj import CRS
import datetime
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

        
class GmuPrismStatic800m(RasterDataset):
    """
        PRISM Climate Date at 800m or 30s resolution
        original data: prism_ppt_us_30s_20200101.zip
    """
    filename_glob = "PRISM_*_800m_*.bil"
    filename_regex = (r"PRISM_(?P<region>[a-z]+)_(?P<band>[a-zA-Z0-9]+)_"
                      r"(?P<resolution>[a-z0-9]+)_"
                      r"(?P<format>[a-zA-Z0-9]+).bil")
    date_format = "%Y%m%d"
    is_image = True
    separate_files = True
    all_bands = ["dem"]

    PRISM800MDEM_BASE_URL_TEMPLATE=(
        "https://prism.oregonstate.edu/downloads/data/"
        "PRISM_us_dem_800m_bil.zip")    

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
            Initialize a new GmuPrismStatic800m instance.

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
        end_date = datetime.datetime.now().strftime("%Y-%m-%d")
        start_date = end_date
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
        if temp_attr == "dem":
            return self.PRISM800MDEM_BASE_URL_TEMPLATE
        else:
            raise PrismUnknownElement(elem=temp_attr)