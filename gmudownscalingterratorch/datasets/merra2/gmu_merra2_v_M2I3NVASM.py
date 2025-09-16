"""
Module for dataset M2I3NVASM
"""
from typing import Any
from collections.abc import Callable, Iterable, Sequence
from pyproj import CRS
from torchgeo.datasets.utils import (
    Path
)
from torchgeo.datasets import RasterDataset
from gmudownscalingterratorch.util import (
    download_data_merra2_with_template,
    extract_3d_subdatasets_band_from_downloaded_merra2_with_template
)
from .gmu_merra2_exceptions import (
    Merra2BandExtractError,
    Merra2DownloadError,
    Merra2UnknownElement
)


class GmuMerra2M2I3NVASM(RasterDataset):
    """
        inst3_3d_asm_Nv (M2I3NVASM)
        original data: MERRA2_400.inst3_3d_asm_Nv.20200101.nc4

    """
    filename_glob = "MERRA2_[0-9]00.inst3_3d_asm_Nv.*.tif"
    filename_regex = (
        r"MERRA2_[0-9]00\.inst3_3d_asm_Nv\.(?P<olddate>\d{8})"
        r"_sd_(?P<band>[a-zA-Z0-9]+)_(?P<start>\d{4}\d{2}\d{2}T\d{2}\d{2}\d{2})_"
        r"to_(?P<stop>\d{4}\d{2}\d{2}T\d{2}\d{2}\d{2})"
        r"\.tif"
    )

    is_image = True
    date_format = "%Y%m%dT%H%M%S"
    separate_files = True

    all_bands = ("U", "V", "T", "QV", "OMEGA",
                 "PL", "H", "CLOUD", "QI", "QL")
    all_levels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 
                  15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 
                  27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 
                  39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 
                  51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62,
                  63, 64, 65, 66, 67, 68, 69, 70, 71, 72]

    M2I3NVASM_BASE_URL_TEMPLATE=(
        "https://goldsmr5.gesdisc.eosdis.nasa.gov/data/MERRA2/M2I3NVASM.5.12.4/"
        "{year}/{month}/"
        "MERRA2_{YYYY}00.inst3_3d_asm_Nv.{year}{month}{day}.nc4"
        )

    def __init__(
            self,
            paths: Path | Iterable[Path] = "merra2",
            crs: CRS | None = None,
            res: float | tuple[float, float] | None = None,
            bands: Sequence[str] | None = None,
            transforms: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
            cache: bool =True,

            levels: list[float] | None=None,
            download:bool=False,
            target_download_dir:str="download",
            start_date:str=None,
            end_date:str=None,
            username:str=None,
            password:str=None,
            edl_token:str=None,
            extract:bool=False,
            overwrite:bool=False
            ):
        """
            Initialize a new GmuMerra2M2I3NVASM instance

            Args:
                paths: one or more root directories to search or files to load
                crs: :term:`coordinate reference system (CRS)` to warp to
                    (defaults to the CRS of the first file found)
                res: resolution of the dataset in units of CRS
                    (defaults to the resolution of the first file found)
                bands: bands to return (defaults to all bands)
                transforms: a function/transform that takes an input sample
                    and returns a transformed version
                cache: if True, cache file handle to speed up repeated sampling

                download: if True, data between [start_date, end_date] will be
                    downloaded to target_download_dir from MERRA2 data repo.
                target_download_dir: one directory to hold the downloaded data (i.e. nc4
                    files directly downloaded from MERRA2 data repo)
                start_date: start datetime string in ISO 8601 date format
                    to be parsed by dateutil.parser.isoparse
                extract: if True, data between [start_date, end_date] (nc4 files)
                    will be extracted to the destionation directory of paths - 
                    which should be one directory string.
                overwrite: if True, either download or extract operation will be
                    forced to overwrite the data in the destination directory if
                    the files already exist.

            Raises:
                AssertionError: If *bands* are invalid.
                DatasetNotFoundError: If dataset is not found.
                Merra2UnknownElement: If an element or variable is not found in the dataset.
                Merra2DownloadError: Any error occurred during the downloading of MERRA2 data.

        """
        if not self._validate_bands(
            bands=bands,
            all_bands=self.all_bands):
            raise Merra2UnknownElement(
                elem=bands.__str__(),
                msg="not all elements are supported in MERRA2 data repo")

        if not self._validate_levels(
            levels=levels,
            all_levels=self.all_levels):
            raise Merra2UnknownElement(
                elem=bands.__str__(),
                msg="not all levels are supported in MERRA2 data repo")

        if bands:
            self.all_bands = list()
            for band in bands:
                self.all_bands.append(band)

        if levels:
            self.all_levels = list()
            for level in levels:
                self.all_levels.append(level)

        if download:
            # download the dataset
            status = download_data_merra2_with_template(
                url_temp=self.M2I3NVASM_BASE_URL_TEMPLATE,
                start_date=start_date,
                end_date=end_date,
                target_dir=target_download_dir,
                username=username,
                password=password,
                edl_token=edl_token,
                overwrite=overwrite)
            if status < 0:
                raise Merra2DownloadError(
                    nerrors=status,
                    url_temp=self.M2I3NVASM_BASE_URL_TEMPLATE,
                    target_path=target_download_dir
                )
        if extract:
            temp = self.M2I3NVASM_BASE_URL_TEMPLATE
            filename_temp = temp.rsplit("/", 1)[1]
            status = extract_3d_subdatasets_band_from_downloaded_merra2_with_template(
                filename_temp=filename_temp,
                start_date=start_date,
                end_date=end_date,
                source_dir=target_download_dir,
                target_dir=paths,
                subdatasets=self.all_bands,
                levels=self.all_levels,
                overwrite=overwrite)
            if status < 0:
                raise Merra2BandExtractError(
                    nerrors=status,
                    filename_temp=filename_temp,
                    source_path=target_download_dir)

        self.paths=paths
        super().__init__(
            paths=self.paths,
            crs=crs,
            res=res,
            bands=bands,
            transforms=transforms,
            cache=cache)
    
    def _validate_bands(
            self,
            bands:list[str],
            all_bands:list[str]
            )->bool:
        if bands is None:
            return True
        for bnd in bands:
            if bnd not in all_bands:
                return False
        return True


    def _validate_levels(
            self,
            levels:list[str],
            all_levels:list[str]
            )->bool:
        if levels is None:
            return True
        for lvl in levels:
            if lvl not in all_levels:
                return False
        return True
