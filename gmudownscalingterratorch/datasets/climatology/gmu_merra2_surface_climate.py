"""
Module for dataset MERRA-2 Climatology Surface data
"""
from typing import Any
from collections.abc import Callable, Iterable, Sequence
from pyproj import CRS
from torchgeo.datasets.utils import (
    Path
)
import datetime as py_dt
from torchgeo.datasets import (
    RasterDataset, BoundingBox)
from gmudownscalingterratorch.util import (
    download_data_merra2_climatology_from_hugging_with_template,
    extract_subdatasets_band_data_merra2_climatology_from_hugging_with_template
)
from ..merra2.gmu_merra2_exceptions import (
    Merra2BandExtractError,
    Merra2DownloadError,
    Merra2UnknownElement
)
class GmuMerra2SurfaceClimate(RasterDataset):
    """
        MERRA-2 Climatology Data with surface
        original data: climate_surface_doy001_hour00.nc
        Will be managed using year 2020 which has 366 days.
    """
    filename_glob = "climate_surface_doy*_hour*_sd_*_*_to_*.tif"
    filename_regex = (
        r"climate_surface_doy(?P<doy>[0-3][0-9][0-9])_hour(?P<hod>[0-2][0-9])"
        r"_sd_(?P<band>[a-zA-Z0-9]+)_(?P<start>\d{4}\d{2}\d{2}T\d{2}\d{2}\d{2})_"
        r"to_(?P<stop>\d{4}\d{2}\d{2}T\d{2}\d{2}\d{2})"
        r"\.tif"
    )

    is_image = True
    date_format = "%Y%m%dT%H%M%S"
    separate_files = True
    surface_vars = [
        "EFLUX",
        "GWETROOT",
        "HFLUX",
        "LAI",
        "LWGAB",
        "LWGEM",
        "LWTUP",
        "PS",
        "QV2M",
        "SLP",
        "SWGNT",
        "SWTNT",
        "T2M",
        "TQI",
        "TQL",
        "TQV",
        "TS",
        "U10M",
        "V10M",
        "Z0M",
    ]

    all_bands = (
        "EFLUX",
        "GWETROOT",
        "HFLUX",
        "LAI",
        "LWGAB",
        "LWGEM",
        "LWTUP",
        "PS",
        "QV2M",
        "SLP",
        "SWGNT",
        "SWTNT",
        "T2M",
        "TQI",
        "TQL",
        "TQV",
        "TS",
        "U10M",
        "V10M",
        "Z0M",
        )

    REPO_ID="ibm-nasa-geospatial/Prithvi-WxC-1.0-2300M"
    HF_RESOURCE_PATTERN="climatology/climate_surface_doy{doy}_hour{hod}.nc"

    def __init__(
            self,
            paths: Path | Iterable[Path] = "climatology/merra2",
            crs: CRS | None = None,
            res: float | tuple[float, float] | None = None,
            bands: Sequence[str] | None = None,
            transforms: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
            cache: bool =True,

            download:bool=False,
            target_download_dir:str="climatology/download",
            start_date:str=None,
            end_date:str=None,
            extract:bool=False,
            overwrite:bool=False
            ):
        """
            Initialize a new GmuMerra2SurfaceClimate instance

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
                msg="not all elements are supported in MERRA-2 climatology data")

        if bands:
            self.all_bands = list()
            for band in bands:
                self.all_bands.append(band)

        if download:
            # download the dataset
            status = download_data_merra2_climatology_from_hugging_with_template(
                repo_id=self.REPO_ID,
                hf_resource_temp=self.HF_RESOURCE_PATTERN,
                start_date=start_date,
                end_date=end_date,
                target_dir=target_download_dir,
                overwrite=overwrite)
            if status < 0:
                raise Merra2DownloadError(
                    nerrors=status,
                    url_temp=self.HF_RESOURCE_PATTERN,
                    target_path=target_download_dir
                )
        if extract:
            status = extract_subdatasets_band_data_merra2_climatology_from_hugging_with_template(
                hf_resource_temp=self.HF_RESOURCE_PATTERN,
                start_date=start_date,
                end_date=end_date,
                source_dir=target_download_dir,
                target_dir=paths,
                subdatasets=self.all_bands,
                overwrite=overwrite,
                date_format=self.date_format)
            if status < 0:
                raise Merra2BandExtractError(
                    nerrors=status,
                    filename_temp=self.HF_RESOURCE_PATTERN,
                    source_path=target_download_dir)

        self.paths=paths
        super().__init__(
            paths=self.paths,
            crs=crs,
            res=res,
            bands=bands,
            transforms=transforms,
            cache=cache)
    def __getitem__(self, query:BoundingBox):
        minx = query.minx
        maxx = query.maxx
        miny = query.miny
        maxy = query.maxy
        mint = self._update_datetime_to_2020(
            py_dt.datetime.fromtimestamp(query.mint)
            ).timestamp()
        maxt = self._update_datetime_to_2020(
            py_dt.datetime.fromtimestamp(query.maxt)
            ).timestamp()

        return super().__getitem__(
            BoundingBox(minx,maxx,miny,maxy,mint,maxt))
    
    def _update_datetime_to_2020(
            self,
            old_time: py_dt.datetime)->py_dt.datetime:
        target_year = py_dt.datetime(
            year=2020,
            month=1,
            day=1)
        old_year = py_dt.datetime(
            year=old_time.year,
            month=1,
            day=1)
        return old_time+(target_year-old_year)
    
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
