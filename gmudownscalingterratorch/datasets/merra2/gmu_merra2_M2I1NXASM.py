"""
Module for dataset M2I1NXASM
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
    extract_subdatasets_band_from_downloaded_merra2_with_template
)
from .gmu_merra2_exceptions import (
    Merra2BandExtractError,
    Merra2DownloadError,
    Merra2UnknownElement
)
class GmuMerra2M2I1NXASM(RasterDataset):
    """
        M2I1NXASM (inst1_2d_asm_Nx)
        original data: MERRA2_400.inst1_2d_asm_Nx.20200101.nc4
    """
    filename_glob = "MERRA2_[0-9]00.inst1_2d_asm_Nx.*.tif"
    filename_regex = (
        r"MERRA2_[0-9]00\.inst1_2d_asm_Nx\.(?P<olddate>\d{8})"
        r"_sd_(?P<band>[a-zA-Z0-9]+)_(?P<start>\d{4}\d{2}\d{2}T\d{2}\d{2}\d{2})_"
        r"to_(?P<stop>\d{4}\d{2}\d{2}T\d{2}\d{2}\d{2})"
        r"\.tif"
    )

    is_image = True
    date_format = "%Y%m%dT%H%M%S"
    separate_files = True
    all_bands = ("U10M", "V10M", "T2M", "QV2M",
                 "PS", "SLP", "TS", "TQI", "TQL", 
                 "TQV")

    M2I1NXASM_BASE_URL_TEMPLATE=("https://goldsmr4.gesdisc.eosdis.nasa.gov/"
                                 "data/MERRA2"
                                 "/M2I1NXASM.5.12.4/{year}/{month}/"
                                 "MERRA2_{YYYY}00.inst1_2d_asm_Nx.{year}{month}{day}.nc4")
    def __init__(
            self,
            paths: Path | Iterable[Path] = "merra2",
            crs: CRS | None = None,
            res: float | tuple[float, float] | None = None,
            bands: Sequence[str] | None = None,
            transforms: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
            cache: bool =True,

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
            Initialize a new GmuMerra2M2I1NXASM instance

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

        if bands:
            self.all_bands = list()
            for band in bands:
                self.all_bands.append(band)

        if download:
            # download the dataset
            status = download_data_merra2_with_template(
                url_temp=self.M2I1NXASM_BASE_URL_TEMPLATE,
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
                    url_temp=self.M2I1NXASM_BASE_URL_TEMPLATE,
                    target_path=target_download_dir
                )
        if extract:
            temp = self.M2I1NXASM_BASE_URL_TEMPLATE
            filename_temp = temp.rsplit("/", 1)[1]
            status = extract_subdatasets_band_from_downloaded_merra2_with_template(
                filename_temp=filename_temp,
                start_date=start_date,
                end_date=end_date,
                source_dir=target_download_dir,
                target_dir=paths,
                subdatasets=bands,
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
