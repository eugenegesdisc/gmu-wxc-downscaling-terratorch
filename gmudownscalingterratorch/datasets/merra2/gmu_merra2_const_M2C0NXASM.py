"""
Module for dataset M2C0NXASM
"""
from typing import Any
from collections.abc import Callable, Iterable, Sequence
from pyproj import CRS
import datetime
from torchgeo.datasets.utils import (
    Path
)
from torchgeo.datasets import RasterDataset
from gmudownscalingterratorch.util import (
    download_data_merra2_with_template,
    extract_subdatasets_band_from_downloaded_const_merra2_with_template
)
from .gmu_merra2_exceptions import (
    Merra2BandExtractError,
    Merra2DownloadError,
    Merra2UnknownElement
)

class GmuMerra2M2C0NXASM(RasterDataset):
    """
        const_2d_asm_Nx (M2C0NXASM)
        original data: MERRA2_101.const_2d_asm_Nx.00000000.nc4
    """

    filename_glob = "MERRA2_[0-9][0-9][0-9].const_2d_asm_Nx.*.tif"
    #Leave out 'date' group to allow RasterDataset to set it to maximum temporal range
    filename_regex = (
        r"MERRA2_[0-9][0-9][0-9]\.const_2d_asm_Nx\."
        r"(?P<date00>[0-9]{8})_sd_(?P<band>[a-zA-Z0-9]+)\.tif")

    is_image = True
    separate_files = True


    all_bands = ("PHIS", "FRLAND","FROCEAN") #, "FRACI")

    M2C0NXASM_BASE_URL_TEMPLATE=(
        "https://goldsmr4.gesdisc.eosdis.nasa.gov/data/MERRA2_MONTHLY/M2C0NXASM.5.12.4/1980/"
        "MERRA2_101.const_2d_asm_Nx.00000000.nc4")

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

        #start_date="1981-01-01"
        end_date = datetime.datetime.now().strftime("%Y-%m-%d")
        start_date = end_date

        if download:
            # download the dataset
            status = download_data_merra2_with_template(
                url_temp=self.M2C0NXASM_BASE_URL_TEMPLATE,
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
                    url_temp=self.M2C0NXASM_BASE_URL_TEMPLATE,
                    target_path=target_download_dir
                )
        if extract:
            temp = self.M2C0NXASM_BASE_URL_TEMPLATE
            filename_temp = temp.rsplit("/", 1)[1]
            status = extract_subdatasets_band_from_downloaded_const_merra2_with_template(
                filename_temp=filename_temp,
                start_date=start_date,
                end_date=end_date,
                source_dir=target_download_dir,
                target_dir=paths,
                subdatasets=self.all_bands,
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
