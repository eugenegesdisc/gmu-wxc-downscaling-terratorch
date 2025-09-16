from typing import Any
from torch.utils.data import Dataset
from torch import Tensor


class GmuMerra2DatasetForWxc(Dataset):
    """MERRA2 dataset for directly used by Prithvi WxC. 
    The dataset unifies surface and vertical data 
    as well as optional climatology.

    Samples come in the form of a dictionary. Not all keys support all
    variables, yet the general ordering of dimensions is
    parameter, level, time, lat, lon

    Note:
        Data is assumed to be in NetCDF files containing daily data at 3-hourly
        intervals. These follow the naming patterns
        MERRA2_sfc_YYYYMMHH.nc and MERRA_pres_YYYYMMHH.nc and can be located in
        two different locations. Optional climatology data comes from files
        climate_surface_doyDOY_hourHOD.nc and
        climate_vertical_doyDOY_hourHOD.nc.


    Note:
        `_get_valid_timestamps` assembles a set of all timestamps for which
        there is data (with hourly resolutions). The result is stored in
        `_valid_timestamps`. `_get_valid_climate_timestamps` does the same with
        climatology data and stores it in `_valid_climate_timestamps`.

        Based on this information, `samples` generates a list of valid samples,
        stored in `samples`. Here the format is::

            [
                [
                    (timestamp 1, lead time A),
                    (timestamp 1, lead time B),
                    (timestamp 1, lead time C),
                ],
                [
                    (timestamp 2, lead time D),
                    (timestamp 2, lead time E),
                ]
            ]

        That is, the outer list iterates over timestamps (init times), the
        inner over lead times. Only valid entries are stored.
    """

    valid_vertical_vars = [
        "CLOUD",
        "H",
        "OMEGA",
        "PL",
        "QI",
        "QL",
        "QV",
        "T",
        "U",
        "V",
    ]
    valid_surface_vars = [
        "EFLUX",
        "GWETROOT",
        "HFLUX",
        "LAI",
        "LWGAB",
        "LWGEM",
        "LWTUP",
        "PRECTOT",
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
    valid_static_surface_vars = ["FRACI", "FRLAND", "FROCEAN", "PHIS"]

    valid_levels = [
        34.0,
        39.0,
        41.0,
        43.0,
        44.0,
        45.0,
        48.0,
        51.0,
        53.0,
        56.0,
        63.0,
        68.0,
        71.0,
        72.0,
    ]

    def __init__(
        self,*args:Any, **kwargs:Any
    ) -> None:
        """
        Args:
            data_path_surface: Location of surface data.
            data_path_vertical: Location of vertical data.
            climatology_path_surface: Location of (optional) surface
                climatology.
            climatology_path_vertical: Location of (optional) vertical
                climatology.
            surface_vars: Surface variables.
            static_surface_vars: Static surface variables.
            vertical_vars: Vertical variables.
            levels: Levels.
            time_range: Used to subset data.
            lead_times: Lead times for generalized forecasting.
            roll_longitudes: Set to non-zero value to data by random amount
                along longitude dimension.
            position_encoding: possible values are
              ['absolute' (default), 'fourier'].
                'absolute' returns lat lon encoded in 3 dimensions using sine
                  and cosine
                'fourier' returns lat/lon to be encoded by model
                <any other key> returns lat/lon to be encoded by model
            rtype: numpy data type used during read
            dtype: torch data type of data output
        """
        pass



    def __getitem__(self, idx: int) -> dict[str, Tensor | int]:
        """
        Loads data based on sample index and random choice of sample.
        Args:
            idx: Sample index.
         Returns:
            Dictionary with keys 'sur_static', 'sur_vals', 'sur_tars',
                'ulv_vals', 'ulv_tars', 'sur_climate', 'ulv_climate',
                'lead_time', 'input_time'.
        """
        #sample_set = self.samples[idx]
        #timestamp, input_time, lead_time, *nsteps = random.choice(sample_set)
        #sample_data = self.get_data(timestamp, input_time, lead_time)
        data = Tensor(1,1,2,2)
        sample_data = {'sur_static': data, 'sur_vals': data, 'sur_tars':data,
                'ulv_vals':data, 'ulv_tars':data, 'sur_climate':data, 'ulv_climate':data,
                'lead_time':data, 'input_time':data }
        return sample_data

    def __len__(self):
        #return len(self.samples)
        return 2
