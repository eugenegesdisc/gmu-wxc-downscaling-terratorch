import torch
import pandas as pd
import math
import logging
from torch.utils.data import Dataset
from ..merra2 import (
    GmuMerra2M2T1NXFLX,
    GmuMerra2M2I1NXASM,
    GmuMerra2M2T1NXLND,
    GmuMerra2M2T1NXRAD,
    GmuMerra2M2C0NXCTM,
    GmuMerra2M2I3NVASM)
from ..climatology import (
    GmuMerra2SurfaceClimate,
    GmuMerra2VerticalClimate
)
from ..prism import (
    GmuPrismDaily4km,
    GmuPrismStatic4km
)
from dateutil import parser as dt_parser
import datetime as py_dt
import functools as py_ft
from torchgeo.datasets import (
    BoundingBox as torchgeo_bbx,
    RasterDataset
)
import torch.nn.functional as nn_F
logger = logging.getLogger(__name__)

class GmuMerra2ToPrism4km(Dataset):
    climate_surface_vars = [
        "EFLUX", # GmuMerra2M2T1NXFLX - 0
        "GWETROOT", # GmuMerra2M2T1NXLND - 0
        "HFLUX", # GmuMerra2M2T1NXFLX - 1
        "LAI", # GmuMerra2M2T1NXLND - 1
        "LWGAB", # GmuMerra2M2T1NXRAD - 1
        "LWGEM", # GmuMerra2M2T1NXRAD - 0
        "LWTUP", # GmuMerra2M2T1NXRAD - 2
        "PS", # GmuMerra2M2I1NXASM - 4
        "QV2M", # GmuMerra2M2I1NXASM - 3
        "SLP", # GmuMerra2M2I1NXASM - 5
        "SWGNT", # GmuMerra2M2T1NXRAD - 3
        "SWTNT", # GmuMerra2M2T1NXRAD - 4
        "T2M", # GmuMerra2M2I1NXASM - 2
        "TQI", # GmuMerra2M2I1NXASM - 7
        "TQL", # GmuMerra2M2I1NXASM - 8
        "TQV", # GmuMerra2M2I1NXASM - 9
        "TS", # GmuMerra2M2I1NXASM - 6
        "U10M", # GmuMerra2M2I1NXASM - 0
        "V10M", # GmuMerra2M2I1NXASM - 1
        "Z0M", # GmuMerra2M2T1NXFLX - 2
    ]
    climate_static_surface_vars = [
        "FRACI", # GmuMerra2M2C0NXCTM - 3
        "FRLAND", # GmuMerra2M2C0NXCTM - 1
        "FROCEAN", # GmuMerra2M2C0NXCTM - 2
        "PHIS", # GmuMerra2M2C0NXCTM - 0
        ]
    climate_vertical_vars = [
        "CLOUD", # GmuMerra2M2I3NVASM - 7
        "H", # GmuMerra2M2I3NVASM - 6
        "OMEGA", # GmuMerra2M2I3NVASM - 4
        "PL",  # GmuMerra2M2I3NVASM - 5
        "QI", # GmuMerra2M2I3NVASM - 8
        "QL", # GmuMerra2M2I3NVASM - 9
        "QV", # GmuMerra2M2I3NVASM - 3
        "T", # GmuMerra2M2I3NVASM - 2
        "U", # GmuMerra2M2I3NVASM - 0
        "V", # GmuMerra2M2I3NVASM - 1
        ]
    climate_levels = [
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

    surface_vars = [
        "EFLUX", # GmuMerra2M2T1NXFLX - 0
        "GWETROOT", # GmuMerra2M2T1NXLND - 0
        "HFLUX", # GmuMerra2M2T1NXFLX - 1
        "LAI", # GmuMerra2M2T1NXLND - 1
        "LWGAB", # GmuMerra2M2T1NXRAD - 1
        "LWGEM", # GmuMerra2M2T1NXRAD - 0
        "LWTUP", # GmuMerra2M2T1NXRAD - 2
        "PS", # GmuMerra2M2I1NXASM - 4
        "QV2M", # GmuMerra2M2I1NXASM - 3
        "SLP", # GmuMerra2M2I1NXASM - 5
        "SWGNT", # GmuMerra2M2T1NXRAD - 3
        "SWTNT", # GmuMerra2M2T1NXRAD - 4
        "T2M", # GmuMerra2M2I1NXASM - 2
        "TQI", # GmuMerra2M2I1NXASM - 7
        "TQL", # GmuMerra2M2I1NXASM - 8
        "TQV", # GmuMerra2M2I1NXASM - 9
        "TS", # GmuMerra2M2I1NXASM - 6
        "U10M", # GmuMerra2M2I1NXASM - 0
        "V10M", # GmuMerra2M2I1NXASM - 1
        "Z0M", # GmuMerra2M2T1NXFLX - 2
    ]
    static_surface_vars = [
        "FRACI", # GmuMerra2M2C0NXCTM - 3
        "FRLAND", # GmuMerra2M2C0NXCTM - 1
        "FROCEAN", # GmuMerra2M2C0NXCTM - 2
        "PHIS", # GmuMerra2M2C0NXCTM - 0
        ]
    vertical_vars = [
        "CLOUD", # GmuMerra2M2I3NVASM - 7
        "H", # GmuMerra2M2I3NVASM - 6
        "OMEGA", # GmuMerra2M2I3NVASM - 4
        "PL",  # GmuMerra2M2I3NVASM - 5
        "QI", # GmuMerra2M2I3NVASM - 8
        "QL", # GmuMerra2M2I3NVASM - 9
        "QV", # GmuMerra2M2I3NVASM - 3
        "T", # GmuMerra2M2I3NVASM - 2
        "U", # GmuMerra2M2I3NVASM - 0
        "V", # GmuMerra2M2I3NVASM - 1
        ]
    levels = [
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
    old_levels = [
        72.0,
        71.0,
        68.0,
        63.0,
        56.0,
        53.0,
        51.0,
        48.0,
        45.0,
        44.0,
        43.0,
        41.0,
        39.0,
        34.0,
    ]
    prism_vars = [
        "ppt", 
        "tdmean", 
        "tmax", 
        "tmean", 
        "tmin",
        "vpdmax", 
        "vpdmin",
        ]
    high_res_aux_vars = [
        "dem"
    ]
    def __init__(
            self,
            start_date:str,
            end_date:str,
            scale:int=16,
            n_lat_pixels:int=360,
            n_lon_pixels:int=576,
            work_on_subset:bool=False,
            work_min_lon_or_x:float=None,
            work_max_lon_or_x:float=None,
            work_min_lat_or_y:float=None,
            work_max_lat_or_y:float=None,
            predict_mode:bool=False,
            use_climatology:bool=True,
            use_hr_auxiliary:bool=False,
            lead_times: list[int]=[3],
            input_times: list[int]=[-3],
            download:bool=False,
            extract:bool=False,
            convert:bool=False,
            overwrite:bool=False,
            username:str=None,
            password:str=None,
            edl_token:str=None,
            m2t1nxflx_paths:str|list[str]="m2t1nxflx/merra2",
            m2t1nxflx_down_dir:str="m2t1nxflx/download",
            m2i1nxasm_paths:str|list[str]="m2i1nxasm/merra2",
            m2i1nxasm_down_dir:str="m2i1nxasm/download",
            m2t1nxlnd_paths:str|list[str]="m2t1nxlnd/merra2",
            m2t1nxlnd_down_dir:str="m2t1nxlnd/download",
            m2t1nxrad_paths:str|list[str]="m2t1nxrad/merra2",
            m2t1nxrad_down_dir:str="m2t1nxrad/download",
            m2c0nxctm_paths:str|list[str]="m2c0nxctm/merra2",
            m2c0nxctm_down_dir:str="m2c0nxctm/download",
            m2i3nvasm_paths:str|list[str]="m2i3nvasm/merra2",
            m2i3nvasm_down_dir:str="m2i3nvasm/download",
            climate_surface_paths:str="climatology/surface",
            climate_surface_down_dir="climatology/download",
            climate_vertical_paths:str="climatology3d/vertical",
            climate_vertical_down_dir:str="climatology3d/download",
            prism_paths:str|list[str]="dataprism4km/merra2",
            prism_down_dir:str="dataprism4km/download",
            prism_static_paths:str|list[str]="staticprism4km/prism",
            prism_static_down_dir:str="staticprism4km/download",
            position_encoding:str="absolute",
            merra2_valid_range:tuple[float,float]=(-9.9999999e+14,9.9999999e+14),
            prism_valid_range:tuple[float|None,float|None]=(-9999, None),
            use_prism_mask:bool=True,
            prism_vars:list[str]=None,
            ):
        """
            Initialize a GmuMerra2ToPrism4km class.
                M2I1NXASM: ("U10M", "V10M", "T2M", "QV2M", "PS", "SLP", "TS", "TQI", "TQL", 
                            "TQV")
                M2T1NXFLX: ("EFLUX", "HFLUX", "Z0M", "PRECTOT")
                M2T1NXLND: ("GWETROOT", "LAI")
                M2T1NXRAD: ("LWGEM", "LWGAB", "LWTUP", "SWGNT",
                            "SWTNT")
                M2I3NVASM: ("U", "V", "T", "QV", "OMEGA",
                            "PL", "H", "CLOUD", "QI", "QL")
                M2C0NXCTM: ("PHIS", "FRLAND","FROCEAN", "FRACI")

            Args:
                start_date: An ISO datetime string (to be parsed by Dateutil.parser.isoparse)
                end_date: An ISO datetime string (to be parsed by Dateutil.parser.isoparse)
                download: If true, all downloadable component dataset will be downloaded.
                extract: If true, all component dataset will be extracted.
                overwrite: If true and target exists, download/extract will be re-done.


        """
        self.start_datetime = dt_parser.isoparse(start_date)
        self.end_datetime = dt_parser.isoparse(end_date)
        assert self.end_datetime >= self.start_datetime
        self.position_encoding = position_encoding
        self.predict_mode = predict_mode
        self.use_climatology = use_climatology
        self.use_hr_auxiliary = use_hr_auxiliary
        self.scale = scale
        self.n_lat_pixels = n_lat_pixels
        self.n_lon_pixels = n_lon_pixels
        self.merra2_valid_range = merra2_valid_range
        self.prism_valid_range = prism_valid_range
        self.use_prism_mask = use_prism_mask
        if prism_vars:
            self.prism_vars = prism_vars

        self.input_times = input_times
        self.lead_itmes = lead_times
        self.number_of_timestamps = int((self.end_datetime 
                                      - self.start_datetime
                                      + py_dt.timedelta(hours=1)
                                      )/(py_dt.timedelta(hours=3)))
        # ("U10M", "V10M", "T2M", "QV2M",
        #    "PS", "SLP", "TS", "TQI", "TQL", 
        #    "TQV")
        self.sur_m2i1nxasm_dataset = GmuMerra2M2I1NXASM(
            start_date=start_date,
            end_date=end_date,
            username=username,
            password=password,
            edl_token=edl_token,
            paths=m2i1nxasm_paths,
            target_download_dir=m2i1nxasm_down_dir,
            download=download,
            extract=extract,
            overwrite=overwrite)
        self.work_bounds = self.sur_m2i1nxasm_dataset.bounds
        bb =  self.sur_m2i1nxasm_dataset.bounds
        self.work_bounds =torchgeo_bbx(
            minx = -180,
            maxx=180,
            miny=-90,
            maxy=90,
            mint=bb.mint,
            maxt=bb.maxt)

        if work_on_subset:
            new_minx = self.work_bounds.minx
            if work_min_lon_or_x is not None:
                new_minx = work_min_lon_or_x
            new_maxx = self.work_bounds.maxx
            if work_max_lon_or_x is not None:
                new_maxx = work_max_lon_or_x
            new_miny = self.work_bounds.miny
            if work_min_lat_or_y is not None:
                new_miny = work_min_lat_or_y
            new_maxy = self.work_bounds.maxy
            if work_max_lat_or_y is not None:
                new_maxy = work_max_lat_or_y
            self.work_bounds =torchgeo_bbx(
                minx=new_minx,
                maxx=new_maxx,
                miny=new_miny,
                maxy=new_maxy,
                mint=bb.mint,
                maxt=bb.maxt)

        self.input_x_channel = (len(self.surface_vars) + 
                    len((self.levels)*len(self.vertical_vars)))

        #("EFLUX", "HFLUX", "Z0M", "PRECTOT")
        self.sur_m2t1nxflx_dataset = GmuMerra2M2T1NXFLX(
            start_date=start_date,
            end_date=end_date,
            username=username,
            password=password,
            edl_token=edl_token,
            paths=m2t1nxflx_paths,
            target_download_dir=m2t1nxflx_down_dir,
            download=download,
            extract=extract,
            overwrite=overwrite)
        # ("GWETROOT", "LAI")
        self.sur_m2t1nxlnd_dataset = GmuMerra2M2T1NXLND(
            start_date=start_date,
            end_date=end_date,
            username=username,
            password=password,
            edl_token=edl_token,
            paths=m2t1nxlnd_paths,
            target_download_dir=m2t1nxlnd_down_dir,
            download=download,
            extract=extract,
            overwrite=overwrite)
        # ("LWGEM", "LWGAB", "LWTUP", "SWGNT", "SWTNT")
        self.sur_m2t1nxrad_dataset = GmuMerra2M2T1NXRAD(
            start_date=start_date,
            end_date=end_date,
            username=username,
            password=password,
            edl_token=edl_token,
            paths=m2t1nxrad_paths,
            target_download_dir=m2t1nxrad_down_dir,
            download=download,
            extract=extract,
            overwrite=overwrite)
        # ("PHIS", "FRLAND","FROCEAN", "FRACI")
        self.sta_m2c0nxctm_dataset = GmuMerra2M2C0NXCTM(
            username=username,
            password=password,
            edl_token=edl_token,
            paths=m2c0nxctm_paths,
            target_download_dir=m2c0nxctm_down_dir,
            download=download,
            extract=extract,
            overwrite=overwrite)
        # ("U", "V", "T", "QV", "OMEGA",
        # "PL", "H", "CLOUD", "QI", "QL")
        self.ver_m2i3nvasm_dataset = GmuMerra2M2I3NVASM(
            start_date=start_date,
            end_date=end_date,
            username=username,
            password=password,
            edl_token=edl_token,
            paths=m2i3nvasm_paths,
            target_download_dir=m2i3nvasm_down_dir,
            download=download,
            extract=extract,
            overwrite=overwrite)
        if self.use_climatology:
            self.sur_climatology_dataset = GmuMerra2SurfaceClimate(
                start_date=start_date,
                end_date=end_date,
                paths=climate_surface_paths,
                target_download_dir=climate_surface_down_dir,
                download=download,
                extract=extract,
                overwrite=overwrite
            )
            self.ver_climatology_dataset = GmuMerra2VerticalClimate(
                start_date=start_date,
                end_date=end_date,
                paths=climate_vertical_paths,
                target_download_dir=climate_vertical_down_dir,
                download=download,
                extract=extract,
                overwrite=overwrite)

        # ("ppt", "tdmean", "tmax", "tmean", "tmin",
        #   "vpdmax", "vpdmin")
        if not self.predict_mode:
            self.sur_prism_dataset = GmuPrismDaily4km(
                paths=prism_paths,
                start_date=start_date,
                end_date=end_date,
                target_download_dir=prism_down_dir,
                download=download,
                extract=extract,
                overwrite=overwrite,
                bands=self.prism_vars)
        # (dem)
        if self.use_hr_auxiliary:
            self.sta_prism_dataset = GmuPrismStatic4km(
                paths=prism_static_paths,
                target_download_dir=prism_static_down_dir,
                download=download,
                extract=extract,
                convert=convert,
                overwrite=overwrite)

    @py_ft.cached_property
    def samples(self)->list[tuple[py_dt.datetime, int, int]]:
        """
            A list of all valid samples: A sample is represented with 
            a timestamp (), a inputtime (in hours), and a lead time (in hours)

            A sample: (timestamp, it, lt) - For forward forecasting, "it" should 
                be negative and lt should be positive. Example: 
                (dt_parser.isoparse("2000-12-31T03:00:00"), -3, 6)==>
                with input pairs at timestamp ("2000-12-31T00:00:00", "2000-12-31T03:00:00") to 
                forcast "2000-12-31T09:00:00".
                For retrospective forecasting, "it" should be positive and "lt" should be negative.
                Exmaple:  (dt_parser.isoparse("2000-12-31T12:00:00"), 3, -6)==>
                with input pairs at timestamp ("2000-12-31T15:00:00", "2000-12-31T12:00:00") to 
                retrospectively estimate "2000-12-31T06:00:00".
        """
        valid_samples = set()
        dts = [(it, lt) for it in self.input_times for lt in self.lead_itmes]
        start_datetime= self.start_datetime
        tdelta_3h = py_dt.timedelta(hours=3)
        for tstep in range(self.number_of_timestamps):
            timestamp_samples = set()
            for it, lt in dts:
                if self._verify_in_range(
                    start_datetime, it, lt):
                    timestamp_samples.add(
                        (start_datetime, it, lt))
            if timestamp_samples:
                valid_samples = valid_samples.union(timestamp_samples)
            start_datetime = start_datetime + tdelta_3h
        return list(valid_samples)

    def _verify_in_range(
            self,
            anchor_datetime:py_dt.datetime,
            input_hour:int,
            lead_hour:int)->bool:
        """
            verify if the sample(anchor_datetime.timestamp, input_hour, lead_hour)
            has data at datetime in the range [self.start_datetime, self.end_datetime]
        """
        input_delta = py_dt.timedelta(hours=input_hour)
        input_time = anchor_datetime + input_delta
        lead_delta = py_dt.timedelta(hours=lead_hour)
        lead_time = anchor_datetime + lead_delta

        if (anchor_datetime > self.end_datetime or
            anchor_datetime < self.start_datetime):
            return False
        
        if (input_time > self.end_datetime or
            input_time < self.start_datetime):
            return False

        if (lead_time > self.end_datetime or
            lead_time < self.start_datetime):
            return False

        return True
    
    def get_param_index_in_x(
            self,
            param:str,
            level:str|int|float=None)->int:
        ret_idx = 0
        if level:
            if param not in self.vertical_vars:
                logger.error("Error: not find %s in vertical_vars", param)
                return 0
            ret_idx = len(self.surface_vars)
            p_idx = self.vertical_vars.index(param)
            if level not in self.levels:
                logger.error("Error: not find level '%s'", level)
                return 0
            l_idx = self.levels.index(level)
            ret_idx = ret_idx + p_idx * len(self.levels) + l_idx
            return ret_idx
        if param not in self.surface_vars:
            logger.error("Error: not find %s in surface_vars", param)
            return 0
        return_idx = self.surface_vars.index(param)
        return ret_idx

    def _get_sample(self):
        """
            Use the start_date and end_date to estimate number of total samples.
            Assuming 3-hour intevarls. This does not check verify if data are 
            available (downloaded, extracted, and/or cached).

            
            Returns:

        """
        pass

    def _get_data(
            self,
            index):
        ret_val = {}
        sample = self.samples[index]
        ref_time = sample[0]
        input_t1 = ref_time + py_dt.timedelta(hours=sample[1])
        input_t2 = ref_time
        target_time = ref_time + py_dt.timedelta(hours=sample[2])
        ret_val["input_time"] = torch.tensor([float(-sample[1])])
        ret_val["lead_time"] = torch.tensor([float(sample[2])])

        input_x = self._read_input_x(
            input_t1=input_t1,
            input_t2=input_t2)
        ret_val["x"] = torch.flip(input_x, dims=(1,))[0]

        # retrieve y (low resolution) for backbone: keep it for reference or validation of the backbone
        output_y = self._read_output_y(
            target_time=target_time)

        ret_val["y"] = torch.flip(output_y, dims=(1,))[0]

        # form and read input (low resolution) static data
        static_data = self._read_static_data(
            bounds=self.work_bounds,
            n_lat_pixels = self.n_lat_pixels,
            n_lon_pixels = self.n_lon_pixels,
            input_time=input_t1,
            position_encoding=self.position_encoding)
        ret_val["static"] = torch.flip(static_data, dims=(1,))[0]

        # form and read target high resolution static data
        hr_satatic_data = self._read_hr_static_data(
            bounds=self.work_bounds,
            n_lat_pixels = self.n_lat_pixels*self.scale,
            n_lon_pixels = self.n_lon_pixels*self.scale,
            input_time=input_t1,
            position_encoding=self.position_encoding)
        ret_val["hr_static"] = torch.flip(hr_satatic_data, dims=(1,))[0]

        # read target high resolution data: resize to match scaled data from low resolution
        # The scale ratio is chosen as the closet and power of 2 since each step is by upscale
        # by 2 for now
        if not self.predict_mode:
            hr_prism_data = self._read_prism_data(
                bounds=self.work_bounds,
                n_lat_pixels = self.n_lat_pixels*self.scale,
                n_lon_pixels = self.n_lon_pixels*self.scale,
                input_time=target_time)
            ret_val["hr_prism"] = torch.flip(hr_prism_data, dims=(1,))[0]
            if self.use_prism_mask:
                if self.prism_valid_range[0] is not None:
                    prism_mask = ret_val["hr_prism"] > self.prism_valid_range[0]
                    if self.prism_valid_range[1] is not None:
                        prism_mask = prism_mask & (ret_val["hr_prism"] < self.prism_valid_range[1])
                    ret_val["hr_prism_mask"] = prism_mask
                elif self.prism_valid_range[1] is not None:
                    prism_mask = ret_val["hr_prism"] < self.prism_valid_range[1]
                    ret_val["hr_prism_mask"] = prism_mask
        # read high resolution auxiliary data
        if self.use_hr_auxiliary:
            hr_aux_data = self._read_dem_data(
                bounds=self.work_bounds,
                n_lat_pixels = self.n_lat_pixels*self.scale,
                n_lon_pixels = self.n_lon_pixels*self.scale,
                input_time=target_time)
            ret_val["hr_aux"] = torch.flip(hr_aux_data, dims=(1,))[0]
        # read low resolution climate data (for replacement or inputs)
        if self.use_climatology:
            climate_x_data = self._read_climate_x_data(
                input_time1=input_t1,
                input_time2=input_t2)
            ret_val["climate_x"] = torch.flip(climate_x_data, dims=(1,))[0]
            if not self.predict_mode:
                climate_y_data = self._read_climate_y_data(
                    input_time=target_time)
                ret_val["climate"] = torch.flip(climate_y_data, dims=(1,))[0]
        return ret_val
    
    def _read_input_x(
            self,
            input_t1: py_dt.datetime,
            input_t2: py_dt.datetime,
    )->torch.Tensor:
        input_x = torch.empty(
            1, 2, 
            self.input_x_channel, 
            self.n_lat_pixels,
            self.n_lon_pixels)
        # m2t1nxflx
        self._set_5d_data(
            bounds=self.work_bounds,
            time_idx=0,
            input_time=input_t1,
            dataset=self.sur_m2t1nxflx_dataset,
            target_param_set=self.surface_vars,
            target_tensor=input_x)

        self._set_5d_data(
            bounds=self.work_bounds,
            time_idx=1,
            input_time=input_t2,
            dataset=self.sur_m2t1nxflx_dataset,
            target_param_set=self.surface_vars,
            target_tensor=input_x)

        # m2i1nxasm
        self._set_5d_data(
            bounds=self.work_bounds,
            time_idx=0,
            input_time=input_t1,
            dataset=self.sur_m2i1nxasm_dataset,
            target_param_set=self.surface_vars,
            target_tensor=input_x)

        self._set_5d_data(
            bounds=self.work_bounds,
            time_idx=1,
            input_time=input_t2,
            dataset=self.sur_m2i1nxasm_dataset,
            target_param_set=self.surface_vars,
            target_tensor=input_x)

        #m2t1nxlnd
        self._set_5d_data(
            bounds=self.work_bounds,
            time_idx=0,
            input_time=input_t1,
            dataset=self.sur_m2t1nxlnd_dataset,
            target_param_set=self.surface_vars,
            target_tensor=input_x)

        self._set_5d_data(
            bounds=self.work_bounds,
            time_idx=1,
            input_time=input_t2,
            dataset=self.sur_m2t1nxlnd_dataset,
            target_param_set=self.surface_vars,
            target_tensor=input_x)

        #m2t1nxrad
        self._set_5d_data(
            bounds=self.work_bounds,
            time_idx=0,
            input_time=input_t1,
            dataset=self.sur_m2t1nxrad_dataset,
            target_param_set=self.surface_vars,
            target_tensor=input_x)

        self._set_5d_data(
            bounds=self.work_bounds,
            time_idx=1,
            input_time=input_t2,
            dataset=self.sur_m2t1nxrad_dataset,
            target_param_set=self.surface_vars,
            target_tensor=input_x)
        #m2i3nvasm
        self._set_5d_data_plus_levels(
            bounds=self.work_bounds,
            time_idx=0,
            input_time=input_t1,
            dataset=self.ver_m2i3nvasm_dataset,
            target_tensor=input_x,
            target_param_set=self.vertical_vars,
            target_levels=self.levels,
            param_index_shift=len(self.surface_vars)
        )
        self._set_5d_data_plus_levels(
            bounds=self.work_bounds,
            time_idx=1,
            input_time=input_t2,
            dataset=self.ver_m2i3nvasm_dataset,
            target_tensor=input_x,
            target_param_set=self.vertical_vars,
            target_levels=self.levels,
            param_index_shift=len(self.surface_vars)
        )
        return input_x

    def _read_output_y(
            self,
            target_time:py_dt.datetime,
            )->torch.Tensor:
        output_y = torch.empty(
            1, self.input_x_channel,
            self.n_lat_pixels, 
            self.n_lon_pixels)
        # m2t1nxflx
        self._set_4d_data(
            bounds=self.work_bounds,
            input_time=target_time,
            dataset=self.sur_m2t1nxflx_dataset,
            target_param_set=self.surface_vars,
            target_tensor=output_y)

        # m2i1nxasm
        self._set_4d_data(
            bounds=self.work_bounds,
            input_time=target_time,
            dataset=self.sur_m2i1nxasm_dataset,
            target_param_set=self.surface_vars,
            target_tensor=output_y)

        #m2t1nxlnd
        self._set_4d_data(
            bounds=self.work_bounds,
            input_time=target_time,
            dataset=self.sur_m2t1nxlnd_dataset,
            target_param_set=self.surface_vars,
            target_tensor=output_y)

        #m2t1nxrad
        self._set_4d_data(
            bounds=self.work_bounds,
            input_time=target_time,
            dataset=self.sur_m2t1nxrad_dataset,
            target_param_set=self.surface_vars,
            target_tensor=output_y)

        #m2i3nvasm
        self._set_4d_data_plus_levels(
            bounds=self.work_bounds,
            input_time=target_time,
            dataset=self.ver_m2i3nvasm_dataset,
            target_tensor=output_y,
            target_param_set=self.vertical_vars,
            target_levels=self.levels,
            param_index_shift=len(self.surface_vars)
        )
        return output_y

    def _read_climate_x_data(
            self,
            input_time1:py_dt.datetime,
            input_time2:py_dt.datetime)->torch.Tensor:
        the_bbx1 = torchgeo_bbx(
            minx=self.work_bounds.minx,
            maxx=self.work_bounds.maxx,
            miny=self.work_bounds.miny,
            maxy=self.work_bounds.maxy,
            mint=input_time1.timestamp(),
            maxt=input_time1.timestamp())
        
        surf = self.sur_climatology_dataset.__getitem__(
            the_bbx1)
        n_input_channels = len(self.surface_vars) + len(self.vertical_vars) * len(self.levels)
        climate_shape = surf["image"][0].shape
        print("climate shape=", climate_shape)
        ret_data = torch.empty(1, 2, n_input_channels, climate_shape[0], climate_shape[1])

         # read-static-surface-variables
        self._set_5d_data(
            bounds=self.work_bounds,
            time_idx=0,
            input_time=input_time1,
            dataset=self.sur_climatology_dataset,
            target_param_set=self.surface_vars,
            target_tensor=ret_data)
        self._set_5d_data_plus_levels(
            bounds=self.work_bounds,
            time_idx=0,
            input_time=input_time1,
            dataset=self.ver_climatology_dataset,
            target_tensor=ret_data,
            target_param_set=self.vertical_vars,
            target_levels=self.levels,
            param_index_shift=len(self.surface_vars)
        )
        self._set_5d_data(
            bounds=self.work_bounds,
            time_idx=1,
            input_time=input_time2,
            dataset=self.sur_climatology_dataset,
            target_param_set=self.surface_vars,
            target_tensor=ret_data)
        self._set_5d_data_plus_levels(
            bounds=self.work_bounds,
            time_idx=1,
            input_time=input_time2,
            dataset=self.ver_climatology_dataset,
            target_tensor=ret_data,
            target_param_set=self.vertical_vars,
            target_levels=self.levels,
            param_index_shift=len(self.surface_vars)
        )
        return ret_data


    def _read_climate_y_data(
            self,
            input_time:py_dt.datetime)->torch.Tensor:
        the_bbx1 = torchgeo_bbx(
            minx=self.work_bounds.minx,
            maxx=self.work_bounds.maxx,
            miny=self.work_bounds.miny,
            maxy=self.work_bounds.maxy,
            mint=input_time.timestamp(),
            maxt=input_time.timestamp())
        
        surf = self.sur_climatology_dataset.__getitem__(
            the_bbx1)
        n_input_channels = len(self.surface_vars) + len(self.vertical_vars) * len(self.levels)
        climate_shape = surf["image"][0].shape
        print("climate shape=", climate_shape)
        ret_data = torch.empty(1, n_input_channels, climate_shape[0], climate_shape[1])

         # read-static-surface-variables
        self._set_4d_data(
            bounds=self.work_bounds,
            input_time=input_time,
            dataset=self.sur_climatology_dataset,
            target_param_set=self.surface_vars,
            target_tensor=ret_data)
        self._set_4d_data_plus_levels(
            bounds=self.work_bounds,
            input_time=input_time,
            dataset=self.ver_climatology_dataset,
            target_param_set=self.vertical_vars,
            target_levels=self.levels,
            target_tensor=ret_data,
            param_index_shift=len(self.surface_vars))
        return ret_data
        
    def _read_dem_data(
        self,
        bounds:torchgeo_bbx,
        input_time:py_dt.datetime,
        n_lat_pixels:int = None,
        n_lon_pixels:int = None,
        )->torch.Tensor:
        """
            Read static auxilary variables: 

            Returns:
                Tensor (batch, hr-prism-param, hr-lat, hr-lon)

        """
        the_bbx1 = torchgeo_bbx(
            minx=bounds.minx,
            maxx=bounds.maxx,
            miny=bounds.miny,
            maxy=bounds.maxy,
            mint=input_time.timestamp(),
            maxt=input_time.timestamp())
        
        dem = self.sta_prism_dataset.__getitem__(
            the_bbx1)
        hr_aux_channel = len(self.high_res_aux_vars)
        print("dem shape=", dem["image"].shape)
        dem_0_shape = dem["image"][0].shape
        if n_lat_pixels is None:
            n_lat_pixels = dem_0_shape[0]
        if n_lon_pixels is None:
            n_lon_pixels = dem_0_shape[1]
        #hr_aux = torch.empty(1, hr_aux_channel, dem_0_shape[0], dem_0_shape[1])
        hr_aux = torch.empty(
            1, hr_aux_channel,
            n_lat_pixels,
            n_lon_pixels)

         # read-static-surface-variables
        self._set_4d_data_with_resize(
            bounds=bounds,
            n_lat_pixels=n_lat_pixels,
            n_lon_pixels=n_lon_pixels,
            input_time=input_time,
            dataset=self.sta_prism_dataset,
            target_param_set=self.high_res_aux_vars,
            target_tensor=hr_aux)
        return hr_aux

    def _read_prism_data(
        self,
        bounds:torchgeo_bbx,
        input_time:py_dt.datetime,
        n_lat_pixels:int=None,
        n_lon_pixels:int=None,
        )->torch.Tensor:
        """
            Read high resolution data - variables at target resolution. 

            Args:
                bounds: Working BoundingBox (for spatial only)
                input_time: time to read (matching day)
                n_lat_pixels: The number of pixels along lat at high resolution.
                    It equals to the number of pixels along lat at low resolution multiplies
                    scale (a convenient scale to be upscaled with NN).                
                n_lon_pixels: The number of pixels along lon at high resolution.
                    It equals to the number of pixels along lon at low resolution multiplies
                    scale (a convenient scale to be upscaled with NN).
            Returns:
                Tensor (batch, hr-prism-param, hr-lat, hr-lon)

        """
        the_bbx1 = torchgeo_bbx(
            minx=bounds.minx,
            maxx=bounds.maxx,
            miny=bounds.miny,
            maxy=bounds.maxy,
            mint=input_time.timestamp(),
            maxt=input_time.timestamp())
        
        prism = self.sur_prism_dataset.__getitem__(
            the_bbx1)
        output_y_channel = len(self.prism_vars)
        print("prism shape=", prism["image"].shape)
        prism_0_shape = prism["image"][0].shape
        if n_lat_pixels is None:
            n_lat_pixels = prism_0_shape[0]
        if n_lon_pixels is None:
            n_lon_pixels = prism_0_shape[1]
        #output_y = torch.empty(1, output_y_channel, prism_0_shape[0], prism_0_shape[1])
        output_y = torch.empty(
            1, output_y_channel,
            n_lat_pixels,
            n_lon_pixels)
        print("output_y.shape=", output_y.shape)
         # read-static-surface-variables
        self._set_4d_data_with_resize(
            bounds=bounds,
            n_lat_pixels=n_lat_pixels,
            n_lon_pixels=n_lon_pixels,
            input_time=input_time,
            dataset=self.sur_prism_dataset,
            target_param_set=self.prism_vars,
            target_tensor=output_y)
        return output_y

    def _read_static_data(
        self,
        bounds:torchgeo_bbx,
        n_lat_pixels:int,
        n_lon_pixels:int,
        input_time:py_dt.datetime,
        position_encoding:str="absolute")->torch.Tensor:
        """
            Read static variables: 
            1. positiion encoding, 
            2. temporal encoding, 
            3. static surface variables

        month_of_year:int,        
        day_of_year:int,
        hour_of_day:int,
            Returns:
                Tensor (batch, static-param, lat, lon)

        """
        pos_static = self._position_encoding(
            bounds=bounds,
            n_latitudes=n_lat_pixels, 
            n_longitudes=n_lon_pixels,
            position_encoding=position_encoding)
 
        n_stat_position = len(pos_static)
        n_stat_time = 4
        n_stat_surface_vars = len(self.static_surface_vars)
        n_stat_all = n_stat_position + n_stat_time + n_stat_surface_vars
        ret_sta_data = torch.empty(
            1, n_stat_all, n_lat_pixels,n_lon_pixels)
        # position-encoding - 2 (lat, lon) or 3 (cos(lat),sin(lon),cos(lon))
        ret_sta_data[0, 0:n_stat_position] = pos_static
        # temporal-encoding - cos(doy), sin(doy), cos(hod), sin(hod)
        ret_sta_data[0, n_stat_position + 0] = math.cos(2 * torch.pi * input_time.day / 366) #torch.cos(2 * torch.pi * day_of_year / 366)
        ret_sta_data[0, n_stat_position + 1] = math.sin(2 * torch.pi * input_time.day / 366) #torch.sin(2 * torch.pi * day_of_year / 366)
        ret_sta_data[0, n_stat_position + 2] = math.cos(2 * torch.pi * input_time.hour / 24)
        ret_sta_data[0, n_stat_position + 3] = math.sin(2 * torch.pi * input_time.hour / 24)
        # read-static-surface-variables
        self._set_4d_data_plus_month(
            bounds=bounds,
            input_time=input_time,
            dataset=self.sta_m2c0nxctm_dataset,
            target_param_set=self.static_surface_vars,
            target_tensor=ret_sta_data,
            param_index_shift=(n_stat_position + n_stat_time))

        return ret_sta_data
        
    @py_ft.cache
    def _position_encoding(
            self,
            bounds:torchgeo_bbx,
            n_latitudes:int,
            n_longitudes:int,
            position_encoding:str="absolute"
            )->torch.Tensor:
        """
            Create position features.

            Returns:
                Tesnosr of dimensions (parameter, lat, lon)
                Parameter can be - (sin(lat), cos(lon), sin(lon)) if "absolute"
                    or (latitudes, longtitudes)

        """
        lat_res = (bounds.maxy - bounds.miny) / n_latitudes
        lon_res = (bounds.maxx - bounds.minx) / n_longitudes
        lats = torch.empty(n_latitudes)
        for k in range(n_latitudes):
            lats[k] = bounds.miny+lat_res*k+lat_res/2
        lons = torch.empty(n_longitudes)
        for k in range(n_longitudes):
            lons[k] = bounds.minx+lon_res*k+lon_res/2
        latitudes, longitudes = torch.meshgrid(
            lats, lons, indexing="ij")
        if position_encoding == "absolute":
            latitudes = latitudes / 360. * 2.0 * torch.pi
            longitudes = longitudes / 360. * 2.0 * torch.pi
            pos_static = torch.stack(
                (torch.sin(latitudes), 
                 torch.cos(longitudes), 
                 torch.sin(longitudes)), dim=0)
        else:
            pos_static = torch.stack(
                (latitudes / 360. * 2.0 * torch.pi, longitudes / 360. * 2.0 * torch.pi), dim=0)
        print("lats: lat0=", lats[0], " lat-n=", lats[n_latitudes-1], "lat_res=", lat_res)
        print("lons: lon0=", lons[0], " lon-n=", lons[n_longitudes-1], "lon_res=", lon_res)

        return pos_static

    def _set_4d_data(
            self,
            bounds:torchgeo_bbx,
            input_time: py_dt.datetime,
            dataset: RasterDataset,
            target_tensor: torch.Tensor,
            target_param_set:list[str],
            param_index_shift:int=0
            ):
        """
            Args:
                target_tensor: dimension (batch, parameter, lat, lon)
        """
        the_bbx = torchgeo_bbx(
            minx=bounds.minx,
            maxx=bounds.maxx,
            miny=bounds.miny,
            maxy=bounds.maxy,
            mint=input_time.timestamp(),
            maxt=input_time.timestamp())
        
        data = dataset.__getitem__(the_bbx)
        for bnd in dataset.all_bands:
            ds_bnd_idx = dataset.all_bands.index(bnd)
            if bnd in target_param_set:
                bnd_idx = target_param_set.index(bnd) + param_index_shift
                target_tensor[0,bnd_idx] = data["image"][ds_bnd_idx]
        
        return target_tensor

    def _set_4d_data_with_resize(
            self,
            bounds:torchgeo_bbx,
            n_lat_pixels:int,
            n_lon_pixels:int,
            input_time: py_dt.datetime,
            dataset: RasterDataset,
            target_tensor: torch.Tensor,
            target_param_set:list[str],
            param_index_shift:int=0
            ):
        """
            Args:
                target_tensor: dimension (batch, parameter, lat, lon)
        """
        the_bbx = torchgeo_bbx(
            minx=bounds.minx,
            maxx=bounds.maxx,
            miny=bounds.miny,
            maxy=bounds.maxy,
            mint=input_time.timestamp(),
            maxt=input_time.timestamp())
        
        output_size = (n_lat_pixels, n_lon_pixels)
        data = dataset.__getitem__(the_bbx)
        tensor_bchw = data['image'].unsqueeze(0)
        #print("before resize data['image'].shape=", tensor_bchw.shape)
        data_img = nn_F.interpolate(
            input=tensor_bchw,
            size=output_size,
            mode='nearest')
        #print("after resize=", data_img.shape)
        for bnd in dataset.all_bands:
            ds_bnd_idx = dataset.all_bands.index(bnd)
            if bnd in target_param_set:
                bnd_idx = target_param_set.index(bnd) + param_index_shift
                target_tensor[0,bnd_idx] = data_img[0,bnd_idx]
        
        return target_tensor


    def _read_hr_static_data(
        self,
        bounds:torchgeo_bbx,
        n_lat_pixels:int,
        n_lon_pixels:int,
        input_time:py_dt.datetime,
        position_encoding:str="absolute")->torch.Tensor:
        """
            Read static variables: 
            1. positiion encoding, 
            2. temporal encoding, 

        month_of_year:int,        
        day_of_year:int,
        hour_of_day:int, - this may be removed for 
            Returns:
                Tensor (batch, hr-static-param, lat, lon)

        """
        pos_static = self._position_encoding(
            bounds=bounds,
            n_latitudes=n_lat_pixels, 
            n_longitudes=n_lon_pixels,
            position_encoding=position_encoding)
 
        n_stat_position = len(pos_static)
        n_stat_time = 4
        n_stat_all = n_stat_position + n_stat_time
        ret_sta_data = torch.empty(
            1, n_stat_all, n_lat_pixels,n_lon_pixels)
        # position-encoding - 2 (lat, lon) or 3 (cos(lat),sin(lon),cos(lon))
        ret_sta_data[0, 0:n_stat_position] = pos_static
        # temporal-encoding - cos(doy), sin(doy), cos(hod), sin(hod)
        ret_sta_data[0, n_stat_position + 0] = math.cos(2 * torch.pi * input_time.day / 366) #torch.cos(2 * torch.pi * day_of_year / 366)
        ret_sta_data[0, n_stat_position + 1] = math.sin(2 * torch.pi * input_time.day / 366) #torch.sin(2 * torch.pi * day_of_year / 366)
        ret_sta_data[0, n_stat_position + 2] = math.cos(2 * torch.pi * input_time.hour / 24)
        ret_sta_data[0, n_stat_position + 3] = math.sin(2 * torch.pi * input_time.hour / 24)

        return ret_sta_data


    def _set_4d_data_plus_month(
            self,
            bounds:torchgeo_bbx,
            input_time: py_dt.datetime,
            dataset: RasterDataset,
            target_tensor: torch.Tensor,
            target_param_set:list[str],
            param_index_shift:int=0
            ):
        """
            Args:
                target_tensor: dimension (batch, parameter, lat, lon)
        """
        the_bbx = torchgeo_bbx(
            minx=bounds.minx,
            maxx=bounds.maxx,
            miny=bounds.miny,
            maxy=bounds.maxy,
            mint=input_time.timestamp(),
            maxt=input_time.timestamp())
        
        data = dataset.__getitem__(the_bbx)
        for bnd in dataset.all_bands:
            ds_bnd_idx = dataset.all_bands.index(bnd)*12+input_time.month-1
            if bnd in target_param_set:
                bnd_idx = target_param_set.index(bnd) + param_index_shift
                target_tensor[0,bnd_idx] = data["image"][ds_bnd_idx]
        
        return target_tensor

    def _set_5d_data_plus_levels(
            self,
            bounds:torchgeo_bbx,
            input_time: py_dt.datetime,
            time_idx: int,
            dataset: RasterDataset,
            target_tensor: torch.Tensor,
            target_param_set:list[str],
            target_levels:list[str|int|float]=None,
            param_index_shift:int=0
            ):
        """
            Args:
                input_x: target tensor with dim (batch, time, parameter, lat, lon)
        """
        the_bbx = torchgeo_bbx(
            minx=bounds.minx,
            maxx=bounds.maxx,
            miny=bounds.miny,
            maxy=bounds.maxy,
            mint=input_time.timestamp(),
            maxt=input_time.timestamp())
        
        data = dataset.__getitem__(the_bbx)
        src_levels = dataset.all_levels
        n_src_levels = len(src_levels)
        n_tgt_levels = len(target_levels)
        for bnd in dataset.all_bands:
            ds_bnd_idx = dataset.all_bands.index(bnd)*n_src_levels
            if bnd in target_param_set:
                tgt_bnd_idx = target_param_set.index(bnd)*n_tgt_levels + param_index_shift
                for tgt_level in target_levels:
                    tgt_lvl_idx = target_levels.index(tgt_level)
                    if tgt_level not in src_levels:
                        logger.error("Error- level %s not found in src", tgt_level)
                        continue
                    src_lvl_idx = src_levels.index(tgt_level)
                    src_lvl_data = data['image'][ds_bnd_idx+src_lvl_idx]
                    target_tensor[0,time_idx,tgt_bnd_idx+tgt_lvl_idx] = src_lvl_data
        
        return target_tensor


    def _set_4d_data_plus_levels(
            self,
            bounds:torchgeo_bbx,
            input_time: py_dt.datetime,
            dataset: RasterDataset,
            target_tensor: torch.Tensor,
            target_param_set:list[str],
            target_levels:list[str|int|float]=None,
            param_index_shift:int=0
            ):
        """
            Args:
                input_x: target tensor with dim (batch, time, parameter, lat, lon)
        """
        the_bbx = torchgeo_bbx(
            minx=bounds.minx,
            maxx=bounds.maxx,
            miny=bounds.miny,
            maxy=bounds.maxy,
            mint=input_time.timestamp(),
            maxt=input_time.timestamp())
        
        data = dataset.__getitem__(the_bbx)
        src_levels = dataset.all_levels
        n_src_levels = len(src_levels)
        n_tgt_levels = len(target_levels)
        for bnd in dataset.all_bands:
            ds_bnd_idx = dataset.all_bands.index(bnd)*n_src_levels
            if bnd in target_param_set:
                tgt_bnd_idx = target_param_set.index(bnd)*n_tgt_levels + param_index_shift
                for tgt_level in target_levels:
                    tgt_lvl_idx = target_levels.index(tgt_level)
                    if tgt_level not in src_levels:
                        logger.error("Error- level %s not found in src", tgt_level)
                        continue
                    src_lvl_idx = src_levels.index(tgt_level)
                    src_lvl_data = data['image'][ds_bnd_idx+src_lvl_idx]
                    target_tensor[0,tgt_bnd_idx+tgt_lvl_idx] = src_lvl_data
        
        return target_tensor

    def _set_5d_data(
            self,
            bounds:torchgeo_bbx,
            input_time: py_dt.datetime,
            time_idx: int,
            dataset: RasterDataset,
            target_tensor: torch.Tensor,
            target_param_set:list[str],
            param_index_shift:int=0
            ):
        """
            Args:
                input_x: target tensor with dim (batch, time, parameter, lat, lon)
        """
        the_bbx = torchgeo_bbx(
            minx=bounds.minx,
            maxx=bounds.maxx,
            miny=bounds.miny,
            maxy=bounds.maxy,
            mint=input_time.timestamp(),
            maxt=input_time.timestamp())
        
        data = dataset.__getitem__(the_bbx)
        for bnd in dataset.all_bands:
            ds_bnd_idx = dataset.all_bands.index(bnd)
            if bnd in target_param_set:
                bnd_idx = target_param_set.index(bnd) + param_index_shift
                target_tensor[0,time_idx,bnd_idx] = data["image"][ds_bnd_idx]
        
        return target_tensor


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        """
        x: [batch, time, parameter, lat, lon]
        y: [batch, parameter, lat, lon]
        static: [batch, channel_static, lat, lon]
        climate: [batch, parameter, lat, lon]
        input_time: [batch]
        lead_time: [batch]
        """
        return self._get_data(index)
        #return super().__getitem__(index)


class GmuMerra2ToPrism4kmSet(Dataset):
    component_elems=list()
    def __init__(self):
        super().__init__()
    
    def append(self, other:GmuMerra2ToPrism4km):
        self.component_elems.append(other)

    def __getitem__(self, index):
        
        _cmp, _index = self._get_component(index)
        return _cmp.__getitem__(_index)

    def _get_component(self, index)->tuple[GmuMerra2ToPrism4km, int]:
        _index = 0
        for _c in self.component_elems:
            for _ic in range(_c.__len__()):
                if _index == index:
                    return _c, _ic
                _index += 1
        return None, None
    
    def __len__(self):
        _len = 0
        for item in self.component_elems:
            _len += item.__len__()
        return _len

