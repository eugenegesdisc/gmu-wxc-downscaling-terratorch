import lightning as pl
import os
import logging
import yaml
#from torchgeo.datamodules import BaseDataModule
from torch.utils.data import DataLoader
from gmudownscalingterratorch.datasets.downscaling import (
    GmuMerra2ToPrism4km, GmuMerra2ToPrism4kmSet
)
import gmudownscalingterratorch.util
from gmudownscalingterratorch.util import (
    torch_seed_worker,
    torch_generator
)

import datetime as py_dt
from dateutil import parser as dt_parser

logger = logging.getLogger(__name__)

class GmuMerra2Downscale2Prism4kmDataModule(pl.LightningDataModule):
    """

    """
    def __init__(
            self,
            train_data_periods:list[str]=["2019-01-01; 2019-01-02T23:50:50", "2019-01-03"],
            validate_data_periods:list[str]=["2020-01-01; 2020-01-02T23:50:50", "2020-01-03"],
            test_data_periods:list[str]=["2021-01-01; 2021-01-02T23:50:50", "2021-01-03"],
            predict_data_periods:list[str]=["2022-01-01; 2022-01-02T23:50:50", "2020-01-03"],
            earthdata_login_config:str="~/.gmu_downscaling_earthdata_login.cfg",
            download:bool=False,
            extract:bool=False,
            convert:bool=False,
            use_climatology:bool=True,
            use_hr_auxiliary:bool=True,
            data_root_path:str="data",
            train_data_folder:str="train",
            validate_data_folder:str="validate",
            test_data_folder:str="test",
            predict_data_folder:str="predict",
            position_encoding:str="fourier",
            #prism_vars:list[str]=None,
            ):
        self.train_data_periods = train_data_periods
        self.validate_data_periods = validate_data_periods
        self.test_data_periods = test_data_periods
        self.predict_data_periods = predict_data_periods
        self.earthdata_login_config = earthdata_login_config
        self.download=download
        self.extract=extract
        self.convert=convert
        self.use_climatology=use_climatology
        self.use_hr_auxiliary=use_hr_auxiliary
        self.data_root_path = data_root_path
        self.train_data_folder = train_data_folder
        self.validate_data_folder = validate_data_folder
        self.test_data_folder = test_data_folder
        self.predict_data_folder = predict_data_folder
        self.position_encoding = position_encoding
        #self.prism_vars=prism_vars

        self.edl_token = None
        self.username = None
        self.password = None
        self._load_earthdata_authentication()
        self.dataset_train=None
        self.dataset_validate=None
        self.dataset_test=None
        self.dataset_predict=None

        super().__init__()
    
    def _load_earthdata_authentication(
            self):
        with open(self.earthdata_login_config, 'r') as file:
            data = yaml.safe_load(file)
            if "edl_token" in data:
                self.edl_token = data["edl_token"]
            if "username" in data:
                self.username = data["username"]
            if "password" in data:
                self.password = data["password"]
    
    def setup(self, stage):
        self.m2t1nxflx_paths  = os.path.join(
           self.data_root_path,"m2t1nxflx")
        self.m2t1nxflx_down_dir =  os.path.join(
           self.data_root_path,"m2t1nxflx",
           "download")
        self.m2i1nxasm_paths =  os.path.join(
           self.data_root_path,"m2i1nxasm")
        self.m2i1nxasm_down_dir =  os.path.join(
           self.data_root_path,"m2i1nxasm",
           "download")
        self.m2t1nxlnd_paths =  os.path.join(
           self.data_root_path,"m2t1nxlnd")
        self.m2t1nxlnd_down_dir =  os.path.join(
           self.data_root_path,"m2t1nxlnd",
           "download")
        self.m2t1nxrad_paths =  os.path.join(
           self.data_root_path,"m2t1nxrad")
        self.m2t1nxrad_down_dir =  os.path.join(
           self.data_root_path,"m2t1nxrad",
           "download")
        self.m2c0nxctm_paths =  os.path.join(
           self.data_root_path,"m2c0nxctm")
        self.m2c0nxctm_down_dir =  os.path.join(
           self.data_root_path,"m2c0nxctm",
           "download")
        self.m2i3nvasm_paths =  os.path.join(
           self.data_root_path,"m2i3nvasm")
        self.m2i3nvasm_down_dir =  os.path.join(
           self.data_root_path,"m2i3nvasm",
           "download")
        self.climate_surface_paths= os.path.join(
           self.data_root_path,"climatology",
           "surface")
        self.climate_surface_down_dir= os.path.join(
           self.data_root_path,"climatology",
           "download")
        self.climate_vertical_paths= os.path.join(
           self.data_root_path,"climatology3d",
           "vertical")
        self.climate_vertical_down_dir= os.path.join(
           self.data_root_path,"climatology3d",
           "download")
        self.prism_paths =  os.path.join(
           self.data_root_path,"dataprism4km")
        self.prism_down_dir =  os.path.join(
           self.data_root_path,"dataprism4km",
           "download")
        self.prism_static_paths =  os.path.join(
           self.data_root_path,"staticprism4km")
        self.prism_static_down_dir =  os.path.join(
           self.data_root_path,"staticprism4km",
           "download")
        if not os.path.exists(self.m2i1nxasm_paths):
            data_download_and_extract = True
        else:
            data_download_and_extract = False
        if data_download_and_extract:
            download = True
            extract = True
            convert = True
        else:
            download = self.download
            extract = self.extract
            convert = self.convert

        if (stage == "fit"):
            self.dataset_train = self._pre_download_and_extract_data(
                data_periods=self.train_data_periods,
                target_data_folder=self.train_data_folder,
                predict_mode=False,
                download=download,
                extract=extract,
                convert=convert)
        
        if (stage == "validate"):
            self.dataset_validate = self._pre_download_and_extract_data(
                data_periods=self.validate_data_periods,
                target_data_folder=self.validate_data_folder,
                predict_mode=False,
                download=download,
                extract=extract,
                convert=convert)

        if (stage == "test"):
            self.dataset_test = self._pre_download_and_extract_data(
                data_periods=self.test_data_periods,
                target_data_folder=self.test_data_folder,
                predict_mode=False,
                download=download,
                extract=extract,
                convert=convert)

        if (stage == "predict"):
            self.dataset_predict = self._pre_download_and_extract_data(
                data_periods=self.predict_data_periods,
                target_data_folder=self.predict_data_folder,
                predict_mode=True,
                download=download,
                extract=extract,
                convert=convert)
        #return super().setup(stage)
    
    def _pre_download_and_extract_data(
            self,
            data_periods:list[str],
            target_data_folder:str,
            predict_mode:bool=False,
            download:bool=True,
            extract:bool=True,
            convert:bool=True,
            ):
        ret_dataset = GmuMerra2ToPrism4kmSet()
        input_data_periods = list()
        for a_str in data_periods:
            if ";" in a_str:
                a_strs = a_str.split(";")
                if len(a_strs) != 2:
                    logger.warning("time period malformatted: '{a_str}'")
                start_str = a_strs[0].strip()
                end_str = a_strs[1].strip()
                input_data_periods.append((start_str, end_str))
            else:
                #if one date string, expand it to cover the whole day
                date_format = "%Y%m%d %H:%M:%S"
                c_date = dt_parser.isoparse(a_str)
                start_time = py_dt.datetime(
                    year=c_date.year,
                    month=c_date.month,
                    day=c_date.day)
                end_time = py_dt.datetime(
                    year=c_date.year,
                    month=c_date.month,
                    day=c_date.day,
                    hour=23,
                    minute=59,
                    second=59
                    )
                start_str = py_dt.datetime.strftime(start_time,date_format)
                end_str = py_dt.datetime.strftime(end_time,date_format)
                input_data_periods.append((start_str,end_str))
        
        for tp in input_data_periods:
            ret_data = GmuMerra2ToPrism4km(
                start_date=tp[0],
                end_date=tp[1],
                download=download,
                extract=extract,
                convert=convert,
                predict_mode=predict_mode,
                use_climatology=self.use_climatology,
                use_hr_auxiliary=self.use_hr_auxiliary,
                username=self.username,
                password=self.password,
                edl_token=self.edl_token,
                m2t1nxflx_paths = os.path.join(
                    self.m2t1nxflx_paths,
                    target_data_folder),
                m2t1nxflx_down_dir = self.m2t1nxflx_down_dir,
                m2i1nxasm_paths = os.path.join(
                    self.m2i1nxasm_paths,
                    target_data_folder),
                m2i1nxasm_down_dir = self.m2i1nxasm_down_dir,
                m2t1nxlnd_paths = os.path.join(
                    self.m2t1nxlnd_paths,
                    target_data_folder),
                m2t1nxlnd_down_dir = self.m2t1nxlnd_down_dir,
                m2t1nxrad_paths = os.path.join(
                    self.m2t1nxrad_paths,
                    target_data_folder),
                m2t1nxrad_down_dir = self.m2t1nxrad_down_dir,
                m2c0nxctm_paths = os.path.join(
                    self.m2c0nxctm_paths,
                    target_data_folder),
                m2c0nxctm_down_dir = self.m2c0nxctm_down_dir,
                m2i3nvasm_paths = os.path.join(
                    self.m2i3nvasm_paths,
                    target_data_folder),
                m2i3nvasm_down_dir = self.m2i3nvasm_down_dir,
                climate_surface_paths = self.climate_surface_paths,
                climate_surface_down_dir = self.climate_surface_down_dir,
                climate_vertical_paths = self.climate_vertical_paths,
                climate_vertical_down_dir = self.climate_vertical_down_dir,
                prism_paths = os.path.join(
                    self.prism_paths,
                    target_data_folder),
                prism_down_dir = self.prism_down_dir,
                prism_static_paths = os.path.join(
                    self.prism_static_paths,
                    target_data_folder),
                prism_static_down_dir = self.prism_static_down_dir,
                position_encoding=self.position_encoding,
                #prism_vars=self.prism_vars
            )
            ret_dataset.append(ret_data)
        return ret_dataset

    def train_dataloader(self):
        #return super().train_dataloader()
        if not self.dataset_train:
            self.setup("fit")
        return DataLoader(
            dataset=self.dataset_train,
            batch_size=1,
            worker_init_fn=torch_seed_worker,
            generator=torch_generator)
    
    def val_dataloader(self):
        #return super().val_dataloader()
        if not self.dataset_validate:
            self.setup("validate")
        return DataLoader(
            dataset=self.dataset_validate,
            batch_size=1,
            worker_init_fn=torch_seed_worker,
            generator=torch_generator)
    
    def predict_dataloader(self):
        #return super().predict_dataloader()
        if not self.dataset_predict:
            self.setup("predict")
        #return self.dataset_predict
        return DataLoader(
            dataset=self.dataset_predict,
            batch_size=1,
            worker_init_fn=torch_seed_worker,
            generator=torch_generator
        )
    
    def test_dataloader(self):
        #return super().test_dataloader()
        if not self.dataset_test:
            self.setup("test")
        #return self.dataset_test
        return DataLoader(
            dataset=self.dataset_test,
            batch_size=1,
            worker_init_fn=torch_seed_worker,
            generator=torch_generator)
    
    def teardown(self, stage):
        return super().teardown(stage)
