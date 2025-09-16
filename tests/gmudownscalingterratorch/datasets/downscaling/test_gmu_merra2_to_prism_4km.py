"""
Module for testing Gmu dataset
"""
import logging
import pytest
import torch
from dateutil import parser as dtu_parser
from torchgeo.datasets import BoundingBox
import random

from gmudownscalingterratorch.datasets.downscaling import (
    GmuMerra2ToPrism4km)

from gmudownscalingterratorch.util.decorators import (
    decorator_exceptions_to_logger,
    decorator_timeit_to_logger)

from gmudownscalingterratorch.util import (
    read_earthdata_authentication
)

logger = logging.getLogger(__name__)

@pytest.fixture(scope="session")
def gmu_downscaling_earthdata_login_file(pytestconfig):
    return pytestconfig.getoption("--earthdata-login-config")

class DecoTest2GmuMerra2ToPrism4km:
    """
    Test GmuMerra2ToPrism4km dataset
    """
    @decorator_timeit_to_logger(logger=logger)
    @decorator_exceptions_to_logger(logger=logger)
    def deco_test_download_and_extract(
        self,
        gmu_downscaling_earthdata_login_file)->bool:
        """
        Test download
        """
        my_ed_username, my_ed_password, my_ed_token = read_earthdata_authentication(
            earthdata_login_config=gmu_downscaling_earthdata_login_file)
        ds = GmuMerra2ToPrism4km(
            start_date="2000-12-31",
            end_date="2001-01-02T23:59:59",
            download=True,
            extract=True,
            convert=True,
            username=my_ed_username,
            password=my_ed_password,
            edl_token=my_ed_token,
            m2t1nxflx_paths = "../exp/m2t1nxflx/merra2",
            m2t1nxflx_down_dir = "../exp/m2t1nxflx/download",
            m2i1nxasm_paths = "../exp/m2i1nxasm/merra2",
            m2i1nxasm_down_dir = "../exp/m2i1nxasm/download",
            m2t1nxlnd_paths = "../exp/m2t1nxlnd/merra2",
            m2t1nxlnd_down_dir = "../exp/m2t1nxlnd/download",
            m2t1nxrad_paths = "../exp/m2t1nxrad/merra2",
            m2t1nxrad_down_dir = "../exp/m2t1nxrad/download",
            m2c0nxctm_paths = "../exp/m2c0nxctm/merra2",
            m2c0nxctm_down_dir = "../exp/m2c0nxctm/download",
            m2i3nvasm_paths = "../exp/m2i3nvasm/merra2",
            m2i3nvasm_down_dir = "../exp/m2i3nvasm/download",
            climate_surface_paths="../exp/climatology/surface",
            climate_surface_down_dir="../exp/climatology/download",
            climate_vertical_paths="../exp/climatology3d/vertical",
            climate_vertical_down_dir="../exp/climatology3d/download",
            prism_paths = "../exp/dataprism4km/merra2",
            prism_down_dir = "../exp/dataprism4km/download",
            prism_static_paths = "../exp/staticprism4km/prism",
            prism_static_down_dir = "../exp/staticprism4km/download",
            )
        if not ds:
            return False
        return True

    @decorator_timeit_to_logger(logger=logger)
    @decorator_exceptions_to_logger(logger=logger)
    def deco_test_get_len(
        self,
        gmu_downscaling_earthdata_login_file)->bool:
        """
        Test download
        """
        my_ed_username, my_ed_password, my_ed_token = read_earthdata_authentication(
            earthdata_login_config=gmu_downscaling_earthdata_login_file)
        ds = GmuMerra2ToPrism4km(
            start_date="2000-12-31T06:00:00",
            end_date="2001-01-02T23:59:59",
            download=False,
            extract=False,
            convert=False,
            use_hr_auxiliary=True,
            use_climatology=True,
            lead_times=[3],
            input_times=[-3],
            username=my_ed_username,
            password=my_ed_password,
            edl_token=my_ed_token,
            m2t1nxflx_paths = "../exp/m2t1nxflx/merra2",
            m2t1nxflx_down_dir = "../exp/m2t1nxflx/download",
            m2i1nxasm_paths = "../exp/m2i1nxasm/merra2",
            m2i1nxasm_down_dir = "../exp/m2i1nxasm/download",
            m2t1nxlnd_paths = "../exp/m2t1nxlnd/merra2",
            m2t1nxlnd_down_dir = "../exp/m2t1nxlnd/download",
            m2t1nxrad_paths = "../exp/m2t1nxrad/merra2",
            m2t1nxrad_down_dir = "../exp/m2t1nxrad/download",
            m2c0nxctm_paths = "../exp/m2c0nxctm/merra2",
            m2c0nxctm_down_dir = "../exp/m2c0nxctm/download",
            m2i3nvasm_paths = "../exp/m2i3nvasm/merra2",
            m2i3nvasm_down_dir = "../exp/m2i3nvasm/download",
            climate_surface_paths="../exp/climatology/surface",
            climate_surface_down_dir="../exp/climatology/download",
            climate_vertical_paths="../exp/climatology3d/vertical",
            climate_vertical_down_dir="../exp/climatology3d/download",
            prism_paths = "../exp/dataprism4km/merra2",
            prism_down_dir = "../exp/dataprism4km/download",
            prism_static_paths = "../exp/staticprism4km/prism",
            prism_static_down_dir = "../exp/staticprism4km/download",
            )
        print("len=", ds.__len__())
        print("samples=", ds.samples)
        if ds.__len__() < 1:
            return False
        return True

    @decorator_timeit_to_logger(logger=logger)
    @decorator_exceptions_to_logger(logger=logger)
    def deco_test_getitem(
        self,
        gmu_downscaling_earthdata_login_file)->bool:
        """
        Test super().__getitem__
        """
        my_ed_username, my_ed_password, my_ed_token = read_earthdata_authentication(
            earthdata_login_config=gmu_downscaling_earthdata_login_file)
        ds = GmuMerra2ToPrism4km(
            start_date="2000-12-31T06:00:00",
            end_date="2001-01-02T23:59:59",
            download=False,
            extract=False,
            convert=False,
            use_climatology=True,
            use_hr_auxiliary=True,
            lead_times=[3],
            input_times=[-3],
            username=my_ed_username,
            password=my_ed_password,
            edl_token=my_ed_token,
            m2t1nxflx_paths = "../exp/m2t1nxflx/merra2",
            m2t1nxflx_down_dir = "../exp/m2t1nxflx/download",
            m2i1nxasm_paths = "../exp/m2i1nxasm/merra2",
            m2i1nxasm_down_dir = "../exp/m2i1nxasm/download",
            m2t1nxlnd_paths = "../exp/m2t1nxlnd/merra2",
            m2t1nxlnd_down_dir = "../exp/m2t1nxlnd/download",
            m2t1nxrad_paths = "../exp/m2t1nxrad/merra2",
            m2t1nxrad_down_dir = "../exp/m2t1nxrad/download",
            m2c0nxctm_paths = "../exp/m2c0nxctm/merra2",
            m2c0nxctm_down_dir = "../exp/m2c0nxctm/download",
            m2i3nvasm_paths = "../exp/m2i3nvasm/merra2",
            m2i3nvasm_down_dir = "../exp/m2i3nvasm/download",
            climate_surface_paths="../exp/climatology/surface",
            climate_surface_down_dir="../exp/climatology/download",
            climate_vertical_paths="../exp/climatology3d/vertical",
            climate_vertical_down_dir="../exp/climatology3d/download",
            prism_paths = "../exp/dataprism4km/merra2",
            prism_down_dir = "../exp/dataprism4km/download",
            prism_static_paths = "../exp/staticprism4km/prism",
            prism_static_down_dir = "../exp/staticprism4km/download",
            )
        # non-extended boundingbox may miss one pixels
        n_item = random.choice(range(ds.__len__()))
        data = ds.__getitem__(n_item)
        print("keys=", data.keys())
        
        print("input_time.shape=",data["input_time"].shape)
        print("input_time=", data["input_time"])
        print("lead_time.shape=",data["lead_time"].shape)
        print("lead_time=", data["lead_time"])

        print("x-shape:", data["x"].shape)
        #print(data["x"][0, 0, 0, 30:40, 30:40]) # mon
        print("m2t1nxflx - Z0M:")
        self._show_x_elem(
            data_tensor=data["x"],
            param_idx=ds.get_param_index_in_x("Z0M"))
        print("m2i1nxasm - V10M:")
        self._show_x_elem(
            data_tensor=data["x"],
            param_idx=ds.get_param_index_in_x("V10M"))
        print("m2i1nxasm @t2 - V10M:")
        self._show_x_elem(
            data_tensor=data["x"],
            time_idx=1,
            param_idx=ds.get_param_index_in_x("V10M"))
        print("m2t1nxlnd - GWETROOT:")
        self._show_x_elem(
            data_tensor=data["x"],
            param_idx=ds.get_param_index_in_x("GWETROOT"))
        print("m2t1nxrad - SWGNT:")
        self._show_x_elem(
            data_tensor=data["x"],
            param_idx=ds.get_param_index_in_x("SWGNT"))

        print("m2i3nvasm - QI:")
        self._show_x_elem(
            data_tensor=data["x"],
            time_idx=0,
            param_idx=ds.get_param_index_in_x(
                param="QI",level=72.0))

        print("m2i3nvasm @t2 - V:")
        self._show_x_elem(
            data_tensor=data["x"],
            time_idx=1,
            param_idx=ds.get_param_index_in_x(
                param="V",level=72.0))

        print("static-shape:", data["static"].shape)
        print("static - position-encoding (lat or cos(lat))")
        self._show_4d_elem(
            data_tensor=data["static"],
            param_idx=0)
        print("static - temporal-encoding (sin(hod))")
        t_hod_idx = data["static"].shape[0]-len(ds.static_surface_vars)-1
        print("t_hod_idx = ", t_hod_idx)
        self._show_4d_elem(
            data_tensor=data["static"],
            param_idx=t_hod_idx)

        print("static - FRACI")
        t_param_idx = (data["static"].shape[0]
                       -len(ds.static_surface_vars)
                       +ds.static_surface_vars.index("FRACI"))
        print("t_param_idx=", t_param_idx)
        self._show_4d_elem(
            data_tensor=data["static"],
            param_idx=t_param_idx)

        print("cliamte_x=")
        if "climate_x" in data:
            print("climate-x-shape", data["climate_x"].shape)
            print("show sample:")
            y_param_idx = ds.get_param_index_in_x(
                param="T2M")
            start_x=int(data["climate_x"].shape[3]*1/4)
            start_y=int(data["climate_x"].shape[2]*1/4)
            end_x=int(data["climate_x"].shape[3]*1/4+10)
            end_y=int(data["climate_x"].shape[2]*1/4+5)
            print("showing range (",
                  start_x, end_x, start_y, end_y, ")")
            self._show_x_elem(
                data_tensor=data["climate_x"],
                time_idx = 0,
                param_idx = y_param_idx,
                start_x=start_x,
                start_y=start_y,
                end_x=end_x,
                end_y=end_y)

        print("climate(climate_y)=")
        if "climate" in data:
            print("climate-y-shape", data["climate"].shape)
            print("show sample:")
            y_param_idx = ds.get_param_index_in_x(
                param="T2M")
            start_x=int(data["climate"].shape[2]*1/4)
            start_y=int(data["climate"].shape[1]*1/4)
            end_x=int(data["climate"].shape[2]*1/4+10)
            end_y=int(data["climate"].shape[1]*1/4+5)
            print("showing range (",
                  start_x, end_x, start_y, end_y, ")")
            self._show_4d_elem(
                data_tensor=data["climate"],
                param_idx = y_param_idx,
                start_x=start_x,
                start_y=start_y,
                end_x=end_x,
                end_y=end_y)

        print("hr_prism = ")
        if "hr_prism" in data:
            print("prism-shape:", data["hr_prism"].shape)
            y_param_idx = ds.prism_vars.index("tmin")
            start_x=int(data["hr_prism"].shape[2]*1/4)
            start_y=int(data["hr_prism"].shape[1]*1/4)
            end_x=int(data["hr_prism"].shape[2]*1/4+10)
            end_y=int(data["hr_prism"].shape[1]*1/4+5)
            print("showing range (",
                  start_x, end_x, start_y, end_y, ")")
            self._show_4d_elem(
                data_tensor=data["hr_prism"],
                param_idx = y_param_idx,
                start_x=start_x,
                start_y=start_y,
                end_x=end_x,
                end_y=end_y)

        print("hr_prism_mask = ")
        if "hr_prism_mask" in data:
            print("hr-prism-mask-shape:", data["hr_prism_mask"].shape)
            y_param_idx = ds.prism_vars.index("tmin")
            start_x=int(data["hr_prism_mask"].shape[2]*1/4)
            start_y=int(data["hr_prism_mask"].shape[1]*1/4)
            end_x=int(data["hr_prism_mask"].shape[2]*1/4+10)
            end_y=int(data["hr_prism_mask"].shape[1]*1/4+5)
            print("showing range (",
                  start_x, end_x, start_y, end_y, ")")
            self._show_4d_elem(
                data_tensor=data["hr_prism_mask"],
                param_idx = y_param_idx,
                start_x=start_x,
                start_y=start_y,
                end_x=end_x,
                end_y=end_y)


        print("y-shape:", data["y"].shape)

        print("dem = ")
        if "hr_aux" in data:
            print("hr_aux-shape:", data["hr_aux"].shape)
            aux_param_idx = ds.high_res_aux_vars.index("dem")
            start_x=int(data["hr_aux"].shape[2]*1/4)
            start_y=int(data["hr_aux"].shape[1]*1/4)
            end_x=int(data["hr_aux"].shape[2]*1/4+10)
            end_y=int(data["hr_aux"].shape[1]*1/4+5)
            print("showing range (",
                  start_x, end_x, start_y, end_y, ")")
            self._show_4d_elem(
                data_tensor=data["hr_aux"],
                param_idx = aux_param_idx,
                start_x=start_x,
                start_y=start_y,
                end_x=end_x,
                end_y=end_y)

        print("hr_static = ")
        if "hr_static" in data:
            print("hr_static-shape:", data["hr_static"].shape)
            aux_param_idx = 1 # 
            start_x=int(data["hr_static"].shape[2]*1/4)
            start_y=int(data["hr_static"].shape[1]*1/4)
            end_x=int(data["hr_static"].shape[2]*1/4+10)
            end_y=int(data["hr_static"].shape[1]*1/4+5)
            print("showing range (",
                  start_x, end_x, start_y, end_y, ")")
            self._show_4d_elem(
                data_tensor=data["hr_static"],
                param_idx = aux_param_idx,
                start_x=start_x,
                start_y=start_y,
                end_x=end_x,
                end_y=end_y)

        if not data:
            return False
        return True
    def _show_x_elem(
        self,
        data_tensor:torch.Tensor,
        #batch_idx:int=0,
        time_idx:int=0,
        param_idx:int=0,
        start_y:int=30,
        end_y:int=35,
        start_x:int=30,
        end_x:int=40
        ):
        print(data_tensor[
            #batch_idx, 
            time_idx, param_idx, start_y:end_y, start_x:end_x]) # mon

    def _show_4d_elem(
        self,
        data_tensor:torch.Tensor,
        #batch_idx:int=0,
        param_idx:int=0,
        start_y:int=30,
        end_y:int=35,
        start_x:int=30,
        end_x:int=40
        ):
        print(data_tensor[
            #batch_idx, 
            param_idx, start_y:end_y, start_x:end_x]) # mon

#@pytest.mark.option1
class TestGmuMerra2ToPrism4km:
    """
    Test GmuMerra2ToPrism4km datasets
    """
    @pytest.mark.option1
    def test_download_and_extract(
        self,
        gmu_downscaling_earthdata_login_file):
        dec_cls = DecoTest2GmuMerra2ToPrism4km()
        result = dec_cls.deco_test_download_and_extract(gmu_downscaling_earthdata_login_file)
        assert result


    @pytest.mark.option1
    def test_get_len(
        self,
        gmu_downscaling_earthdata_login_file):
        """
            Test super().__len__
        """
        dec_cls = DecoTest2GmuMerra2ToPrism4km()
        result = dec_cls.deco_test_get_len(gmu_downscaling_earthdata_login_file)
        assert result


    #@pytest.mark.option1
    def test_getitem(
            self,
            gmu_downscaling_earthdata_login_file):
        """
            Test super().__len__
        """
        dec_cls = DecoTest2GmuMerra2ToPrism4km()
        result = dec_cls.deco_test_getitem(gmu_downscaling_earthdata_login_file)
        assert result
