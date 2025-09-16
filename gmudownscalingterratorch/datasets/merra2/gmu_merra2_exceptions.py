"""
Custom exceptions for download Merra2 data, extrat bands from Merra2 files.
"""

class Merra2UnknownElement(Exception):
    """
    Unknown element type. CUrrently, it only supports:
    M2I1NXASM: ("U10M", "V10M", "T2M", "QV2M", "PS", "SLP", "TS", "TQI", "TQL", 
                 "TQV")
    M2T1NXFLX: ("EFLUX", "HFLUX", "Z0M", "PRECTOT")
    M2T1NXLND: ("GWETROOT", "LAI")
    M2T1NXRAD: ("LWGEM", "LWGAB", "LWTUP", "SWGNT",
                 "SWTNT")
    M2I3NVASM: ("U", "V", "T", "QV", "OMEGA",
                 "PL", "H", "CLOUD", "QI", "QL")
    M2C0NXCTM: ("PHIS", "FRLAND","FROCEAN", "FRACI")
    """
    def __init__(
            self,
            elem:str,
            msg="Unsupported element in MERRA2 data repo"):
        self.elem = elem
        self.msg = msg
        super().__init__(self.msg)
    
    def __str__(self):
        return f'{self.elem} -> {self.msg}'

class Merra2DownloadError(Exception):
    """
    Any error occurred during the downloading process.
    """
    def __init__(
            self,
            nerrors:int,
            url_temp:str="",
            target_path:str="",
            msg="Failures occurred in the downloading of data from MERRA2 climate data repo."):
        self.nerrors = nerrors
        self.url_temp = url_temp
        self.target_download_path = target_path
        self.msg = msg
        super().__init__(self.msg)
    
    def __str__(self):
        return (f'{self.nerrors} errors during downloading files'
                f' from {self.url_temp} to {self.target_download_path} -> {self.msg}')
    
class Merra2BandExtractError(Exception):
    """
    Any error occurred during extracting bands from MERRA2 datasets
    """
    def __init__(
            self,
            nerrors:int,
            filename_temp:str="",
            source_path:str="",
            msg=("Failures occurred when extracting data"
                 " from downloaded MERRA2 multidimensioal dataset.")
            ):
        self.nerrors = nerrors
        self.filename_temp = filename_temp
        self.source_path = source_path
        self.msg = msg
        super().__init__(self.msg)
    
    def __str__(self):
        return (f'{self.nerrors} errors during extracting files'
                f' from {self.source_path}/{self.filename_temp} -> {self.msg}')
