"""
Custom exceptions for download, unzip PRISM files.
"""

class PrismUnknownElement(Exception):
    def __init__(
            self,
            elem:str,
            msg="Unsupported element in PRISM climate data repo"):
        self.elem = elem
        self.msg = msg
        super().__init__(self.msg)
    
    def __str__(self):
        return f'{self.elem} -> {self.msg}'

class PrismDownloadError(Exception):
    def __init__(
            self,
            nerrors:int,
            url_temp:str="",
            target_path:str="",
            msg="Failures occurred in the downloading of data from PRISM climate data repo."):
        self.nerrors = nerrors
        self.url_temp = url_temp
        self.target_download_path = target_path
        self.msg = msg
        super().__init__(self.msg)
    
    def __str__(self):
        return (f'{self.nerrors} errors during downloading files'
                f' from {self.url_temp} to {self.target_download_path} -> {self.msg}')
    
class PrismZipfileExtractError(Exception):
    def __init__(
            self,
            nerrors:int,
            filename_temp:str="",
            zip_source_path:str="",
            msg=("Failures occurred when extracting data"
                 " from downloaded PRISM climate data zipped files.")
            ):
        self.nerrors = nerrors
        self.filename_temp = filename_temp
        self.zip_source_path = zip_source_path
        self.msg = msg
        super().__init__(self.msg)
    
    def __str__(self):
        return (f'{self.nerrors} errors during extracting files'
                f' from {self.zip_source_path}/{self.filename_temp} -> {self.msg}')
