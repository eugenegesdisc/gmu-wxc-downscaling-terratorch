import re
import os
import datetime
import logging
import pyproj
import glob
from osgeo import gdal

logger = logging.getLogger(__name__)

gdal.UseExceptions()

def gdal_convert_to_geotiff(
    src_dir:str,
    src_file_temp:str,
    target_dir:str,
    overwrite:bool=False)->bool:
    #---debug 1: checking the script's working directory---
    #print(f"DEBUG 1 for Unsupported format failure:current working directory:{os.getcwd()}")
    #---debug 2: environment variables---
    #print(f"DEBUG 2 for Unsupported format failure:GDAL_DATA env variable:{os.environ.get('GDAL_DATA')}")

    #---debug 3: add GDAL_DEBUG
    #os.environ["GDAL_DEBUG"]="ON"
    try:
        if src_dir:
            src_file_temp = os.path.join(src_dir,src_file_temp)
        src_files = glob.glob(src_file_temp)#, root_dir=src_dir)
        driver = gdal.GetDriverByName("GTiff")
        #gdal.GetDriverByName("EHdr").Register()
        #print(gdal.VersionInfo())
        for src_file in src_files:
            b_filename = os.path.splitext(os.path.basename(src_file))[0]
            target_file = os.path.join(target_dir, f"{b_filename}.tif")
            if ( os.path.exists(target_file) and
                not overwrite):
                continue
            #options = gdal.TranslateOptions(
            #    format="GTiff",
            #    creationOptions=["COMPRESS=DEFLATE"]
            #)
            src_ds = gdal.Open(src_file)
            if not src_ds:
                raise ValueError("Unsupported file format")
            dst_ds = driver.CreateCopy(
                target_file, src_ds,
                strict=0,
                options=["COMPRESS=DEFLATE"])
            src_ds = None
            dst_ds = None
            #gdal.Translate(
            #    destName=target_file,
            #    srcDS=src_file,
            #    options=options)
        return True
    except ValueError as e:
        return False
    except Exception as e:
        print(e)
        return False
    return False

def gdal_get_subdatasets(filename:str)->dict:
    sds = []
    with gdal.Open(filename) as ds:
        sds = ds.GetSubDatasets()
    if not sds:
        return None
    ret_dict = dict()
    for sd in sds:
        key = sd[0].rsplit(':', 1)[1]
        ret_dict[key] = sd
    return ret_dict


def gdal_get_elem_names(filename:str)->list:
    sds = []
    with gdal.Open(filename) as ds:
        sds = ds.GetSubDatasets()
    if not sds:
        return None
    ret_list = list()
    for sd in sds:
        key = sd[0].rsplit(':', 1)[1]
        ret_list.append(key)
    return ret_list

def gdal_get_subdataset_timebands(
        subdataset_name:str,
        metadata_time_dim:str="NETCDF_DIM_time_VALUES",
        start_datetime_pattern:str=r"(?P<date>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})",
        start_datetime_metadata_attr:str=r"time#units",
        start_datetime_format:str="%Y-%m-%d %H:%M:%S",
        start_datetime_pattern_in_sd_name:str=r"(?P<date>\d{4}\d{2}\d{2})",
        start_datetime_format_in_sd_name:str="%Y%m%d"
        )->list[tuple[int,str]]:
    """
        Retrieve the list of bands (temporal)

        Args:
            subdataset_name: A Gdal acceptable subdataset name.
            metadata_time_dim: The attribute name of time value in
                pattern like "{'t1', 't2', 't3', ...}", where t1, t2, 
                t3, ..., are integer representing start in minutes 
                counting from start_datetime
            start_datetime_pattern: A regular expression to extract 
                start datetime from metadata as group <date>.
            start_datetime_metadata_attr: The attribute name in metadata
                for the start datetime. If None or "", start_datetime will
                attempted using subdataset_name.
            start_datetime_format: A datetime format string for converting 
                from the datetime string extracted from metadata.
            start_datetime_pattern_in_sd_name: A regular expression to extract 
                start datetime from subdataset_name as group <date>
            start_datetime_format_in_sd_name: A datetime format string for 
                converting the datetime string extracted from subdataset_name
        
        Returns:
            A list of temporal start and end tuples to be usable directly 
            in band name as "<start_datetime>_to_<end_datetime>". None is returned if
            any error occurrs.
    """
    ret_list = list()
    with gdal.Open(subdataset_name) as sds:
        if metadata_time_dim not in sds.GetMetadata():
            logger.error("No time field '%s' found in neither subdataset-name '%s'",
                            metadata_time_dim,
                            subdataset_name)
            return None          
        time_array = sds.GetMetadata()[metadata_time_dim][1:][:-1].split(",")
        time_values = list(map(int, time_array))
        start_datetime = None
        if (start_datetime_metadata_attr and 
            start_datetime_metadata_attr in sds.GetMetadata()):
            time_metadata_str = sds.GetMetadata()[start_datetime_metadata_attr]
            time_match = re.search(
                pattern=start_datetime_pattern,
                string=time_metadata_str)
            if time_match:
                start_datetime = datetime.datetime.strptime(
                    time_match.group("date"), start_datetime_format)
        if not start_datetime:
            time_match2 = re.search(
                pattern=start_datetime_pattern_in_sd_name,
                string=subdataset_name)
            if not time_match2:
                logger.error("No start datetime found in neither subdataset-name '%s'"
                             " nor metadata",
                             subdataset_name)
                return None
            start_datetime = datetime.datetime.strptime(
                time_match2.group("date"), start_datetime_format_in_sd_name)
        #add_minutes = lambda mm: start_datetime + datetime.timedelta(minutes=mm)
        time_values = list(map(
            lambda mm: start_datetime + datetime.timedelta(minutes=mm),
            time_values))
        # get time interval - assuming datetime_values have more than two elements
        # and their intevals are equal
        time_invterv = time_values[1] - time_values[0] - datetime.timedelta(seconds=1)
        out_format = '%Y%m%dT%H%M%S'
        interv_time_str = lambda x: (x.strftime(out_format), (
            x+time_invterv).strftime(out_format))
        ret_list = list(map(
            lambda x: (x.strftime(out_format), (
                x+time_invterv).strftime(out_format)),
            time_values))
    return ret_list

def gdal_get_3d_subdataset_level_bands(
        subdataset_name:str,
        metadata_lev_dim:str="NETCDF_DIM_lev_VALUES"
        )->list[float]:
    """
        Retrieve the list of bands (temporal)

        Args:
            subdataset_name: A Gdal acceptable subdataset name.
            metadata_lev_dim: The attribute name of level value in
                pattern like "{'t1', 't2', 't3', ...}", where t1, t2, 
                t3, ..., are integer representing pressure levels in
                most cases.
        
        Returns:
            A list of integers reprsensting the pressure levels. None is returned if
            any error occurrs.
    """
    ret_list = list()
    with gdal.Open(subdataset_name) as sds:
        if metadata_lev_dim not in sds.GetMetadata():
            logger.error("No time field '%s' found in neither subdataset-name '%s'",
                            metadata_lev_dim,
                            subdataset_name)
            return None            
        lev_array = sds.GetMetadata()[metadata_lev_dim][1:][:-1].split(",")
        ret_list = list(map(float, lev_array))
    return ret_list


def gdal_export_const_subdataset_to_separate_geotiff(
        subdataset_name:str,
        elem_name:str,
        target_basename:str="",
        target_dir:str=".",
        overwrite:bool=False,
        )->bool:

    # create target directory
    os.makedirs(target_dir,exist_ok=True)

    outfile_name = (target_basename +"_sd_"+elem_name+".tif")
    output_raster_path = os.path.join(
        target_dir,outfile_name)
    if (os.path.exists(output_raster_path) and
        (not overwrite)):
        logger.info("Target file %s exixsts. Not overwritten.",
                    output_raster_path)
        return True
    driver = gdal.GetDriverByName("GTiff")
    creation_options = [
        "COMPRESS=ZSTD", # Or "COMPRESS=DEFLATE", "COMPRESS=LZW", "COMPRESS=PACKBITS", "COMPRESS=ZSTD", "COMPRESS=JPEG", "COMPRESS=LERC"
        #"PREDICTOR=2", # For integer data
        "PREDICTOR=3", # For floating point data
        # "JPEG_QUALITY=90" # If using JPEG compression
        #"ZSTD_LEVEL=9" # If using ZSTD compression
        # "MAX_Z_ERROR=0.001" # If using LERC compression
        ]
    with gdal.Open(subdataset_name, gdal.GA_ReadOnly) as sds:
        nbands = sds.RasterCount
        geotransform = sds.GetGeoTransform()
        projection = sds.GetProjection()
        if not projection:
            projection = pyproj.CRS.from_epsg(4326).to_wkt()
        cols = sds.RasterXSize
        rows = sds.RasterYSize
        output_dataset = driver.Create(
            output_raster_path,
            cols,
            rows,
            nbands,
            gdal.GDT_Float32,
            creation_options)
        output_dataset.SetGeoTransform(geotransform)
        output_dataset.SetProjection(projection)
        for nb in range(nbands):
            band = sds.GetRasterBand(nb+1)
            band_array = band.ReadAsArray()
            output_band = output_dataset.GetRasterBand(nb+1)
            output_band.WriteArray(band_array)
        output_dataset = None
    return True

def gdal_export_3d_subdatasets_to_separate_geotiffs_with_fixes(
        src_filepath:str,
        elem_names:list[str]|None=None,
        levels:list[float]|None=None,
        target_prefix:str="",
        target_suffix:str="",
        target_dir:str=".",
        overwrite:bool=False
        )->tuple[int,int]:
    if elem_names is None:
        elem_names = gdal_get_elem_names(src_filepath)
    ret_errors = 0
    ret_extracts = 0

    for elem in elem_names:
        outfile_full_path = os.path.join(
            target_dir,
            f"{target_prefix}{elem}{target_suffix}")
        if (os.path.exists(outfile_full_path) and
            (not overwrite)):
            logger.info("Target file %s exixsts. Not overwritten.",
                        outfile_full_path)
            continue
        subdataset_name = f'NETCDF:"{src_filepath}":{elem}'
        if gdal_export_3d_subdataset_to_separate_geotiff_with_fixes(
            subdataset_name=subdataset_name,
            out_tiff_filepath=outfile_full_path,
            levels=levels):
            ret_extracts += 1
        else:
            ret_errors -= 1
    return (ret_extracts, ret_errors)


def gdal_export_subdatasets_to_separate_geotiffs_with_fixes(
        src_filepath:str,
        elem_names:list[str]|None=None,
        target_prefix:str="",
        target_suffix:str="",
        target_dir:str=".",
        overwrite:bool=False
        )->tuple[int,int]:
    if elem_names is None:
        elem_names = gdal_get_elem_names(src_filepath)
    ret_errors = 0
    ret_extracts = 0

    for elem in elem_names:
        outfile_full_path = os.path.join(
            target_dir,
            f"{target_prefix}{elem}{target_suffix}")
        if (os.path.exists(outfile_full_path) and
            (not overwrite)):
            logger.info("Target file %s exixsts. Not overwritten.",
                        outfile_full_path)
            continue
        subdataset_name = f'NETCDF:"{src_filepath}":{elem}'
        if gdal_export_subdataset_to_separate_geotiff_with_fixes(
            subdataset_name=subdataset_name,
            out_tiff_filepath=outfile_full_path):
            ret_extracts += 1
        else:
            ret_errors -= 1
    return (ret_extracts, ret_errors)

def gdal_export_subdataset_to_separate_geotiff_with_fixes(
        subdataset_name:str,
        out_tiff_filepath:str)->bool:
    driver = gdal.GetDriverByName("GTiff")
    creation_options = [
        "COMPRESS=ZSTD", # Or "COMPRESS=DEFLATE", "COMPRESS=LZW", "COMPRESS=PACKBITS", "COMPRESS=ZSTD", "COMPRESS=JPEG", "COMPRESS=LERC"
        #"PREDICTOR=2", # For integer data
        "PREDICTOR=3", # For floating point data
        # "JPEG_QUALITY=90" # If using JPEG compression
        #"ZSTD_LEVEL=9" # If using ZSTD compression
        # "MAX_Z_ERROR=0.001" # If using LERC compression
        ]
    with gdal.Open(subdataset_name, gdal.GA_ReadOnly) as sds:
        nbands = sds.RasterCount
        if (nbands != 1):
            logger.warning("This function was intended for surface band which should have only 1 band. Get %d bands", nbands)
        geotransform = sds.GetGeoTransform()
        projection = sds.GetProjection()
        if not projection:
            projection = pyproj.CRS.from_epsg(4326).to_wkt()
        cols = sds.RasterXSize
        rows = sds.RasterYSize
        output_dataset = driver.Create(
            out_tiff_filepath,
            cols,
            rows,
            nbands,
            gdal.GDT_Float32,
            creation_options)
        output_dataset.SetGeoTransform(geotransform)
        output_dataset.SetProjection(projection)
        for nb in range(nbands):
            band = sds.GetRasterBand(nb+1)
            band_array = band.ReadAsArray()
            output_band = output_dataset.GetRasterBand(nb+1)
            output_band.WriteArray(band_array)
        output_dataset = None
    return True

def _valid_levels(
        level_list:list[float],
        available_level_list:list[float]):
    if level_list is None:
        return True
    for l_it in level_list:
        if l_it not in available_level_list:
            return False
    return True

def gdal_export_3d_subdataset_to_separate_geotiff_with_fixes(
        subdataset_name:str,
        out_tiff_filepath:str,
        levels:list[float]|None=None,
        metadata_3rd_dim:str="NETCDF_DIM_lev_VALUES",
        )->bool:
    pres_levels = gdal_get_3d_subdataset_level_bands(
        subdataset_name=subdataset_name,
        metadata_lev_dim=metadata_3rd_dim
        )
    if not levels:
        levels=pres_levels
    if not _valid_levels(
        level_list=levels,
        available_level_list=pres_levels):
        logger.error("Not all levels available: %s", levels)
        return False
    n_out_levels = len(levels)
    driver = gdal.GetDriverByName("GTiff")
    creation_options = [
        "COMPRESS=ZSTD", # Or "COMPRESS=DEFLATE", "COMPRESS=LZW", "COMPRESS=PACKBITS", "COMPRESS=ZSTD", "COMPRESS=JPEG", "COMPRESS=LERC"
        #"PREDICTOR=2", # For integer data
        "PREDICTOR=3", # For floating point data
        # "JPEG_QUALITY=90" # If using JPEG compression
        #"ZSTD_LEVEL=9" # If using ZSTD compression
        # "MAX_Z_ERROR=0.001" # If using LERC compression
        ]
    with gdal.Open(subdataset_name, gdal.GA_ReadOnly) as sds:
        nbands = sds.RasterCount # this should be idenditcal to its number of levels
        assert(nbands==len(pres_levels))
        assert(n_out_levels<=nbands)
        geotransform = sds.GetGeoTransform()
        projection = sds.GetProjection()
        if not projection:
            projection = pyproj.CRS.from_epsg(4326).to_wkt()
        cols = sds.RasterXSize
        rows = sds.RasterYSize
        output_dataset = driver.Create(
            out_tiff_filepath,
            cols,
            rows,
            n_out_levels,
            gdal.GDT_Float32,
            creation_options)
        output_dataset.SetGeoTransform(geotransform)
        output_dataset.SetProjection(projection)
        output_dataset.SetMetadata({"levels":levels})        
        for nb in range(n_out_levels):
            nb_old = pres_levels.index(levels[nb])
            band = sds.GetRasterBand(nb_old+1)
            band_array = band.ReadAsArray()
            output_band = output_dataset.GetRasterBand(nb+1)
            output_band.WriteArray(band_array)
        output_dataset = None
    return True


def gdal_export_subdataset_to_separate_geotiff(
        subdataset_name:str,
        elem_name:str,
        target_basename:str="",
        target_dir:str=".",
        overwrite:bool=False,
        metadata_time_dim:str="NETCDF_DIM_time_VALUES",
        start_datetime_pattern:str=r"(?P<date>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})",
        start_datetime_metadata_attr:str=r"time#units",
        start_datetime_format:str="%Y-%m-%d %H:%M:%S",
        start_datetime_pattern_in_sd_name:str=r"(?P<date>\d{4}\d{2}\d{2})",
        start_datetime_format_in_sd_name:str="%Y%m%d"
        )->bool:
    time_bands = gdal_get_subdataset_timebands(
        subdataset_name=subdataset_name,
        metadata_time_dim=metadata_time_dim,
        start_datetime_pattern=start_datetime_pattern,
        start_datetime_metadata_attr=start_datetime_metadata_attr,
        start_datetime_format=start_datetime_format,
        start_datetime_pattern_in_sd_name=start_datetime_pattern_in_sd_name,
        start_datetime_format_in_sd_name=start_datetime_format_in_sd_name
        )
    if not time_bands:
        logger.error("No time bands found in this subdataset %s",
                     subdataset_name)
        return False
    # create target directory
    os.makedirs(target_dir,exist_ok=True)

    with gdal.Open(subdataset_name, gdal.GA_ReadOnly) as sds:
        nbands = sds.RasterCount
        if len(time_bands) != nbands:
            logger.error("Band number not match %d vs %d",
                         nbands, len(time_bands))
            return False
        for nb in range(nbands):
            band = sds.GetRasterBand(nb+1)
            geotransform = sds.GetGeoTransform()
            projection = sds.GetProjection()
            if not projection:
                projection = pyproj.CRS.from_epsg(4326).to_wkt()
            cols = sds.RasterXSize
            rows = sds.RasterYSize
            band_array = band.ReadAsArray()
            outfile_name = (target_basename +"_sd_"+elem_name+"_"
                            +time_bands[nb][0]+"_to_"+time_bands[nb][1]+".tif")
            output_raster_path = os.path.join(
                target_dir,outfile_name)
            if (os.path.exists(output_raster_path) and
                (not overwrite)):
                logger.info("Target file %s exixsts. Not overwritten.",
                            output_raster_path)
                continue
            driver = gdal.GetDriverByName("GTiff")
            creation_options = [
                "COMPRESS=ZSTD", # Or "COMPRESS=DEFLATE", "COMPRESS=LZW", "COMPRESS=PACKBITS", "COMPRESS=ZSTD", "COMPRESS=JPEG", "COMPRESS=LERC"
                #"PREDICTOR=2", # For integer data
                "PREDICTOR=3", # For floating point data
                # "JPEG_QUALITY=90" # If using JPEG compression
                #"ZSTD_LEVEL=9" # If using ZSTD compression
                # "MAX_Z_ERROR=0.001" # If using LERC compression
                ]
            output_dataset = driver.Create(
                output_raster_path,
                cols,
                rows,
                1,
                gdal.GDT_Float32,
                creation_options)
            output_dataset.SetGeoTransform(geotransform)
            output_dataset.SetProjection(projection)
            output_band = output_dataset.GetRasterBand(1)
            output_band.WriteArray(band_array)
            output_dataset = None
    return True

def gdal_export_3d_subdataset_to_separate_geotiff(
        subdataset_name:str,
        elem_name:str,
        levels:list[float]|None = None,
        target_basename:str="",
        target_dir:str=".",
        overwrite:bool=False,
        metadata_time_dim:str="NETCDF_DIM_time_VALUES",
        metadata_3rd_dim:str="NETCDF_DIM_lev_VALUES",
        start_datetime_pattern:str=r"(?P<date>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})",
        start_datetime_metadata_attr:str=r"time#units",
        start_datetime_format:str="%Y-%m-%d %H:%M:%S",
        start_datetime_pattern_in_sd_name:str=r"(?P<date>\d{4}\d{2}\d{2})",
        start_datetime_format_in_sd_name:str="%Y%m%d"
        )->bool:
    """
        Original: [subdataset, time, pressure, lon, lat]
    """
    time_bands = gdal_get_subdataset_timebands(
        subdataset_name=subdataset_name,
        metadata_time_dim=metadata_time_dim,
        start_datetime_pattern=start_datetime_pattern,
        start_datetime_metadata_attr=start_datetime_metadata_attr,
        start_datetime_format=start_datetime_format,
        start_datetime_pattern_in_sd_name=start_datetime_pattern_in_sd_name,
        start_datetime_format_in_sd_name=start_datetime_format_in_sd_name
        )
    if not time_bands:
        logger.error("No time bands found in this subdataset %s",
                     subdataset_name)
        return False
    n_timebands = len(time_bands)
    pres_levels = gdal_get_3d_subdataset_level_bands(
        subdataset_name=subdataset_name,
        metadata_lev_dim=metadata_3rd_dim
        )
    n_3dlevels = len(pres_levels) # Total levels in netcdf file

    if levels is None:
        levels = pres_levels

    #if not pres_levels:
    #    logger.warning("No levels '%s' found in this subdataset %s",
    #                   metadata_3rd_dim,
    #                   subdataset_name)
    #    n_3dlevels = 0
    #else:
    #    n_3dlevels = len(pres_levels)
    #if n_3dlevels == 0:
    #    logger.warning("Number of levels returned zero for %s. "
    #                   "It will be assuming equivalently 1 level.",
    #                   subdataset_name)
    #    n_3dlevels = 1
    n_out_levels = len(levels)
    # create target directory
    os.makedirs(target_dir,exist_ok=True)
    gdal.UseExceptions()

    with gdal.Open(subdataset_name, gdal.GA_ReadOnly) as sds:
        nbands = sds.RasterCount
        if (n_timebands*n_3dlevels) != nbands:
            logger.error("Band number not match %d vs %d * %d",
                         nbands, n_timebands, n_3dlevels)
            return False
        assert((n_out_levels*n_timebands)<=nbands)
        for tm in range(n_timebands):
            nb0 = tm*n_3dlevels
            band = sds.GetRasterBand(nb0+1)
            geotransform = sds.GetGeoTransform()
            projection = sds.GetProjection()
            if not projection:
                projection = pyproj.CRS.from_epsg(4326).to_wkt()
            cols = sds.RasterXSize
            rows = sds.RasterYSize
            outfile_name = (target_basename +"_sd_"+elem_name+"_"
                            +time_bands[tm][0]+"_to_"+time_bands[tm][1]+".tif")
            output_raster_path = os.path.join(
                target_dir,outfile_name)
            if (os.path.exists(output_raster_path) and
                (not overwrite)):
                logger.info("Target file %s exixsts. Not overwritten.",
                            output_raster_path)
                continue
            creation_options = [
                "COMPRESS=ZSTD", # Or "COMPRESS=DEFLATE", "COMPRESS=LZW", "COMPRESS=PACKBITS", "COMPRESS=ZSTD", "COMPRESS=JPEG", "COMPRESS=LERC"
                #"PREDICTOR=2", # For integer data
                "PREDICTOR=3", # For floating point data
                # "JPEG_QUALITY=90" # If using JPEG compression
                #"ZSTD_LEVEL=9" # If using ZSTD compression
                # "MAX_Z_ERROR=0.001" # If using LERC compression
                ]
            gdriver=gdal.GetDriverByName("GTiff")
            if gdriver is None:
                print("ERRROR")
            output_dataset = gdriver.Create(
                output_raster_path,
                cols,
                rows,
                n_out_levels,
                gdal.GDT_Float32,
                creation_options)
            #if pres_levels:
            #    output_dataset.SetMetadata({"levels":pres_levels})
            output_dataset.SetMetadata({"levels":levels})
            output_dataset.SetGeoTransform(geotransform)
            output_dataset.SetProjection(projection)

            for nb in range(n_out_levels):
                nb_old = pres_levels.index(levels[nb])
                band = sds.GetRasterBand(nb0+nb_old+1)
                band_array = band.ReadAsArray()
                output_band = output_dataset.GetRasterBand(nb+1)
                output_band.WriteArray(band_array)
            output_dataset = None
    return True

