""" Module for custom decorators """
import resource
import inspect
import functools

from  gmudownscalingterratorch.datasets.prism import (
    PrismDownloadError,
    PrismUnknownElement,
    PrismZipfileExtractError
)

from gmudownscalingterratorch.datasets.merra2 import (
    Merra2UnknownElement,
    Merra2BandExtractError,
    Merra2DownloadError
)

def decorator_prism_exceptions_to_logger(logger):
    """
        try-catch-decorator-with-parameter for prism
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            #wrapper.__name__ = func.__name__
            try:
                return func(*args, **kwargs)
            except PrismDownloadError as e:
                logger.error("PrismDownloadError - %s", e)
                return False
            except PrismUnknownElement as e:
                logger.error("PrismUnknownElement - %s", e)
                return False
            except PrismZipfileExtractError as e:
                logger.error("PrismZipfileExtractError - %s", e)
                return False
            except Exception as e:
                logger.error("An unknown exceeption - %s", e)
                return False
        return wrapper
    return decorator


def decorator_prism_exceptions(func):
    """
        try-catch-decorator-with-parameter for prism
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        #wrapper.__name__ = func.__name__
        try:
            return func(*args, **kwargs)
        except PrismDownloadError as e:
            print(f"PrismDownloadError - {e}")
            return False
        except PrismUnknownElement as e:
            print(f"PrismUnknownElement - {e}")
            return False
        except PrismZipfileExtractError as e:
            print(f"PrismZipfileExtractError - {e}")
            return False
        except Exception as e:
            print(f"An unknown exceeption - {e}")
            return False
    return wrapper


def decorator_timeit_to_logger(logger):
    """
        time the funciton with output to a given logger
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            wrapper.__name__ = func.__name__
            start_usage = resource.getrusage(resource.RUSAGE_SELF)
            result = func(*args, **kwargs)
            end_usage = resource.getrusage(resource.RUSAGE_SELF)

            user_time = end_usage.ru_utime - start_usage.ru_utime
            system_time = end_usage.ru_stime - start_usage.ru_stime
            cpu_time = user_time + system_time

            currrent_frame = inspect.currentframe()
            function_name = currrent_frame.f_code.co_name

            logger.info("---%s---",decorator_timeit_to_logger.__name__)
            logger.info("---%s---", function_name)
            logger.info("---%s---", func.__name__)
            logger.info("User time: %s seconds", user_time)
            logger.info("System time: %s seconds", system_time)
            logger.info("CPU time: %s seconds", cpu_time)
            return result
        return wrapper
    return decorator

def decorator_timeit(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        wrapper.__name__ = func.__name__
        start_usage = resource.getrusage(resource.RUSAGE_SELF)
        result = func(*args, **kwargs)
        end_usage = resource.getrusage(resource.RUSAGE_SELF)

        user_time = end_usage.ru_utime - start_usage.ru_utime
        system_time = end_usage.ru_stime - start_usage.ru_stime
        cpu_time = user_time + system_time

        currrent_frame = inspect.currentframe()
        function_name = currrent_frame.f_code.co_name

        print(f"---{decorator_timeit_to_logger.__name__}---")
        print(f"---{function_name}---")
        print(f"---{func.__name__}---")
        print(f"User time: {user_time} seconds")
        print(f"System time: {system_time} seconds")
        print(f"CPU time: {cpu_time} seconds")
        return result
    return wrapper



def decorator_exceptions_to_logger(logger):
    """
        try-catch-decorator-with-parameter
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            #wrapper.__name__ = func.__name__
            try:
                return func(*args, **kwargs)
            except Merra2DownloadError as e:
                logger.error("Merra2DownloadError - %s", e)
                return False
            except Merra2UnknownElement as e:
                logger.error("Merra2UnknownElement - %s", e)
                return False
            except Merra2BandExtractError as e:
                logger.error("Merra2BandExtractError - %s", e)
                return False
            except PrismDownloadError as e:
                logger.error("PrismDownloadError - %s", e)
                return False
            except PrismUnknownElement as e:
                logger.error("PrismUnknownElement - %s", e)
                return False
            except PrismZipfileExtractError as e:
                logger.error("PrismZipfileExtractError - %s", e)
                return False
            except Exception as e:
                logger.error("An unknown exceeption - %s", e)
                return False
        return wrapper
    return decorator



def decorator_merra2_exceptions_to_logger(logger):
    """
        try-catch-decorator-with-parameter for merra2
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            #wrapper.__name__ = func.__name__
            try:
                return func(*args, **kwargs)
            except Merra2DownloadError as e:
                logger.error("Merra2DownloadError - %s", e)
                return False
            except Merra2UnknownElement as e:
                logger.error("Merra2UnknownElement - %s", e)
                return False
            except Merra2BandExtractError as e:
                logger.error("Merra2BandExtractError - %s", e)
                return False
            except Exception as e:
                logger.error("An unknown exceeption - %s", e)
                return False
        return wrapper
    return decorator


def decorator_merra2_exceptions(func):
    """
        try-catch-decorator-with-parameter for merra2
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        #wrapper.__name__ = func.__name__
        try:
            return func(*args, **kwargs)
        except Merra2DownloadError as e:
            print(f"Merra2DownloadError - {e}")
            return False
        except Merra2UnknownElement as e:
            print(f"Merra2UnknownElement - {e}")
            return False
        except Merra2BandExtractError as e:
            print(f"Merra2BandExtractError - {e}")
            return False
        except Exception as e:
            print(f"An unknown exceeption - {e}")
            return False
    return wrapper

