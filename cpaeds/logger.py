import logging

class LoggerFactory(object):

    _LOG = None

    @staticmethod
    def __create_logger(log_file,  log_level):
        """
        A private method that interacts with the python
        logging module
        --------------------------------------------------
        log_file: Python module
            any kind of module, needing a logger initialization
        log_level: DEBUG, INFO, WARNING, ERROR, CRITICAL
            DEBUG: Detailed information, for diagnosing problems. Value=10.
            INFO: Confirm things are working as expected. Value=20.
            WARNING: Something unexpected happened, or indicative of some problem. But the software is still working as expected. Value=30.
            ERROR: More serious problem, the software is not able to perform some function. Value=40
            CRITICAL: A serious error, the program itself may be unable to continue running. Value=50
        """

        # set the logging format
        log_format = "%(asctime)s -  %(name)s - %(levelname)s - %(message)s" 
        # Initialize the class variable with logger object
        LoggerFactory._LOG = logging.getLogger(log_file)
        # setting basicConfig to info
        logging.basicConfig(level=logging.INFO, format=log_format, datefmt="%Y-%m-%d %H:%M:%S")
        
        # set the logging level based on the user selection
        if log_level == "DEBUG":
            LoggerFactory._LOG.setLevel(logging.DEBUG)
        elif log_level == "INFO":
            LoggerFactory._LOG.setLevel(logging.INFO)
        elif log_level == "WARNING":
            LoggerFactory._LOG.setLevel(logging.WARNING)
        elif log_level == "ERROR":
            LoggerFactory._LOG.setLevel(logging.ERROR)
        elif log_level == "CRITICAL":
            LoggerFactory._LOG.setLevel(logging.CRITICAL)

        return LoggerFactory._LOG

    @staticmethod
    def get_logger(log_file, log_level, file_name=None):
        """
        A static method called by other modules to initialize logger in
        their own module
        """
        logger = LoggerFactory.__create_logger(log_file, log_level)

        # Create log file for debugging runs 
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        if file_name and log_level=="DEBUG":
            fh = logging.FileHandler(file_name)
            fh.setFormatter(formatter)
            logger.addHandler(fh)
        
        # return the logger object
        return logger