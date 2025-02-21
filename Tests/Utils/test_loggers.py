"""
Tests for the loggers module.
"""

import logging
import SkiNet.Utils.loggers as loggers


def test_stdout_logging_default_params():  
    
    """
    Test that stdout_logging creates a StreamHandler, attaches it to the logger, and
    the log levels of the logger and of the handler are as per default log level
    """

    # reset the module-level variable stdout_logging_handler to None
    loggers.stdout_logging_handler = None
    # call the function under test using default log level
    loggers.stdout_logging()
    # assert the global variable is set after calling the function
    assert loggers.stdout_logging_handler is not None

    # Retrieeve the logger and check how the handlers are set
    logger = logging.getLogger()
    assert len(logger.handlers) == 1
    assert isinstance(logger.handlers[0], logging.StreamHandler)

     # Check that the logger's level is set to INFO
    assert logger.level == logging.INFO
    # Check that the attached handler's level is also set to INFO
    assert logger.handlers[0].level == logging.INFO



def test_stdout_logging_set_debug_log_level():
    """
    Test that calling stdout_logging(logging.DEBUG) sets the level to DEBUG.
    """
    loggers.stdout_logging_handler = None
    loggers.stdout_logging(logging.DEBUG)
    
    # Check that both the logger and the handler are set to DEBUG.
    assert logging.getLogger().level == logging.DEBUG
    assert logging.getLogger().handlers[0].level == logging.DEBUG

def test_stdout_logging_no_duplicate_handlers():
    """
    Ensure that calling stdout_logging multiple times doesn't add duplicate handlers.
    """
    loggers.stdout_logging_handler = None
    loggers.stdout_logging()
    initial_count = len(logging.getLogger().handlers)
    
    # Call the function again with a different log level.
    loggers.stdout_logging(logging.DEBUG)
    
    # There should still be only one handler attached.
    assert len(logging.getLogger().handlers) == initial_count


def test_file_logging_default_params(tmp_path):
    """
    Test that file_logging creates a FileHandler, attaches it to the logger, and 
    the log levels of the logger and of the handler are as per default log level
    """

    # create a temporary log file
    test_log_file = tmp_path / "test.log"

    # reset the module-level variable file_logging_handler to None
    loggers.file_logging_handler = None
    # reset stdout handler
    loggers.stdout_logging()

    # call the function under test using default log level
    loggers.file_logging(log_file=test_log_file)
    # assert the global variable is set after calling the function
    assert loggers.stdout_logging_handler is not None

    # Retrieeve the logger and check if file handler is there 
    logger = logging.getLogger()
    file_handlers = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
    assert len(file_handlers) == 1, "Expected one FileHandler to be attached"

     # Check that the logger's level is set to INFO
    assert logger.level == logging.INFO
    # Check that the attached handler's level is also set to INFO
    assert file_handlers[0].level == logging.INFO

    # Check that the file handler is writing to the correct file
    # handler.baseFilename contains the absolute path of the log file
    assert file_handlers[0].baseFilename == str(test_log_file.resolve())


def test_file_logging_default_path(tmp_path):
    """
    Test that file_logging, when called without specifying a file,
    uses the default log file path (LOG_FILE) defined in the module.
    """
    # reset the module-level variable file_logging_handler and stdout_logging_handler to None
    loggers.file_logging_handler = None
    loggers.stdout_logging_handler = None

    # reset stdout handler
    loggers.stdout_logging()

    # call file_logging without specifying a file; it should use LOG_FILE.
    loggers.file_logging()

    # Retrieeve the logger and its file handler (should be 1)
    logger = logging.getLogger()
    file_handlers = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
    assert len(file_handlers) == 1, "Expected one FileHandler to be attached"

    # Get the default log file from the module.
    default_log_file = loggers.LOG_FILE

    # Verify that the file handler's baseFilename matches the default log file's resolved path.
    assert file_handlers[0].baseFilename == str(default_log_file.resolve()), (
        f"Expected file handler to write to {default_log_file.resolve()}, but got {file_handlers[0].baseFilename}"
    )


def test_file_logging_uses_stdout_level(tmp_path):
    """
    Test that file_logging uses the stdout logging level if stdout_logging_handler is set.
    """
    # create a temporary log file
    test_log_file = tmp_path / "test.log"

    # reset the module-level variable file_logging_handler and stdout_logging_handler to None
    loggers.file_logging_handler = None
    loggers.stdout_logging_handler = None
    # reset stdout handler so that it has logging.DEBUG
    loggers.stdout_logging(log_level=logging.DEBUG)

    # call the function under test using default log level that is expected to be equal  to that of stdout logging
    loggers.file_logging(log_file=test_log_file)
    
    # Retrieeve the logger and check if file handler is there 
    logger = logging.getLogger()
    file_handlers = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
    
     # Check that the file logger's level is set to that of stdout logger
    assert file_handlers[0].level == logging.DEBUG


def test_file_logging_no_duplicate_handlers(tmp_path):
    """
    Test that calling file_logging multiple times does not add duplicate file handlers.
    """
    # Create a temporary log file
    test_log_file = tmp_path / "test_duplicate.log"
    
    # reset the module-level variable file_logging_handler to None
    loggers.file_logging_handler = None 

    # call file_logging for the 1st time
    loggers.file_logging(log_file=test_log_file)
    
    # Retrieeve the logger and check if file handler is there 
    logger = logging.getLogger()
    initial_file_handlers = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
    initial_count = len(initial_file_handlers)
    
    # Call file_logging a second time
    loggers.file_logging(log_file=test_log_file)
    
    # There should be no additional file handler added.
    file_handlers = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
    assert len(file_handlers) == initial_count, "Expected no duplicate FileHandler to be added"
