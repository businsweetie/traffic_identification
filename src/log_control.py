import os
import logging
import logging.handlers

class LogController:
    @staticmethod
    def initialize(log_directory_path: str, base_log_name: str, logging_level: int):
        if not os.path.exists(log_directory_path):
            os.makedirs(log_directory_path)
        full_path = os.path.join(log_directory_path, base_log_name)

        rolling_file_handler = logging.handlers.TimedRotatingFileHandler(full_path, 's', 60*60*24, utc=True)

        log_message_format = '%(asctime)-15s[%(levelname)s]: %(message)s'
        log_message_formatter = logging.Formatter(log_message_format)

        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging_level)
        stream_handler.setFormatter(log_message_formatter)

        rolling_file_handler.setFormatter(log_message_formatter)
        rolling_file_handler.setLevel(logging_level)

        logging.getLogger().setLevel(logging_level)

        logging.getLogger().addHandler(rolling_file_handler)
        logging.getLogger().addHandler(stream_handler)