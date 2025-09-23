import logging
import json
import sys

class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "message": record.getMessage(),
            "logger_name": record.name,
            "file": f"{record.filename}:{record.lineno}",
        }
        if record.exc_info:
            log_record["exc_info"] = self.formatException(record.exc_info)
        if record.stack_info:
            log_record["stack_info"] = self.formatStack(record.stack_info)
        
        # Add custom fields if present in record (e.g., from extra dict)
        for key, value in record.__dict__.items():
            if key not in ['name', 'levelname', 'pathname', 'filename', 'lineno', 'asctime',
                           'threadName', 'processName', 'message', 'args', 'exc_info',
                           'exc_text', 'stack_info', 'funcName', 'created', 'msecs', 'relativeCreated',
                           'thread', 'process', 'levelno', 'r_name', 'msg', 'module', 'pathname',
                           'funcName', 'lineno', 'created', 'msecs', 'relativeCreated', 'thread',
                           'threadName', 'processName', 'process', 'levelno', 'exc_text', 'stack_info',
                           'extra', 'taskName']: # Standard attributes to exclude
                if not key.startswith('_'): # Exclude internal attributes
                    log_record[key] = value

        return json.dumps(log_record)

def setup_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Remove existing handlers to avoid duplicate logs
    if logger.hasHandlers():
        logger.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    formatter = JsonFormatter(datefmt="%Y-%m-%dT%H:%M:%S%z")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger
