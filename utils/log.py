import sys
from datetime import datetime
from loguru import logger as mylogger
from pathlib import Path
import pytz
import os

# 定义一个函数用于格式化时间，设置为东八区
def format_time(record):
    tz = pytz.timezone('Asia/Shanghai')
    now = datetime.now(tz)
    record["extra"]["formatted_time"] = now.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    return "{extra[formatted_time]} | {level: <8} | {message}\n"

def get_logger(file_path,print_level="DEBUG", logfile_level="DEBUG"):
    mylogger.remove()
    mylogger.add(sys.stderr, level=print_level, format=format_time)
    mylogger.add(file_path, level=logfile_level, format=format_time)
    return mylogger

if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parent # 表示当前文件夹
    file_path = os.path.join(PROJECT_ROOT,'log.txt')

    main_logger = get_logger(file_path)
    
    main_logger.info("Starting application")
    main_logger.debug("Debug message")
    main_logger.warning("Warning message")
    main_logger.error("Error message")
    main_logger.critical("Critical message")
