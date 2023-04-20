import logging
from .utils import init_log

logger_name = 'av-env_modeling-motion_forecasting'

init_log('./logs/av-env_modeling-motion_forecasting.log',
         logger_name=logger_name)

logger = logging.getLogger(logger_name)