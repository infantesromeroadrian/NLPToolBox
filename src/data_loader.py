import pandas as pd
from utils import log_execution_time, logger


class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None

    @log_execution_time
    def load_data(self):
        try:
            self.data = pd.read_csv(self.file_path)
            logger.info(f"Data loaded successfully from {self.file_path}")
        except Exception as e:
            logger.error(f"Failed to load data from {self.file_path}: {e}")
            raise

    def get_data(self):
        return self.data

    @log_execution_time
    def info(self):
        try:
            info = self.data.info()
            logger.info("DataFrame info retrieved successfully")
            return info
        except Exception as e:
            logger.error(f"Failed to retrieve DataFrame info: {e}")
            raise

    @log_execution_time
    def describe(self):
        try:
            description = self.data.describe()
            logger.info("DataFrame description retrieved successfully")
            return description
        except Exception as e:
            logger.error(f"Failed to retrieve DataFrame description: {e}")
            raise

    @log_execution_time
    def check_missing_values(self):
        try:
            missing_values = self.data.isnull().sum()
            logger.info("Missing values check completed successfully")
            return missing_values
        except Exception as e:
            logger.error(f"Failed to check missing values: {e}")
            raise
