import pandas as pd
import re
from utils import log_execution_time, logger


class DataPreprocessor:
    def __init__(self, data):
        self.data = data

    @log_execution_time
    def remove_unnecessary_columns(self, columns):
        try:
            self.data.drop(columns, axis=1, inplace=True)
            logger.info(f"Removed unnecessary columns: {columns}")
        except Exception as e:
            logger.error(f"Error removing columns {columns}: {e}")
            raise

    @log_execution_time
    def handle_missing_values(self):
        try:
            self.data['Review_Title'].fillna('No Title', inplace=True)
            self.data.dropna(subset=['Review'], inplace=True)
            logger.info("Handled missing values in 'Review_Title' and 'Review'")
        except Exception as e:
            logger.error(f"Error handling missing values: {e}")
            raise

    @log_execution_time
    def convert_date(self):
        try:
            self.data['Review_Date'] = pd.to_datetime(self.data['Review_Date'].str.extract(r'(\d{2}/\d{2}/\d{2})')[0],
                                                      format='%m/%d/%y', errors='coerce')
            logger.info("Converted 'Review_Date' to datetime")
        except Exception as e:
            logger.error(f"Error converting 'Review_Date': {e}")
            raise

    @log_execution_time
    def clean_text(self, text_column):
        try:
            self.data[text_column] = self.data[text_column].apply(lambda x: re.sub(r'[^\w\s]', '', x.lower()))
            self.data[text_column] = self.data[text_column].apply(lambda x: re.sub(r'\s+', ' ', x).strip())
            logger.info(f"Cleaned text in column: {text_column}")
        except Exception as e:
            logger.error(f"Error cleaning text in column {text_column}: {e}")
            raise

    @log_execution_time
    def generate_binary_labels(self, rating_column):
        try:
            self.data['Sentiment'] = self.data[rating_column].apply(lambda x: 1 if x >= 4 else 0)
            logger.info("Generated binary labels from 'Rating'")
            self.data.drop(columns=[rating_column], inplace=True)
            logger.info(f"Removed column: {rating_column}")
        except Exception as e:
            logger.error(f"Error generating binary labels from 'Rating': {e}")
            raise

    @log_execution_time
    def combine_text_columns(self, text_columns, new_column_name):
        try:
            self.data[new_column_name] = self.data[text_columns].apply(lambda row: ' '.join(row.values.astype(str)),
                                                                       axis=1)
            self.clean_text(new_column_name)
            self.data.drop(text_columns, axis=1, inplace=True)
            logger.info(f"Combined text columns {text_columns} into {new_column_name} and removed original columns")
        except Exception as e:
            logger.error(f"Error combining text columns {text_columns} into {new_column_name}: {e}")
            raise

    def get_preprocessed_data(self):
        return self.data
