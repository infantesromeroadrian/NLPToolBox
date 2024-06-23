from transformers import pipeline
from utils import log_execution_time, logger


class Summarizer:
    def __init__(self, model_name='facebook/bart-large-cnn'):
        try:
            self.summarizer = pipeline('summarization', model=model_name)
            logger.info(f"Summarization pipeline initialized with model: {model_name}")
        except Exception as e:
            logger.error(f"Error initializing summarization pipeline: {e}")
            raise

    @log_execution_time
    def summarize(self, text, max_length=55, min_length=50):
        try:
            if len(text.split()) < min_length:
                logger.warning(f"Text too short for summarization. Minimum length required: {min_length} words.")
                return "Text too short to summarize effectively."

            summary = self.summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
            summarized_text = summary[0]['summary_text']
            logger.info(f"Text summarized successfully.")
            return summarized_text
        except Exception as e:
            logger.error(f"Error in summarize: {e}")
            return "Error occurred while summarizing the text."