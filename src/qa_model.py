from transformers import pipeline
from utils import log_execution_time, logger

class QAModel:
    def __init__(self, model_name='deepset/minilm-uncased-squad2'):
        try:
            self.qa_pipeline = pipeline('question-answering', model=model_name, tokenizer=model_name)
            logger.info(f"Pipeline initialized with model: {model_name}")
        except Exception as e:
            logger.error(f"Error initializing pipeline: {e}")
            raise

    @log_execution_time
    def get_answer(self, question, context):
        try:
            result = self.qa_pipeline(question=question, context=context)
            answer = result.get('answer', 'No answer found')
            logger.info(f"Question: {question}, Answer: {answer}")
            return answer
        except Exception as e:
            logger.error(f"Error in get_answer: {e}")
            return "Error occurred while finding the answer"