from transformers import MarianMTModel, MarianTokenizer
from datasets import load_metric
from functools import wraps
from utils import log_execution_time, logger

class Translator:
    def __init__(self, model_name='Helsinki-NLP/opus-mt-en-es'):
        self.tokenizer = MarianTokenizer.from_pretrained(model_name)
        self.model = MarianMTModel.from_pretrained(model_name)
        self.bleu = load_metric('bleu')

    @log_execution_time
    def translate(self, text):
        try:
            # Tokenizar y generar la traducción
            inputs = self.tokenizer.encode(text, return_tensors='pt', truncation=True)
            translated_tokens = self.model.generate(inputs)
            translated_text = self.tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
            return translated_text
        except Exception as e:
            logger.error(f"Error in translate: {e}")
            raise

    @log_execution_time
    def calculate_bleu(self, prediction, references):
        try:
            # Tokenización para el cálculo del BLEU score
            prediction_tokens = prediction.split()
            reference_tokens = [ref.split() for ref in references]
            bleu_score = self.bleu.compute(predictions=[prediction_tokens], references=[reference_tokens])
            return bleu_score
        except Exception as e:
            logger.error(f"Error in calculate_bleu: {e}")
            raise