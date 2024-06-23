from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from utils import log_execution_time, logger

class SentimentClassifier:
    def __init__(self, model_name='nlptown/bert-base-multilingual-uncased-sentiment'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.classifier = pipeline('sentiment-analysis', model=self.model, tokenizer=self.tokenizer)

    def truncate_reviews(self, texts, max_length=512):
        truncated_texts = []
        for text in texts:
            inputs = self.tokenizer(text, truncation=True, max_length=max_length, return_tensors='pt')
            truncated_text = self.tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)
            truncated_texts.append(truncated_text)
        return truncated_texts

    @log_execution_time
    def classify_sentiments(self, texts):
        try:
            truncated_texts = self.truncate_reviews(texts)
            predictions = self.classifier(truncated_texts)
            predicted_labels = [1 if pred['label'] in ['4 stars', '5 stars'] else 0 for pred in predictions]
            return predicted_labels
        except Exception as e:
            logger.error(f"Error in classify_sentiments: {e}")
            raise

    @log_execution_time
    def evaluate_metrics(self, true_labels, predicted_labels):
        try:
            accuracy = accuracy_score(true_labels, predicted_labels)
            f1 = f1_score(true_labels, predicted_labels)
            return accuracy, f1
        except Exception as e:
            logger.error(f"Error in evaluate_metrics: {e}")
            raise