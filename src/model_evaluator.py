import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from utils import logger


class ModelEvaluator:
    def __init__(self, sentiment_classifier, translator, qa_model, summarizer):
        self.sentiment_classifier = sentiment_classifier
        self.translator = translator
        self.qa_model = qa_model
        self.summarizer = summarizer

    def evaluate_sentiment_classifier(self, texts, true_labels):
        predicted_labels = self.sentiment_classifier.classify_sentiments(texts)
        accuracy = accuracy_score(true_labels, predicted_labels)
        f1 = f1_score(true_labels, predicted_labels)
        cm = confusion_matrix(true_labels, predicted_labels)
        report = classification_report(true_labels, predicted_labels, target_names=['Negative', 'Positive'])

        logger.info(f"Sentiment Classifier Accuracy: {accuracy:.4f}")
        logger.info(f"Sentiment Classifier F1 Score: {f1:.4f}")
        logger.info(f"Sentiment Classifier Confusion Matrix:\n{cm}")
        logger.info(f"Sentiment Classifier Classification Report:\n{report}")

        # Plot confusion matrix
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'],
                    yticklabels=['Negative', 'Positive'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.show()

        return accuracy, f1, cm, report

    def evaluate_translator(self, text, references):
        translated_text = self.translator.translate(text)
        bleu_score = self.translator.calculate_bleu(translated_text, references)

        logger.info(f"Translated Text: {translated_text}")
        logger.info(f"BLEU Score: {bleu_score}")

        # Plot BLEU score
        plt.figure(figsize=(6, 4))
        plt.bar(['BLEU Score'], [bleu_score['bleu']])
        plt.ylim(0, 1)
        plt.ylabel('Score')
        plt.title('BLEU Score for Translation')
        plt.show()

        return translated_text, bleu_score

    def evaluate_qa_model(self, question, context, true_answer):
        predicted_answer = self.qa_model.get_answer(question, context)

        logger.info(f"Question: {question}")
        logger.info(f"Context: {context}")
        logger.info(f"True Answer: {true_answer}")
        logger.info(f"Predicted Answer: {predicted_answer}")

        # Print answers for comparison
        print(f"Question: {question}")
        print(f"Context: {context}")
        print(f"True Answer: {true_answer}")
        print(f"Predicted Answer: {predicted_answer}")

        return predicted_answer

    def evaluate_summarizer(self, text, true_summary):
        predicted_summary = self.summarizer.summarize(text)

        logger.info(f"Original Text: {text}")
        logger.info(f"True Summary: {true_summary}")
        logger.info(f"Predicted Summary: {predicted_summary}")

        # Print summaries for comparison
        print(f"Original Text: {text}")
        print(f"True Summary: {true_summary}")
        print(f"Predicted Summary: {predicted_summary}")

        return predicted_summary
