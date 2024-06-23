import pandas as pd
from data_loader import DataLoader
from data_preprocessor import DataPreprocessor
from embeddings_generator import EmbeddingsGenerator
from sentiment_classifier import SentimentClassifier
from translator import Translator
from qa_model import QAModel
from summarizer import Summarizer
from model_evaluator import ModelEvaluator
from utils import logger


class NLPToolBoxPipeline:
    def __init__(self, data_path, models_path, processed_data_path):
        self.data_path = data_path
        self.models_path = models_path
        self.processed_data_path = processed_data_path

    def run_pipeline(self):
        # Step 1: Load Data
        data_loader = DataLoader(self.data_path)
        data_loader.load_data()

        # Step 2: Preprocess Data
        data_preprocessor = DataPreprocessor(data_loader.get_data())
        data_preprocessor.remove_unnecessary_columns(['Unnamed: 0.1', 'Unnamed: 0', 'Author_Name'])
        data_preprocessor.handle_missing_values()
        data_preprocessor.convert_date()
        data_preprocessor.combine_text_columns(['Vehicle_Title', 'Review_Title', 'Review'], 'Combined_Review')
        data_preprocessor.generate_binary_labels('Rating')
        preprocessed_data = data_preprocessor.get_preprocessed_data()

        # Step 3: Generate Embeddings
        embeddings_generator = EmbeddingsGenerator()
        embeddings = embeddings_generator.generate_embeddings(preprocessed_data['Combined_Review'].tolist())
        preprocessed_data['Embeddings'] = embeddings
        logger.info("Embeddings added to the preprocessed data")

        # Save processed data
        preprocessed_data.to_csv(f"{self.processed_data_path}/data.csv", index=False)
        logger.info("Preprocessed data saved successfully")

        # Step 4: Train and Evaluate Models
        sentiment_classifier = SentimentClassifier()
        sentiment_classifier.model.save_pretrained(f"{self.models_path}/sentiment_classifier")

        translator = Translator()
        translator.model.save_pretrained(f"{self.models_path}/translator")

        qa_model = QAModel()
        qa_model.qa_pipeline.model.save_pretrained(f"{self.models_path}/qa_model")

        summarizer = Summarizer()
        summarizer.summarizer.model.save_pretrained(f"{self.models_path}/summarizer")

        # Step 5: Evaluate Models
        model_evaluator = ModelEvaluator(sentiment_classifier, translator, qa_model, summarizer)
        texts = preprocessed_data['Combined_Review'].tolist()
        true_labels = preprocessed_data['Sentiment'].tolist()
        model_evaluator.evaluate_sentiment_classifier(texts, true_labels)

        text = "This car is very reliable and I love its performance."
        references = ["Este coche es muy fiable y me encanta su rendimiento."]
        model_evaluator.evaluate_translator(text, references)

        question = "What is the performance of the car?"
        context = "This car is very reliable and I love its performance. It has a powerful engine and smooth handling."
        true_answer = "I love its performance."
        model_evaluator.evaluate_qa_model(question, context, true_answer)

        summary_text = """
        The car is very reliable and offers great performance. It has a powerful engine and smooth handling.
        The interior is spacious and comfortable, making it perfect for long drives. The advanced safety features
        provide peace of mind, and the fuel efficiency is excellent for its class. Overall, this car is an
        excellent choice for anyone looking for a combination of performance, comfort, and safety.
        """
        true_summary = "The car is reliable with great performance, comfortable interior, advanced safety features, and excellent fuel efficiency."
        model_evaluator.evaluate_summarizer(summary_text, true_summary)
