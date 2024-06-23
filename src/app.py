import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import streamlit as st
import pandas as pd
from pipeline import NLPToolBoxPipeline
from sentiment_classifier import SentimentClassifier
from translator import Translator
from qa_model import QAModel
from summarizer import Summarizer

# Paths to the data and models
data_path = "/Users/adrianinfantes/Desktop/AIR/CollegeStudies/MachineLearningPath/DataCamp/Projects/NLPToolBox/data/CarReviews.csv"
models_path = "/Users/adrianinfantes/Desktop/AIR/CollegeStudies/MachineLearningPath/DataCamp/Projects/NLPToolBox/models"
processed_data_path = "/Users/adrianinfantes/Desktop/AIR/CollegeStudies/MachineLearningPath/DataCamp/Projects/NLPToolBox/data/ProcessedData"

# Initialize pipeline
pipeline = NLPToolBoxPipeline(data_path, models_path, processed_data_path)

# Streamlit app
st.set_page_config(page_title="NLP ToolBox", layout="wide", page_icon=":rocket:")

st.title("NLP ToolBox :rocket:")
st.markdown("""
Welcome to the NLP ToolBox! This app provides a comprehensive suite of Natural Language Processing tools including:
- **Sentiment Analysis**
- **Translation**
- **Question Answering**
- **Summarization**

Use the sidebar to navigate through different functionalities.
""")

# Sidebar navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Choose the app mode",
                                ["Run Pipeline", "Sentiment Analysis", "Translation", "Question Answering", "Summarization"])

if app_mode == "Run Pipeline":
    st.header("Run Pipeline")
    st.markdown("Click the button below to run the entire pipeline which includes data preprocessing, embeddings generation, and model evaluation.")
    if st.button("Run Pipeline"):
        with st.spinner('Running the pipeline...'):
            pipeline.run_pipeline()
        st.success("Pipeline executed successfully and models saved!")

elif app_mode == "Sentiment Analysis":
    st.header("Sentiment Analysis")
    st.markdown("Enter a car review to analyze its sentiment.")
    review_text = st.text_area("Car Review:", "This car is very reliable and I love its performance.")
    if st.button("Analyze Sentiment"):
        sentiment_classifier = SentimentClassifier()
        prediction = sentiment_classifier.classify_sentiments([review_text])[0]
        sentiment = "Positive" if prediction == 1 else "Negative"
        st.write(f"The sentiment of the review is: **{sentiment}**")
        st.balloons()

elif app_mode == "Translation":
    st.header("Translation")
    st.markdown("Enter text to translate it from English to Spanish.")
    text_to_translate = st.text_area("Text to Translate:", "This car is very reliable and I love its performance.")
    if st.button("Translate"):
        translator = Translator()
        translated_text = translator.translate(text_to_translate)
        st.write(f"Translated text: **{translated_text}**")
        st.balloons()

elif app_mode == "Question Answering":
    st.header("Question Answering")
    st.markdown("Enter a question and a context to get an answer.")
    question = st.text_input("Question:", "What does Transformers provide?")
    context = st.text_area("Context:", "Transformers provides thousands of pretrained models to perform tasks on different modalities such as text, vision, and audio.")
    if st.button("Get Answer"):
        qa_model = QAModel()
        answer = qa_model.get_answer(question, context)
        st.write(f"Answer: **{answer}**")
        st.balloons()

elif app_mode == "Summarization":
    st.header("Summarization")
    st.markdown("Enter text to summarize it.")
    text_to_summarize = st.text_area("Text to Summarize:", """
The car is very reliable and offers great performance. It has a powerful engine and smooth handling.
The interior is spacious and comfortable, making it perfect for long drives. The advanced safety features
provide peace of mind, and the fuel efficiency is excellent for its class. Overall, this car is an
excellent choice for anyone looking for a combination of performance, comfort, and safety.
""")
    if st.button("Summarize"):
        summarizer = Summarizer()
        summary = summarizer.summarize(text_to_summarize)
        st.write(f"Summary: **{summary}**")
        st.balloons()

st.sidebar.markdown("Created by Adrian Infantes")
st.sidebar.markdown("Data source: Car Reviews Dataset from Kaggle")
st.sidebar.markdown("Model source: Hugging Face Transformers")
st.sidebar.markdown("For more information, visit my [GitHub repository](https://github.com/adrianinfantes/CarReviewsAnalysis)")
