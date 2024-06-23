from pipeline import NLPToolBoxPipeline

def main():
    data_path = "/Users/adrianinfantes/Desktop/AIR/CollegeStudies/MachineLearningPath/DataCamp/Projects/CarReviews/data/RawData/CarReviews.csv"
    models_path = "/Users/adrianinfantes/Desktop/AIR/CollegeStudies/MachineLearningPath/DataCamp/Projects/CarReviews/models"
    processed_data_path = "/Users/adrianinfantes/Desktop/AIR/CollegeStudies/MachineLearningPath/DataCamp/Projects/CarReviews/data/ProcessedData"

    pipeline = NLPToolBoxPipeline(data_path, models_path, processed_data_path)
    pipeline.run_pipeline()

if __name__ == "__main__":
    main()
